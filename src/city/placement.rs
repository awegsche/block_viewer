//! Ghost preview and validity (ticket 047, roadmap E3): the B2 mesh of the
//! currently selected building, rotated per B3, shown translucent at the
//! hovered tile and tinted by whether it could actually be placed there.
//!
//! ## No build menu yet
//!
//! G1 (roadmap group G) is what will eventually let a player click a
//! catalogue entry. It doesn't exist yet, so [`PlacementSelection`] is
//! driven by a small keyboard stand-in ([`cycle_selection`]): number keys
//! `1`-`9` pick the *n*th [`BuildingCatalogue`](blueprint::BuildingCatalogue)
//! entry (sorted by id, for a mapping that doesn't depend on `HashMap`
//! iteration order), `R` rotates 90° clockwise, `Escape` clears the
//! selection. This is explicitly not G1 — it's the least that lets this
//! ticket (and E4 after it) be exercised and demoed before a real menu
//! exists, the same role W8's paint command played for the write path
//! before any UI did.
//!
//! ## Manual height (ticket 048)
//!
//! [`PlacementSelection::y_offset`] shifts a placement up or down from
//! [`grid::fit_footprint`]'s own auto-fit height — `Page Up`/`Page Down` step
//! it by one block, `Home` resets it to zero, all in [`cycle_selection`]
//! alongside the rest of the keyboard stand-in. Not the mouse wheel: Rts's
//! camera already owns scroll for zoom (ticket 045), and every zoom would
//! otherwise also nudge the height. The offset resets to zero on `Escape`
//! and whenever a new catalogue entry is picked, so a fresh selection always
//! starts at the terrain's own fit rather than wherever the last one was
//! left. [`resolve_placement`] is where it's applied — see that function's
//! docs.
//!
//! ## Validity is two answers, ANDed
//!
//! `city::grid::fit_footprint` (E2, terrain) and `state::City::is_tile_free`
//! (D1, occupancy) each own one question about a footprint. This module
//! doesn't reinvent either — [`resolve_placement`] just combines both
//! answers into the one green/red signal the ghost needs. A refused fit
//! still produces a ghost (red, at the hovered block's height) rather than
//! nothing, so an unbuildable spot reads as "no" instead of "the game didn't
//! notice I moved the cursor here."
//!
//! ## One entity, updated in place
//!
//! The ghost is spawned once ([`GhostState::entity`]) and its
//! `Mesh3d`/`MeshMaterial3d`/`Transform`/`Visibility` are overwritten every
//! frame rather than despawned and respawned — a held selection shouldn't
//! churn entities 60 times a second.
//!
//! ## Caching, not per-frame work
//!
//! [`GhostState::meshes`] caches a `Handle<Mesh>` per `(catalogue id,
//! Rotation)` — [`blueprint::rotate_blueprint`] and [`blueprint::mesh_blueprint`]
//! only run once per combination actually visited, not once per frame at
//! the cursor. A rotation the palette can't support
//! ([`blueprint::RotationError::UnrotatableProperty`]) is cached as `None`
//! right alongside successful entries, logged once rather than every frame
//! the player leaves that combination selected — see [`ghost_mesh`].
//! [`GhostState::materials`] is the same idea for the two translucent
//! materials themselves: built once, off whatever texture
//! [`chunk_pipeline::TerrainMaterial`] is currently using, not reallocated
//! every frame.
//!
//! `Rotation` needed a `Hash` impl to be a cache key at all, and
//! `PlacementSelection` needed a starting rotation before the player rotates
//! anything — both landed directly on `blueprint::rotate::Rotation` (see its
//! doc comment) rather than a mirror type here.

use std::collections::HashMap;

use bevy::prelude::*;

use crate::blueprint::{self, BuildingCatalogue, CatalogueEntry, Rotation};
use crate::camera;
use crate::chunk_pipeline::{SharedAtlasIndex, SharedColorMaps, TerrainMaterial};
use crate::world::{self, BiomeColors};
use crate::DecodedWorld;

use super::definition::BuildingDefinitions;
use super::grid::{self, FootprintFit};
use super::picking::{HoveredBlock, PickingSet};
use super::state;
use super::tool::ActiveTool;

/// What building is selected to place, at what rotation, and how far its
/// height has been nudged from the terrain's own auto-fit — see the module
/// docs' "No build menu yet" and ticket 048's "Manual height" for how these
/// get set today.
#[derive(Resource, Debug, Default, Clone, PartialEq, Eq)]
pub struct PlacementSelection {
    pub catalogue_id: Option<String>,
    /// Which *definition* (`assets/city/buildings/<id>.ron`) the selection
    /// came from, when it came from one at all — the build menu sets it,
    /// [`cycle_selection`]'s keyboard stand-in leaves it `None`.
    ///
    /// [`catalogue_id`](Self::catalogue_id) alone can't stand in for it: a
    /// `.ron` names the `.nbt` it uses, and the two stems are free to
    /// differ, so a blueprint doesn't identify the game data placed with it.
    /// Ticket 073 needs the definition to charge the placement's `cost`; a
    /// selection with no definition behind it has no cost, no requirements
    /// and no production, and is placed free.
    pub definition_id: Option<String>,
    pub rotation: Rotation,
    /// Added to whichever Y [`resolve_placement`] would otherwise have used
    /// — [`grid::fit_footprint`]'s `base_y` on a fit, the hovered block's
    /// height on a refusal. Positive raises the placement, negative sinks
    /// it. Zero (the default) is the terrain's own auto-fit, untouched.
    pub y_offset: i32,
}

/// Marks the one ghost preview entity — see the module docs' "One entity,
/// updated in place".
#[derive(Component)]
struct GhostPreview;

/// The two translucent materials the ghost swaps between, built once (see
/// [`ensure_materials`]) off whatever texture [`TerrainMaterial`] is
/// currently using.
struct GhostMaterials {
    valid: Handle<StandardMaterial>,
    invalid: Handle<StandardMaterial>,
}

/// Everything [`update_ghost_preview`] keeps between frames: the spawned
/// entity (once it exists), the lazily-built materials, and the mesh cache
/// keyed by `(catalogue id, Rotation)` — `None` for a combination that
/// failed to rotate, so the failure itself is cached and not retried every
/// frame (see the module docs).
#[derive(Resource, Default)]
struct GhostState {
    entity: Option<Entity>,
    materials: Option<GhostMaterials>,
    meshes: HashMap<(String, Rotation), Option<Handle<Mesh>>>,
}

/// What [`update_ghost_preview`] should do to the ghost entity this frame —
/// computed by [`resolve_ghost`] first, then applied, so the decision logic
/// stays callable from a test without an `App`/`Commands` in the loop.
enum GhostUpdate {
    Hidden,
    Shown { mesh: Handle<Mesh>, material: Handle<StandardMaterial>, transform: Transform },
}

pub struct PlacementPlugin;

impl Plugin for PlacementPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<PlacementSelection>()
            .init_resource::<GhostState>()
            // After `picking::PickingSet` so this reads *this* frame's
            // `HoveredBlock`, not last frame's — see that set's docs.
            .add_systems(Update, (cycle_selection, update_ghost_preview).chain().after(PickingSet));
    }
}

/// The keyboard interim for G1's build menu — see the module docs.
const NUMBER_KEYS: [KeyCode; 9] = [
    KeyCode::Digit1,
    KeyCode::Digit2,
    KeyCode::Digit3,
    KeyCode::Digit4,
    KeyCode::Digit5,
    KeyCode::Digit6,
    KeyCode::Digit7,
    KeyCode::Digit8,
    KeyCode::Digit9,
];

/// Catalogue ids in a stable order — `BuildingCatalogue::iter` walks a
/// `HashMap`, whose order isn't something a numbered key mapping can depend
/// on.
fn sorted_catalogue_ids(catalogue: &BuildingCatalogue) -> Vec<&str> {
    let mut ids: Vec<&str> = catalogue.iter().map(|entry| entry.id.as_str()).collect();
    ids.sort_unstable();
    ids
}

/// This rotation after one 90° clockwise turn — `Rotation` itself has no
/// public "next" step (`blueprint::rotate`'s own turn count is private,
/// internal to its geometry/property remap), so the small four-way match
/// lives here instead.
fn rotate_clockwise(rotation: Rotation) -> Rotation {
    match rotation {
        Rotation::Deg0 => Rotation::Deg90,
        Rotation::Deg90 => Rotation::Deg180,
        Rotation::Deg180 => Rotation::Deg270,
        Rotation::Deg270 => Rotation::Deg0,
    }
}

/// `Rotation` as a degree figure for display — same reason [`rotate_clockwise`]
/// exists next to it: `blueprint::rotate::Rotation` has no public numeric
/// view of itself, `quarter_turns` being private to that module's own
/// geometry remap. `pub(super)`: `city::ui::build_menu` (ticket 050, roadmap
/// G1) is the one other caller, for the "Rotation: 90°" line next to the
/// build menu's selected entry.
pub(super) fn rotation_degrees(rotation: Rotation) -> u16 {
    match rotation {
        Rotation::Deg0 => 0,
        Rotation::Deg90 => 90,
        Rotation::Deg180 => 180,
        Rotation::Deg270 => 270,
    }
}

/// Drives [`PlacementSelection`] off the keyboard — see the module docs'
/// "No build menu yet".
fn cycle_selection(
    keys: Res<ButtonInput<KeyCode>>,
    egui_input: Res<camera::EguiInputCapture>,
    catalogue: Option<Res<BuildingCatalogue>>,
    mut selection: ResMut<PlacementSelection>,
    mut tool: Option<ResMut<ActiveTool>>,
) {
    // Same guard `camera.rs`'s own input systems use — a keystroke egui is
    // already handling (typing into a panel, say) shouldn't also drive the
    // game underneath it.
    if egui_input.keyboard {
        return;
    }

    if keys.just_pressed(KeyCode::Escape) {
        selection.catalogue_id = None;
        selection.definition_id = None;
        selection.y_offset = 0;
        // Ticket 083: clearing a placement is one of the two ways back to
        // the resting state — see `tool`'s own module docs.
        if let Some(tool) = tool.as_deref_mut() {
            *tool = ActiveTool::Inspect;
        }
    }
    if keys.just_pressed(KeyCode::KeyR) {
        selection.rotation = rotate_clockwise(selection.rotation);
    }
    // Ticket 048's height keys — see the module docs' "Manual height". Not
    // gated on a selection existing: harmless either way, and simpler than
    // special-casing "no building picked yet."
    if keys.just_pressed(KeyCode::PageUp) {
        selection.y_offset = selection.y_offset.saturating_add(1);
    }
    if keys.just_pressed(KeyCode::PageDown) {
        selection.y_offset = selection.y_offset.saturating_sub(1);
    }
    if keys.just_pressed(KeyCode::Home) {
        selection.y_offset = 0;
    }

    let Some(catalogue) = catalogue else { return };
    let ids = sorted_catalogue_ids(&catalogue);
    for (index, key) in NUMBER_KEYS.iter().enumerate() {
        if keys.just_pressed(*key) {
            if let Some(&id) = ids.get(index) {
                selection.catalogue_id = Some(id.to_string());
                // A catalogue entry picked straight off the keyboard has no
                // definition behind it — see `definition_id`'s own docs.
                selection.definition_id = None;
                selection.y_offset = 0;
            }
        }
    }
}

/// A plain-white fallback for the (practically unreachable — `BiomeRegistry::new`
/// always interns plains at id 0) case where the biome tint table has
/// nothing at [`world::BiomeRegistry::PLAINS`].
fn white_biome() -> BiomeColors {
    BiomeColors { grass: LinearRgba::WHITE, foliage: LinearRgba::WHITE, water: LinearRgba::WHITE }
}

/// [`mesh_blueprint`](blueprint::mesh_blueprint)'s biome argument, resolved
/// against plains — B2's own module docs call this "a reasonable default
/// for a preview, not a claim about where the building will actually
/// stand." Only called from [`ghost_mesh`]'s cache-miss path, not once a
/// frame — see the module docs' "Caching, not per-frame work".
fn plains_biome_colors(world: &DecodedWorld, maps: &world::ColorMaps) -> BiomeColors {
    let registry = world.biomes.lock().expect("biome registry mutex poisoned");
    let table = world::build_biome_tint_table(&registry, maps);
    table.get(world::BiomeRegistry::PLAINS.0 as usize).copied().unwrap_or_else(white_biome)
}

/// Resolves (and caches) the mesh for `id` at `rotation` — builds it on the
/// first request for that combination and reuses the handle after, per the
/// module docs. `Deg0` never touches `rotate_blueprint` (mirrors B3's own
/// identity-case shortcut: it can't fail on a property nothing recognises).
#[allow(clippy::too_many_arguments)]
fn ghost_mesh(
    ghost: &mut GhostState,
    id: &str,
    rotation: Rotation,
    entry: &CatalogueEntry,
    atlas: &world::AtlasUvIndex,
    world: &DecodedWorld,
    color_maps: &world::ColorMaps,
    meshes: &mut Assets<Mesh>,
) -> Option<Handle<Mesh>> {
    let key = (id.to_string(), rotation);
    if let Some(cached) = ghost.meshes.get(&key) {
        return cached.clone();
    }

    let rotated;
    let blueprint = if rotation == Rotation::Deg0 {
        &entry.blueprint
    } else {
        match blueprint::rotate_blueprint(&entry.blueprint, rotation) {
            Ok(b) => {
                rotated = b;
                &rotated
            }
            Err(err) => {
                println!("block_viewer: ghost preview: {id} can't rotate to {rotation:?}: {err}");
                ghost.meshes.insert(key, None);
                return None;
            }
        }
    };

    let biome = plains_biome_colors(world, color_maps);
    let handle = blueprint::mesh_blueprint(blueprint, atlas, biome).map(|mesh| meshes.add(mesh));
    ghost.meshes.insert(key, handle.clone());
    handle
}

/// The two translucent ghost materials, built the first time they're
/// needed off whatever texture [`TerrainMaterial`] currently uses —
/// `AlphaMode::Blend` rather than the terrain's own `Mask(0.5)` cutout,
/// since a ghost is meant to be seen through, not just have its edges cut
/// cleanly. `unlit: true` so the tint reads the same green/red regardless of
/// the sky's current light level (ticket 011's day/night cycle) — legibility
/// matters more than the ghost looking lit here.
fn ensure_materials<'a>(
    ghost: &'a mut GhostState,
    terrain_material: &Handle<StandardMaterial>,
    materials: &mut Assets<StandardMaterial>,
) -> &'a GhostMaterials {
    ghost.materials.get_or_insert_with(|| {
        // One `ResMut<Assets<StandardMaterial>>` read-then-write, not a
        // second `Res` alongside it — see the ticket's "watch out" on why
        // that would be a conflicting-access panic at app-build time.
        let texture = materials.get(terrain_material).and_then(|m| m.base_color_texture.clone());
        let valid = materials.add(StandardMaterial {
            base_color: Color::srgba(0.35, 0.95, 0.4, 0.55),
            base_color_texture: texture.clone(),
            alpha_mode: AlphaMode::Blend,
            unlit: true,
            ..default()
        });
        let invalid = materials.add(StandardMaterial {
            base_color: Color::srgba(0.95, 0.3, 0.3, 0.55),
            base_color_texture: texture,
            alpha_mode: AlphaMode::Blend,
            unlit: true,
            ..default()
        });
        GhostMaterials { valid, invalid }
    })
}

/// Where a footprint would land, and whether it's actually placeable — see
/// the module docs' "Validity is two answers, ANDed". `pub(super)`: ticket
/// 048's `city::commit` recomputes exactly this on a click, so a commit
/// never disagrees with the ghost that was on screen when it happened.
pub(super) struct GhostPlacement {
    pub(super) origin: IVec3,
    pub(super) valid: bool,
}

/// ANDs E2's terrain fit and D1's occupancy check for `footprint`/`rotation`
/// at `hovered` (the solid block the cursor is over — one below where a
/// building's floor would sit, the same `+1` [`grid::ground_height_at`]
/// already applies), then shifts down by `ground_level` (ticket 085) and
/// applies `y_offset` (ticket 048) on top of that. A refused fit still
/// returns a placement — at `hovered`'s height (shifted and offset the same
/// way), invalid — so the caller always has *something* to show; see the
/// module docs. `saturating_sub`/`saturating_add` rather than plain
/// arithmetic: neither a definition's `ground_level` nor an offset built
/// purely from key-press counts can overflow in a real session, but nothing
/// here should panic if either somehow did.
///
/// `ground_level` shifts where the blueprint's own `y=0` lands, not where the
/// player's cursor is: a fit's `base_y` is one above the *terrain's* topmost
/// block, which is where the blueprint's ground-level layer (not
/// necessarily its `y=0`) belongs — see [`BuildingDefinitions::ground_level`]
/// for where the value itself comes from.
///
/// `pub(super)`: `city::commit` (ticket 048) is a second caller, on a click
/// rather than every frame — see [`GhostPlacement`]'s own docs.
pub(super) fn resolve_placement(
    hovered: IVec3,
    footprint: IVec2,
    rotation: Rotation,
    y_offset: i32,
    ground_level: i32,
    world: &DecodedWorld,
    city: &state::City,
) -> GhostPlacement {
    let probe = IVec3::new(hovered.x, hovered.y + 1, hovered.z);
    let (mut origin, terrain_ok) = match grid::fit_footprint(probe, footprint, rotation, world) {
        FootprintFit::Fits { base_y } => (IVec3::new(hovered.x, base_y, hovered.z), true),
        FootprintFit::Refused(_) => (probe, false),
    };
    origin.y = origin.y.saturating_sub(ground_level).saturating_add(y_offset);
    let occupancy_ok = state::footprint_tiles(origin, footprint, rotation).all(|tile| city.is_tile_free(tile));
    GhostPlacement { origin, valid: terrain_ok && occupancy_ok }
}

/// The world-space transform for a blueprint placed at Minecraft-space
/// `origin` — the same `bevy.z = -mc.z` translation
/// [`crate::chunk_pipeline`]'s chunk mesh spawn uses, since B2's mesh
/// vertices already bake the per-vertex flip
/// ([`crate::world::mesh::face_geometry`]) and only need a plain
/// translation on top.
fn ghost_transform(origin: IVec3) -> Transform {
    Transform::from_xyz(origin.x as f32, origin.y as f32, -(origin.z as f32))
}

/// Computes what the ghost should look like this frame — `Hidden` if
/// nothing is selected, hovered, or resolvable; `Shown` with a mesh,
/// a validity-tinted material, and a transform otherwise. Kept separate
/// from [`update_ghost_preview`] so it's callable from a test with plain
/// `Assets`, no `App`/`Commands`/`Query` involved.
///
/// `tool` (ticket 055, roadmap F2/F3): `Hidden` outright when it isn't
/// [`ActiveTool::Building`] — `city::road_build` owns the preview while the
/// road tool is active, and the two must never both draw at once. A plain
/// value, not a resource: [`update_ghost_preview`] is the one place that
/// reads [`ActiveTool`] off the world (defaulting to `Building` when the
/// resource is absent — see [`super::tool`]'s module docs), so every other
/// caller of this function, tests included, stays explicit about which tool
/// it's asking about.
#[allow(clippy::too_many_arguments)]
fn resolve_ghost(
    ghost: &mut GhostState,
    selection: &PlacementSelection,
    catalogue: Option<&BuildingCatalogue>,
    definitions: Option<&BuildingDefinitions>,
    hovered: Option<IVec3>,
    world: &DecodedWorld,
    color_maps: &world::ColorMaps,
    city: &state::City,
    atlas: &world::AtlasUvIndex,
    terrain_material: &Handle<StandardMaterial>,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    tool: ActiveTool,
) -> GhostUpdate {
    if tool != ActiveTool::Building {
        return GhostUpdate::Hidden;
    }
    let Some(id) = selection.catalogue_id.as_deref() else { return GhostUpdate::Hidden };
    let Some(catalogue) = catalogue else { return GhostUpdate::Hidden };
    let Some(entry) = catalogue.get(id) else { return GhostUpdate::Hidden };
    let Some(hovered) = hovered else { return GhostUpdate::Hidden };

    let Some(mesh) = ghost_mesh(ghost, id, selection.rotation, entry, atlas, world, color_maps, meshes) else {
        return GhostUpdate::Hidden;
    };

    // Ticket 085: `0` with no `BuildingDefinitions` resource at all — the
    // same tolerant default every other optional resource in this module
    // falls back to, so a minimal test `App` without one still previews at
    // the old, unshifted height.
    let ground_level = definitions.map(|defs| defs.ground_level(selection.definition_id.as_deref())).unwrap_or(0);
    let placement =
        resolve_placement(hovered, entry.footprint, selection.rotation, selection.y_offset, ground_level, world, city);
    let ghost_materials = ensure_materials(ghost, terrain_material, materials);
    let material = if placement.valid { ghost_materials.valid.clone() } else { ghost_materials.invalid.clone() };

    GhostUpdate::Shown { mesh, material, transform: ghost_transform(placement.origin) }
}

/// Applies a [`GhostUpdate`]: hides the entity (if it exists), or shows it —
/// spawning it the first time, updating its components in place after (see
/// the module docs' "One entity, updated in place").
fn apply_ghost_update(
    update: GhostUpdate,
    commands: &mut Commands,
    ghost: &mut GhostState,
    query: &mut Query<(&mut Visibility, &mut Transform, &mut Mesh3d, &mut MeshMaterial3d<StandardMaterial>), With<GhostPreview>>,
) {
    match update {
        GhostUpdate::Hidden => {
            if let Some(entity) = ghost.entity {
                if let Ok((mut visibility, ..)) = query.get_mut(entity) {
                    *visibility = Visibility::Hidden;
                }
            }
        }
        GhostUpdate::Shown { mesh, material, transform } => match ghost.entity {
            Some(entity) => {
                if let Ok((mut visibility, mut existing_transform, mut mesh3d, mut material3d)) = query.get_mut(entity) {
                    *visibility = Visibility::Visible;
                    *existing_transform = transform;
                    mesh3d.0 = mesh;
                    material3d.0 = material;
                }
            }
            None => {
                let entity = commands
                    .spawn((
                        Name::new("Ghost preview"),
                        Mesh3d(mesh),
                        MeshMaterial3d(material),
                        transform,
                        Visibility::Visible,
                        GhostPreview,
                    ))
                    .id();
                ghost.entity = Some(entity);
            }
        },
    }
}

/// The per-frame system: resolves what the ghost should look like (see
/// [`resolve_ghost`]) and applies it. Missing atlas/colormap/terrain-material
/// resources (all inserted unconditionally by `lib.rs::setup_world`, but
/// guarded the same `Option<Res<_>>` way `chunk_pipeline`'s own systems are,
/// so a minimal test app doesn't need the whole citybuilder wired up) hide
/// the ghost rather than panic.
#[allow(clippy::too_many_arguments)]
fn update_ghost_preview(
    mut commands: Commands,
    hovered: Res<HoveredBlock>,
    selection: Res<PlacementSelection>,
    catalogue: Option<Res<BuildingCatalogue>>,
    definitions: Option<Res<BuildingDefinitions>>,
    world: Res<DecodedWorld>,
    city: Res<state::City>,
    atlas: Option<Res<SharedAtlasIndex>>,
    color_maps: Option<Res<SharedColorMaps>>,
    terrain_material: Option<Res<TerrainMaterial>>,
    tool: Option<Res<ActiveTool>>,
    mut ghost: ResMut<GhostState>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut query: Query<(&mut Visibility, &mut Transform, &mut Mesh3d, &mut MeshMaterial3d<StandardMaterial>), With<GhostPreview>>,
) {
    let (Some(atlas), Some(color_maps), Some(terrain_material)) = (atlas, color_maps, terrain_material) else {
        apply_ghost_update(GhostUpdate::Hidden, &mut commands, &mut ghost, &mut query);
        return;
    };

    let update = resolve_ghost(
        &mut ghost,
        &selection,
        catalogue.as_deref(),
        definitions.as_deref(),
        hovered.0,
        &world,
        &color_maps.0,
        &city,
        &atlas.0,
        &terrain_material.0,
        &mut meshes,
        &mut materials,
        tool.map(|t| *t).unwrap_or_default(),
    );
    apply_ghost_update(update, &mut commands, &mut ghost, &mut query);
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;
    use crate::blueprint::{write_structure_file, BlockState, Blueprint};
    use crate::world::{BiomeRegistry, BlockId, BlockRegistry, ChunkColumn, ChunkSection};
    use std::sync::{Arc, Mutex};

    // --- fixtures -----------------------------------------------------

    fn state_named(name: &str) -> BlockState {
        name.parse().unwrap()
    }

    /// A `size`-sized blueprint, solid stone at index 0..volume except that
    /// `[0]` stays air-backed by the palette's first entry always being air —
    /// mirrors the shape `blueprint::mesh`'s own tests build.
    fn solid_blueprint(size: IVec3) -> Blueprint {
        let volume = (size.x * size.y * size.z) as usize;
        Blueprint {
            size,
            origin: IVec3::ZERO,
            palette: vec![BlockState::air(), state_named("minecraft:stone")],
            blocks: vec![1; volume],
            data_version: 0,
            failed_columns: 0,
        }
    }

    /// A blueprint carrying a property `blueprint::rotate`'s table has no
    /// rewrite rule for — the same fixture `blueprint::rotate`'s own tests
    /// use to exercise `UnrotatableProperty`, reused here to prove
    /// [`ghost_mesh`] caches the failure rather than panicking on it.
    fn unrotatable_blueprint() -> Blueprint {
        let unrotatable =
            BlockState { name: "minecraft:made_up_block".to_string(), properties: vec![("orientation".to_string(), "north_up".to_string())] };
        Blueprint { size: IVec3::ONE, origin: IVec3::ZERO, palette: vec![unrotatable], blocks: vec![0], data_version: 0, failed_columns: 0 }
    }

    fn catalogue_entry(id: &str, blueprint: Blueprint) -> CatalogueEntry {
        let footprint = IVec2::new(blueprint.size.x, blueprint.size.z);
        CatalogueEntry { id: id.to_string(), path: std::path::PathBuf::new(), blueprint, footprint }
    }

    fn temp_dir(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("block_viewer_test_placement_{name}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("should create temp dir");
        dir
    }

    /// A real [`BuildingCatalogue`] loaded off temp `.nbt` files — its
    /// `entries` field is private, so going through the public
    /// load-from-disk path (same as `blueprint::catalogue`'s own tests) is
    /// how a test builds one at all.
    fn catalogue_with(dir: &Path, entries: &[(&str, IVec3)]) -> BuildingCatalogue {
        for (id, size) in entries {
            write_structure_file(&dir.join(format!("{id}.nbt")), &solid_blueprint(*size)).expect("should write fixture");
        }
        let (catalogue, skipped) = blueprint::load_catalogue_dir(dir);
        assert!(skipped.is_empty(), "{skipped:?}");
        catalogue
    }

    fn registry_with_stone() -> (BlockRegistry, BlockId) {
        let mut registry = BlockRegistry::new();
        let stone = registry.intern("minecraft:stone");
        (registry, stone)
    }

    /// A flat, fully-decoded 16x16 chunk of ground at `y`, covering the
    /// footprints this module's tests place — same shape `city::grid`'s own
    /// `flat_chunk` fixture builds.
    fn flat_world(y: i32) -> DecodedWorld {
        let (registry, stone) = registry_with_stone();
        let size = world::SECTION_SIZE as i32;
        let section_y = y.div_euclid(size) as i8;
        let local_y = y.rem_euclid(size) as usize;

        let mut section = ChunkSection {
            y: section_y,
            blocks: Box::new([BlockRegistry::AIR; world::SECTION_VOLUME]),
            biomes: Box::new([BiomeRegistry::PLAINS; world::BIOME_GRID_VOLUME]),
        };
        for x in 0..size as usize {
            for z in 0..size as usize {
                section.blocks[ChunkSection::index(x, local_y, z)] = stone;
            }
        }
        let column = ChunkColumn { x: 0, z: 0, sections: vec![section], floor_y: world::WORLD_MIN_Y };

        DecodedWorld {
            registry: Arc::new(Mutex::new(registry)),
            biomes: Arc::new(Mutex::new(BiomeRegistry::new())),
            columns: [((0, 0), column)].into_iter().collect(),
        }
    }

    /// A full-size (if blank) colormap pair — `biome_colors_for`'s plains
    /// path indexes straight into `grass`/`foliage` by `(temperature,
    /// downfall)` position, so an empty `Vec` (unlike an empty atlas, which
    /// `world::atlas::resolve_faces` tolerates via its fallback texture)
    /// would panic rather than just look wrong.
    fn atlas_and_maps() -> (world::AtlasUvIndex, world::ColorMaps) {
        let blank = vec![[0u8, 0, 0]; 256 * 256];
        (world::AtlasUvIndex::default(), world::ColorMaps { grass: blank.clone(), foliage: blank })
    }

    // --- resolve_placement ---------------------------------------------

    #[test]
    fn resolve_placement_is_valid_on_flat_free_ground() {
        let world = flat_world(63);
        let city = state::City::default();
        let placement = resolve_placement(IVec3::new(2, 63, 2), IVec2::new(3, 3), Rotation::Deg0, 0, 0, &world, &city);
        assert!(placement.valid);
        assert_eq!(placement.origin, IVec3::new(2, 64, 2));
    }

    #[test]
    fn resolve_placement_is_invalid_when_a_tile_is_occupied() {
        let world = flat_world(63);
        let mut city = state::City::default();
        city.place_building("house01", None, IVec3::new(2, 64, 2), Rotation::Deg0, IVec2::new(1, 1)).unwrap();

        let placement = resolve_placement(IVec3::new(2, 63, 2), IVec2::new(3, 3), Rotation::Deg0, 0, 0, &world, &city);
        assert!(!placement.valid, "the footprint overlaps an already-placed building");
    }

    #[test]
    fn resolve_placement_falls_back_to_the_hovered_height_when_refused() {
        // No ground at all under this tile — every sample is `NotLoaded`.
        let world = flat_world(63);
        let city = state::City::default();
        let placement = resolve_placement(IVec3::new(500, 63, 500), IVec2::new(3, 3), Rotation::Deg0, 0, 0, &world, &city);
        assert!(!placement.valid);
        assert_eq!(placement.origin, IVec3::new(500, 64, 500), "still places the ghost somewhere, one above the hovered block");
    }

    #[test]
    fn resolve_placement_applies_a_positive_y_offset_on_a_fitting_placement() {
        let world = flat_world(63);
        let city = state::City::default();
        let placement = resolve_placement(IVec3::new(2, 63, 2), IVec2::new(3, 3), Rotation::Deg0, 5, 0, &world, &city);
        assert!(placement.valid, "the offset moves the building, it doesn't touch terrain/occupancy validity");
        assert_eq!(placement.origin, IVec3::new(2, 69, 2), "base_y (64) + the offset (5)");
    }

    #[test]
    fn resolve_placement_applies_a_negative_y_offset_on_the_refused_fallback() {
        let world = flat_world(63);
        let city = state::City::default();
        let placement = resolve_placement(IVec3::new(500, 63, 500), IVec2::new(3, 3), Rotation::Deg0, -3, 0, &world, &city);
        assert!(!placement.valid);
        assert_eq!(placement.origin, IVec3::new(500, 61, 500), "the hovered-height fallback (64) minus the offset (3)");
    }

    // --- resolve_placement + ground_level (ticket 085) ------------------

    #[test]
    fn resolve_placement_sinks_the_origin_by_ground_level_on_a_fitting_placement() {
        // lumber.ron's own shape: two below-grade layers (ground_level: 2)
        // under an otherwise-ordinary fit. base_y is still 64 (one above the
        // terrain at 63); the origin (the blueprint's y=0) has to land two
        // below that so the blueprint's own y=2 layer is what sits on the
        // terrain.
        let world = flat_world(63);
        let city = state::City::default();
        let placement = resolve_placement(IVec3::new(2, 63, 2), IVec2::new(3, 3), Rotation::Deg0, 0, 2, &world, &city);
        assert!(placement.valid);
        assert_eq!(placement.origin, IVec3::new(2, 62, 2), "base_y (64) - ground_level (2)");
    }

    #[test]
    fn resolve_placement_applies_ground_level_before_the_manual_y_offset() {
        let world = flat_world(63);
        let city = state::City::default();
        let placement = resolve_placement(IVec3::new(2, 63, 2), IVec2::new(3, 3), Rotation::Deg0, 5, 2, &world, &city);
        assert!(placement.valid);
        assert_eq!(placement.origin, IVec3::new(2, 67, 2), "base_y (64) - ground_level (2) + y_offset (5)");
    }

    #[test]
    fn resolve_placement_applies_ground_level_on_the_refused_fallback_too() {
        let world = flat_world(63);
        let city = state::City::default();
        let placement = resolve_placement(IVec3::new(500, 63, 500), IVec2::new(3, 3), Rotation::Deg0, 0, 2, &world, &city);
        assert!(!placement.valid);
        assert_eq!(placement.origin, IVec3::new(500, 62, 500), "hovered fallback (64) - ground_level (2), same as a fitting placement");
    }

    // --- ghost_mesh ------------------------------------------------------

    #[test]
    fn ghost_mesh_builds_and_caches_a_handle() {
        let mut ghost = GhostState::default();
        let entry = catalogue_entry("house01", solid_blueprint(IVec3::new(2, 2, 2)));
        let world = flat_world(63);
        let (atlas, maps) = atlas_and_maps();
        let mut meshes = Assets::<Mesh>::default();

        let handle = ghost_mesh(&mut ghost, "house01", Rotation::Deg0, &entry, &atlas, &world, &maps, &mut meshes);
        assert!(handle.is_some());
        assert_eq!(ghost.meshes.len(), 1);
    }

    #[test]
    fn ghost_mesh_reuses_the_cached_handle_on_a_second_call() {
        let mut ghost = GhostState::default();
        let entry = catalogue_entry("house01", solid_blueprint(IVec3::new(2, 2, 2)));
        let world = flat_world(63);
        let (atlas, maps) = atlas_and_maps();
        let mut meshes = Assets::<Mesh>::default();

        let first = ghost_mesh(&mut ghost, "house01", Rotation::Deg90, &entry, &atlas, &world, &maps, &mut meshes);
        let second = ghost_mesh(&mut ghost, "house01", Rotation::Deg90, &entry, &atlas, &world, &maps, &mut meshes);
        assert_eq!(first, second);
        assert_eq!(ghost.meshes.len(), 1, "one cache entry for one (id, rotation) pair, however many times it's asked for");
    }

    #[test]
    fn ghost_mesh_caches_a_rotation_failure_as_none_without_panicking() {
        let mut ghost = GhostState::default();
        let entry = catalogue_entry("weird", unrotatable_blueprint());
        let world = flat_world(63);
        let (atlas, maps) = atlas_and_maps();
        let mut meshes = Assets::<Mesh>::default();

        let first = ghost_mesh(&mut ghost, "weird", Rotation::Deg90, &entry, &atlas, &world, &maps, &mut meshes);
        let second = ghost_mesh(&mut ghost, "weird", Rotation::Deg90, &entry, &atlas, &world, &maps, &mut meshes);
        assert!(first.is_none());
        assert!(second.is_none());
        assert_eq!(ghost.meshes.len(), 1, "the failure itself is the cached entry, not retried");

        // `Deg0` never touches the rotation table at all (B3's own identity
        // shortcut), so the same unrotatable-at-90-degrees blueprint still
        // meshes fine here — a single boundary block with every face
        // exposed.
        let identity = ghost_mesh(&mut ghost, "weird", Rotation::Deg0, &entry, &atlas, &world, &maps, &mut meshes);
        assert!(identity.is_some());
        assert_eq!(ghost.meshes.len(), 2, "Deg90's failure and Deg0's success are two distinct cache entries");
    }

    // --- resolve_ghost -----------------------------------------------------

    #[test]
    fn resolve_ghost_is_hidden_with_no_selection() {
        let mut ghost = GhostState::default();
        let selection = PlacementSelection::default();
        let dir = temp_dir("hidden_no_selection");
        let catalogue = catalogue_with(&dir, &[("house01", IVec3::new(2, 2, 2))]);
        let world = flat_world(63);
        let (atlas, maps) = atlas_and_maps();
        let city = state::City::default();
        let mut meshes = Assets::<Mesh>::default();
        let mut materials = Assets::<StandardMaterial>::default();
        let terrain_material = materials.add(StandardMaterial::default());

        let update = resolve_ghost(
            &mut ghost,
            &selection,
            Some(&catalogue),
            None,
            Some(IVec3::new(2, 63, 2)),
            &world,
            &maps,
            &city,
            &atlas,
            &terrain_material,
            &mut meshes,
            &mut materials,
            ActiveTool::Building,
        );
        assert!(matches!(update, GhostUpdate::Hidden));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn resolve_ghost_is_hidden_with_no_hovered_block() {
        let mut ghost = GhostState::default();
        let selection = PlacementSelection { catalogue_id: Some("house01".to_string()), definition_id: None, rotation: Rotation::Deg0, y_offset: 0 };
        let dir = temp_dir("hidden_no_hover");
        let catalogue = catalogue_with(&dir, &[("house01", IVec3::new(2, 2, 2))]);
        let world = flat_world(63);
        let (atlas, maps) = atlas_and_maps();
        let city = state::City::default();
        let mut meshes = Assets::<Mesh>::default();
        let mut materials = Assets::<StandardMaterial>::default();
        let terrain_material = materials.add(StandardMaterial::default());

        let update = resolve_ghost(
            &mut ghost, &selection, Some(&catalogue), None, None, &world, &maps, &city, &atlas, &terrain_material, &mut meshes, &mut materials,
            ActiveTool::Building,
        );
        assert!(matches!(update, GhostUpdate::Hidden));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn resolve_ghost_is_hidden_while_the_road_tool_is_active() {
        // Ticket 055, roadmap F2/F3: the two tools' previews must never both
        // draw at once — `city::road_build` owns the preview here.
        let mut ghost = GhostState::default();
        let selection = PlacementSelection { catalogue_id: Some("house01".to_string()), definition_id: None, rotation: Rotation::Deg0, y_offset: 0 };
        let dir = temp_dir("hidden_road_tool");
        let catalogue = catalogue_with(&dir, &[("house01", IVec3::new(2, 2, 2))]);
        let world = flat_world(63);
        let (atlas, maps) = atlas_and_maps();
        let city = state::City::default();
        let mut meshes = Assets::<Mesh>::default();
        let mut materials = Assets::<StandardMaterial>::default();
        let terrain_material = materials.add(StandardMaterial::default());

        let update = resolve_ghost(
            &mut ghost,
            &selection,
            Some(&catalogue),
            None,
            Some(IVec3::new(2, 63, 2)),
            &world,
            &maps,
            &city,
            &atlas,
            &terrain_material,
            &mut meshes,
            &mut materials,
            ActiveTool::Road,
        );
        assert!(matches!(update, GhostUpdate::Hidden));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn resolve_ghost_shows_the_valid_material_on_buildable_free_ground() {
        let mut ghost = GhostState::default();
        let selection = PlacementSelection { catalogue_id: Some("house01".to_string()), definition_id: None, rotation: Rotation::Deg0, y_offset: 0 };
        let dir = temp_dir("shown_valid");
        let catalogue = catalogue_with(&dir, &[("house01", IVec3::new(2, 2, 2))]);
        let world = flat_world(63);
        let (atlas, maps) = atlas_and_maps();
        let city = state::City::default();
        let mut meshes = Assets::<Mesh>::default();
        let mut materials = Assets::<StandardMaterial>::default();
        let terrain_material = materials.add(StandardMaterial::default());

        let update = resolve_ghost(
            &mut ghost,
            &selection,
            Some(&catalogue),
            None,
            Some(IVec3::new(2, 63, 2)),
            &world,
            &maps,
            &city,
            &atlas,
            &terrain_material,
            &mut meshes,
            &mut materials,
            ActiveTool::Building,
        );
        let GhostUpdate::Shown { material, .. } = update else { panic!("expected a ghost to be shown") };
        assert_eq!(material, ghost.materials.as_ref().unwrap().valid);
    }

    #[test]
    fn resolve_ghost_shows_the_invalid_material_when_occupied() {
        let mut ghost = GhostState::default();
        let selection = PlacementSelection { catalogue_id: Some("house01".to_string()), definition_id: None, rotation: Rotation::Deg0, y_offset: 0 };
        let dir = temp_dir("shown_invalid");
        let catalogue = catalogue_with(&dir, &[("house01", IVec3::new(2, 2, 2))]);
        let world = flat_world(63);
        let (atlas, maps) = atlas_and_maps();
        let mut city = state::City::default();
        city.place_building("house01", None, IVec3::new(2, 64, 2), Rotation::Deg0, IVec2::new(1, 1)).unwrap();
        let mut meshes = Assets::<Mesh>::default();
        let mut materials = Assets::<StandardMaterial>::default();
        let terrain_material = materials.add(StandardMaterial::default());

        let update = resolve_ghost(
            &mut ghost,
            &selection,
            Some(&catalogue),
            None,
            Some(IVec3::new(2, 63, 2)),
            &world,
            &maps,
            &city,
            &atlas,
            &terrain_material,
            &mut meshes,
            &mut materials,
            ActiveTool::Building,
        );
        let GhostUpdate::Shown { material, .. } = update else { panic!("expected a ghost to be shown") };
        assert_eq!(material, ghost.materials.as_ref().unwrap().invalid);
        std::fs::remove_dir_all(&dir).ok();
    }

    // --- rotate_clockwise / sorted_catalogue_ids -----------------------

    #[test]
    fn rotate_clockwise_cycles_through_all_four_and_wraps() {
        let mut r = Rotation::Deg0;
        for expected in [Rotation::Deg90, Rotation::Deg180, Rotation::Deg270, Rotation::Deg0] {
            r = rotate_clockwise(r);
            assert_eq!(r, expected);
        }
    }

    // --- cycle_selection (a minimal App, per camera.rs's own test style) --

    /// No `InputPlugin` — same call `camera.rs`'s own `rts_test_app` makes.
    /// `InputPlugin` clears `just_pressed`/`just_released` once a frame as
    /// part of processing real OS input events; a manual `.press()` right
    /// before `app.update()` races that clear, where a bare
    /// `ButtonInput<KeyCode>` resource with nothing else touching it doesn't.
    fn selection_test_app() -> App {
        let mut app = App::new();
        app.init_resource::<ButtonInput<KeyCode>>()
            .init_resource::<camera::EguiInputCapture>()
            .init_resource::<PlacementSelection>()
            .add_systems(Update, cycle_selection);
        app
    }

    fn press(app: &mut App, key: KeyCode) {
        app.world_mut().resource_mut::<ButtonInput<KeyCode>>().press(key);
        app.update();
        app.world_mut().resource_mut::<ButtonInput<KeyCode>>().release(key);
    }

    #[test]
    fn pressing_a_number_key_selects_the_nth_catalogue_entry_alphabetically() {
        let dir = temp_dir("select_number");
        let catalogue = catalogue_with(&dir, &[("zzz", IVec3::ONE), ("aaa", IVec3::ONE), ("mmm", IVec3::ONE)]);

        let mut app = selection_test_app();
        app.insert_resource(catalogue);
        press(&mut app, KeyCode::Digit2); // sorted: aaa, mmm, zzz -> index 1 -> "mmm"

        assert_eq!(app.world().resource::<PlacementSelection>().catalogue_id.as_deref(), Some("mmm"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn pressing_escape_clears_the_selection() {
        let mut app = selection_test_app();
        app.world_mut().resource_mut::<PlacementSelection>().catalogue_id = Some("house01".to_string());
        press(&mut app, KeyCode::Escape);
        assert_eq!(app.world().resource::<PlacementSelection>().catalogue_id, None);
    }

    /// Ticket 083: clearing a placement is one of the two ways back to the
    /// resting state.
    #[test]
    fn pressing_escape_resets_the_tool_to_inspect() {
        let mut app = selection_test_app();
        app.insert_resource(ActiveTool::Building);
        press(&mut app, KeyCode::Escape);
        assert_eq!(*app.world().resource::<ActiveTool>(), ActiveTool::Inspect);
    }

    /// The same tolerant shape every other optional resource in this module
    /// gets: a minimal test `App` that never adds `ToolPlugin` (like
    /// [`selection_test_app`] itself) must not panic on `Escape`.
    #[test]
    fn pressing_escape_without_a_tool_resource_does_not_panic() {
        let mut app = selection_test_app();
        press(&mut app, KeyCode::Escape);
        assert_eq!(app.world().resource::<PlacementSelection>().catalogue_id, None);
    }

    #[test]
    fn pressing_r_rotates_the_selection() {
        let mut app = selection_test_app();
        press(&mut app, KeyCode::KeyR);
        assert_eq!(app.world().resource::<PlacementSelection>().rotation, Rotation::Deg90);
        press(&mut app, KeyCode::KeyR);
        assert_eq!(app.world().resource::<PlacementSelection>().rotation, Rotation::Deg180);
    }

    #[test]
    fn a_number_key_past_the_catalogues_size_does_nothing() {
        let dir = temp_dir("select_out_of_range");
        let catalogue = catalogue_with(&dir, &[("only_one", IVec3::ONE)]);

        let mut app = selection_test_app();
        app.insert_resource(catalogue);
        press(&mut app, KeyCode::Digit9);

        assert_eq!(app.world().resource::<PlacementSelection>().catalogue_id, None);
        std::fs::remove_dir_all(&dir).ok();
    }
}
