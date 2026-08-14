//! The sun and moon discs (ticket 016): two unlit, alpha-masked billboards
//! on the sky layer, positioned from [`super::SkyPalette::sun_direction`]
//! so they stay consistent with the actual lighting for free — including
//! once 018 starts animating that direction on a clock.

use std::path::Path;

use bevy::image::ImageSampler;
use bevy::pbr::{NotShadowCaster, NotShadowReceiver};
use bevy::prelude::*;
use bevy::render::{
    mesh::{Indices, PrimitiveTopology},
    render_asset::RenderAssetUsages,
    render_resource::{Extent3d, TextureDimension, TextureFormat},
    view::RenderLayers,
};

use super::dome;

/// Billboard side length (world units) at [`dome::RADIUS`]. Vanilla's sun is
/// drawn far larger than the real ~0.5° it should subtend — matched here by
/// eye against the game's look rather than by any real angular size (see
/// the parent ticket).
const SUN_SIZE: f32 = 24.0;
const MOON_SIZE: f32 = 20.0;

const MOON_ATLAS_COLS: u32 = 4;
const MOON_ATLAS_ROWS: u32 = 2;
/// `moon_phases.png` is a 4x2 atlas of the eight lunar phases, cell 0 (top
/// left) being a full moon (vanilla's day-count-mod-8 phase 0). The real
/// phase depends on the world's day count, which nothing decodes yet (see
/// 018's `level.dat` note) — hardcoded here until that lands.
const MOON_PHASE: u32 = 0;

/// Which billboard an entity is, and therefore which side of
/// [`super::SkyPalette::sun_direction`] it sits on — see
/// [`celestial_transform`].
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CelestialBody {
    Sun,
    Moon,
}

/// A single quad in the local XY plane, `size` on a side, centred on the
/// origin, textured with `(u0, v0)..(u1, v1)` — [`spawn_one`] renders it
/// double-sided via the material's `cull_mode`, not extra geometry here. No
/// vertex colour attribute — unlike the terrain meshes in
/// [`super::super::world::mesh`] this material is never mixed with a
/// coloured-vertex pipeline, so there's nothing to keep the vertex layout
/// consistent with.
fn quad_mesh(size: f32, (u0, v0, u1, v1): (f32, f32, f32, f32)) -> Mesh {
    let h = size / 2.0;
    let positions = vec![[-h, -h, 0.0], [h, -h, 0.0], [h, h, 0.0], [-h, h, 0.0]];
    let normals = vec![[0.0, 0.0, 1.0]; 4];
    // (0,0) in image space is the texture's top-left corner, which is the
    // quad's top-left (-h, h) corner once faced toward the viewer.
    let uvs = vec![[u0, v1], [u1, v1], [u1, v0], [u0, v0]];

    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
    .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs)
    .with_inserted_indices(Indices::U32(vec![0, 1, 2, 0, 2, 3]))
}

/// The atlas rect for [`MOON_PHASE`] inside `moon_phases.png`'s 4x2 grid.
fn moon_uv_rect() -> (f32, f32, f32, f32) {
    let col = (MOON_PHASE % MOON_ATLAS_COLS) as f32;
    let row = (MOON_PHASE / MOON_ATLAS_COLS) as f32;
    let cell_w = 1.0 / MOON_ATLAS_COLS as f32;
    let cell_h = 1.0 / MOON_ATLAS_ROWS as f32;
    (
        col * cell_w,
        row * cell_h,
        (col + 1.0) * cell_w,
        (row + 1.0) * cell_h,
    )
}

/// Decodes a PNG straight into a Bevy [`Image`] the same way
/// [`super::super::world::atlas::build`] and [`super::super::world::tint::load_color_maps`]
/// do — this app never routes textures through `AssetServer`, so sun/moon
/// don't start now either. Nearest sampling matches every other texture in
/// the app (blocky source art, no filtering).
fn load_texture(path: &Path) -> std::io::Result<Image> {
    let decoded = image::open(path)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?
        .to_rgba8();
    let (width, height) = decoded.dimensions();
    let mut image = Image::new(
        Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        decoded.into_raw(),
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.sampler = ImageSampler::nearest();
    Ok(image)
}

/// World-space transform for a billboard sitting at `direction *`
/// [`dome::RADIUS`], facing back toward the origin — where the sky camera
/// always sits (see
/// [`super::sync_sky_camera_rotation`]), so this only ever needs computing
/// once per palette change rather than every frame like a typical billboard
/// that tracks a moving camera.
///
/// `looking_to(direction, up)` points local -Z along `direction` (i.e.
/// outward, away from the origin), which leaves local +Z — the quad's
/// texture-facing side, see [`quad_mesh`] — pointing back at the origin.
fn celestial_transform(direction: Vec3) -> Transform {
    let direction = direction.normalize();
    // `looking_to` degenerates when `direction` is parallel to `up`; that
    // only happens when the sun/moon sits at the zenith/nadir, which the
    // default palette never does, but 018's day/night animation might swing
    // through it, so fall back to a second axis rather than emit a NaN
    // transform.
    let up = if direction.dot(Vec3::Y).abs() > 0.999 {
        Vec3::Z
    } else {
        Vec3::Y
    };
    Transform::from_translation(direction * dome::RADIUS).looking_to(direction, up)
}

/// Spawns the sun and moon billboards on [`super::SKY_LAYER`], loading their
/// textures the same eager, panic-on-failure way `lib.rs::setup_world` loads the
/// block atlas and biome colormaps — both textures are checked into the
/// repo, so a missing one means a broken checkout, not a recoverable
/// runtime condition.
pub(crate) fn spawn_celestial_bodies(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    images: &mut Assets<Image>,
    sun_direction: Vec3,
) {
    let sun_image = load_texture(Path::new("assets/minecraft/textures/environment/sun.png"))
        .expect("failed to load the sun texture");
    let moon_image = load_texture(Path::new(
        "assets/minecraft/textures/environment/moon_phases.png",
    ))
    .expect("failed to load the moon texture");

    spawn_one(
        commands,
        meshes,
        materials,
        images,
        CelestialBody::Sun,
        SUN_SIZE,
        (0.0, 0.0, 1.0, 1.0),
        sun_image,
        -sun_direction,
    );
    spawn_one(
        commands,
        meshes,
        materials,
        images,
        CelestialBody::Moon,
        MOON_SIZE,
        moon_uv_rect(),
        moon_image,
        sun_direction,
    );
}

#[allow(clippy::too_many_arguments)]
fn spawn_one(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    images: &mut Assets<Image>,
    body: CelestialBody,
    size: f32,
    uv: (f32, f32, f32, f32),
    image: Image,
    direction: Vec3,
) {
    let mesh = meshes.add(quad_mesh(size, uv));
    let material = materials.add(StandardMaterial {
        base_color_texture: Some(images.add(image)),
        unlit: true,
        alpha_mode: AlphaMode::Mask(0.5),
        // Winding the billboard correctly is unverifiable without actually
        // running the app (see CLAUDE.md on manual/visual checks) — double
        // siding it costs nothing on a two-triangle mesh and removes an
        // entire class of "invisible sun" bug.
        cull_mode: None,
        ..default()
    });

    commands.spawn((
        Name::new(format!("{body:?}")),
        Mesh3d(mesh),
        MeshMaterial3d(material),
        celestial_transform(direction),
        RenderLayers::layer(super::SKY_LAYER),
        NotShadowCaster,
        NotShadowReceiver,
        body,
    ));
}

/// Repositions the sun and moon whenever [`super::SkyPalette`] changes —
/// same `is_changed()` gate as [`super::sync_sky_palette`] and
/// [`dome::rebuild_dome_on_palette_change`], so all three stay in lockstep
/// on whichever frame the palette actually moves.
pub(crate) fn sync_celestial_positions(
    palette: Res<super::SkyPalette>,
    mut query: Query<(&mut Transform, &CelestialBody)>,
) {
    if !palette.is_changed() {
        return;
    }
    for (mut transform, body) in &mut query {
        let direction = match body {
            CelestialBody::Sun => -palette.sun_direction,
            CelestialBody::Moon => palette.sun_direction,
        };
        *transform = celestial_transform(direction);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Ticket 016: "sun and moon positions are antipodal and follow
    /// `sun_direction`."
    #[test]
    fn sun_and_moon_are_antipodal_and_track_sun_direction() {
        let sun_direction = Vec3::new(-0.35, -0.85, -0.4).normalize();

        let sun_transform = celestial_transform(-sun_direction);
        let moon_transform = celestial_transform(sun_direction);

        assert!(
            sun_transform
                .translation
                .normalize()
                .dot(moon_transform.translation.normalize())
                < -0.999,
            "sun and moon should sit on opposite sides of the dome"
        );
        assert!((sun_transform.translation.length() - dome::RADIUS).abs() < 1e-3);
        assert!((moon_transform.translation.length() - dome::RADIUS).abs() < 1e-3);

        // The sun sits opposite the direction the light travels (it shines
        // *from* the sun *toward* the ground).
        assert!(sun_transform.translation.normalize().dot(-sun_direction) > 0.999);
        assert!(moon_transform.translation.normalize().dot(sun_direction) > 0.999);
    }

    /// Each billboard's local +Z (its texture-facing side, see [`quad_mesh`])
    /// must point back at the origin, wherever it's placed on the dome.
    #[test]
    fn billboard_faces_the_origin() {
        for direction in [
            Vec3::new(1.0, 0.2, 0.3).normalize(),
            Vec3::new(-0.6, 0.1, -0.4).normalize(),
            Vec3::new(0.0, 1.0, 0.0), // straight up: exercises the degenerate-up fallback
        ] {
            let transform = celestial_transform(direction);
            let facing = transform.rotation * Vec3::Z;
            let toward_origin = (-transform.translation).normalize();
            assert!(
                facing.dot(toward_origin) > 0.999,
                "billboard at {direction:?} should face the origin, faced {facing:?} instead"
            );
        }
    }

    #[test]
    fn moon_phase_zero_is_the_atlas_top_left_cell() {
        let (u0, v0, u1, v1) = moon_uv_rect();
        assert_eq!((u0, v0), (0.0, 0.0));
        assert_eq!((u1, v1), (0.25, 0.5));
    }

    /// Sanity check that the real vendored textures decode — catches a
    /// broken checkout the same way [`super::super::world::atlas`]'s
    /// equivalent test does for the block atlas.
    #[test]
    fn loads_the_real_vendored_sun_and_moon_textures() {
        let base =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/minecraft/textures/environment");
        load_texture(&base.join("sun.png")).expect("sun.png should decode");
        load_texture(&base.join("moon_phases.png")).expect("moon_phases.png should decode");
    }
}
