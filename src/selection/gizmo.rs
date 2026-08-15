//! Draws the selection as a wireframe box (ticket 019), in two passes so the
//! part of it that's underground is distinguishable from the part that isn't
//! (ticket 026).
//!
//! Immediate-mode [`Gizmos`] rather than a spawned wireframe mesh entity:
//! the box changes on nearly every keystroke (ticket 020), and a mesh entity
//! would need a material, a despawn path on every bounds change, and space
//! in the same `Assets<Mesh>` the streaming pipeline is already churning
//! through. Gizmos have no entity lifecycle to keep in sync with
//! [`Selection`]. (`bevy::pbr::wireframe` isn't an option at all — it draws
//! *existing* meshes as wireframes, and there's no mesh here.)
//!
//! ## Why two passes
//!
//! 019 drew the box once, always in front of terrain. That makes a box
//! extending into the ground unreadable: every edge is at full strength over
//! the hillside, so the wireframe reads as a flat outline pasted on the
//! screen and there's no telling which edges are buried or where it enters
//! the ground. Depth-testing it instead just hides the buried part, which is
//! worse — that's the part you're extending when you can't see it.
//!
//! So the same geometry goes out twice: [`SelectionGizmos`] depth tested and
//! solid for the part in open air, [`BuriedSelectionGizmos`] always-in-front
//! and dotted for the part behind terrain. The boundary between the two
//! styles is the line where the box enters the ground, which is the thing
//! that was missing.

use bevy::prelude::*;

use super::{block_bevy_aabb, Selection, SelectionBounds, CHUNK_STEP};

/// The visible pass: depth tested, so terrain occludes it.
///
/// Kept out of [`DefaultGizmoConfigGroup`] so the depth bias below is scoped
/// to this feature rather than silently applying to any gizmo anything else
/// in the app draws later.
#[derive(Default, Reflect, GizmoConfigGroup)]
struct SelectionGizmos;

/// The buried pass: always in front of everything, dotted and dimmed.
#[derive(Default, Reflect, GizmoConfigGroup)]
struct BuriedSelectionGizmos;

/// Just enough bias to stop the solid pass z-fighting where the box is
/// coplanar with block faces — which is constantly, since every selection
/// boundary *is* a block boundary.
///
/// **This number has to be tiny** (ticket 029). Bevy's gizmo shader applies a
/// negative bias as `clip.z * (clip.w / clip.z)^-bias`, and under a reverse-Z
/// infinite projection `clip.z` is the near plane and `clip.w` the view
/// distance — so the line is pulled forward by a *fraction* of how far away it
/// is, roughly `-bias * ln(distance / near)`. With `near` at 0.1 (this app
/// leaves [`PerspectiveProjection`] at its default) that fraction is about
/// `6 * -bias` across the whole useful range. 026's original `-0.02` was
/// therefore a ~12% pull: nearly a block through terrain at 10 blocks out and
/// almost six at 50, which meant terrain stopped occluding this pass entirely
/// and the two passes drew identically. That's what made the box unreadable.
///
/// At `-0.0002` the pull is ~6 cm at 50 blocks — invisible, but still some
/// four orders of magnitude more than float32 reverse-Z needs to win a
/// coplanar depth test, so the z-fighting this exists for stays fixed.
const SOLID_DEPTH_BIAS: f32 = -0.0002;

/// The far end of [`GizmoConfig::depth_bias`]'s range: always in front of
/// everything, however deep it's buried.
const BURIED_DEPTH_BIAS: f32 = -1.0;

const LINE_WIDTH: f32 = 2.5;

/// The buried pass's width — which is also its *dash* length (ticket 029).
///
/// [`GizmoLineStyle::Dotted`] has no spacing of its own: the shader's
/// `fragment_dotted` measures the pattern in units of `line_width` with a
/// period of 2, so a dash is `line_width` pixels long and so is the gap after
/// it. At 026's 2.0 that's a 2px-on/2px-off stipple, which reads as a solid
/// line at half brightness rather than as a dashed one — no use at all as the
/// thing that distinguishes this pass. 4px dashes are unambiguous.
///
/// The cost is that the buried pass is now *wider* than the solid one, which
/// is backwards where they overlap in open air; [`BURIED_ALPHA`] pays for it.
const BURIED_LINE_WIDTH: f32 = 4.0;

/// How far the buried pass is dimmed.
///
/// Lower than it would need to be on its own, because the two passes overlap
/// wherever the box is in open air and this one is drawn on top at four
/// pixels wide. At 0.4 that overlap reads as a faint broken halo around the
/// crisp solid line, rather than as a fat dashed line smothering it.
const BURIED_ALPHA: f32 = 0.4;

/// Warm yellow: reads against grass, stone and water alike, and isn't a
/// colour the vanilla texture set has much of.
const BOX_COLOR: Srgba = Srgba::rgb(1.0, 0.85, 0.2);

/// The anchor block's marker, in a hotter shade so the two are
/// distinguishable where they overlap.
const ANCHOR_COLOR: Srgba = Srgba::rgb(1.0, 0.4, 0.1);

/// The anchor cube is drawn inset inside its block rather than filling it.
/// On a fresh 1x1x1 selection the anchor cube and the selection box cover
/// the same block, and two coincident wireframes z-fight into a mess; inset,
/// it reads as a cube nested inside the box.
const ANCHOR_INSET: f32 = 0.8;

/// The slice plates (see [`slice_heights`]) are dimmer than the box's own
/// edges so the outline still dominates — they're a depth cue, not a second
/// thing to read.
const SLICE_ALPHA: f32 = 0.45;

/// How many slice plates one box may draw before [`slice_step`] coarsens the
/// step.
///
/// Y is clamped to the build limits, so a full-height selection is 384
/// blocks — 24 plates at a 16-block step, which is a stack dense enough to
/// read as fill rather than as depth.
const MAX_SLICES: i32 = 12;

pub struct SelectionGizmoPlugin;

impl Plugin for SelectionGizmoPlugin {
    fn build(&self, app: &mut App) {
        app.insert_gizmo_config(
            SelectionGizmos,
            GizmoConfig {
                depth_bias: SOLID_DEPTH_BIAS,
                line_width: LINE_WIDTH,
                // `render_layers` deliberately left at the default (layer 0
                // only). The sky camera (ticket 016) is a second `Camera3d`
                // rendering only `RenderLayers::layer(1)`, and gizmos are
                // drawn by every camera whose layers intersect theirs — on
                // layer 1 the box would be drawn by the sky pass too, where
                // the dome would then paint over it.
                ..default()
            },
        )
        .insert_gizmo_config(
            BuriedSelectionGizmos,
            GizmoConfig {
                depth_bias: BURIED_DEPTH_BIAS,
                line_width: BURIED_LINE_WIDTH,
                line_style: GizmoLineStyle::Dotted,
                ..default()
            },
        )
        .add_systems(Update, draw_selection_box);
    }
}

/// Draws the box, the anchor marker and the slice plates — once depth tested
/// and once in front of everything.
///
/// Both passes draw the *same* geometry rather than trying to work out which
/// edges are buried: that's what the depth buffer is for, and a box against
/// streamed terrain has no cheap answer anyway. Where the box is in open air
/// the two overlap, and the solid pass is what reads.
///
/// Draws nothing at all when there's no selection.
fn draw_selection_box(
    mut solid: Gizmos<SelectionGizmos>,
    mut buried: Gizmos<BuriedSelectionGizmos>,
    selection: Res<Selection>,
) {
    let Some(bounds) = selection.0 else {
        return;
    };

    draw_selection(&mut solid, bounds, 1.0);
    draw_selection(&mut buried, bounds, BURIED_ALPHA);
}

/// One pass's worth of geometry, at `alpha`.
///
/// Generic over the config group so the two passes can't drift apart —
/// they differ only in their [`GizmoConfig`] and how far they're dimmed.
fn draw_selection<C: GizmoConfigGroup>(
    gizmos: &mut Gizmos<'_, '_, C>,
    bounds: SelectionBounds,
    alpha: f32,
) {
    let (min, max) = bounds.bevy_aabb();
    gizmos.cuboid(box_transform(min, max, 1.0), BOX_COLOR.with_alpha(alpha));

    let (anchor_min, anchor_max) = block_bevy_aabb(bounds.anchor);
    gizmos.cuboid(
        box_transform(anchor_min, anchor_max, ANCHOR_INSET),
        ANCHOR_COLOR.with_alpha(alpha),
    );

    let slice_color = BOX_COLOR.with_alpha(alpha * SLICE_ALPHA);
    for y in slice_heights(bounds) {
        // A closed rectangle across the box at this height. Minecraft and
        // Bevy agree on Y, so the slice height needs no conversion — only X
        // and Z come from the already-flipped AABB.
        let y = y as f32;
        gizmos.linestrip(
            [
                Vec3::new(min.x, y, min.z),
                Vec3::new(max.x, y, min.z),
                Vec3::new(max.x, y, max.z),
                Vec3::new(min.x, y, max.z),
                Vec3::new(min.x, y, min.z),
            ],
            slice_color,
        );
    }
}

/// The heights of the horizontal slice plates: every multiple of
/// [`slice_step`] *strictly inside* the box.
///
/// Strictly inside because the box's own top and bottom faces are already
/// drawn by [`Gizmos::cuboid`] — a plate coincident with one of them would
/// only z-fight against it.
///
/// Chunk-aligned in world space rather than relative to the box, so the
/// plates line up with the terrain's own 16-block grid and stay put as a
/// face is extended past them.
fn slice_heights(bounds: SelectionBounds) -> Vec<i32> {
    // The box's top face is one block above its max block — the same `+1`
    // `block_bevy_aabb` applies.
    let (bottom, top) = (bounds.min.y, bounds.max.y + 1);
    let step = slice_step(top - bottom);

    let mut y = bottom.div_euclid(step) * step;
    let mut heights = Vec::new();
    while y <= bottom {
        y += step;
    }
    while y < top {
        heights.push(y);
        y += step;
    }
    heights
}

/// How far apart the slice plates are for a box `height` blocks tall:
/// [`CHUNK_STEP`], doubled until there are at most [`MAX_SLICES`] of them.
///
/// Doubling rather than giving up past a threshold: the tall boxes are
/// exactly the ones whose depth is hardest to read, so they should keep
/// their plates and lose only the resolution.
fn slice_step(height: i32) -> i32 {
    let mut step = CHUNK_STEP;
    while height / step > MAX_SLICES {
        step *= 2;
    }
    step
}

/// [`Gizmos::cuboid`] draws a *unit* cube centred on its transform's
/// translation and scaled by its scale, so an AABB has to be handed over as
/// centre + extent rather than as its two corners.
fn box_transform(min: Vec3, max: Vec3, shrink: f32) -> Transform {
    Transform::from_translation((min + max) * 0.5).with_scale((max - min) * shrink)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::selection::{WORLD_MAX_Y, WORLD_MIN_Y};

    fn bounds(min_y: i32, max_y: i32) -> SelectionBounds {
        SelectionBounds::from_corners(
            IVec3::new(0, min_y, 0),
            IVec3::new(0, min_y, 0),
            IVec3::new(3, max_y, 3),
        )
    }

    #[test]
    fn box_transform_centres_on_the_aabb_and_scales_to_its_extent() {
        let bounds = SelectionBounds::from_corners(
            IVec3::new(2, 10, 3),
            IVec3::new(2, 10, 3),
            IVec3::new(4, 11, 6),
        );
        let (min, max) = bounds.bevy_aabb();
        let transform = box_transform(min, max, 1.0);

        // Centre of `x 2..5`, `y 10..12`, `z -7..-3`.
        assert_eq!(transform.translation, Vec3::new(3.5, 11.0, -5.0));
        // Scale is the block count per axis — a unit cube scaled by this
        // covers exactly the selected blocks.
        assert_eq!(transform.scale, bounds.size().as_vec3());
    }

    #[test]
    fn the_anchor_cube_is_inset_within_its_block() {
        let (min, max) = block_bevy_aabb(IVec3::new(0, 0, 0));
        let transform = box_transform(min, max, ANCHOR_INSET);

        assert_eq!(transform.scale, Vec3::splat(ANCHOR_INSET));
        // Still centred on the block it marks, not shrunk toward a corner.
        assert_eq!(transform.translation, Vec3::new(0.5, 0.5, -0.5));
    }

    /// The plates sit on world chunk boundaries, not on the box's own
    /// bottom — so they stay put while a face is extended past them.
    #[test]
    fn slices_land_on_chunk_boundaries_inside_the_box() {
        assert_eq!(slice_heights(bounds(60, 75)), vec![64]);
        assert_eq!(slice_heights(bounds(60, 100)), vec![64, 80, 96]);
    }

    /// The box's own top and bottom faces are drawn by the cuboid; a plate
    /// coincident with either would only z-fight against it.
    #[test]
    fn slices_never_land_on_the_boxs_own_faces() {
        // Exactly one chunk tall, both faces on a boundary: no plates.
        assert_eq!(slice_heights(bounds(64, 79)), Vec::<i32>::new());
        // Two chunks tall: only the boundary between them.
        assert_eq!(slice_heights(bounds(64, 95)), vec![80]);
    }

    /// A box that doesn't reach a boundary at all gets nothing, rather than
    /// a plate clamped to one of its faces.
    #[test]
    fn a_box_inside_one_chunk_layer_has_no_slices() {
        assert_eq!(slice_heights(bounds(65, 70)), Vec::<i32>::new());
    }

    /// Below Y 0 the "first multiple at or above the bottom" arithmetic is
    /// the `div_euclid` path, which truncating division would get wrong.
    #[test]
    fn slices_are_correct_below_y_zero() {
        assert_eq!(slice_heights(bounds(-60, -40)), vec![-48]);
        assert_eq!(slice_heights(bounds(-40, 20)), vec![-32, -16, 0, 16]);
    }

    /// A tall box keeps its plates and loses resolution instead.
    #[test]
    fn the_slice_step_coarsens_rather_than_flooding_a_tall_box() {
        assert_eq!(slice_step(100), CHUNK_STEP);
        assert_eq!(slice_step(MAX_SLICES * CHUNK_STEP), CHUNK_STEP);
        assert_eq!(slice_step((MAX_SLICES + 1) * CHUNK_STEP), CHUNK_STEP * 2);

        // The tallest selection the world allows still stays in budget.
        let full_height = bounds(WORLD_MIN_Y, WORLD_MAX_Y);
        assert!(slice_heights(full_height).len() <= MAX_SLICES as usize);
    }
}
