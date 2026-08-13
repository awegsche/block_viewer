//! Draws the selection as a wireframe box (ticket 019).
//!
//! Immediate-mode [`Gizmos`] rather than a spawned wireframe mesh entity:
//! the box changes on nearly every keystroke (ticket 020), and a mesh entity
//! would need a material, a despawn path on every bounds change, and space
//! in the same `Assets<Mesh>` the streaming pipeline is already churning
//! through. Gizmos have no entity lifecycle to keep in sync with
//! [`Selection`]. (`bevy::pbr::wireframe` isn't an option at all — it draws
//! *existing* meshes as wireframes, and there's no mesh here.)

use bevy::prelude::*;

use super::{block_bevy_aabb, Selection};

/// The selection box's own gizmo config, kept separate from
/// [`DefaultGizmoConfigGroup`] so the depth bias below is scoped to this
/// feature rather than silently applying to any gizmo anything else in the
/// app draws later.
#[derive(Default, Reflect, GizmoConfigGroup)]
struct SelectionGizmos;

/// Draw the box in front of terrain rather than inside it.
///
/// A selection box you can't see because it's behind a hill is useless, and
/// the box is a UI affordance rather than part of the scene. `-1.0` is the
/// far end of [`GizmoConfig::depth_bias`]'s range: always in front of
/// everything. Whether that's the right look — versus a small negative bias,
/// which would let terrain occlude it while stopping it z-fighting against
/// block faces it's coplanar with — is a judgement call that needs a human
/// at the window; it's in `todo.md`.
const DEPTH_BIAS: f32 = -1.0;

const LINE_WIDTH: f32 = 2.5;

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

pub struct SelectionGizmoPlugin;

impl Plugin for SelectionGizmoPlugin {
    fn build(&self, app: &mut App) {
        app.insert_gizmo_config(
            SelectionGizmos,
            GizmoConfig {
                depth_bias: DEPTH_BIAS,
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
        .add_systems(Update, draw_selection_box);
    }
}

/// One wireframe box around the whole selection, plus a small cube marking
/// the anchor block so ticket 020's keyboard extrusion is legible (which
/// block did I click, and which way is the box growing away from it).
///
/// Draws nothing at all when there's no selection.
fn draw_selection_box(mut gizmos: Gizmos<SelectionGizmos>, selection: Res<Selection>) {
    let Some(bounds) = selection.0 else {
        return;
    };

    let (min, max) = bounds.bevy_aabb();
    gizmos.cuboid(box_transform(min, max, 1.0), BOX_COLOR);

    let (anchor_min, anchor_max) = block_bevy_aabb(bounds.anchor);
    gizmos.cuboid(box_transform(anchor_min, anchor_max, ANCHOR_INSET), ANCHOR_COLOR);
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
    use crate::selection::SelectionBounds;

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
}
