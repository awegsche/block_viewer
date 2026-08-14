//! The skydome (ticket 016): an inverted UV-sphere with the zenith->horizon
//! gradient baked straight into `Mesh::ATTRIBUTE_COLOR` per vertex, instead
//! of a cubemap or a shader — see the parent ticket's "why not the obvious
//! options". At ~500 vertices, rebuilding the whole mesh every time
//! [`super::SkyPalette`] changes (018's day/night clock does this every
//! frame it's running) is free.

use bevy::color::Mix;
use bevy::prelude::*;
use bevy::render::{
    mesh::{Indices, PrimitiveTopology},
    render_asset::RenderAssetUsages,
};

/// Horizontal/vertical resolution of the dome's UV-sphere. 32x16 -> 33*17 =
/// 561 vertices (the "~500" the parent ticket budgets for).
const SECTORS: u32 = 32;
const STACKS: u32 = 16;

/// World-space radius of the dome, and — per the parent ticket — of the
/// sun/moon billboards in [`super::bodies`] too, so both stay on the same
/// sphere. Comfortably past any terrain the streaming pipeline loads, and
/// (translation pinned to the origin, see [`super::sync_sky_camera_rotation`])
/// never closer to the sky camera than that regardless of how far the main
/// camera flies.
pub(crate) const RADIUS: f32 = 100.0;

/// Marks the one skydome entity — [`rebuild_dome_on_palette_change`] doesn't
/// actually need to query it (it goes through [`SkyDomeMesh`] instead), but
/// this makes the entity findable/debuggable the same way every other
/// tagged entity in the app is.
#[derive(Component)]
pub(crate) struct SkyDome;

/// The dome mesh's asset handle, stashed at spawn time so
/// [`rebuild_dome_on_palette_change`] can reach back into [`Assets<Mesh>`]
/// and overwrite it in place — cheaper than despawning/respawning the
/// entity every time the palette changes.
#[derive(Resource)]
pub(crate) struct SkyDomeMesh(pub(crate) Handle<Mesh>);

/// Builds the dome mesh: normals point inward (the camera only ever sees
/// the inside of it), and vertex colour is `horizon` lerped toward `zenith`
/// on the vertex's normalised Y, curved (`sqrt`) so the horizon band reads
/// as a tight strip rather than a linear wash — see the parent ticket's
/// "colouring the dome". Y < 0 (below the horizon ring) holds flat
/// `horizon` colour: there's terrain down there, but the dome is still
/// visible past the render-distance edge when looking down from height.
///
/// Colour math happens in linear space (`Color::to_linear`/[`Mix`]) per
/// [`super::super::world::mesh`]'s vertex-colour-channel convention — feeding
/// sRGB straight into `ATTRIBUTE_COLOR` would wash the gradient out.
pub(crate) fn build_dome_mesh(zenith: Color, horizon: Color) -> Mesh {
    let zenith = zenith.to_linear();
    let horizon = horizon.to_linear();

    let vertex_count = ((STACKS + 1) * (SECTORS + 1)) as usize;
    let mut positions = Vec::with_capacity(vertex_count);
    let mut normals = Vec::with_capacity(vertex_count);
    let mut uvs = Vec::with_capacity(vertex_count);
    let mut colors = Vec::with_capacity(vertex_count);

    for i in 0..=STACKS {
        // phi sweeps from the top pole (0 rad, +Y) to the bottom pole (PI
        // rad, -Y); y = cos(phi) is therefore monotonically decreasing in i.
        let phi = i as f32 / STACKS as f32 * std::f32::consts::PI;
        let (ring_radius, y) = phi.sin_cos();
        let t = y.max(0.0).sqrt();
        let color = horizon.mix(&zenith, t);
        let color = [color.red, color.green, color.blue, color.alpha];

        for j in 0..=SECTORS {
            let theta = j as f32 / SECTORS as f32 * std::f32::consts::TAU;
            let (sin_t, cos_t) = theta.sin_cos();
            let dir = Vec3::new(ring_radius * cos_t, y, ring_radius * sin_t);

            positions.push((dir * RADIUS).to_array());
            normals.push((-dir).to_array());
            uvs.push([j as f32 / SECTORS as f32, i as f32 / STACKS as f32]);
            colors.push(color);
        }
    }

    // Two triangles per sector/stack cell, wound so the front face (Bevy:
    // CCW as seen from the viewer) faces inward — the sky camera never
    // leaves the origin, i.e. always views the dome from inside.
    let row = SECTORS + 1;
    let mut indices = Vec::with_capacity((STACKS * SECTORS * 6) as usize);
    for i in 0..STACKS {
        for j in 0..SECTORS {
            let a = i * row + j;
            let b = a + row;
            let c = a + 1;
            let d = b + 1;
            indices.extend_from_slice(&[a, b, c, b, d, c]);
        }
    }

    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
    .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs)
    .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, colors)
    .with_inserted_indices(Indices::U32(indices))
}

/// Rebuilds the dome's vertex colours in place whenever [`super::SkyPalette`]
/// changes (`Res::is_changed()` — true the frame it's first inserted too, so
/// the dome is coloured correctly from frame one). 018's day/night clock
/// ticks this every frame it's running; at ~500 vertices that's free, which
/// is the entire reason this bakes colour into geometry instead of a shader
/// or a cubemap (see the parent ticket).
///
/// `dome_mesh` is `Option` because this system is wired up by [`crate::sky::SkyPlugin`]
/// (`crate::sky::SkyPlugin`) itself, before [`super::spawn_sky_scene`] has
/// necessarily run — a test app that adds the plugin without ever spawning
/// the scene should just have nothing to rebuild, not panic on a missing
/// resource.
pub(crate) fn rebuild_dome_on_palette_change(
    palette: Res<super::SkyPalette>,
    dome_mesh: Option<Res<SkyDomeMesh>>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    if !palette.is_changed() {
        return;
    }
    let Some(dome_mesh) = dome_mesh else {
        return;
    };
    if let Some(mesh) = meshes.get_mut(&dome_mesh.0) {
        *mesh = build_dome_mesh(palette.zenith_color, palette.horizon_color);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::render::mesh::VertexAttributeValues;

    fn dome_colors(mesh: &Mesh) -> Vec<[f32; 4]> {
        let VertexAttributeValues::Float32x4(colors) =
            mesh.attribute(Mesh::ATTRIBUTE_COLOR).unwrap()
        else {
            panic!("expected Float32x4 vertex colours");
        };
        colors.clone()
    }

    fn dome_positions(mesh: &Mesh) -> Vec<[f32; 3]> {
        let VertexAttributeValues::Float32x3(positions) =
            mesh.attribute(Mesh::ATTRIBUTE_POSITION).unwrap()
        else {
            panic!("expected Float32x3 positions");
        };
        positions.clone()
    }

    /// Ticket 016's "the dome mesh's vertex colours interpolate
    /// zenith->horizon monotonically in Y": walking the vertices from the
    /// bottom ring to the top pole, red should never *decrease* for this
    /// zenith/horizon pair (zenith is redder than horizon here), since `t`
    /// only ever grows as Y grows.
    #[test]
    fn vertex_colours_interpolate_monotonically_with_y() {
        let horizon = Color::linear_rgba(0.0, 0.0, 0.0, 1.0);
        let zenith = Color::linear_rgba(1.0, 0.0, 0.0, 1.0);
        let mesh = build_dome_mesh(zenith, horizon);

        let positions = dome_positions(&mesh);
        let colors = dome_colors(&mesh);

        let mut by_y: Vec<(f32, f32)> = positions
            .iter()
            .zip(colors.iter())
            .map(|(p, c)| (p[1], c[0]))
            .collect();
        by_y.sort_by(|a, b| a.0.total_cmp(&b.0));

        let mut last_red = f32::NEG_INFINITY;
        for (_, red) in by_y {
            assert!(
                red + 1e-6 >= last_red,
                "red channel decreased going up in Y: {red} after {last_red}"
            );
            last_red = red;
        }
    }

    /// The bottom ring (Y = -RADIUS) must be *exactly* `horizon_color` —
    /// ticket 016 calls this out explicitly because it's what has to match
    /// the fog colour with no visible seam.
    #[test]
    fn bottom_ring_is_exactly_horizon_color() {
        let horizon = Color::srgb(0.2, 0.4, 0.6);
        let zenith = Color::srgb(0.9, 0.9, 1.0);
        let mesh = build_dome_mesh(zenith, horizon);

        let positions = dome_positions(&mesh);
        let colors = dome_colors(&mesh);
        let expected = {
            let c = horizon.to_linear();
            [c.red, c.green, c.blue, c.alpha]
        };

        for (p, c) in positions.iter().zip(colors.iter()) {
            if p[1] <= -RADIUS + 1e-3 {
                assert_eq!(*c, expected);
            }
        }
    }

    /// The top pole (Y = +RADIUS) must be exactly `zenith_color` — the curve
    /// (`y.max(0.0).sqrt()`) still reaches exactly 1.0 there.
    #[test]
    fn top_pole_is_exactly_zenith_color() {
        let horizon = Color::srgb(0.2, 0.4, 0.6);
        let zenith = Color::srgb(0.9, 0.9, 1.0);
        let mesh = build_dome_mesh(zenith, horizon);

        let positions = dome_positions(&mesh);
        let colors = dome_colors(&mesh);
        let expected = {
            let c = zenith.to_linear();
            [c.red, c.green, c.blue, c.alpha]
        };

        for (p, c) in positions.iter().zip(colors.iter()) {
            if p[1] >= RADIUS - 1e-3 {
                assert_eq!(*c, expected);
            }
        }
    }

    /// Ticket 016: "changing `SkyPalette` rebuilds the dome colours."
    #[test]
    fn palette_change_rebuilds_the_dome_mesh() {
        let mut app = App::new();
        app.init_resource::<Assets<Mesh>>()
            .init_resource::<super::super::SkyPalette>()
            .add_systems(Update, rebuild_dome_on_palette_change);

        let handle = app
            .world_mut()
            .resource_mut::<Assets<Mesh>>()
            .add(build_dome_mesh(Color::BLACK, Color::BLACK));
        app.insert_resource(SkyDomeMesh(handle.clone()));

        app.update();
        {
            let meshes = app.world().resource::<Assets<Mesh>>();
            let colors = dome_colors(meshes.get(&handle).unwrap());
            let default_palette = super::super::SkyPalette::default();
            let expected = {
                let c = default_palette.horizon_color.to_linear();
                [c.red, c.green, c.blue, c.alpha]
            };
            assert!(
                colors.contains(&expected),
                "expected the default palette's horizon colour after the first tick"
            );
        }

        {
            let mut palette = app.world_mut().resource_mut::<super::super::SkyPalette>();
            palette.horizon_color = Color::srgb(1.0, 0.0, 1.0);
            palette.zenith_color = Color::srgb(0.0, 1.0, 1.0);
        }
        app.update();

        let meshes = app.world().resource::<Assets<Mesh>>();
        let colors = dome_colors(meshes.get(&handle).unwrap());
        let expected = {
            let c = Color::srgb(1.0, 0.0, 1.0).to_linear();
            [c.red, c.green, c.blue, c.alpha]
        };
        assert!(
            colors.contains(&expected),
            "expected the mesh to be rebuilt with the new horizon colour"
        );
    }
}
