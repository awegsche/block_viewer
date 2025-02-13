use crate::camera::CameraPlugin;
use bevy::{
    image::{ImageLoaderSettings, ImageSampler},
    prelude::*,
};
use chunks::{
    check_block_texture_loading, check_for_texture_atlas, load_block_textures, update_chunk_system, Chunk, Chunks, RenderBlock
};
use ranvil::{get_saves, Save};

mod camera;
mod chunks;

#[derive(Debug, States, Default, Clone, Eq, PartialEq, Hash)]
enum AppState {
    #[default]
    LoadingTextures,
    Running,
    TexturesLoaded,
    AtlasInserted,
}

#[derive(Component)]
struct StateText;

fn main() {
    println!("Hello, world!");

    let mut saves = get_saves().unwrap();

    for save in saves.iter() {
        println!("{}", save);
    }

    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(CameraPlugin)
        .init_state::<AppState>()
        .add_systems(OnEnter(AppState::LoadingTextures), load_block_textures)
        .add_systems(
            Update,
            check_block_texture_loading.run_if(in_state(AppState::LoadingTextures)),
        )
        .add_systems(Update, check_for_texture_atlas.run_if(in_state(AppState::TexturesLoaded)))
        .add_systems(Startup, setup)
        //.add_systems(OnEnter(AppState::AtlasInserted), load_chunks)
        .add_systems(Update, update_chunk_system.run_if(in_state(AppState::AtlasInserted)))
        .run();
}


fn setup(mut commands: Commands) {
    commands.spawn((Text::new("Startup"), TextFont::default(), StateText));

    let new_world = ranvil::get_save("Jan25").unwrap();

    Chunks::from_savegame(new_world, &mut commands).unwrap();

}

fn coord_to_idx(x: i32, y: i32, z: i32, width: i32, depth: i32) -> usize {
    (x + z * width + y * width * depth) as usize
}

pub enum Side {
    Top,
    Bottom,
    Left,
    Right,
    Front,
    Back,
    None,
}
