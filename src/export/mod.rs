//! Export module for saving terrain data to various file formats.
//!
//! Supports 16-bit PNG for universal compatibility, RAW formats
//! for game engine imports, and plate visualization maps.

mod png;
mod raw;
mod plate_map;
mod biome_map;

pub use png::{
    export_face_png,
    export_planet_png,
    export_face_scalar_png_f32,
    export_face_mask_png_u8,
    PngExportOptions,
};
pub use raw::{export_face_raw, export_planet_raw, RawFormat};
pub use plate_map::{
    export_face_plate_map, export_planet_plate_map,
    export_face_boundary_map, export_planet_boundary_map,
    generate_plate_colors, generate_plate_colors_by_type,
    PlateMapOptions, PlateMapError,
};
pub use biome_map::{
    export_face_biome_map_png, export_planet_biome_map_png,
    BiomeMapOptions, BiomeMapError,
};
