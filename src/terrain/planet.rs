//! Planet and CubeFace data structures.

use serde::{Deserialize, Serialize};
use crate::geometry::CubeFaceId;
use crate::tectonics::{TectonicPlate, SphericalVoronoi, PlateBoundary};

/// Represents a procedurally generated planet.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Planet {
    /// Planet radius in kilometers.
    pub radius: f32,
    /// Master random seed for generation.
    pub seed: u64,
    /// The six cube faces containing terrain data.
    pub faces: [CubeFace; 6],
    /// Tectonic plates (populated after tectonic stage).
    #[serde(skip)]
    pub plates: Option<Vec<TectonicPlate>>,
    /// Plate boundaries (populated after tectonic stage).
    #[serde(skip)]
    pub boundaries: Option<Vec<PlateBoundary>>,
    /// Voronoi tessellation for plate assignment.
    #[serde(skip)]
    pub voronoi: Option<SphericalVoronoi>,
}

impl Planet {
    /// Creates a new planet with the specified resolution and seed.
    ///
    /// # Arguments
    /// * `resolution` - Width/height of each cube face in pixels
    /// * `seed` - Random seed for reproducible generation
    /// * `radius` - Planet radius in kilometers (default Earth-like: 6371.0)
    pub fn new(resolution: u32, seed: u64, radius: f32) -> Self {
        Self {
            radius,
            seed,
            faces: [
                CubeFace::new(CubeFaceId::PosX, resolution),
                CubeFace::new(CubeFaceId::NegX, resolution),
                CubeFace::new(CubeFaceId::PosY, resolution),
                CubeFace::new(CubeFaceId::NegY, resolution),
                CubeFace::new(CubeFaceId::PosZ, resolution),
                CubeFace::new(CubeFaceId::NegZ, resolution),
            ],
            plates: None,
            boundaries: None,
            voronoi: None,
        }
    }

    /// Returns true if tectonic data has been generated.
    pub fn has_tectonics(&self) -> bool {
        self.plates.is_some() && self.boundaries.is_some()
    }

    /// Returns the number of tectonic plates, or 0 if tectonics not generated.
    pub fn num_plates(&self) -> usize {
        self.plates.as_ref().map(|p| p.len()).unwrap_or(0)
    }

    /// Creates a new Earth-like planet.
    pub fn earth_like(resolution: u32, seed: u64) -> Self {
        Self::new(resolution, seed, 6371.0)
    }

    /// Returns the resolution of each face (all faces have same resolution).
    pub fn resolution(&self) -> u32 {
        self.faces[0].resolution
    }

    /// Returns a reference to a specific face.
    pub fn face(&self, id: CubeFaceId) -> &CubeFace {
        &self.faces[id.index()]
    }

    /// Returns a mutable reference to a specific face.
    pub fn face_mut(&mut self, id: CubeFaceId) -> &mut CubeFace {
        &mut self.faces[id.index()]
    }

    /// Computes the global min and max height values across all faces.
    pub fn height_range(&self) -> (f32, f32) {
        let mut min = f32::MAX;
        let mut max = f32::MIN;

        for face in &self.faces {
            for &height in &face.heights {
                min = min.min(height);
                max = max.max(height);
            }
        }

        (min, max)
    }
}

/// Represents a single face of the cube-sphere with terrain data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CubeFace {
    /// Which face of the cube this represents.
    pub id: CubeFaceId,
    /// Resolution (width and height) in pixels.
    pub resolution: u32,
    /// Height values stored in row-major order.
    pub heights: Vec<f32>,
    /// Water depth per pixel (populated after erosion stage).
    #[serde(skip)]
    pub water: Option<Vec<f32>>,
    /// Suspended sediment per pixel (populated after erosion stage).
    #[serde(skip)]
    pub sediment: Option<Vec<f32>>,
    /// Net deposited material per pixel (optional; pop. after erosion stage).
    #[serde(skip)]
    pub deposition: Option<Vec<f32>>,
    /// Flow accumulation per pixel (populated after river extraction).
    #[serde(skip)]
    pub flow_accum: Option<Vec<u32>>,
    /// River mask per pixel (0/255) (populated after river extraction).
    #[serde(skip)]
    pub river_mask: Option<Vec<u8>>,
    /// Plate assignment per pixel (populated after tectonic stage).
    #[serde(skip)]
    pub plate_ids: Option<Vec<usize>>,
    /// Tectonic uplift values per pixel in km (populated after tectonic stage).
    #[serde(skip)]
    pub tectonic_uplift: Option<Vec<f32>>,

    // --- Climate (Phase 4) annual summaries ---
    /// Coastline distance (km). Ocean pixels are 0.0 (populated after climate stage).
    #[serde(skip)]
    pub coast_distance_km: Option<Vec<f32>>,
    /// Annual mean temperature (°C) (populated after climate stage).
    #[serde(skip)]
    pub temperature_mean_c: Option<Vec<f32>>,
    /// Annual total precipitation (mm/year) (populated after climate stage).
    #[serde(skip)]
    pub precip_annual_mm: Option<Vec<f32>>,
    /// Minimum monthly temperature across the year (°C) (populated after climate stage).
    #[serde(skip)]
    pub temp_min_month_c: Option<Vec<f32>>,
    /// Maximum monthly temperature across the year (°C) (populated after climate stage).
    #[serde(skip)]
    pub temp_max_month_c: Option<Vec<f32>>,
}

impl CubeFace {
    /// Creates a new cube face with the given resolution.
    ///
    /// Heights are initialized to 0.0.
    pub fn new(id: CubeFaceId, resolution: u32) -> Self {
        let size = (resolution as usize) * (resolution as usize);
        Self {
            id,
            resolution,
            heights: vec![0.0; size],
            water: None,
            sediment: None,
            deposition: None,
            flow_accum: None,
            river_mask: None,
            plate_ids: None,
            tectonic_uplift: None,
            coast_distance_km: None,
            temperature_mean_c: None,
            precip_annual_mm: None,
            temp_min_month_c: None,
            temp_max_month_c: None,
        }
    }

    /// Initializes tectonic data arrays.
    pub fn init_tectonic_data(&mut self) {
        let size = self.pixel_count();
        self.plate_ids = Some(vec![0; size]);
        self.tectonic_uplift = Some(vec![0.0; size]);
    }

    /// Returns true if tectonic data has been initialized.
    pub fn has_tectonic_data(&self) -> bool {
        self.plate_ids.is_some() && self.tectonic_uplift.is_some()
    }

    /// Applies tectonic uplift to the height values.
    pub fn apply_tectonic_uplift(&mut self) {
        if let Some(uplift) = &self.tectonic_uplift {
            for (height, &up) in self.heights.iter_mut().zip(uplift.iter()) {
                *height += up;
            }
        }
    }

    /// Gets the plate ID at a pixel, or None if tectonic data not initialized.
    pub fn get_plate_id(&self, x: u32, y: u32) -> Option<usize> {
        self.plate_ids.as_ref().map(|ids| {
            ids[(y * self.resolution + x) as usize]
        })
    }

    /// Sets the plate ID at a pixel.
    pub fn set_plate_id(&mut self, x: u32, y: u32, plate_id: usize) {
        if let Some(ids) = &mut self.plate_ids {
            ids[(y * self.resolution + x) as usize] = plate_id;
        }
    }

    /// Gets the tectonic uplift at a pixel, or None if not initialized.
    pub fn get_tectonic_uplift(&self, x: u32, y: u32) -> Option<f32> {
        self.tectonic_uplift.as_ref().map(|uplift| {
            uplift[(y * self.resolution + x) as usize]
        })
    }

    /// Sets the tectonic uplift at a pixel.
    pub fn set_tectonic_uplift(&mut self, x: u32, y: u32, value: f32) {
        if let Some(uplift) = &mut self.tectonic_uplift {
            uplift[(y * self.resolution + x) as usize] = value;
        }
    }

    /// Returns the height at the given pixel coordinate.
    ///
    /// # Panics
    /// Panics if x or y is out of bounds.
    pub fn get_height(&self, x: u32, y: u32) -> f32 {
        debug_assert!(x < self.resolution && y < self.resolution);
        self.heights[(y * self.resolution + x) as usize]
    }

    /// Sets the height at the given pixel coordinate.
    ///
    /// # Panics
    /// Panics if x or y is out of bounds.
    pub fn set_height(&mut self, x: u32, y: u32, height: f32) {
        debug_assert!(x < self.resolution && y < self.resolution);
        self.heights[(y * self.resolution + x) as usize] = height;
    }

    /// Converts a pixel coordinate to UV coordinates in [0, 1] range.
    ///
    /// The returned UV maps the pixel center (hence +0.5).
    pub fn pixel_to_uv(&self, x: u32, y: u32) -> (f32, f32) {
        let u = (x as f32 + 0.5) / self.resolution as f32;
        let v = (y as f32 + 0.5) / self.resolution as f32;
        (u, v)
    }

    /// Converts UV coordinates to the nearest pixel coordinate.
    pub fn uv_to_pixel(&self, u: f32, v: f32) -> (u32, u32) {
        let x = ((u * self.resolution as f32) as u32).min(self.resolution - 1);
        let y = ((v * self.resolution as f32) as u32).min(self.resolution - 1);
        (x, y)
    }

    /// Returns the minimum height value in this face.
    pub fn min_height(&self) -> f32 {
        self.heights.iter().cloned().fold(f32::MAX, f32::min)
    }

    /// Returns the maximum height value in this face.
    pub fn max_height(&self) -> f32 {
        self.heights.iter().cloned().fold(f32::MIN, f32::max)
    }

    /// Returns (min, max) height range for this face.
    pub fn height_range(&self) -> (f32, f32) {
        (self.min_height(), self.max_height())
    }

    /// Returns the total number of pixels in this face.
    pub fn pixel_count(&self) -> usize {
        (self.resolution as usize) * (self.resolution as usize)
    }

    /// Returns an iterator over all (x, y) pixel coordinates.
    pub fn pixel_coords(&self) -> impl Iterator<Item = (u32, u32)> + '_ {
        let res = self.resolution;
        (0..res).flat_map(move |y| (0..res).map(move |x| (x, y)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_planet_creation() {
        let planet = Planet::new(256, 42, 6371.0);
        assert_eq!(planet.resolution(), 256);
        assert_eq!(planet.seed, 42);
        assert_eq!(planet.radius, 6371.0);
        assert_eq!(planet.faces.len(), 6);
    }

    #[test]
    fn test_cube_face_creation() {
        let face = CubeFace::new(CubeFaceId::PosX, 128);
        assert_eq!(face.resolution, 128);
        assert_eq!(face.heights.len(), 128 * 128);
        assert!(face.heights.iter().all(|&h| h == 0.0));
    }

    #[test]
    fn test_get_set_height() {
        let mut face = CubeFace::new(CubeFaceId::PosZ, 64);
        face.set_height(10, 20, 0.5);
        assert_eq!(face.get_height(10, 20), 0.5);
    }

    #[test]
    fn test_pixel_to_uv() {
        let face = CubeFace::new(CubeFaceId::PosY, 100);

        // Center pixel should be near (0.5, 0.5)
        let (u, v) = face.pixel_to_uv(49, 49);
        assert!((u - 0.495).abs() < 0.01);
        assert!((v - 0.495).abs() < 0.01);

        // Corner pixels
        let (u, v) = face.pixel_to_uv(0, 0);
        assert!((u - 0.005).abs() < 0.01);
        assert!((v - 0.005).abs() < 0.01);
    }

    #[test]
    fn test_height_range() {
        let mut face = CubeFace::new(CubeFaceId::NegX, 32);
        face.set_height(0, 0, -0.5);
        face.set_height(31, 31, 1.5);

        let (min, max) = face.height_range();
        assert_eq!(min, -0.5);
        assert_eq!(max, 1.5);
    }

    #[test]
    fn test_pixel_coords_iterator() {
        let face = CubeFace::new(CubeFaceId::PosZ, 4);
        let coords: Vec<_> = face.pixel_coords().collect();

        assert_eq!(coords.len(), 16);
        assert_eq!(coords[0], (0, 0));
        assert_eq!(coords[1], (1, 0));
        assert_eq!(coords[4], (0, 1));
        assert_eq!(coords[15], (3, 3));
    }
}
