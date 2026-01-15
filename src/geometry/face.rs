//! Cube face identification and enumeration.

use serde::{Deserialize, Serialize};

/// Identifies which face of the cube a point belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum CubeFaceId {
    /// +X face (right)
    PosX = 0,
    /// -X face (left)
    NegX = 1,
    /// +Y face (top)
    PosY = 2,
    /// -Y face (bottom)
    NegY = 3,
    /// +Z face (front)
    PosZ = 4,
    /// -Z face (back)
    NegZ = 5,
}

impl CubeFaceId {
    /// Returns all six cube faces in order.
    pub const fn all() -> [CubeFaceId; 6] {
        [
            CubeFaceId::PosX,
            CubeFaceId::NegX,
            CubeFaceId::PosY,
            CubeFaceId::NegY,
            CubeFaceId::PosZ,
            CubeFaceId::NegZ,
        ]
    }

    /// Returns the face index (0-5).
    pub const fn index(self) -> usize {
        self as usize
    }

    /// Creates a face from an index (0-5).
    pub const fn from_index(index: usize) -> Option<CubeFaceId> {
        match index {
            0 => Some(CubeFaceId::PosX),
            1 => Some(CubeFaceId::NegX),
            2 => Some(CubeFaceId::PosY),
            3 => Some(CubeFaceId::NegY),
            4 => Some(CubeFaceId::PosZ),
            5 => Some(CubeFaceId::NegZ),
            _ => None,
        }
    }

    /// Returns a short name for the face (e.g., "posx", "negy").
    pub const fn short_name(self) -> &'static str {
        match self {
            CubeFaceId::PosX => "posx",
            CubeFaceId::NegX => "negx",
            CubeFaceId::PosY => "posy",
            CubeFaceId::NegY => "negy",
            CubeFaceId::PosZ => "posz",
            CubeFaceId::NegZ => "negz",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_faces() {
        let faces = CubeFaceId::all();
        assert_eq!(faces.len(), 6);
        for (i, face) in faces.iter().enumerate() {
            assert_eq!(face.index(), i);
        }
    }

    #[test]
    fn test_from_index() {
        for i in 0..6 {
            let face = CubeFaceId::from_index(i).unwrap();
            assert_eq!(face.index(), i);
        }
        assert!(CubeFaceId::from_index(6).is_none());
    }

    #[test]
    fn test_short_names() {
        assert_eq!(CubeFaceId::PosX.short_name(), "posx");
        assert_eq!(CubeFaceId::NegY.short_name(), "negy");
    }
}
