//! Shared helpers for climate computations.

use glam::Vec3;

/// Returns `(east, north)` tangent unit vectors at sphere point `p`.
///
/// - `east` points along increasing longitude.\n/// - `north` points toward geographic north.\n///
/// Both are tangent to the sphere at `p`. If `p` is too close to the poles, a stable fallback is used.
pub fn local_tangent_basis(p: Vec3) -> (Vec3, Vec3) {
    // Project global up onto tangent plane for “north”.
    let up = Vec3::Y;
    let mut north = up - p * up.dot(p);
    let nlen = north.length();
    if nlen < 1e-6 {
        // At/near poles: choose an arbitrary orthonormal basis.
        // Pick axis least aligned with p.
        let a = if p.x.abs() < 0.9 { Vec3::X } else { Vec3::Z };
        north = (a - p * a.dot(p)).normalize_or_zero();
    } else {
        north /= nlen;
    }
    let east = p.cross(north).normalize_or_zero();
    (east, north)
}

/// Seasonal phase sine wave for `month_idx` with `months` samples and fractional phase shift.
///
/// Returns values in [-1, 1].
pub fn month_phase_sin(month_idx: u8, months: u8, phase: f32) -> f32 {
    let m = month_idx as f32;
    let months = months.max(1) as f32;
    let frac = (m + 0.5) / months; // sample at month center
    (std::f32::consts::TAU * (frac + phase)).sin()
}

/// Great-circle step along a tangent unit vector.
///
/// For unit sphere point `p`, tangent unit `t`, and angular step `a` (radians),
/// returns normalized `p' = p*cos(a) + t*sin(a)`.
pub fn great_circle_step(p: Vec3, tangent_dir: Vec3, angle_rad: f32) -> Vec3 {
    let c = angle_rad.cos();
    let s = angle_rad.sin();
    (p * c + tangent_dir * s).normalize_or_zero()
}

