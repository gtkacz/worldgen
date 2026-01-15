// Hydraulic erosion compute kernels (Mei et al.-inspired pipe model).
//
// This shader expects `neighbors.wgsl` to be concatenated before it so that
// `neighbor4(...)` is available.

struct Params {
  resolution: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
  p0: vec4<f32>, // rainfall, evaporation, Ks, Kd
  p1: vec4<f32>, // Kc, dt, talus, thermal_strength
};

@group(0) @binding(0) var<uniform> params: Params;

// Inputs (sampled)
@group(0) @binding(1) var height_in: texture_2d_array<f32>;
@group(0) @binding(2) var water_in: texture_2d_array<f32>;
@group(0) @binding(3) var sediment_in: texture_2d_array<f32>;
@group(0) @binding(4) var flux_in: texture_2d_array<f32>; // rgba = (E,W,N,S)

// Outputs (storage)
@group(0) @binding(5) var height_out: texture_storage_2d_array<r32float, write>;
@group(0) @binding(6) var water_out: texture_storage_2d_array<r32float, write>;
@group(0) @binding(7) var sediment_out: texture_storage_2d_array<r32float, write>;
@group(0) @binding(8) var flux_out: texture_storage_2d_array<rgba32float, write>;

fn load_scalar(tex: texture_2d_array<f32>, face: u32, x: u32, y: u32) -> f32 {
  return textureLoad(tex, vec2<i32>(i32(x), i32(y)), i32(face), 0).x;
}

fn load_flux(face: u32, x: u32, y: u32) -> vec4<f32> {
  return textureLoad(flux_in, vec2<i32>(i32(x), i32(y)), i32(face), 0);
}

fn store_height(face: u32, x: u32, y: u32, v: f32) {
  textureStore(height_out, vec2<i32>(i32(x), i32(y)), i32(face), vec4<f32>(v, 0.0, 0.0, 0.0));
}

fn store_water(face: u32, x: u32, y: u32, v: f32) {
  textureStore(water_out, vec2<i32>(i32(x), i32(y)), i32(face), vec4<f32>(v, 0.0, 0.0, 0.0));
}

fn store_sediment(face: u32, x: u32, y: u32, v: f32) {
  textureStore(sediment_out, vec2<i32>(i32(x), i32(y)), i32(face), vec4<f32>(v, 0.0, 0.0, 0.0));
}

fn store_flux(face: u32, x: u32, y: u32, f: vec4<f32>) {
  textureStore(flux_out, vec2<i32>(i32(x), i32(y)), i32(face), f);
}

fn in_bounds(x: u32, y: u32, res: u32) -> bool {
  return x < res && y < res;
}

// Pass 1: rainfall/source
@compute @workgroup_size(8, 8, 1)
fn rainfall(@builtin(global_invocation_id) gid: vec3<u32>) {
  let res = params.resolution;
  if (!in_bounds(gid.x, gid.y, res) || gid.z >= 6u) { return; }

  let face = gid.z;
  let x = gid.x;
  let y = gid.y;

  let h = load_scalar(height_in, face, x, y);
  let w = load_scalar(water_in, face, x, y);
  let s = load_scalar(sediment_in, face, x, y);
  let f = load_flux(face, x, y);

  store_height(face, x, y, h);
  store_water(face, x, y, w + params.p0.x);
  store_sediment(face, x, y, s);
  store_flux(face, x, y, f);
}

// Pass 2: flow simulation (compute outflow fluxes)
@compute @workgroup_size(8, 8, 1)
fn flow(@builtin(global_invocation_id) gid: vec3<u32>) {
  let res = params.resolution;
  if (!in_bounds(gid.x, gid.y, res) || gid.z >= 6u) { return; }

  let face = gid.z;
  let x = gid.x;
  let y = gid.y;

  let h = load_scalar(height_in, face, x, y);
  let w = load_scalar(water_in, face, x, y);
  let s = load_scalar(sediment_in, face, x, y);

  let total = h + w;

  // 4-neighborhood
  let e = neighbor4(res, face, x, y, 1, 0);
  let wv = neighbor4(res, face, x, y, -1, 0);
  let n = neighbor4(res, face, x, y, 0, 1);
  let sv = neighbor4(res, face, x, y, 0, -1);

  let te = load_scalar(height_in, e.x, e.y, e.z) + load_scalar(water_in, e.x, e.y, e.z);
  let tw = load_scalar(height_in, wv.x, wv.y, wv.z) + load_scalar(water_in, wv.x, wv.y, wv.z);
  let tn = load_scalar(height_in, n.x, n.y, n.z) + load_scalar(water_in, n.x, n.y, n.z);
  let ts = load_scalar(height_in, sv.x, sv.y, sv.z) + load_scalar(water_in, sv.x, sv.y, sv.z);

  var fE = max(0.0, total - te);
  var fW = max(0.0, total - tw);
  var fN = max(0.0, total - tn);
  var fS = max(0.0, total - ts);

  // Scale so we never send more water than we have.
  let sum = fE + fW + fN + fS;
  if (sum > 1e-6 && sum > w) {
    let scale = w / sum;
    fE *= scale;
    fW *= scale;
    fN *= scale;
    fS *= scale;
  }

  store_height(face, x, y, h);
  store_water(face, x, y, w);
  store_sediment(face, x, y, s);
  store_flux(face, x, y, vec4<f32>(fE, fW, fN, fS));
}

// Pass 3: update water using flux divergence + evaporation
@compute @workgroup_size(8, 8, 1)
fn water_update(@builtin(global_invocation_id) gid: vec3<u32>) {
  let res = params.resolution;
  if (!in_bounds(gid.x, gid.y, res) || gid.z >= 6u) { return; }

  let face = gid.z;
  let x = gid.x;
  let y = gid.y;

  let h = load_scalar(height_in, face, x, y);
  let w0 = load_scalar(water_in, face, x, y);
  let s = load_scalar(sediment_in, face, x, y);
  let f = load_flux(face, x, y);

  let outflow = f.x + f.y + f.z + f.w;

  let west = neighbor4(res, face, x, y, -1, 0);
  let east = neighbor4(res, face, x, y, 1, 0);
  let south = neighbor4(res, face, x, y, 0, -1);
  let north = neighbor4(res, face, x, y, 0, 1);

  // Inflow from neighbors: from west neighbor's east component, etc.
  let f_west = load_flux(west.x, west.y, west.z).x;
  let f_east = load_flux(east.x, east.y, east.z).y;
  let f_south = load_flux(south.x, south.y, south.z).z;
  let f_north = load_flux(north.x, north.y, north.z).w;

  let inflow = f_west + f_east + f_south + f_north;

  let evap = params.p0.y;
  let w1 = max(0.0, w0 + inflow - outflow);
  let w2 = w1 * (1.0 - evap);

  store_height(face, x, y, h);
  store_water(face, x, y, w2);
  store_sediment(face, x, y, s);
  store_flux(face, x, y, f);
}

// Pass 4: erosion / deposition
@compute @workgroup_size(8, 8, 1)
fn erosion_deposition(@builtin(global_invocation_id) gid: vec3<u32>) {
  let res = params.resolution;
  if (!in_bounds(gid.x, gid.y, res) || gid.z >= 6u) { return; }

  let face = gid.z;
  let x = gid.x;
  let y = gid.y;

  let h0 = load_scalar(height_in, face, x, y);
  let w = load_scalar(water_in, face, x, y);
  let sed0 = load_scalar(sediment_in, face, x, y);
  let f = load_flux(face, x, y);

  let total = h0 + w;

  // Estimate slope via steepest descent (4-neighborhood).
  let e = neighbor4(res, face, x, y, 1, 0);
  let wv = neighbor4(res, face, x, y, -1, 0);
  let n = neighbor4(res, face, x, y, 0, 1);
  let sv = neighbor4(res, face, x, y, 0, -1);

  let te = load_scalar(height_in, e.x, e.y, e.z) + load_scalar(water_in, e.x, e.y, e.z);
  let tw = load_scalar(height_in, wv.x, wv.y, wv.z) + load_scalar(water_in, wv.x, wv.y, wv.z);
  let tn = load_scalar(height_in, n.x, n.y, n.z) + load_scalar(water_in, n.x, n.y, n.z);
  let ts = load_scalar(height_in, sv.x, sv.y, sv.z) + load_scalar(water_in, sv.x, sv.y, sv.z);

  let min_n = min(min(te, tw), min(tn, ts));
  let slope = max(0.0, total - min_n);

  // Velocity proxy from flux imbalance.
  let vx = f.x - f.y;
  let vy = f.z - f.w;
  let vel = sqrt(vx * vx + vy * vy);

  let Ks = params.p0.z;
  let Kd = params.p0.w;
  let Kc = params.p1.x;

  let capacity = Kc * vel * slope;

  var h1 = h0;
  var sed1 = sed0;
  if (sed0 > capacity) {
    let dep = Kd * (sed0 - capacity);
    sed1 = sed0 - dep;
    h1 = h0 + dep;
  } else {
    let ero = Ks * (capacity - sed0);
    sed1 = sed0 + ero;
    h1 = h0 - ero;
  }

  store_height(face, x, y, h1);
  store_water(face, x, y, w);
  store_sediment(face, x, y, sed1);
  store_flux(face, x, y, f);
}

// Pass 5: sediment transport (simple nearest-neighbor backtrace)
@compute @workgroup_size(8, 8, 1)
fn transport(@builtin(global_invocation_id) gid: vec3<u32>) {
  let res = params.resolution;
  if (!in_bounds(gid.x, gid.y, res) || gid.z >= 6u) { return; }

  let face = gid.z;
  let x = gid.x;
  let y = gid.y;

  let h = load_scalar(height_in, face, x, y);
  let w = load_scalar(water_in, face, x, y);
  let f = load_flux(face, x, y);

  let vx = f.x - f.y;
  let vy = f.z - f.w;

  var src = vec3<u32>(face, x, y);
  if (abs(vx) > abs(vy)) {
    // Flow to +x means upstream is -x, etc.
    let step = select(0, -1, vx > 0.0) + select(0, 1, vx < 0.0);
    if (step != 0) {
      src = neighbor4(res, face, x, y, step, 0);
    }
  } else {
    let step = select(0, -1, vy > 0.0) + select(0, 1, vy < 0.0);
    if (step != 0) {
      src = neighbor4(res, face, x, y, 0, step);
    }
  }

  let sed_src = load_scalar(sediment_in, src.x, src.y, src.z);
  let sed_dst = sed_src; // simple copy (non-conservative but stable)

  store_height(face, x, y, h);
  store_water(face, x, y, w);
  store_sediment(face, x, y, sed_dst);
  store_flux(face, x, y, f);
}

// Optional pass: extra evaporation (can be a no-op if already applied).
@compute @workgroup_size(8, 8, 1)
fn evaporation(@builtin(global_invocation_id) gid: vec3<u32>) {
  let res = params.resolution;
  if (!in_bounds(gid.x, gid.y, res) || gid.z >= 6u) { return; }

  let face = gid.z;
  let x = gid.x;
  let y = gid.y;

  let h = load_scalar(height_in, face, x, y);
  let w = load_scalar(water_in, face, x, y);
  let s = load_scalar(sediment_in, face, x, y);
  let f = load_flux(face, x, y);

  let evap = params.p0.y;
  store_height(face, x, y, h);
  store_water(face, x, y, w * (1.0 - 0.25 * evap));
  store_sediment(face, x, y, s);
  store_flux(face, x, y, f);
}

// Thermal erosion: reduce slopes above the talus threshold.
//
// This is a conservative *slope clamp* style relaxation (gather-only), which
// stabilizes steep edges without explicit mass conservation.
@compute @workgroup_size(8, 8, 1)
fn thermal(@builtin(global_invocation_id) gid: vec3<u32>) {
  let res = params.resolution;
  if (!in_bounds(gid.x, gid.y, res) || gid.z >= 6u) { return; }

  let face = gid.z;
  let x = gid.x;
  let y = gid.y;

  let h0 = load_scalar(height_in, face, x, y);
  let w = load_scalar(water_in, face, x, y);
  let s = load_scalar(sediment_in, face, x, y);
  let f = load_flux(face, x, y);

  let talus = params.p1.z;
  let strength = clamp(params.p1.w, 0.0, 1.0);

  let e = neighbor4(res, face, x, y, 1, 0);
  let wv = neighbor4(res, face, x, y, -1, 0);
  let n = neighbor4(res, face, x, y, 0, 1);
  let sv = neighbor4(res, face, x, y, 0, -1);

  let he = load_scalar(height_in, e.x, e.y, e.z);
  let hw = load_scalar(height_in, wv.x, wv.y, wv.z);
  let hn = load_scalar(height_in, n.x, n.y, n.z);
  let hs = load_scalar(height_in, sv.x, sv.y, sv.z);

  let min_neighbor = min(min(he, hw), min(hn, hs));
  let max_allowed = min_neighbor + talus;

  var h1 = h0;
  if (h0 > max_allowed) {
    h1 = mix(h0, max_allowed, strength);
  }

  store_height(face, x, y, h1);
  store_water(face, x, y, w);
  store_sediment(face, x, y, s);
  store_flux(face, x, y, f);
}

