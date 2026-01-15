// Seam-aware neighbor mapping for cube-sphere faces.
// Must match `src/geometry/neighbors.rs`.

fn map_edge(resolution: u32, face: u32, x: u32, y: u32, edge: u32) -> vec3<u32> {
  let r: u32 = resolution - 1u;

  // edge: 0=Left,1=Right,2=Down,3=Up
  // face: 0=PosX,1=NegX,2=PosY,3=NegY,4=PosZ,5=NegZ

  // PosX
  if (face == 0u && edge == 0u) { return vec3<u32>(4u, r, y); }
  if (face == 0u && edge == 1u) { return vec3<u32>(5u, 0u, y); }
  if (face == 0u && edge == 2u) { return vec3<u32>(3u, r, x); }
  if (face == 0u && edge == 3u) { return vec3<u32>(2u, r, r - x); }

  // NegX
  if (face == 1u && edge == 0u) { return vec3<u32>(5u, r, y); }
  if (face == 1u && edge == 1u) { return vec3<u32>(4u, 0u, y); }
  if (face == 1u && edge == 2u) { return vec3<u32>(3u, 0u, r - x); }
  if (face == 1u && edge == 3u) { return vec3<u32>(2u, 0u, x); }

  // PosY
  if (face == 2u && edge == 0u) { return vec3<u32>(1u, y, r); }
  if (face == 2u && edge == 1u) { return vec3<u32>(0u, r - y, r); }
  if (face == 2u && edge == 2u) { return vec3<u32>(5u, r - x, r); }
  if (face == 2u && edge == 3u) { return vec3<u32>(4u, x, r); }

  // NegY
  if (face == 3u && edge == 0u) { return vec3<u32>(1u, r - y, 0u); }
  if (face == 3u && edge == 1u) { return vec3<u32>(0u, y, 0u); }
  if (face == 3u && edge == 2u) { return vec3<u32>(4u, x, 0u); }
  if (face == 3u && edge == 3u) { return vec3<u32>(5u, r - x, 0u); }

  // PosZ
  if (face == 4u && edge == 0u) { return vec3<u32>(1u, r, y); }
  if (face == 4u && edge == 1u) { return vec3<u32>(0u, 0u, y); }
  if (face == 4u && edge == 2u) { return vec3<u32>(3u, x, 0u); }
  if (face == 4u && edge == 3u) { return vec3<u32>(2u, x, r); }

  // NegZ
  if (face == 5u && edge == 0u) { return vec3<u32>(0u, r, y); }
  if (face == 5u && edge == 1u) { return vec3<u32>(1u, 0u, y); }
  if (face == 5u && edge == 2u) { return vec3<u32>(3u, r - x, r); }
  // (face == 5u && edge == 3u)
  return vec3<u32>(2u, r - x, 0u);
}

fn neighbor4(resolution: u32, face: u32, x: u32, y: u32, dx: i32, dy: i32) -> vec3<u32> {
  let nx: i32 = i32(x) + dx;
  let ny: i32 = i32(y) + dy;
  let res_i: i32 = i32(resolution);

  if (nx >= 0 && nx < res_i && ny >= 0 && ny < res_i) {
    return vec3<u32>(face, u32(nx), u32(ny));
  }

  if (nx < 0) { return map_edge(resolution, face, x, y, 0u); }
  if (nx >= res_i) { return map_edge(resolution, face, x, y, 1u); }
  if (ny < 0) { return map_edge(resolution, face, x, y, 2u); }
  return map_edge(resolution, face, x, y, 3u);
}

