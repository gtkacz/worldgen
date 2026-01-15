"""
Blender demo importer for `worldgen` cube-face exports.

This script creates:
- an Ico Sphere
- a material that samples the 6 cube-face images using direction-based cube mapping

It is intended as a quick visualization tool (preview maps, plates, boundaries, normals, etc.).

### Usage
1) Generate outputs from `worldgen` into a folder, e.g.:
   - biome preview: `{name}_biomes_{face}.png`
   - plates:        `{name}_plates_{face}.png`
   - boundaries:    `{name}_boundaries_{face}.png`
   - height png:    `{name}_{face}.png`
   - normal png:    `{name}_normal_{face}.png`

2) In Blender: Scripting workspace → open this file → edit CONFIG below → Run Script.

Notes:
- The cube-face suffixes are: posx, negx, posy, negy, posz, negz.
- This is a simple cube-map sampler; seam perfection is not the goal here.
"""

import os
import bpy


# ----------------------------
# CONFIG (edit these)
# ----------------------------

OUTPUT_DIR = r"./output"  # folder containing `{BASE_NAME}_...` images
BASE_NAME = "planet"

# Choose one:
#   "biomes"     -> `{BASE_NAME}_biomes_{face}.png`
#   "plates"     -> `{BASE_NAME}_plates_{face}.png`
#   "boundaries" -> `{BASE_NAME}_boundaries_{face}.png`
#   "height"     -> `{BASE_NAME}_{face}.png`
#   "normal"     -> `{BASE_NAME}_normal_{face}.png`
MAP_KIND = "biomes"

# Sphere quality
SPHERE_SUBDIVISIONS = 6
SPHERE_RADIUS = 1.0


FACE_SUFFIXES = ["posx", "negx", "posy", "negy", "posz", "negz"]


def _abspath(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def _face_filename(base: str, kind: str, face: str) -> str:
    if kind == "biomes":
        return f"{base}_biomes_{face}.png"
    if kind == "plates":
        return f"{base}_plates_{face}.png"
    if kind == "boundaries":
        return f"{base}_boundaries_{face}.png"
    if kind == "height":
        return f"{base}_{face}.png"
    if kind == "normal":
        return f"{base}_normal_{face}.png"
    raise ValueError(f"Unknown MAP_KIND: {kind}")


def load_face_images(output_dir: str, base: str, kind: str):
    output_dir = _abspath(output_dir)
    images = {}
    missing = []

    for face in FACE_SUFFIXES:
        fn = _face_filename(base, kind, face)
        path = os.path.join(output_dir, fn)
        if not os.path.exists(path):
            missing.append(path)
            continue
        img = bpy.data.images.load(path, check_existing=True)
        images[face] = img

    if missing:
        raise FileNotFoundError(
            "Missing cube-face images:\n" + "\n".join(missing)
        )

    return images


def ensure_cycles():
    # For best results (especially displacement), use Cycles.
    bpy.context.scene.render.engine = "CYCLES"


def new_ico_sphere(name: str):
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=SPHERE_SUBDIVISIONS, radius=SPHERE_RADIUS)
    obj = bpy.context.active_object
    obj.name = name
    return obj


def _new_node(tree, node_type, name=None, location=(0, 0)):
    n = tree.nodes.new(node_type)
    if name:
        n.name = name
        n.label = name
    n.location = location
    return n


def _link(tree, out_socket, in_socket):
    tree.links.new(out_socket, in_socket)


def build_cubemap_sampler_group(group_name: str, images_by_face: dict):
    """
    Creates a node group which:
    - takes a direction vector (world/object space)
    - selects the major axis face
    - computes UV for that face
    - samples the corresponding Image Texture
    """
    # Reuse if exists
    if group_name in bpy.data.node_groups:
        bpy.data.node_groups.remove(bpy.data.node_groups[group_name], do_unlink=True)

    g = bpy.data.node_groups.new(group_name, "ShaderNodeTree")

    # Group IO
    g.inputs.new("NodeSocketVector", "Dir")
    g.outputs.new("NodeSocketColor", "Color")

    n_in = _new_node(g, "NodeGroupInput", location=(-1200, 0))
    n_out = _new_node(g, "NodeGroupOutput", location=(1200, 0))

    # Normalize dir
    n_norm = _new_node(g, "ShaderNodeVectorMath", name="NormalizeDir", location=(-1000, 0))
    n_norm.operation = "NORMALIZE"
    _link(g, n_in.outputs["Dir"], n_norm.inputs[0])

    n_sep = _new_node(g, "ShaderNodeSeparateXYZ", name="SepXYZ", location=(-800, 0))
    _link(g, n_norm.outputs["Vector"], n_sep.inputs["Vector"])

    # abs(dir)
    n_abs = _new_node(g, "ShaderNodeVectorMath", name="AbsDir", location=(-800, -200))
    n_abs.operation = "ABSOLUTE"
    _link(g, n_norm.outputs["Vector"], n_abs.inputs[0])

    n_abs_sep = _new_node(g, "ShaderNodeSeparateXYZ", name="SepAbsXYZ", location=(-600, -200))
    _link(g, n_abs.outputs["Vector"], n_abs_sep.inputs["Vector"])

    # major axis tests (>=)
    def ge(a, b, loc):
        n = _new_node(g, "ShaderNodeMath", location=loc)
        n.operation = "GREATER_THAN"
        n.inputs[0].default_value = 0.0
        n.inputs[1].default_value = 0.0
        _link(g, a, n.inputs[0])
        _link(g, b, n.inputs[1])
        return n

    def mul(a, b, loc):
        n = _new_node(g, "ShaderNodeMath", location=loc)
        n.operation = "MULTIPLY"
        _link(g, a, n.inputs[0])
        _link(g, b, n.inputs[1])
        return n

    def sub_one(a, loc):
        n = _new_node(g, "ShaderNodeMath", location=loc)
        n.operation = "SUBTRACT"
        n.inputs[0].default_value = 1.0
        _link(g, a, n.inputs[1])
        return n

    def add(a, b, loc):
        n = _new_node(g, "ShaderNodeMath", location=loc)
        n.operation = "ADD"
        _link(g, a, n.inputs[0])
        _link(g, b, n.inputs[1])
        return n

    def div(a, b, loc):
        n = _new_node(g, "ShaderNodeMath", location=loc)
        n.operation = "DIVIDE"
        _link(g, a, n.inputs[0])
        _link(g, b, n.inputs[1])
        return n

    def neg(a, loc):
        n = _new_node(g, "ShaderNodeMath", location=loc)
        n.operation = "MULTIPLY"
        n.inputs[0].default_value = -1.0
        _link(g, a, n.inputs[1])
        return n

    def remap01(a, loc):
        # 0.5 * a + 0.5
        n_mul = _new_node(g, "ShaderNodeMath", location=loc)
        n_mul.operation = "MULTIPLY"
        n_mul.inputs[1].default_value = 0.5
        _link(g, a, n_mul.inputs[0])
        n_add = _new_node(g, "ShaderNodeMath", location=(loc[0] + 180, loc[1]))
        n_add.operation = "ADD"
        n_add.inputs[1].default_value = 0.5
        _link(g, n_mul.outputs[0], n_add.inputs[0])
        return n_add

    absx = n_abs_sep.outputs["X"]
    absy = n_abs_sep.outputs["Y"]
    absz = n_abs_sep.outputs["Z"]

    # majorX = absx >= absy AND absx >= absz
    ge_x_y = ge(absx, absy, (-400, -40))
    ge_x_z = ge(absx, absz, (-400, -80))
    major_x = mul(ge_x_y.outputs[0], ge_x_z.outputs[0], (-200, -60))

    # majorY = absy > absx AND absy >= absz (use > to reduce ties)
    gt_y_x = _new_node(g, "ShaderNodeMath", location=(-400, -120))
    gt_y_x.operation = "GREATER_THAN"
    _link(g, absy, gt_y_x.inputs[0])
    _link(g, absx, gt_y_x.inputs[1])
    ge_y_z = ge(absy, absz, (-400, -160))
    major_y = mul(gt_y_x.outputs[0], ge_y_z.outputs[0], (-200, -140))

    # majorZ = 1 - max(majorX, majorY)
    n_max_xy = _new_node(g, "ShaderNodeMath", location=(-200, -220))
    n_max_xy.operation = "MAXIMUM"
    _link(g, major_x.outputs[0], n_max_xy.inputs[0])
    _link(g, major_y.outputs[0], n_max_xy.inputs[1])
    major_z = sub_one(n_max_xy.outputs[0], (0, -220))

    # sign tests
    def ge_zero(sock, loc):
        n = _new_node(g, "ShaderNodeMath", location=loc)
        n.operation = "GREATER_THAN"
        _link(g, sock, n.inputs[0])
        n.inputs[1].default_value = -1e-20  # treat 0 as positive
        return n

    sx = ge_zero(n_sep.outputs["X"], (-600, 120))
    sy = ge_zero(n_sep.outputs["Y"], (-600, 80))
    sz = ge_zero(n_sep.outputs["Z"], (-600, 40))

    sxn = sub_one(sx.outputs[0], (-400, 120))
    syn = sub_one(sy.outputs[0], (-400, 80))
    szn = sub_one(sz.outputs[0], (-400, 40))

    mask_posx = mul(major_x.outputs[0], sx.outputs[0], (-200, 120))
    mask_negx = mul(major_x.outputs[0], sxn.outputs[0], (-200, 90))
    mask_posy = mul(major_y.outputs[0], sy.outputs[0], (-200, 60))
    mask_negy = mul(major_y.outputs[0], syn.outputs[0], (-200, 30))
    mask_posz = mul(major_z.outputs[0], sz.outputs[0], (-200, 0))
    mask_negz = mul(major_z.outputs[0], szn.outputs[0], (-200, -30))

    # denom with epsilon
    def denom(abs_sock, loc):
        n = _new_node(g, "ShaderNodeMath", location=loc)
        n.operation = "ADD"
        _link(g, abs_sock, n.inputs[0])
        n.inputs[1].default_value = 1e-6
        return n

    dx = denom(absx, (-600, -320))
    dy = denom(absy, (-600, -360))
    dz = denom(absz, (-600, -400))

    x = n_sep.outputs["X"]
    y = n_sep.outputs["Y"]
    z = n_sep.outputs["Z"]

    # Per-face UVs (cube-map conventions; may be flipped depending on your expectations)
    # +X: u = -z/absx, v =  y/absx
    u_posx = remap01(div(neg(z, (-400, -500)).outputs[0], dx.outputs[0], (-220, -500)).outputs[0], (0, -500))
    v_posx = remap01(div(y, dx.outputs[0], (-220, -540)).outputs[0], (0, -540))

    # -X: u =  z/absx, v =  y/absx
    u_negx = remap01(div(z, dx.outputs[0], (-220, -600)).outputs[0], (0, -600))
    v_negx = remap01(div(y, dx.outputs[0], (-220, -640)).outputs[0], (0, -640))

    # +Y: u =  x/absy, v = -z/absy
    u_posy = remap01(div(x, dy.outputs[0], (-220, -700)).outputs[0], (0, -700))
    v_posy = remap01(div(neg(z, (-400, -740)).outputs[0], dy.outputs[0], (-220, -740)).outputs[0], (0, -740))

    # -Y: u =  x/absy, v =  z/absy
    u_negy = remap01(div(x, dy.outputs[0], (-220, -800)).outputs[0], (0, -800))
    v_negy = remap01(div(z, dy.outputs[0], (-220, -840)).outputs[0], (0, -840))

    # +Z: u =  x/absz, v =  y/absz
    u_posz = remap01(div(x, dz.outputs[0], (-220, -900)).outputs[0], (0, -900))
    v_posz = remap01(div(y, dz.outputs[0], (-220, -940)).outputs[0], (0, -940))

    # -Z: u = -x/absz, v =  y/absz
    u_negz = remap01(div(neg(x, (-400, -980)).outputs[0], dz.outputs[0], (-220, -980)).outputs[0], (0, -980))
    v_negz = remap01(div(y, dz.outputs[0], (-220, -1020)).outputs[0], (0, -1020))

    def image_tex(face, uv_u, uv_v, loc):
        n_uv = _new_node(g, "ShaderNodeCombineXYZ", location=(loc[0] - 220, loc[1]))
        _link(g, uv_u.outputs[0], n_uv.inputs["X"])
        _link(g, uv_v.outputs[0], n_uv.inputs["Y"])
        n_uv.inputs["Z"].default_value = 0.0

        tex = _new_node(g, "ShaderNodeTexImage", name=f"Tex_{face}", location=loc)
        tex.image = images_by_face[face]
        tex.interpolation = "Linear"
        _link(g, n_uv.outputs["Vector"], tex.inputs["Vector"])
        return tex

    tex_posx = image_tex("posx", u_posx, v_posx, (200, 120))
    tex_negx = image_tex("negx", u_negx, v_negx, (200, 80))
    tex_posy = image_tex("posy", u_posy, v_posy, (200, 40))
    tex_negy = image_tex("negy", u_negy, v_negy, (200, 0))
    tex_posz = image_tex("posz", u_posz, v_posz, (200, -40))
    tex_negz = image_tex("negz", u_negz, v_negz, (200, -80))

    def mix(prev_color, next_color, fac, loc):
        n = _new_node(g, "ShaderNodeMixRGB", location=loc)
        n.blend_type = "MIX"
        n.inputs["Fac"].default_value = 0.0
        _link(g, prev_color, n.inputs["Color1"])
        _link(g, next_color, n.inputs["Color2"])
        _link(g, fac, n.inputs["Fac"])
        return n

    # chain mixes; masks are mutually exclusive (0 or 1)
    base = _new_node(g, "ShaderNodeRGB", location=(520, 140))
    base.outputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)

    m1 = mix(base.outputs["Color"], tex_posx.outputs["Color"], mask_posx.outputs[0], (700, 120))
    m2 = mix(m1.outputs["Color"], tex_negx.outputs["Color"], mask_negx.outputs[0], (880, 120))
    m3 = mix(m2.outputs["Color"], tex_posy.outputs["Color"], mask_posy.outputs[0], (1060, 120))
    m4 = mix(m3.outputs["Color"], tex_negy.outputs["Color"], mask_negy.outputs[0], (1240, 120))
    m5 = mix(m4.outputs["Color"], tex_posz.outputs["Color"], mask_posz.outputs[0], (1420, 120))
    m6 = mix(m5.outputs["Color"], tex_negz.outputs["Color"], mask_negz.outputs[0], (1600, 120))

    _link(g, m6.outputs["Color"], n_out.inputs["Color"])
    return g


def build_material_with_cubemap(images_by_face: dict, mat_name="WorldgenMaterial"):
    mat = bpy.data.materials.new(mat_name)
    mat.use_nodes = True

    nt = mat.node_tree
    nt.nodes.clear()

    n_out = _new_node(nt, "ShaderNodeOutputMaterial", location=(500, 0))
    n_bsdf = _new_node(nt, "ShaderNodeBsdfPrincipled", location=(250, 0))
    _link(nt, n_bsdf.outputs["BSDF"], n_out.inputs["Surface"])

    n_geo = _new_node(nt, "ShaderNodeNewGeometry", location=(-450, 0))
    n_norm = _new_node(nt, "ShaderNodeVectorMath", location=(-250, 0))
    n_norm.operation = "NORMALIZE"
    _link(nt, n_geo.outputs["Position"], n_norm.inputs[0])

    group = build_cubemap_sampler_group("WorldgenCubeMapSampler", images_by_face)
    n_group = _new_node(nt, "ShaderNodeGroup", location=(-20, 0))
    n_group.node_tree = group
    _link(nt, n_norm.outputs["Vector"], n_group.inputs["Dir"])

    _link(nt, n_group.outputs["Color"], n_bsdf.inputs["Base Color"])

    # If user chose normal maps, wire it to the Normal input (basic preview).
    if MAP_KIND == "normal":
        n_nm = _new_node(nt, "ShaderNodeNormalMap", location=(120, -200))
        n_nm.space = "TANGENT"
        _link(nt, n_group.outputs["Color"], n_nm.inputs["Color"])
        _link(nt, n_nm.outputs["Normal"], n_bsdf.inputs["Normal"])

    return mat


def main():
    ensure_cycles()
    images = load_face_images(OUTPUT_DIR, BASE_NAME, MAP_KIND)
    obj = new_ico_sphere("WorldgenPlanet")
    mat = build_material_with_cubemap(images, "Worldgen_" + MAP_KIND)
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)

    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    print(f"[worldgen] Imported cube-face '{MAP_KIND}' maps for '{BASE_NAME}' from { _abspath(OUTPUT_DIR) }")


if __name__ == "__main__":
    main()

