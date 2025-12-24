from pathlib import Path

from pydff.model import DFFModel


def export_obj(model: DFFModel, obj_path, scale: float = 1.0) -> None:
    geometry = model.merged_geometry()

    obj_path = Path(obj_path)
    mtl_path = obj_path.with_suffix(".mtl")

    obj_lines = []
    obj_lines.append(f"mtllib {mtl_path.name}")
    obj_lines.append(f"o {model.name or 'Model'}")

    for v in geometry.vertices:
        x, y, z = v[0] * scale, v[1] * scale, v[2] * scale
        obj_lines.append(f"v {x:.6f} {y:.6f} {z:.6f}")

    if geometry.normals:
        for n in geometry.normals:
            obj_lines.append(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}")

    if geometry.uvs:
        for uv in geometry.uvs:
            obj_lines.append(f"vt {uv[0]:.6f} {uv[1]:.6f}")

    if geometry.materials:
        for mat in geometry.materials:
            obj_lines.append(f"usemtl {mat.name or 'default'}")

    for face in geometry.faces:
        v1, v2, v3 = face[0] + 1, face[1] + 1, face[2] + 1

        if geometry.normals and geometry.uvs:
            obj_lines.append(f"f {v1}/{v1}/{v1} {v2}/{v2}/{v2} {v3}/{v3}/{v3}")
        elif geometry.normals:
            obj_lines.append(f"f {v1}//{v1} {v2}//{v2} {v3}//{v3}")
        elif geometry.uvs:
            obj_lines.append(f"f {v1}/{v1} {v2}/{v2} {v3}/{v3}")
        else:
            obj_lines.append(f"f {v1} {v2} {v3}")

    with open(obj_path, "w") as f:
        f.write("\n".join(obj_lines) + "\n")

    mtl_lines = []
    for mat in geometry.materials:
        mtl_lines.append(f"newmtl {mat.name or 'default'}")
        r, g, b, a = mat.color
        mtl_lines.append(f"Kd {r:.6f} {g:.6f} {b:.6f}")
        mtl_lines.append(f"d {a:.6f}")

        if mat.texture_name:
            mtl_lines.append(f"map_Kd {mat.texture_name}.png")

        mtl_lines.append("")

    with open(mtl_path, "w") as f:
        f.write("\n".join(mtl_lines))