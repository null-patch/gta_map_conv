import os
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import logging


from config import Config
from core.models import SceneObject

# Local imports
try:
    from config import Config
    from core.conversion_pipeline import SceneObject
except ImportError:
    class Config:
        def __init__(self): pass
    class SceneObject:
        def __init__(self): pass


logger = logging.getLogger(__name__)

@dataclass
class OBJVertex:
    x: float
    y: float
    z: float
    w: float = 1.0

    def to_string(self) -> str:
        return f"v {self.x:.6f} {self.y:.6f} {self.z:.6f}"

@dataclass
class OBJNormal:
    x: float
    y: float
    z: float

    def to_string(self) -> str:
        return f"vn {self.x:.6f} {self.y:.6f} {self.z:.6f}"

@dataclass
class OBJTexCoord:
    u: float
    v: float
    w: float = 0.0

    def to_string(self) -> str:
        if self.w != 0.0:
            return f"vt {self.u:.6f} {self.v:.6f} {self.w:.6f}"
        return f"vt {self.u:.6f} {self.v:.6f}"

@dataclass
class OBJFace:
    vertices: List[int]
    normals: Optional[List[int]] = None
    texcoords: Optional[List[int]] = None
    material: str = ""

    def to_string(self) -> str:
        parts = []
        for i, v_idx in enumerate(self.vertices):
            part = str(v_idx)

            if self.texcoords and i < len(self.texcoords):
                t_idx = self.texcoords[i]
                if self.normals and i < len(self.normals):
                    n_idx = self.normals[i]
                    part = f"{v_idx}/{t_idx}/{n_idx}"
                else:
                    part = f"{v_idx}/{t_idx}"
            elif self.normals and i < len(self.normals):
                n_idx = self.normals[i]
                part = f"{v_idx}//{n_idx}"

            parts.append(part)

        return "f " + " ".join(parts)
    
@dataclass
class OBJGroup:
    name: str
    faces: List[OBJFace] = field(default_factory=list)
    material: str = ""

    def add_face(self, face: OBJFace):
        self.faces.append(face)

@dataclass
class OBJObject:
    name: str
    vertices: List[OBJVertex] = field(default_factory=list)
    normals: List[OBJNormal] = field(default_factory=list)
    texcoords: List[OBJTexCoord] = field(default_factory=list)
    groups: Dict[str, OBJGroup] = field(default_factory=dict)
    materials: Dict[str, Any] = field(default_factory=dict)

    def add_vertex(self, v: OBJVertex) -> int:
        self.vertices.append(v)
        return len(self.vertices)

    def add_normal(self, n: OBJNormal) -> int:
        self.normals.append(n)
        return len(self.normals)

    def add_texcoord(self, t: OBJTexCoord) -> int:
        self.texcoords.append(t)
        return len(self.texcoords)

    def get_or_create_group(self, name: str, material: str = "") -> OBJGroup:
        if name not in self.groups:
            self.groups[name] = OBJGroup(name=name, material=material)
        return self.groups[name]

    def get_stats(self) -> Dict[str, int]:
        face_count = sum(len(g.faces) for g in self.groups.values())
        return {
            "vertices": len(self.vertices),
            "normals": len(self.normals),
            "texcoords": len(self.texcoords),
            "groups": len(self.groups),
            "faces": face_count,
            "materials": len(self.materials),
        }

class OBJWriter:
    def __init__(self, config: Config):
        self.config = config
        self.objects: Dict[str, OBJObject] = {}
        self.current_object: Optional[OBJObject] = None

    def create_object(self, name: str) -> OBJObject:
        obj = OBJObject(name=name)
        self.objects[name] = obj
        self.current_object = obj
        return obj

    def set_current_object(self, name: str):
        if name in self.objects:
            self.current_object = self.objects[name]
        else:
            self.create_object(name)

    def add_scene_object(self, scene_obj: SceneObject, model_data: Dict[str, Any]) -> int:
        
        if not self.current_object:
            self.create_object(f"object_{scene_obj.id}")

        obj = self.current_object

        vertex_indices: List[int] = []
        normal_indices: List[int] = []
        texcoord_indices: List[int] = []


        for vertex in model_data.get("vertices", []):
            x, y, z = self._transform_vertex(
                vertex,
                scene_obj.position,
                scene_obj.rotation,
                scene_obj.scale,
            )
            x, y, z = self._convert_coordinate_system(x, y, z)
            s = self.config.conversion.scale_factor
            x *= s
            y *= s
            z *= s
            idx = obj.add_vertex(OBJVertex(x, y, z))
            vertex_indices.append(idx)


        for normal in model_data.get("normals", []):
            nx, ny, nz = self._transform_normal(normal, scene_obj.rotation)
            nx, ny, nz = self._convert_coordinate_system(nx, ny, nz, is_normal=True)
            nidx = obj.add_normal(OBJNormal(nx, ny, nz))
            normal_indices.append(nidx)


        for uv in model_data.get("uvs", []):
            u, v = uv[0], uv[1]
            if self.config.blender.flip_uv_vertical:
                v = 1.0 - v
            tidx = obj.add_texcoord(OBJTexCoord(u, v))
            texcoord_indices.append(tidx)

        group_name = f"obj_{scene_obj.id}"
        material_name = scene_obj.texture_dict or "default_material"
        group = obj.get_or_create_group(group_name, material_name)

        for face in model_data.get("faces", []):
            if len(face) < 3:
                continue

            verts = [vertex_indices[i] for i in face if i < len(vertex_indices)]
            if not verts:
                continue

            obj_face = OBJFace(vertices=verts, material=material_name)

            if normal_indices:
                obj_face.normals = [
                    normal_indices[i] for i in face if i < len(normal_indices)
                ]
            if texcoord_indices:
                obj_face.texcoords = [
                    texcoord_indices[i] for i in face if i < len(texcoord_indices)
                ]

            group.add_face(obj_face)

        return len(vertex_indices)


    def _transform_vertex(
        self,
        vertex: List[float],
        position: Tuple[float, float, float],
        rotation: Tuple[float, float, float],
        scale: Tuple[float, float, float],
    ) -> Tuple[float, float, float]:
        x, y, z = vertex
        x *= scale[0]
        y *= scale[1]
        z *= scale[2]

        rx, ry, rz = rotation
        if any(r != 0 for r in rotation):
            rx = math.radians(rx)
            ry = math.radians(ry)
            rz = math.radians(rz)

            # X
            y, z = (
                y * math.cos(rx) - z * math.sin(rx),
                y * math.sin(rx) + z * math.cos(rx),
            )
            # Y
            x, z = (
                x * math.cos(ry) + z * math.sin(ry),
                -x * math.sin(ry) + z * math.cos(ry),
            )
            # Z
            x, y = (
                x * math.cos(rz) - y * math.sin(rz),
                x * math.sin(rz) + y * math.cos(rz),
            )

        x += position[0]
        y += position[1]
        z += position[2]
        return x, y, z

    def _transform_normal(
        self,
        normal: List[float],
        rotation: Tuple[float, float, float],
    ) -> Tuple[float, float, float]:
        nx, ny, nz = normal
        rx, ry, rz = rotation

        if any(r != 0 for r in rotation):
            rx = math.radians(rx)
            ry = math.radians(ry)
            rz = math.radians(rz)

            ny, nz = (
                ny * math.cos(rx) - nz * math.sin(rx),
                ny * math.sin(rx) + nz * math.cos(rx),
            )
            nx, nz = (
                nx * math.cos(ry) + nz * math.sin(ry),
                -nx * math.sin(ry) + nz * math.cos(ry),
            )
            nx, ny = (
                nx * math.cos(rz) - ny * math.sin(rz),
                nx * math.sin(rz) + ny * math.cos(rz),
            )

        length = math.sqrt(nx * nx + ny * ny + nz * nz)
        if length > 0:
            nx /= length
            ny /= length
            nz /= length

        return nx, ny, nz

    def _convert_coordinate_system(
        self,
        x: float,
        y: float,
        z: float,
        is_normal: bool = False,
    ) -> Tuple[float, float, float]:
        if self.config.conversion.coordinate_system == "y_up":
            new_x = x
            new_y = z
            new_z = -y if not is_normal else y
            return new_x, new_y, new_z
        return x, y, z


    def write_to_file(self, file_path: str, mtl_file_name: str = ""):
        logger.info(f"Writing OBJ file: {file_path}")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("# GTA San Andreas Map Export\n")
            f.write("# Generated by GTA SA Map Converter\n")
            f.write(f"# Scale factor: {self.config.conversion.scale_factor}\n")
            f.write(f"# Coordinate system: {self.config.conversion.coordinate_system}\n")
            if mtl_file_name:
                f.write(f"mtllib {mtl_file_name}\n")
            f.write("\n")

            for obj in self.objects.values():
                self._write_object(f, obj)

    def _write_object(self, f, obj: OBJObject):
        f.write(f"# Object: {obj.name}\n")
        f.write(f"o {obj.name}\n\n")

        for v in obj.vertices:
            f.write(v.to_string() + "\n")
        if obj.vertices:
            f.write("\n")

        for t in obj.texcoords:
            f.write(t.to_string() + "\n")
        if obj.texcoords:
            f.write("\n")

        for n in obj.normals:
            f.write(n.to_string() + "\n")
        if obj.normals:
            f.write("\n")

        for group_name, group in obj.groups.items():
            if not group.faces:
                continue
            f.write(f"# Group: {group_name}\n")
            f.write(f"g {group_name}\n")
            if group.material:
                f.write(f"usemtl {group.material}\n")
            for face in group.faces:
                f.write(face.to_string() + "\n")
            f.write("\n")

    def get_stats(self) -> Dict[str, Any]:
        stats = {
            "objects": len(self.objects),
            "total_vertices": 0,
            "total_normals": 0,
            "total_texcoords": 0,
            "total_groups": 0,
            "total_faces": 0,
        }
        for obj in self.objects.values():
            s = obj.get_stats()
            stats["total_vertices"] += s["vertices"]
            stats["total_normals"] += s["normals"]
            stats["total_texcoords"] += s["texcoords"]
            stats["total_groups"] += s["groups"]
            stats["total_faces"] += s["faces"]
        return stats

class MTLWriter:
    def __init__(self, config: Config):
        self.config = config
        self.materials: Dict[str, Dict[str, Any]] = {}

    def add_material(self, name: str, props: Dict[str, Any]):
        self.materials[name] = props

    def add_texture_material(
        self,
        name: str,
        texture_path: str,
        diffuse_color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        specular_color: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        shininess: float = 30.0,
        transparency: float = 1.0,
    ):
        rel = Path(texture_path).name
        self.materials[name] = {
            "type": "textured",
            "diffuse": diffuse_color,
            "specular": specular_color,
            "shininess": shininess,
            "transparency": transparency,
            "texture": rel,
            "texture_type": "map_Kd",
        }

    def add_color_material(
        self,
        name: str,
        diffuse_color: Tuple[float, float, float] = (0.8, 0.8, 0.8),
        specular_color: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        shininess: float = 30.0,
        transparency: float = 1.0,
    ):
        self.materials[name] = {
            "type": "color",
            "diffuse": diffuse_color,
            "specular": specular_color,
            "shininess": shininess,
            "transparency": transparency,
        }

    def write_to_file(self, file_path: str):
        logger.info(f"Writing MTL file: {file_path}")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("# Material definitions for GTA SA Map\n")
            f.write("# Generated by GTA SA Map Converter\n\n")
            for name, props in self.materials.items():
                self._write_material(f, name, props)

    def _write_material(self, f, name: str, props: Dict[str, Any]):
        f.write(f"newmtl {name}\n")
        diffuse = props.get("diffuse", (0.8, 0.8, 0.8))
        specular = props.get("specular", (0.5, 0.5, 0.5))
        shininess = props.get("shininess", 30.0)
        transparency = props.get("transparency", 1.0)

        f.write("illum 2\n")
        f.write(f"Ka {diffuse[0]:.3f} {diffuse[1]:.3f} {diffuse[2]:.3f}\n")
        f.write(f"Kd {diffuse[0]:.3f} {diffuse[1]:.3f} {diffuse[2]:.3f}\n")
        f.write(f"Ks {specular[0]:.3f} {specular[1]:.3f} {specular[2]:.3f}\n")
        f.write(f"Ns {shininess:.1f}\n")
        f.write(f"d {transparency:.3f}\n")

        if props.get("type") == "textured" and "texture" in props:
            tex = props["texture"]
            f.write(f"{props.get('texture_type', 'map_Kd')} {tex}\n")

        f.write("\n")

class OBJExporter:
    def __init__(self, config: Config):
        self.config = config
        self.obj_writer = OBJWriter(config)
        self.mtl_writer = MTLWriter(config)
        self.export_stats: Dict[str, Any] = {}

    def _get_mat_prop(self, mat: Any, key: str, default: Any = None, alt_keys=()):
        
        if isinstance(mat, dict):
            return mat.get(key, default)

        for k in (key, *alt_keys):
            if hasattr(mat, k):
                return getattr(mat, k)

        return default

    def export_scene(self, scene: Dict[str, Any], output_path: str) -> bool:
        
        logger.info(f"Exporting scene to OBJ: {output_path}")

        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)

        objects = scene.get("objects", [])
        materials = scene.get("materials", {})
        textures = scene.get("textures", {})

        logger.info(f"Scene has {len(objects)} objects.")

        if not objects:
            raise RuntimeError("Scene has no objects to export.")

        self.obj_writer.create_object("GTA_SA_Map")

        self._add_materials_to_mtl(materials, textures, output_dir)

        total_vertices = 0
        for i, obj in enumerate(objects):
            if i % 100 == 0:
                logger.debug(f"Processing object {i+1}/{len(objects)}")

            model_data = obj.model_data
            if not model_data:
                logger.warning(f"Skipping {obj.model_name}: no model_data.")
                continue

            added = self.obj_writer.add_scene_object(obj, model_data)
            total_vertices += added

        if total_vertices == 0:
            logger.warning("No geometry was exported (0 vertices).")
            return True
        mtl_filename = Path(output_path).stem + ".mtl"
        mtl_path = os.path.join(output_dir, mtl_filename)

        self.mtl_writer.write_to_file(mtl_path)
        self.obj_writer.write_to_file(output_path, mtl_filename)

        self.export_stats = {
            "output_path": output_path,
            "mtl_path": mtl_path,
            "total_vertices": total_vertices,
            "object_count": len(objects),
            "obj_stats": self.obj_writer.get_stats(),
            "material_count": len(materials),
            "texture_count": len(textures),
        }

        logger.info(f"Export completed: {output_path}")
        return True

    def export(self, scene: Dict[str, Any], output_path: str) -> bool:

        return self.export_scene(scene, output_path)

    def _add_materials_to_mtl(
        self,
        materials: Dict[str, Any],
        textures: Dict[str, Any],
        output_dir: str,
    ):
        
        self.mtl_writer.add_color_material("default_material")

        for name, mat in materials.items():
            # Get texture value from material (may be path, TextureInfo, etc.)
            tex_val = self._get_mat_prop(
                mat,
                "texture",
                "",
                alt_keys=("texture_path", "image_path"),
            )

            tex_path_str = ""
            if isinstance(tex_val, (str, os.PathLike)):
                tex_path_str = str(tex_val)
            elif tex_val is not None:
                # Try common attributes on texture wrapper objects
                for attr in ("path", "filepath", "file_path", "filename"):
                    if hasattr(tex_val, attr):
                        v = getattr(tex_val, attr)
                        if isinstance(v, (str, os.PathLike)):
                            tex_path_str = str(v)
                            break

            diffuse = self._get_mat_prop(
                mat,
                "diffuse",
                (0.8, 0.8, 0.8),
                alt_keys=("diffuse_color",),
            )
            specular = self._get_mat_prop(
                mat,
                "specular",
                (0.5, 0.5, 0.5),
                alt_keys=("specular_color",),
            )
            shininess = self._get_mat_prop(mat, "shininess", 30.0)
            transparency = self._get_mat_prop(
                mat,
                "transparency",
                1.0,
                alt_keys=("alpha",),
            )

            if tex_path_str and os.path.exists(tex_path_str):
                self.mtl_writer.add_texture_material(
                    name=name,
                    texture_path=tex_path_str,
                    diffuse_color=diffuse,
                    specular_color=specular,
                    shininess=shininess,
                    transparency=transparency,
                )
            else:
                self.mtl_writer.add_color_material(
                    name=name,
                    diffuse_color=diffuse,
                    specular_color=specular,
                    shininess=shininess,
                    transparency=transparency,
                )

    def get_export_stats(self) -> Dict[str, Any]:
        return self.export_stats.copy()

__all__ = ["OBJExporter"]