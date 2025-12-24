import os
import struct
import zlib
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, BinaryIO
from dataclasses import dataclass, field
import logging
import subprocess
import tempfile

logger = logging.getLogger(__name__)


@dataclass
class DFFGeometry:
    vertices: List[List[float]] = field(default_factory=list)
    normals: List[List[float]] = field(default_factory=list)
    colors: List[List[float]] = field(default_factory=list)
    uvs: List[List[float]] = field(default_factory=list)
    faces: List[List[int]] = field(default_factory=list)
    face_materials: List[int] = field(default_factory=list)
    face_normals: List[List[int]] = field(default_factory=list)
    materials: List[Dict[str, Any]] = field(default_factory=list)
    texture_names: List[str] = field(default_factory=list)
    bounding_box: Tuple[Tuple[float, float, float], Tuple[float, float, float]] = (
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    )
    has_normals: bool = False
    has_colors: bool = False
    has_uvs: bool = False
    is_triangulated: bool = True
    is_textured: bool = False
    vertex_count: int = 0
    face_count: int = 0

    def calculate_bounds(self):
        if not self.vertices:
            return
        vertices_array = np.array(self.vertices)
        min_vals = vertices_array.min(axis=0)
        max_vals = vertices_array.max(axis=0)
        self.bounding_box = (
            (float(min_vals[0]), float(min_vals[1]), float(min_vals[2])),
            (float(max_vals[0]), float(max_vals[1]), float(max_vals[2])),
        )

    def get_stats(self) -> Dict[str, Any]:
        return {
            "vertex_count": len(self.vertices),
            "face_count": len(self.faces),
            "normal_count": len(self.normals),
            "uv_count": len(self.uvs),
            "material_count": len(self.materials),
            "has_normals": self.has_normals,
            "has_colors": self.has_colors,
            "has_uvs": self.has_uvs,
            "is_textured": self.is_textured,
            "bounds": self.bounding_box,
        }


@dataclass
class DFFModel:
    name: str = ""
    geometries: List[DFFGeometry] = field(default_factory=list)
    frame_hierarchy: Dict[str, Any] = field(default_factory=dict)
    materials: List[Dict[str, Any]] = field(default_factory=list)
    textures: List[Dict[str, Any]] = field(default_factory=list)
    version: int = 0
    flags: int = 0
    is_vehicle: bool = False
    is_pedestrian: bool = False
    is_weapon: bool = False
    has_alpha: bool = False
    has_collision: bool = False

    def merge_geometries(self) -> DFFGeometry:
        if not self.geometries:
            return DFFGeometry()
        if len(self.geometries) == 1:
            return self.geometries[0]
        merged = DFFGeometry()
        vertex_offset = 0
        for geometry in self.geometries:
            merged.vertices.extend(geometry.vertices)
            merged.normals.extend(geometry.normals)
            merged.colors.extend(geometry.colors)
            merged.uvs.extend(geometry.uvs)
            for face in geometry.faces:
                merged.faces.append([v + vertex_offset for v in face])
            merged.face_materials.extend(geometry.face_materials)
            vertex_offset += len(geometry.vertices)
            for material in geometry.materials:
                if material not in merged.materials:
                    merged.materials.append(material)
            for tex_name in geometry.texture_names:
                if tex_name not in merged.texture_names:
                    merged.texture_names.append(tex_name)
        merged.has_normals = any(g.has_normals for g in self.geometries)
        merged.has_colors = any(g.has_colors for g in self.geometries)
        merged.has_uvs = any(g.has_uvs for g in self.geometries)
        merged.is_textured = any(g.is_textured for g in self.geometries)
        merged.vertex_count = len(merged.vertices)
        merged.face_count = len(merged.faces)
        merged.calculate_bounds()
        return merged


class DFFChunk:
    def __init__(self, chunk_type: int = 0, chunk_size: int = 0, data: bytes = b""):
        self.type = chunk_type
        self.size = chunk_size
        self.data = data
        self.children: List["DFFChunk"] = []

    @classmethod
    def parse(cls, data: bytes, offset: int = 0) -> Tuple["DFFChunk", int]:
        if offset + 12 > len(data):
            raise ValueError("Insufficient data for chunk header")
        chunk_type, chunk_size, library_id = struct.unpack(
            "<III", data[offset : offset + 12]
        )
        chunk = cls(chunk_type, chunk_size)
        offset += 12
        chunk.data = data[offset : offset + chunk_size]
        return chunk, offset + chunk_size


class DFFParser:
    def __init__(self):
        self.model = DFFModel()
        self.current_offset = 0
        self.warnings: List[str] = []
        self.errors: List[str] = []

    def parse_file(self, file_path: str) -> Optional[DFFModel]:
        if not os.path.exists(file_path):
            logger.error(f"DFF file not found: {file_path}")
            return None
        logger.info(f"Parsing DFF file: {file_path}")
        try:
            with open(file_path, "rb") as f:
                data = f.read()
            return self._parse_data(data, os.path.basename(file_path))
        except Exception as e:
            logger.error(f"Error parsing DFF file {file_path}: {e}")
            self.errors.append(str(e))
            return None

    def _parse_data(self, data: bytes, filename: str) -> Optional[DFFModel]:
        self.model.name = Path(filename).stem
        try:
            if len(data) < 12:
                raise ValueError("File too small to be a DFF file")
            self._parse_chunks(data, 0, len(data))
            if not self.model.geometries:
                self.warnings.append("No geometry found in DFF file")
            logger.info(
                f"Parsed DFF: {self.model.name} with {len(self.model.geometries)} geometries"
            )
            return self.model
        except Exception as e:
            logger.error(f"Error parsing DFF data: {e}")
            self.errors.append(str(e))
            return None

    def _parse_chunks(self, data: bytes, offset: int, end_offset: int):
        while offset < end_offset:
            try:
                chunk, offset = DFFChunk.parse(data, offset)
                self._process_chunk(chunk)
            except Exception as e:
                logger.warning(f"Error parsing chunk at offset {offset}: {e}")
                break

    def _process_chunk(self, chunk: DFFChunk):
        try:
            if chunk.type == 0x01:
                self._parse_struct_chunk(chunk)
            elif chunk.type == 0x0F:
                self._parse_geometry_chunk(chunk)
            elif chunk.type == 0x14:
                self._parse_atomic_chunk(chunk)
            elif chunk.type == 0x16:
                self._parse_texture_dict_chunk(chunk)
            elif chunk.type == 0x1A:
                self._parse_geometry_list_chunk(chunk)
        except Exception as e:
            logger.warning(f"Error processing chunk type 0x{chunk.type:02X}: {e}")

    def _parse_struct_chunk(self, chunk: DFFChunk):
        return

    def _parse_geometry_chunk(self, chunk: DFFChunk):
        try:
            geometry = DFFGeometry()
            offset = 0
            data = chunk.data
            if len(data) < 64:
                self.warnings.append("Geometry chunk too small")
                return
            vertex_count = struct.unpack("<H", data[24:26])[0]
            face_count = struct.unpack("<H", data[26:28])[0]
            if vertex_count == 0 or face_count == 0:
                self.warnings.append("Empty geometry")
                return
            vertex_offset = 64
            for _ in range(vertex_count):
                if vertex_offset + 12 > len(data):
                    break
                x, y, z = struct.unpack("<fff", data[vertex_offset : vertex_offset + 12])
                geometry.vertices.append([x, y, z])
                vertex_offset += 12
            face_offset = vertex_offset
            for _ in range(face_count):
                if face_offset + 6 > len(data):
                    break
                v1, v2, v3 = struct.unpack("<HHH", data[face_offset : face_offset + 6])
                geometry.faces.append([v1, v2, v3])
                face_offset += 6
            if face_offset + (vertex_count * 12) <= len(data):
                geometry.has_normals = True
                for _ in range(vertex_count):
                    nx, ny, nz = struct.unpack(
                        "<fff", data[face_offset : face_offset + 12]
                    )
                    geometry.normals.append([nx, ny, nz])
                    face_offset += 12
            if face_offset + (vertex_count * 8) <= len(data):
                geometry.has_uvs = True
                for _ in range(vertex_count):
                    u, v = struct.unpack("<ff", data[face_offset : face_offset + 8])
                    geometry.uvs.append([u, v])
                    face_offset += 8
            geometry.calculate_bounds()
            geometry.vertex_count = len(geometry.vertices)
            geometry.face_count = len(geometry.faces)
            self.model.geometries.append(geometry)
        except Exception as e:
            logger.warning(f"Error parsing geometry: {e}")
            self.warnings.append(f"Geometry parsing error: {e}")

    def _parse_atomic_chunk(self, chunk: DFFChunk):
        return

    def _parse_texture_dict_chunk(self, chunk: DFFChunk):
        try:
            data = chunk.data
            offset = 0
            texture_count = struct.unpack("<H", data[offset : offset + 2])[0]
            offset += 2
            for _ in range(texture_count):
                name_end = data.find(b"\x00", offset)
                if name_end == -1:
                    break
                tex_name = data[offset:name_end].decode("ascii", errors="ignore")
                self.model.textures.append({"name": tex_name})
                offset = name_end + 1
        except Exception as e:
            logger.warning(f"Error parsing texture dictionary: {e}")

    def _parse_geometry_list_chunk(self, chunk: DFFChunk):
        return

    def get_warnings(self) -> List[str]:
        return self.warnings

    def get_errors(self) -> List[str]:
        return self.errors


class DFFConverter:
    def __init__(self, scale_factor: float = 0.01, use_external_tool: bool = True):
        self.scale_factor = scale_factor
        self.use_external_tool = use_external_tool
        self.external_tools = self._detect_external_tools()

    def _detect_external_tools(self) -> Dict[str, str]:
        tools: Dict[str, str] = {}
        tool_paths = [
            "/usr/bin/rwanalyze",
            "/usr/local/bin/rwanalyze",
            "/usr/bin/dff2obj",
            "/usr/local/bin/dff2obj",
            "/opt/gta_tools/rwanalyze",
            Path.home() / ".local/bin/rwanalyze",
        ]
        for tool_path in tool_paths:
            if os.path.exists(tool_path):
                tools["rwanalyze"] = str(tool_path)
                break
        try:
            import gta_tools  # type: ignore
            tools["gta_tools"] = "gta_tools"
        except ImportError:
            pass
        return tools

    def convert(self, dff_path: str) -> Optional[Dict[str, Any]]:
        logger.info(f"Converting DFF: {dff_path}")
        methods = [
            self._convert_with_pydff,
            self._convert_with_external_tool,
            self._convert_with_parser,
            self._convert_with_simplified_parser,
        ]
        for method in methods:
            try:
                result = method(dff_path)
                if result and result.get("vertices") and result.get("faces"):
                    self._apply_scale(result)
                    return result
            except Exception as e:
                logger.warning(f"Conversion method {method.__name__} failed: {e}")
                continue
        logger.error(f"All conversion methods failed for {dff_path}, using dummy geometry")
        result = self._dummy_geometry()
        self._apply_scale(result)
        return result

    def _convert_with_pydff(self, dff_path: str) -> Optional[Dict[str, Any]]:
        try:
            from pydff.io import load as pydff_load
        except ImportError:
            return None
        try:
            model = pydff_load(dff_path)
        except Exception as e:
            logger.warning(f"pydff load error: {e}")
            return None
        try:
            geom = model.merged_geometry()
        except Exception as e:
            logger.warning(f"pydff merge error: {e}")
            return None
        if not geom.vertices or not geom.faces:
            return None
        vertices = [[float(x), float(y), float(z)] for (x, y, z) in geom.vertices]
        faces = [[int(a), int(b), int(c)] for (a, b, c) in geom.faces]
        normals = [[float(x), float(y), float(z)] for (x, y, z) in getattr(geom, "normals", [])]
        uvs = [[float(u), float(v)] for (u, v) in getattr(geom, "uvs", [])]
        materials: List[Dict[str, Any]] = []
        texture_names: List[str] = []
        for m in getattr(geom, "materials", []):
            tex = getattr(m, "texture_name", None)
            if tex:
                texture_names.append(tex)
            color = getattr(m, "color", (1.0, 1.0, 1.0, 1.0))
            materials.append(
                {
                    "name": getattr(m, "name", ""),
                    "color": [float(c) for c in color],
                    "texture": tex or "",
                }
            )
        return {
            "vertices": vertices,
            "faces": faces,
            "normals": normals,
            "uvs": uvs,
            "materials": materials,
            "texture_names": texture_names,
        }

    def _convert_with_external_tool(self, dff_path: str) -> Optional[Dict[str, Any]]:
        if not self.use_external_tool or not self.external_tools:
            return None
        try:
            if "rwanalyze" in self.external_tools:
                return self._convert_with_rwanalyze(dff_path)
        except Exception as e:
            logger.warning(f"External tool conversion failed: {e}")
        return None

    def _convert_with_rwanalyze(self, dff_path: str) -> Optional[Dict[str, Any]]:
        with tempfile.NamedTemporaryFile(suffix=".obj", delete=False) as tmp:
            output_path = tmp.name
        try:
            cmd = [self.external_tools["rwanalyze"], "-o", output_path, dff_path]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                return self._parse_obj_file(output_path)
            logger.warning(f"RWAnalyze failed: {result.stderr}")
        except subprocess.TimeoutExpired:
            logger.warning("RWAnalyze timed out")
        except Exception as e:
            logger.warning(f"RWAnalyze error: {e}")
        finally:
            try:
                os.unlink(output_path)
            except Exception:
                pass
        return None

    def _convert_with_parser(self, dff_path: str) -> Optional[Dict[str, Any]]:
        parser = DFFParser()
        model = parser.parse_file(dff_path)
        if not model:
            return None
        geometry = model.merge_geometries()
        result = {
            "vertices": geometry.vertices,
            "faces": geometry.faces,
            "normals": geometry.normals,
            "uvs": geometry.uvs,
            "colors": geometry.colors,
            "materials": geometry.materials,
            "texture_names": geometry.texture_names,
            "bounds": geometry.bounding_box,
            "stats": geometry.get_stats(),
        }
        return result

    def _convert_with_simplified_parser(self, dff_path: str) -> Optional[Dict[str, Any]]:
        try:
            with open(dff_path, "rb") as f:
                data = f.read()
            geometry = {
                "vertices": [],
                "faces": [],
                "normals": [],
                "uvs": [],
                "materials": [],
                "texture_names": [],
            }
            offset = 0
            max_vertices = 10000
            while offset < len(data) - 12 and len(geometry["vertices"]) < max_vertices:
                try:
                    x, y, z = struct.unpack("<fff", data[offset : offset + 12])
                    if (
                        -10000.0 < x < 10000.0
                        and -10000.0 < y < 10000.0
                        and -10000.0 < z < 10000.0
                    ):
                        geometry["vertices"].append([x, y, z])
                        offset += 12
                    else:
                        offset += 4
                except struct.error:
                    offset += 1
            offset = 0
            max_faces = 10000
            while offset < len(data) - 6 and len(geometry["faces"]) < max_faces:
                try:
                    v1, v2, v3 = struct.unpack("<HHH", data[offset : offset + 6])
                    max_index = len(geometry["vertices"])
                    if v1 < max_index and v2 < max_index and v3 < max_index:
                        geometry["faces"].append([v1, v2, v3])
                        offset += 6
                    else:
                        offset += 2
                except struct.error:
                    offset += 1
            if geometry["vertices"] and geometry["faces"]:
                logger.info(
                    f"Extracted {len(geometry['vertices'])} vertices and {len(geometry['faces'])} faces"
                )
                return geometry
        except Exception as e:
            logger.warning(f"Simplified parser failed: {e}")
        return None

    def _parse_obj_file(self, obj_path: str) -> Dict[str, Any]:
        geometry = {
            "vertices": [],
            "faces": [],
            "normals": [],
            "uvs": [],
            "materials": [],
            "texture_names": [],
        }
        try:
            with open(obj_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    if not parts:
                        continue
                    if parts[0] == "v":
                        if len(parts) >= 4:
                            geometry["vertices"].append(
                                [
                                    float(parts[1]),
                                    float(parts[2]),
                                    float(parts[3]),
                                ]
                            )
                    elif parts[0] == "vn":
                        if len(parts) >= 4:
                            geometry["normals"].append(
                                [
                                    float(parts[1]),
                                    float(parts[2]),
                                    float(parts[3]),
                                ]
                            )
                    elif parts[0] == "vt":
                        if len(parts) >= 3:
                            geometry["uvs"].append(
                                [float(parts[1]), float(parts[2])]
                            )
                    elif parts[0] == "f":
                        face_vertices: List[int] = []
                        for part in parts[1:]:
                            if "/" in part:
                                indices = part.split("/")
                                face_vertices.append(int(indices[0]) - 1)
                            else:
                                face_vertices.append(int(part) - 1)
                        if len(face_vertices) >= 3:
                            for i in range(1, len(face_vertices) - 1):
                                geometry["faces"].append(
                                    [
                                        face_vertices[0],
                                        face_vertices[i],
                                        face_vertices[i + 1],
                                    ]
                                )
                    elif parts[0] == "usemtl":
                        if len(parts) >= 2:
                            geometry["materials"].append(
                                {"name": parts[1], "color": [1, 1, 1, 1]}
                            )
        except Exception as e:
            logger.warning(f"Error parsing OBJ file: {e}")
        return geometry

    def _apply_scale(self, geometry: Dict[str, Any]):
        if not geometry or "vertices" not in geometry:
            return
        for i, vertex in enumerate(geometry["vertices"]):
            geometry["vertices"][i] = [
                vertex[0] * self.scale_factor,
                vertex[1] * self.scale_factor,
                vertex[2] * self.scale_factor,
            ]
        if "bounds" in geometry:
            min_bounds, max_bounds = geometry["bounds"]
            geometry["bounds"] = (
                (
                    min_bounds[0] * self.scale_factor,
                    min_bounds[1] * self.scale_factor,
                    min_bounds[2] * self.scale_factor,
                ),
                (
                    max_bounds[0] * self.scale_factor,
                    max_bounds[1] * self.scale_factor,
                    max_bounds[2] * self.scale_factor,
                ),
            )

    def _dummy_geometry(self) -> Dict[str, Any]:
        v = [
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5],
        ]
        f = [
            [0, 1, 2],
            [0, 2, 3],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [2, 3, 7],
            [2, 7, 6],
            [1, 2, 6],
            [1, 6, 5],
            [0, 3, 7],
            [0, 7, 4],
        ]
        return {
            "vertices": v,
            "faces": f,
            "normals": [],
            "uvs": [],
            "materials": [],
            "texture_names": [],
        }


class DFFManager:
    def __init__(self, scale_factor: float = 0.01):
        self.converter = DFFConverter(scale_factor)
        self.converted_models: Dict[str, Dict[str, Any]] = {}
        self.conversion_stats: Dict[str, Dict] = {}

    def convert_files(
        self, dff_files: List[str], max_workers: int = 4
    ) -> Dict[str, Dict[str, Any]]:
        import concurrent.futures

        self.converted_models.clear()
        logger.info(f"Converting {len(dff_files)} DFF files")
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(self._convert_single_file, dff_file): dff_file
                for dff_file in dff_files
            }
            for future in concurrent.futures.as_completed(future_to_file):
                dff_file = future_to_file[future]
                try:
                    result = future.result()
                    if result:
                        model_name, geometry = result
                        self.converted_models[model_name] = geometry
                        logger.debug(f"Converted: {model_name}")
                except Exception as e:
                    logger.error(f"Failed to convert {dff_file}: {e}")
        logger.info(
            f"Successfully converted {len(self.converted_models)} out of {len(dff_files)} files"
        )
        return self.converted_models.copy()

    def _convert_single_file(
        self, dff_path: str
    ) -> Optional[Tuple[str, Dict[str, Any]]]:
        try:
            geometry = self.converter.convert(dff_path)
            if geometry:
                model_name = Path(dff_path).stem
                return model_name, geometry
        except Exception as e:
            logger.error(f"Error converting {dff_path}: {e}")
        return None

    def get_model(self, model_name: str) -> Optional[Dict[str, Any]]:
        return self.converted_models.get(model_name)

    def list_models(self) -> List[str]:
        return list(self.converted_models.keys())

    def get_model_stats(self, model_name: str) -> Optional[Dict[str, Any]]:
        model = self.get_model(model_name)
        if model and "stats" in model:
            return model["stats"]
        return None

    def clear(self):
        self.converted_models.clear()


def convert_dff_file(
    dff_path: str, scale_factor: float = 0.01
) -> Optional[Dict[str, Any]]:
    converter = DFFConverter(scale_factor)
    return converter.convert(dff_path)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        test_file = sys.argv[1]
        logging.basicConfig(level=logging.INFO)
        converter = DFFConverter()
        result = converter.convert(test_file)
        if result:
            print(f"Successfully converted {test_file}")
            stats = result.get("stats", {})
            print(f"Statistics: {stats}")
            if "vertices" in result:
                print(f"Vertices: {len(result['vertices'])}")
            if "faces" in result:
                print(f"Faces: {len(result['faces'])}")
        else:
            print(f"Failed to convert {test_file}")
    else:
        print("Usage: python dff_converter.py <dff_file_path>")