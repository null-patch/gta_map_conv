from dataclasses import dataclass, field
from typing import Optional


class DFFError(Exception):
    pass

@dataclass
class DFFMaterial:
    name: str
    color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    texture_name: Optional[str] = None

@dataclass
class DFFGeometry:
    vertices: list[tuple[float, float, float]] = field(default_factory=list)
    normals: list[tuple[float, float, float]] = field(default_factory=list)
    uvs: list[tuple[float, float]] = field(default_factory=list)
    faces: list[tuple[int, int, int]] = field(default_factory=list)
    materials: list[DFFMaterial] = field(default_factory=list)

    def vertex_count(self) -> int:
        return len(self.vertices)

    def face_count(self) -> int:
        return len(self.faces)

    def bounds(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        if not self.vertices:
            return ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        xs = [v[0] for v in self.vertices]
        ys = [v[1] for v in self.vertices]
        zs = [v[2] for v in self.vertices]

        return (
            (min(xs), min(ys), min(zs)),
            (max(xs), max(ys), max(zs)),
        )


@dataclass
class DFFModel:
    name: str = ""
    geometries: list[DFFGeometry] = field(default_factory=list)
    version: int = 0

    def merged_geometry(self) -> DFFGeometry:
        merged = DFFGeometry()
        vertex_offset = 0

        for geom in self.geometries:
            merged.vertices.extend(geom.vertices)
            merged.normals.extend(geom.normals)
            merged.uvs.extend(geom.uvs)

            for face in geom.faces:
                merged.faces.append(
                    (
                        face[0] + vertex_offset,
                        face[1] + vertex_offset,
                        face[2] + vertex_offset,
                    )
                )

            merged.materials.extend(geom.materials)
            vertex_offset += len(geom.vertices)

        return merged