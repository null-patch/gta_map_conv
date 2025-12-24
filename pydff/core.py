import struct
from io import BytesIO
from typing import Optional

from pydff.model import DFFError, DFFGeometry, DFFMaterial, DFFModel


RW_STRUCT = 0x0F
RW_FRAME = 0x00
RW_GEOMETRY = 0x08
RW_CLUMP = 0x10
RW_ATOMIC = 0x14
RW_GEOMETRYLIST = 0x1A
RW_MATERIALLIST = 0x08
RW_MATERIAL = 0x07
RW_TEXTURE = 0x06
RW_TEXTUREDICTIONARY = 0x16
RW_TEXTURENATIVE = 0x15


class DFFParser:
    def parse(self, data: bytes, name: str = "") -> DFFModel:
        stream = BytesIO(data)
        model = DFFModel(name=name)

        try:
            while True:
                chunk = self._read_chunk_header(stream)
                if chunk is None:
                    break

                chunk_type, chunk_size, chunk_version = chunk
                chunk_data = stream.read(chunk_size)

                if chunk_type == RW_CLUMP:
                    self._parse_clump(chunk_data, chunk_version, model)
                elif chunk_type == RW_TEXTUREDICTIONARY:
                    self._parse_texture_dictionary(chunk_data, chunk_version)

        except struct.error:
            pass

        return model

    def parse_file(self, path) -> DFFModel:
        with open(path, "rb") as f:
            data = f.read()
        return self.parse(data, name=str(path))

    def _read_chunk_header(self, stream: BytesIO) -> Optional[tuple[int, int, int]]:
        header = stream.read(12)
        if len(header) < 12:
            return None
        chunk_type, chunk_size, chunk_version = struct.unpack("<III", header)
        return chunk_type, chunk_size, chunk_version

    def _parse_clump(
        self, data: bytes, version: int, model: DFFModel
    ) -> None:
        model.version = version
        stream = BytesIO(data)

        while True:
            chunk = self._read_chunk_header(stream)
            if chunk is None:
                break

            chunk_type, chunk_size, chunk_version = chunk
            chunk_data = stream.read(chunk_size)

            if chunk_type == RW_STRUCT:
                self._parse_clump_struct(chunk_data)
            elif chunk_type == RW_FRAME:
                self._parse_frame(chunk_data, chunk_version)
            elif chunk_type == RW_GEOMETRYLIST:
                self._parse_geometry_list(chunk_data, chunk_version, model)

    def _parse_clump_struct(self, data: bytes) -> None:
        if len(data) >= 4:
            struct.unpack("<I", data[:4])[0]

    def _parse_frame(self, data: bytes, version: int) -> None:
        pass

    def _parse_geometry_list(
        self, data: bytes, version: int, model: DFFModel
    ) -> None:
        stream = BytesIO(data)

        chunk = self._read_chunk_header(stream)
        if chunk is None:
            return

        chunk_type, chunk_size, chunk_version = chunk
        if chunk_type != RW_STRUCT:
            return

        struct_data = stream.read(chunk_size)
        if len(struct_data) < 4:
            return

        geom_count = struct.unpack("<I", struct_data[:4])[0]

        for _ in range(geom_count):
            chunk = self._read_chunk_header(stream)
            if chunk is None:
                break

            chunk_type, chunk_size, chunk_version = chunk
            chunk_data = stream.read(chunk_size)

            if chunk_type == RW_GEOMETRY:
                geometry = self._parse_geometry(chunk_data, chunk_version)
                model.geometries.append(geometry)

    def _parse_geometry(self, data: bytes, version: int) -> DFFGeometry:
        geometry = DFFGeometry()
        stream = BytesIO(data)

        chunk = self._read_chunk_header(stream)
        if chunk is None:
            return geometry

        chunk_type, chunk_size, chunk_version = chunk
        if chunk_type != RW_STRUCT:
            return geometry

        struct_data = stream.read(chunk_size)
        (
            flags,
            vertex_count,
            face_count,
            *rest,
        ) = self._parse_geometry_struct(struct_data)

        has_normals = bool(flags & 0x00000010)
        has_uv = bool(flags & 0x00000002)

        vertex_data = self._read_vertices(
            struct_data, vertex_count, has_normals, has_uv
        )
        geometry.vertices = vertex_data["vertices"]
        geometry.normals = vertex_data["normals"]
        geometry.uvs = vertex_data["uvs"]

        while True:
            chunk = self._read_chunk_header(stream)
            if chunk is None:
                break

            chunk_type, chunk_size, chunk_version = chunk
            chunk_data = stream.read(chunk_size)

            if chunk_type == RW_MATERIALLIST:
                self._parse_material_list(chunk_data, chunk_version, geometry)
            elif chunk_type == RW_STRUCT:
                geometry.faces = self._parse_geometry_faces(
                    chunk_data, face_count
                )

        return geometry

    def _parse_geometry_struct(self, data: bytes) -> tuple:
        if len(data) < 28:
            raise DFFError("Geometry struct too short")

        flags, unk1, vertex_count, morph_count = struct.unpack(
            "<IIII", data[:16]
        )
        face_count = struct.unpack("<I", data[24:28])[0]

        return flags, vertex_count, face_count

    def _read_vertices(
        self,
        struct_data: bytes,
        vertex_count: int,
        has_normals: bool,
        has_uv: bool,
    ) -> dict:
        result = {"vertices": [], "normals": [], "uvs": []}

        offset = 28

        if vertex_count == 0:
            return result

        if offset + vertex_count * 12 > len(struct_data):
            return result

        for i in range(vertex_count):
            pos = offset + i * 12
            x, y, z = struct.unpack("<fff", struct_data[pos : pos + 12])
            result["vertices"].append((x, y, z))

        offset += vertex_count * 12

        if has_normals:
            if offset + vertex_count * 12 > len(struct_data):
                return result

            for i in range(vertex_count):
                pos = offset + i * 12
                nx, ny, nz = struct.unpack("<fff", struct_data[pos : pos + 12])
                result["normals"].append((nx, ny, nz))

            offset += vertex_count * 12

        if has_uv:
            if offset + vertex_count * 8 > len(struct_data):
                return result

            for i in range(vertex_count):
                pos = offset + i * 8
                u, v = struct.unpack("<ff", struct_data[pos : pos + 8])
                result["uvs"].append((u, v))

        return result

    def _parse_geometry_faces(self, data: bytes, face_count: int) -> list[tuple[int, int, int]]:
        faces = []

        if face_count == 0:
            return faces

        if len(data) < 4:
            return faces

        offset = 4

        if offset + face_count * 4 > len(data):
            return faces

        for i in range(face_count):
            pos = offset + i * 4
            if pos + 4 > len(data):
                break

            face_data = struct.unpack("<I", data[pos : pos + 4])[0]

            v1 = (face_data >> 0) & 0xFFFF
            v2 = (face_data >> 16) & 0xFFFF

            if i + 1 < face_count and pos + 8 <= len(data):
                next_face_data = struct.unpack("<I", data[pos + 4 : pos + 8])[0]
                v3 = (next_face_data >> 0) & 0xFFFF
                faces.append((v1, v2, v3))

        return faces

    def _parse_material_list(
        self, data: bytes, version: int, geometry: DFFGeometry
    ) -> None:
        stream = BytesIO(data)

        chunk = self._read_chunk_header(stream)
        if chunk is None:
            return

        chunk_type, chunk_size, chunk_version = chunk
        if chunk_type != RW_STRUCT:
            return

        struct_data = stream.read(chunk_size)
        if len(struct_data) < 4:
            return

        material_count = struct.unpack("<I", struct_data[:4])[0]

        for _ in range(material_count):
            chunk = self._read_chunk_header(stream)
            if chunk is None:
                break

            chunk_type, chunk_size, chunk_version = chunk
            chunk_data = stream.read(chunk_size)

            if chunk_type == RW_MATERIAL:
                material = self._parse_material(chunk_data, chunk_version)
                geometry.materials.append(material)

    def _parse_material(self, data: bytes, version: int) -> DFFMaterial:
        material = DFFMaterial(name="")
        stream = BytesIO(data)

        chunk = self._read_chunk_header(stream)
        if chunk is None:
            return material

        chunk_type, chunk_size, chunk_version = chunk
        if chunk_type == RW_STRUCT:
            struct_data = stream.read(chunk_size)
            if len(struct_data) >= 20:
                (
                    _flags,
                    r,
                    g,
                    b,
                    a,
                ) = struct.unpack("<IBBBBxxx", struct_data[:20])
                material.color = (r / 255.0, g / 255.0, b / 255.0, a / 255.0)

        while True:
            chunk = self._read_chunk_header(stream)
            if chunk is None:
                break

            chunk_type, chunk_size, chunk_version = chunk
            chunk_data = stream.read(chunk_size)

            if chunk_type == RW_TEXTURE:
                self._parse_texture(chunk_data, chunk_version, material)

        return material

    def _parse_texture(
        self, data: bytes, version: int, material: DFFMaterial
    ) -> None:
        stream = BytesIO(data)

        chunk = self._read_chunk_header(stream)
        if chunk is None:
            return

        chunk_type, chunk_size, chunk_version = chunk
        if chunk_type == RW_STRUCT:
            struct_data = stream.read(chunk_size)
            if len(struct_data) >= 4:
                filter_flags = struct.unpack("<I", struct_data[:4])[0]

            chunk = self._read_chunk_header(stream)
            if chunk is not None:
                chunk_type, chunk_size, chunk_version = chunk
                if chunk_type == RW_STRUCT:
                    name_data = stream.read(chunk_size)
                    if name_data:
                        try:
                            name_str = (
                                name_data.split(b"\x00")[0]
                                .decode("ascii", errors="ignore")
                                .strip()
                            )
                            if name_str:
                                material.texture_name = name_str
                        except Exception:
                            pass

    def _parse_texture_dictionary(self, data: bytes, version: int) -> None:
        pass