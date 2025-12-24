import os
from pathlib import Path

from pydff.core import DFFParser
from pydff.model import DFFModel


def load(path) -> DFFModel:
    parser = DFFParser()
    return parser.parse_file(path)


def loads(data: bytes, name: str = "") -> DFFModel:
    parser = DFFParser()
    return parser.parse(data, name=name)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m pydff.io <dff_file> [--obj <output_obj>]")
        sys.exit(1)

    dff_path = sys.argv[1]

    try:
        model = load(dff_path)
        print(f"Loaded DFF: {dff_path}")
        print(f"  Name: {model.name}")
        print(f"  Version: {model.version}")
        print(f"  Geometries: {len(model.geometries)}")

        total_vertices = sum(g.vertex_count() for g in model.geometries)
        total_faces = sum(g.face_count() for g in model.geometries)
        total_materials = sum(len(g.materials) for g in model.geometries)

        print(f"  Total vertices: {total_vertices}")
        print(f"  Total faces: {total_faces}")
        print(f"  Total materials: {total_materials}")

        if "--obj" in sys.argv:
            obj_index = sys.argv.index("--obj")
            if obj_index + 1 < len(sys.argv):
                obj_path = sys.argv[obj_index + 1]
                from pydff.obj_export import export_obj

                export_obj(model, obj_path)
                print(f"\nExported to: {obj_path}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)