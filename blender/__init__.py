__all__ = [
    "OBJExporter",
    "MaterialBuilder",
    "BlenderMaterial",
    "TextureInfo",
    "Blender279Compat",
    "CompatibilityChecker",
]

def __getattr__(name):
    if name == "OBJExporter":
        from .obj_exporter import OBJExporter
        return OBJExporter
    if name in ("MaterialBuilder", "BlenderMaterial", "TextureInfo"):
        from .material_builder import MaterialBuilder, BlenderMaterial, TextureInfo
        return {"MaterialBuilder": MaterialBuilder, "BlenderMaterial": BlenderMaterial, "TextureInfo": TextureInfo}[name]
    if name in ("Blender279Compat", "CompatibilityChecker"):
        from .blender279_compat import Blender279Compat, CompatibilityChecker
        return {"Blender279Compat": Blender279Compat, "CompatibilityChecker": CompatibilityChecker}[name]
    raise AttributeError(name)