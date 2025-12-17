"""
GTA SA Map Converter - Blender Module
Blender-specific export tools and compatibility for Blender 2.79
"""

from .obj_exporter import OBJExporter
from .material_builder import MaterialBuilder, BlenderMaterial, TextureInfo
from .blender279_compat import Blender279Compat, CompatibilityChecker

__all__ = [
    # OBJ Export
    'OBJExporter',

    # Material Building
    'MaterialBuilder',
    'BlenderMaterial',
    'TextureInfo',
    
    # Blender 2.79 Compatibility
    'Blender279Compat',
    'CompatibilityChecker',
]
