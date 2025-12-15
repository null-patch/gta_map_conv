"""
GTA SA Map Converter - Material Builder
Creates Blender 2.79 compatible materials from GTA textures
"""

import os
import re
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
import logging
import colorsys

# Try to import image processing libraries
try:
    from PIL import Image, ImageOps
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class TextureInfo:
    """Information about a texture"""
    name: str = ""
    path: str = ""
    width: int = 0
    height: int = 0
    format: str = ""  # png, jpg, tga, etc.
    has_alpha: bool = False
    is_placeholder: bool = False
    
    # Color information (extracted from texture)
    average_color: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    is_dark: bool = False
    is_light: bool = False
    is_transparent: bool = False
    
    def get_size(self) -> Tuple[int, int]:
        """Get texture dimensions"""
        return (self.width, self.height)
    
    def get_aspect_ratio(self) -> float:
        """Get aspect ratio (width/height)"""
        if self.height > 0:
            return self.width / self.height
        return 1.0
    
    def is_valid(self) -> bool:
        """Check if texture is valid"""
        return bool(self.name and self.path and os.path.exists(self.path))


@dataclass
class BlenderMaterial:
    """Blender 2.79 compatible material definition"""
    name: str = ""
    type: str = "SURFACE"  # SURFACE, WIRE, VOLUME, etc.
    diffuse_color: Tuple[float, float, float] = (0.8, 0.8, 0.8)
    specular_color: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    specular_intensity: float = 0.5
    hardness: int = 50  # Shininess (1-511)
    alpha: float = 1.0  # Transparency (0-1)
    
    # Texture information
    texture: Optional[TextureInfo] = None
    texture_blend_type: str = "MIX"  # MIX, MULTIPLY, ADD, etc.
    texture_influence: float = 1.0  # How much texture affects material (0-1)
    
    # UV mapping
    uv_scale: Tuple[float, float] = (1.0, 1.0)
    uv_offset: Tuple[float, float] = (0.0, 0.0)
    uv_rotation: float = 0.0
    
    # Material flags
    use_shadeless: bool = False  # Emit material (no shadows)
    use_transparency: bool = False
    use_mirror: bool = False
    use_ambient_occlusion: bool = False
    
    # GTA-specific properties
    gta_flags: int = 0
    gta_draw_distance: float = 300.0
    gta_time_on: int = 0
    gta_time_off: int = 24
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'name': self.name,
            'type': self.type,
            'diffuse_color': self.diffuse_color,
            'specular_color': self.specular_color,
            'specular_intensity': self.specular_intensity,
            'hardness': self.hardness,
            'alpha': self.alpha,
            'texture': self.texture.name if self.texture else None,
            'texture_path': self.texture.path if self.texture else None,
            'texture_blend_type': self.texture_blend_type,
            'texture_influence': self.texture_influence,
            'uv_scale': self.uv_scale,
            'uv_offset': self.uv_offset,
            'uv_rotation': self.uv_rotation,
            'use_shadeless': self.use_shadeless,
            'use_transparency': self.use_transparency,
            'use_mirror': self.use_mirror,
            'gta_flags': self.gta_flags
        }
    
    def get_mtl_definition(self) -> Dict[str, Any]:
        """Get MTL material definition"""
        mtl_def = {
            'Ka': self.diffuse_color,  # Ambient color
            'Kd': self.diffuse_color,  # Diffuse color
            'Ks': self.specular_color,  # Specular color
            'Ns': float(self.hardness),  # Shininess
            'd': self.alpha,  # Transparency
            'illum': 2  # Highlight on
        }
        
        if self.texture and self.texture.path:
            mtl_def['map_Kd'] = self.texture.path
        
        return mtl_def


class TextureAnalyzer:
    """Analyzes textures to extract color information and properties"""
    
    def __init__(self):
        self.cache: Dict[str, TextureInfo] = {}
        
    def analyze_texture(self, texture_path: str) -> Optional[TextureInfo]:
        """Analyze a texture file"""
        # Check cache first
        if texture_path in self.cache:
            return self.cache[texture_path]
        
        texture = TextureInfo(
            name=Path(texture_path).stem,
            path=texture_path
        )
        
        try:
            if PIL_AVAILABLE and os.path.exists(texture_path):
                with Image.open(texture_path) as img:
                    # Get basic information
                    texture.width, texture.height = img.size
                    texture.format = img.format or Path(texture_path).suffix[1:].upper()
                    
                    # Check for alpha channel
                    if img.mode in ('RGBA', 'LA', 'P'):
                        texture.has_alpha = True
                    
                    # Convert to RGB for analysis
                    if img.mode != 'RGB':
                        rgb_img = img.convert('RGB')
                    else:
                        rgb_img = img
                    
                    # Analyze colors
                    self._analyze_colors(rgb_img, texture)
                    
                    # Check if mostly transparent
                    if texture.has_alpha:
                        self._check_transparency(img, texture)
            
            self.cache[texture_path] = texture
            return texture
            
        except Exception as e:
            logger.warning(f"Error analyzing texture {texture_path}: {e}")
            return None
    
    def _analyze_colors(self, img: Any, texture: TextureInfo):
        """Analyze color information from image"""
        if not PIL_AVAILABLE:
            return
            
        try:
            # Sample colors from image
            sample_size = min(100, img.width * img.height)
            
            # Convert to numpy array if available, otherwise sample
            try:
                import numpy as np
                img_array = np.array(img)
                
                # Calculate average color
                avg_color = img_array.mean(axis=(0, 1)) / 255.0
                texture.average_color = tuple(avg_color[:3])
                
                # Calculate brightness
                brightness = colorsys.rgb_to_hsv(*texture.average_color)[2]
                texture.is_dark = brightness < 0.3
                texture.is_light = brightness > 0.7
                
            except ImportError:
                # Fallback: sample random pixels
                import random
                total_r = total_g = total_b = 0
                
                for _ in range(sample_size):
                    x = random.randint(0, img.width - 1)
                    y = random.randint(0, img.height - 1)
                    r, g, b = img.getpixel((x, y))
                    total_r += r
                    total_g += g
                    total_b += b
                
                avg_r = total_r / sample_size / 255.0
                avg_g = total_g / sample_size / 255.0
                avg_b = total_b / sample_size / 255.0
                
                texture.average_color = (avg_r, avg_g, avg_b)
                
                # Calculate brightness
                brightness = colorsys.rgb_to_hsv(avg_r, avg_g, avg_b)[2]
                texture.is_dark = brightness < 0.3
                texture.is_light = brightness > 0.7
                
        except Exception as e:
            logger.warning(f"Error analyzing colors: {e}")
    
    def _check_transparency(self, img: Any, texture: TextureInfo):
        """Check if texture is mostly transparent"""
        if not PIL_AVAILABLE:
            return
            
        try:
            # Check alpha channel
            if img.mode == 'RGBA':
                alpha = img.getchannel('A')
                
                # Count transparent pixels
                transparent_pixels = sum(1 for pixel in alpha.getdata() if pixel < 128)
                total_pixels = img.width * img.height
                
                texture.is_transparent = (transparent_pixels / total_pixels) > 0.5
                
        except Exception as e:
            logger.warning(f"Error checking transparency: {e}")


class MaterialBuilder:
    """Builds Blender materials from GTA textures"""
    
    # Material presets for common GTA materials
    MATERIAL_PRESETS = {
        # Building materials
        'concrete': {
            'diffuse_color': (0.7, 0.7, 0.7),
            'specular_intensity': 0.2,
            'hardness': 30,
            'use_shadeless': False
        },
        'brick': {
            'diffuse_color': (0.6, 0.4, 0.3),
            'specular_intensity': 0.1,
            'hardness': 20,
            'use_shadeless': False
        },
        'wood': {
            'diffuse_color': (0.5, 0.35, 0.2),
            'specular_intensity': 0.3,
            'hardness': 40,
            'use_shadeless': False
        },
        'metal': {
            'diffuse_color': (0.6, 0.6, 0.6),
            'specular_intensity': 0.8,
            'hardness': 100,
            'use_shadeless': False,
            'use_mirror': True
        },
        'glass': {
            'diffuse_color': (0.9, 0.9, 0.9),
            'specular_intensity': 0.9,
            'hardness': 150,
            'alpha': 0.3,
            'use_transparency': True,
            'use_shadeless': False
        },
        
        # Road and ground materials
        'asphalt': {
            'diffuse_color': (0.2, 0.2, 0.2),
            'specular_intensity': 0.05,
            'hardness': 10,
            'use_shadeless': False
        },
        'grass': {
            'diffuse_color': (0.3, 0.5, 0.2),
            'specular_intensity': 0.1,
            'hardness': 20,
            'use_shadeless': False
        },
        'dirt': {
            'diffuse_color': (0.4, 0.3, 0.2),
            'specular_intensity': 0.05,
            'hardness': 15,
            'use_shadeless': False
        },
        
        # Vehicle materials
        'car_paint': {
            'diffuse_color': (0.8, 0.1, 0.1),  # Red car paint
            'specular_intensity': 0.7,
            'hardness': 80,
            'use_shadeless': False
        },
        'rubber': {
            'diffuse_color': (0.1, 0.1, 0.1),
            'specular_intensity': 0.05,
            'hardness': 10,
            'use_shadeless': False
        },
        
        # Vegetation
        'tree_bark': {
            'diffuse_color': (0.3, 0.2, 0.1),
            'specular_intensity': 0.1,
            'hardness': 25,
            'use_shadeless': False
        },
        'leaf': {
            'diffuse_color': (0.2, 0.4, 0.1),
            'specular_intensity': 0.2,
            'hardness': 30,
            'alpha': 0.8,
            'use_transparency': True,
            'use_shadeless': False
        },
        
        # Water
        'water': {
            'diffuse_color': (0.1, 0.3, 0.5),
            'specular_intensity': 0.9,
            'hardness': 200,
            'alpha': 0.5,
            'use_transparency': True,
            'use_mirror': True,
            'use_shadeless': False
        }
    }
    
    def __init__(self, config):
        self.config = config
        self.texture_analyzer = TextureAnalyzer()
        self.materials: Dict[str, BlenderMaterial] = {}
        self.texture_mapping: Dict[str, str] = {}  # texture_name -> material_name
        
    def create_materials(self, textures: Dict[str, Dict[str, Any]]) -> Dict[str, BlenderMaterial]:
        """
        Create materials from textures
        
        Args:
            textures: Dictionary of texture information
            
        Returns:
            Dictionary of created materials
        """
        logger.info(f"Creating materials from {len(textures)} textures")
        
        self.materials.clear()
        self.texture_mapping.clear()
        
        # Create default material
        default_material = self._create_default_material()
        self.materials['default_material'] = default_material
        
        # Create materials for each texture
        for tex_name, tex_info in textures.items():
            material = self._create_material_from_texture(tex_name, tex_info)
            if material:
                self.materials[material.name] = material
                self.texture_mapping[tex_name] = material.name
        
        logger.info(f"Created {len(self.materials)} materials")
        return self.materials.copy()
    
    def _create_default_material(self) -> BlenderMaterial:
        """Create default material for objects without textures"""
        return BlenderMaterial(
            name="default_material",
            type="SURFACE",
            diffuse_color=(0.8, 0.8, 0.8),
            specular_color=(0.5, 0.5, 0.5),
            specular_intensity=0.5,
            hardness=50,
            alpha=1.0,
            use_shadeless=False
        )
    
    def _create_material_from_texture(self, tex_name: str, 
                                     tex_info: Dict[str, Any]) -> Optional[BlenderMaterial]:
        """Create material from texture information"""
        try:
            # Get texture path
            texture_path = tex_info.get('path', '')
            if not texture_path or not os.path.exists(texture_path):
                logger.warning(f"Texture not found: {tex_name}")
                return None
            
            # Analyze texture
            texture = self.texture_analyzer.analyze_texture(texture_path)
            if not texture:
                return None
            
            # Determine material type from texture name
            material_type = self._guess_material_type(tex_name, texture)
            
            # Create material
            material = BlenderMaterial(
                name=self._sanitize_material_name(tex_name),
                type="SURFACE"
            )
            
            # Set texture
            material.texture = texture
            
            # Apply material preset if available
            if material_type in self.MATERIAL_PRESETS:
                preset = self.MATERIAL_PRESETS[material_type]
                material.diffuse_color = preset['diffuse_color']
                material.specular_intensity = preset['specular_intensity']
                material.hardness = preset['hardness']
                material.use_shadeless = preset.get('use_shadeless', False)
                material.use_mirror = preset.get('use_mirror', False)
                material.use_transparency = preset.get('use_transparency', False)
                material.alpha = preset.get('alpha', 1.0)
            else:
                # Use texture average color
                material.diffuse_color = texture.average_color
                
                # Adjust based on texture properties
                if texture.is_dark:
                    material.specular_intensity = 0.1
                    material.hardness = 20
                elif texture.is_light:
                    material.specular_intensity = 0.3
                    material.hardness = 60
                else:
                    material.specular_intensity = 0.2
                    material.hardness = 40
            
            # Handle transparency
            if texture.has_alpha or texture.is_transparent:
                material.use_transparency = True
                material.alpha = 0.8 if texture.is_transparent else 1.0
            
            # Set UV scaling based on texture aspect ratio
            aspect_ratio = texture.get_aspect_ratio()
            if aspect_ratio > 1.5:  # Wide texture
                material.uv_scale = (1.0, 1.0 / aspect_ratio)
            elif aspect_ratio < 0.67:  # Tall texture
                material.uv_scale = (aspect_ratio, 1.0)
            else:  # Square-ish texture
                material.uv_scale = (1.0, 1.0)
            
            # Set texture influence
            material.texture_influence = 1.0
            
            return material
            
        except Exception as e:
            logger.warning(f"Error creating material from texture {tex_name}: {e}")
            return None
    
    def _guess_material_type(self, tex_name: str, texture: TextureInfo) -> str:
        """Guess material type from texture name and properties"""
        tex_name_lower = tex_name.lower()
        
        # Check for common GTA texture patterns
        if any(pattern in tex_name_lower for pattern in ['concrete', 'cement', 'wall']):
            return 'concrete'
        elif any(pattern in tex_name_lower for pattern in ['brick', 'stone', 'rock']):
            return 'brick'
        elif any(pattern in tex_name_lower for pattern in ['wood', 'log', 'plank']):
            return 'wood'
        elif any(pattern in tex_name_lower for pattern in ['metal', 'steel', 'iron']):
            return 'metal'
        elif any(pattern in tex_name_lower for pattern in ['glass', 'window', 'trans']):
            return 'glass'
        elif any(pattern in tex_name_lower for pattern in ['road', 'asphalt', 'pavement']):
            return 'asphalt'
        elif any(pattern in tex_name_lower for pattern in ['grass', 'lawn', 'field']):
            return 'grass'
        elif any(pattern in tex_name_lower for pattern in ['dirt', 'mud', 'ground']):
            return 'dirt'
        elif any(pattern in tex_name_lower for pattern in ['car', 'vehicle', 'auto']):
            return 'car_paint'
        elif any(pattern in tex_name_lower for pattern in ['rubber', 'tire', 'wheel']):
            return 'rubber'
        elif any(pattern in tex_name_lower for pattern in ['tree', 'bark', 'trunk']):
            return 'tree_bark'
        elif any(pattern in tex_name_lower for pattern in ['leaf', 'foliage', 'plant']):
            return 'leaf'
        elif any(pattern in tex_name_lower for pattern in ['water', 'sea', 'ocean', 'river']):
            return 'water'
        
        # Guess from texture properties
        if texture.is_dark and not texture.has_alpha:
            return 'asphalt'
        elif texture.is_light and texture.has_alpha:
            return 'glass'
        elif texture.has_alpha:
            return 'leaf'
        
        return 'concrete'  # Default
    
    def _sanitize_material_name(self, name: str) -> str:
        """Sanitize material name for Blender compatibility"""
        # Remove invalid characters
        sanitized = re.sub(r'[^\w\.\-]', '_', name)
        
        # Limit length
        if len(sanitized) > 63:  # Blender limit
            sanitized = sanitized[:63]
        
        # Ensure it starts with a letter
        if not sanitized[0].isalpha():
            sanitized = 'mat_' + sanitized
        
        return sanitized
    
    def create_material_for_object(self, obj_name: str, texture_name: str = "",
                                  gta_flags: int = 0) -> BlenderMaterial:
        """Create material for specific object"""
        # Check if we have a texture-based material
        if texture_name and texture_name in self.texture_mapping:
            material_name = self.texture_mapping[texture_name]
            if material_name in self.materials:
                # Return copy with object-specific adjustments
                material = self.materials[material_name]
                return self._adjust_material_for_object(material, obj_name, gta_flags)
        
        # Create new material
        material = BlenderMaterial(
            name=self._sanitize_material_name(f"{obj_name}_material"),
            type="SURFACE"
        )
        
        # Apply GTA flags
        self._apply_gta_flags(material, gta_flags)
        
        return material
    
    def _adjust_material_for_object(self, base_material: BlenderMaterial,
                                   obj_name: str, gta_flags: int) -> BlenderMaterial:
        """Adjust base material for specific object"""
        # Create copy
        material = BlenderMaterial(
            name=self._sanitize_material_name(f"{obj_name}_{base_material.name}"),
            type=base_material.type,
            diffuse_color=base_material.diffuse_color,
            specular_color=base_material.specular_color,
            specular_intensity=base_material.specular_intensity,
            hardness=base_material.hardness,
            alpha=base_material.alpha,
            texture=base_material.texture,
            texture_blend_type=base_material.texture_blend_type,
            texture_influence=base_material.texture_influence,
            uv_scale=base_material.uv_scale,
            uv_offset=base_material.uv_offset,
            uv_rotation=base_material.uv_rotation,
            use_shadeless=base_material.use_shadeless,
            use_transparency=base_material.use_transparency,
            use_mirror=base_material.use_mirror
        )
        
        # Apply GTA flags
        self._apply_gta_flags(material, gta_flags)
        
        return material
    
    def _apply_gta_flags(self, material: BlenderMaterial, gta_flags: int):
        """Apply GTA flags to material properties"""
        material.gta_flags = gta_flags
        
        # Check for specific flags
        if gta_flags & 0x10:  # Alpha transparency 1
            material.use_transparency = True
            material.alpha = 0.7
            
        if gta_flags & 0x20:  # Alpha transparency 2
            material.use_transparency = True
            material.alpha = 0.5
            
        if gta_flags & 0x80:  # Is tunnel
            material.use_shadeless = True
            
        if gta_flags & 0x100:  # Is tunnel transition
            material.use_shadeless = True
            
        if gta_flags & 0x400:  # Has no shadows
            material.use_shadeless = True
            
        if gta_flags & 0x800:  # Has no reflections
            material.use_mirror = False
            
        if gta_flags & 0x1000:  # Has no culling
            # Not directly applicable to materials
            pass
            
        if gta_flags & 0x4000:  # Is breakable glass
            material.use_transparency = True
            material.alpha = 0.3
            material.use_mirror = True
    
    def create_mtl_file(self, output_path: str) -> bool:
        """
        Create MTL file from materials
        
        Args:
            output_path: Path to output MTL file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Creating MTL file: {output_path}")
            
            with open(output_path, 'w', encoding='utf-8') as f:
                # Write header
                f.write("# Material definitions for GTA SA Map\n")
                f.write("# Generated by GTA SA Map Converter\n")
                f.write("\n")
                
                # Write each material
                for material_name, material in self.materials.items():
                    self._write_mtl_material(f, material)
            
            logger.info(f"MTL file created: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating MTL file: {e}")
            return False
    
    def _write_mtl_material(self, f, material: BlenderMaterial):
        """Write material to MTL file"""
        f.write(f"newmtl {material.name}\n")
        
        # Write illumination model
        f.write("illum 2\n")  # Highlight on
        
        # Write colors
        f.write(f"Ka {material.diffuse_color[0]:.3f} {material.diffuse_color[1]:.3f} {material.diffuse_color[2]:.3f}\n")
        f.write(f"Kd {material.diffuse_color[0]:.3f} {material.diffuse_color[1]:.3f} {material.diffuse_color[2]:.3f}\n")
        f.write(f"Ks {material.specular_color[0]:.3f} {material.specular_color[1]:.3f} {material.specular_color[2]:.3f}\n")
        
        # Write shininess
        f.write(f"Ns {float(material.hardness):.1f}\n")
        
        # Write transparency
        f.write(f"d {material.alpha:.3f}\n")
        
        # Write texture if available
        if material.texture and material.texture.path:
            # Use relative path if possible
            texture_path = material.texture.path
            if os.path.isabs(texture_path):
                # Try to make it relative to MTL file
                try:
                    mtl_dir = os.path.dirname(f.name)
                    texture_path = os.path.relpath(texture_path, mtl_dir)
                except:
                    pass
            
            f.write(f"map_Kd {texture_path}\n")
        
        f.write("\n")
    
    def optimize_materials(self, merge_similar: bool = True, 
                          threshold: float = 0.1) -> Dict[str, BlenderMaterial]:
        """
        Optimize materials by merging similar ones
        
        Args:
            merge_similar: Whether to merge similar materials
            threshold: Similarity threshold for merging
            
        Returns:
            Optimized materials dictionary
        """
        if not merge_similar or len(self.materials) <= 1:
            return self.materials.copy()
        
        logger.info(f"Optimizing {len(self.materials)} materials")
        
        optimized = {}
        merged_count = 0
        
        # Group similar materials
        material_groups = {}
        
        for mat_name, material in self.materials.items():
            # Create signature for grouping
            signature = self._get_material_signature(material)
            
            if signature in material_groups:
                material_groups[signature].append((mat_name, material))
            else:
                material_groups[signature] = [(mat_name, material)]
        
        # Create optimized materials
        for signature, group in material_groups.items():
            if len(group) == 1:
                # Single material, keep as is
                mat_name, material = group[0]
                optimized[mat_name] = material
            else:
                # Merge similar materials
                base_name = group[0][0]
                base_material = group[0][1]
                
                # Update texture mapping
                for mat_name, _ in group:
                    if mat_name in self.texture_mapping.values():
                        # Find texture for this material
                        for tex_name, mat_name2 in self.texture_mapping.items():
                            if mat_name2 == mat_name:
                                self.texture_mapping[tex_name] = base_name
                                break
                
                optimized[base_name] = base_material
                merged_count += len(group) - 1
        
        self.materials = optimized
        logger.info(f"Optimized to {len(optimized)} materials (merged {merged_count})")
        
        return optimized.copy()
    
    def _get_material_signature(self, material: BlenderMaterial) -> str:
        """Create signature for material comparison"""
        signature_parts = []
        
        # Color signature
        color_sig = f"{material.diffuse_color[0]:.2f},{material.diffuse_color[1]:.2f},{material.diffuse_color[2]:.2f}"
        signature_parts.append(f"color:{color_sig}")
        
        # Texture signature
        if material.texture:
            texture_sig = material.texture.name
            signature_parts.append(f"texture:{texture_sig}")
        
        # Properties signature
        props_sig = f"spec:{material.specular_intensity:.2f},hard:{material.hardness},alpha:{material.alpha:.2f}"
        signature_parts.append(f"props:{props_sig}")
        
        # Flags signature
        flags_sig = f"shadeless:{material.use_shadeless},trans:{material.use_transparency},mirror:{material.use_mirror}"
        signature_parts.append(f"flags:{flags_sig}")
        
        return "|".join(signature_parts)
    
    def get_material_stats(self) -> Dict[str, Any]:
        """Get material statistics"""
        textured_count = sum(1 for m in self.materials.values() if m.texture)
        transparent_count = sum(1 for m in self.materials.values() if m.use_transparency)
        shadeless_count = sum(1 for m in self.materials.values() if m.use_shadeless)
        
        return {
            'total_materials': len(self.materials),
            'textured_materials': textured_count,
            'transparent_materials': transparent_count,
            'shadeless_materials': shadeless_count,
            'texture_mappings': len(self.texture_mapping)
        }


# Convenience functions
def create_materials_from_textures(textures: Dict[str, Dict[str, Any]], 
                                  config) -> Dict[str, BlenderMaterial]:
    """Create materials from textures"""
    builder = MaterialBuilder(config)
    return builder.create_materials(textures)


def create_mtl_file(materials: Dict[str, BlenderMaterial], 
                   output_path: str) -> bool:
    """Create MTL file from materials"""
    # Create temporary builder
    class TempConfig:
        def __init__(self): pass
    
    builder = MaterialBuilder(TempConfig())
    builder.materials = materials
    return builder.create_mtl_file(output_path)


if __name__ == "__main__":
    # Test the material builder
    import sys
    
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        
        # Create test config
        class TestConfig:
            def __init__(self):
                self.conversion = type('obj', (), {'scale_factor': 0.01})()
                self.blender = type('obj', (), {'flip_uv_vertical': True})()
        
        config = TestConfig()
        
        # Create material builder
        builder = MaterialBuilder(config)
        
        # Create test textures dictionary
        test_textures = {
            'concrete_wall': {
                'path': './test_textures/concrete.png',
                'width': 512,
                'height': 512,
                'format': 'png'
            },
            'glass_window': {
                'path': './test_textures/glass.png',
                'width': 256,
                'height': 256,
                'format': 'png'
            }
        }
        
        # Create materials
        materials = builder.create_materials(test_textures)
        
        # Create MTL file
        mtl_path = os.path.join(output_dir, 'test_materials.mtl')
        success = builder.create_mtl_file(mtl_path)
        
        if success:
            print(f"Test materials created: {len(materials)} materials")
            print(f"MTL file: {mtl_path}")
            
            stats = builder.get_material_stats()
            print(f"Statistics: {stats}")
        else:
            print("Test failed")
    else:
        print("Usage: python material_builder.py <output_dir>")
        print("\nExample:")
        print("  python material_builder.py ./materials")
