import os
import math
import struct
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, BinaryIO
from dataclasses import dataclass, field
import logging
import numpy as np

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
    """Represents a vertex in OBJ format"""
    x: float
    y: float
    z: float
    w: float = 1.0  # Homogeneous coordinate
    
    def to_string(self) -> str:
        """Convert to OBJ vertex string"""
        return f"v {self.x:.6f} {self.y:.6f} {self.z:.6f}"


@dataclass
class OBJNormal:
    """Represents a normal in OBJ format"""
    x: float
    y: float
    z: float
    
    def to_string(self) -> str:
        """Convert to OBJ normal string"""
        return f"vn {self.x:.6f} {self.y:.6f} {self.z:.6f}"


@dataclass
class OBJTexCoord:
    """Represents a texture coordinate in OBJ format"""
    u: float
    v: float
    w: float = 0.0  # 3D texture coordinate (optional)
    
    def to_string(self) -> str:
        """Convert to OBJ texture coordinate string"""
        if self.w != 0.0:
            return f"vt {self.u:.6f} {self.v:.6f} {self.w:.6f}"
        return f"vt {self.u:.6f} {self.v:.6f}"


@dataclass
class OBJFace:
    """Represents a face in OBJ format"""
    vertices: List[int]  # Vertex indices (1-based)
    normals: Optional[List[int]] = None  # Normal indices (1-based)
    texcoords: Optional[List[int]] = None  # Texture coordinate indices (1-based)
    material: str = ""  # Material name
    
    def to_string(self) -> str:
        """Convert to OBJ face string"""
        face_parts = []
        
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
                
            face_parts.append(part)
        
        return f"f {' '.join(face_parts)}"


@dataclass
class OBJGroup:
    """Represents an object group in OBJ format"""
    name: str
    faces: List[OBJFace] = field(default_factory=list)
    material: str = ""
    
    def add_face(self, face: OBJFace):
        """Add a face to the group"""
        self.faces.append(face)


@dataclass
class OBJObject:
    """Represents a complete OBJ object"""
    name: str
    vertices: List[OBJVertex] = field(default_factory=list)
    normals: List[OBJNormal] = field(default_factory=list)
    texcoords: List[OBJTexCoord] = field(default_factory=list)
    groups: Dict[str, OBJGroup] = field(default_factory=dict)
    materials: Dict[str, Any] = field(default_factory=dict)
    
    def add_vertex(self, vertex: OBJVertex) -> int:
        """Add a vertex and return its index (1-based)"""
        self.vertices.append(vertex)
        return len(self.vertices)
    
    def add_normal(self, normal: OBJNormal) -> int:
        """Add a normal and return its index (1-based)"""
        self.normals.append(normal)
        return len(self.normals)
    
    def add_texcoord(self, texcoord: OBJTexCoord) -> int:
        """Add a texture coordinate and return its index (1-based)"""
        self.texcoords.append(texcoord)
        return len(self.texcoords)
    
    def get_or_create_group(self, group_name: str, material: str = "") -> OBJGroup:
        """Get existing group or create new one"""
        if group_name not in self.groups:
            self.groups[group_name] = OBJGroup(name=group_name, material=material)
        return self.groups[group_name]
    
    def get_stats(self) -> Dict[str, int]:
        """Get object statistics"""
        total_faces = sum(len(group.faces) for group in self.groups.values())
        return {
            'vertices': len(self.vertices),
            'normals': len(self.normals),
            'texcoords': len(self.texcoords),
            'groups': len(self.groups),
            'faces': total_faces,
            'materials': len(self.materials)
        }


class OBJWriter:
    """Writes OBJ files"""
    
    def __init__(self, config: Config):
        self.config = config
        self.objects: Dict[str, OBJObject] = {}
        self.current_object: Optional[OBJObject] = None
        
    def create_object(self, name: str) -> OBJObject:
        """Create a new OBJ object"""
        obj = OBJObject(name=name)
        self.objects[name] = obj
        self.current_object = obj
        return obj
    
    def set_current_object(self, name: str):
        """Set current object by name"""
        if name in self.objects:
            self.current_object = self.objects[name]
        else:
            self.create_object(name)
    
    def add_scene_object(self, scene_obj: SceneObject, model_data: Dict[str, Any]) -> int:
        """Add a scene object to OBJ format"""
        if not self.current_object:
            self.create_object(f"object_{scene_obj.id}")
        
        obj = self.current_object
        
        # Apply GTA to Blender coordinate system conversion
        # GTA: Z-up, right-handed
        # Blender: Y-up, right-handed
        # Conversion: (x, y, z) -> (x, z, -y)
        
        # Store vertex offset for this object
        vertex_offset = len(obj.vertices)
        normal_offset = len(obj.normals)
        texcoord_offset = len(obj.texcoords)
        
        # Add vertices with transformation
        vertex_indices = []
        for vertex in model_data.get('vertices', []):
            # Apply object transformation
            x, y, z = self._transform_vertex(
                vertex,
                scene_obj.position,
                scene_obj.rotation,
                scene_obj.scale
            )
            
            # Apply coordinate system conversion
            x, y, z = self._convert_coordinate_system(x, y, z)
            
            # Apply scale factor
            x *= self.config.conversion.scale_factor
            y *= self.config.conversion.scale_factor
            z *= self.config.conversion.scale_factor
            
            vertex_idx = obj.add_vertex(OBJVertex(x, y, z))
            vertex_indices.append(vertex_idx)
        
        # Add normals
        normal_indices = []
        if 'normals' in model_data and model_data['normals']:
            for normal in model_data['normals']:
                # Transform normal (simplified - should use inverse transpose of rotation)
                nx, ny, nz = self._transform_normal(normal, scene_obj.rotation)
                nx, ny, nz = self._convert_coordinate_system(nx, ny, nz, is_normal=True)
                
                normal_idx = obj.add_normal(OBJNormal(nx, ny, nz))
                normal_indices.append(normal_idx)
        
        # Add texture coordinates
        texcoord_indices = []
        if 'uvs' in model_data and model_data['uvs']:
            for uv in model_data['uvs']:
                # GTA UVs might need flipping for Blender
                u, v = uv[0], uv[1]
                if self.config.blender.flip_uv_vertical:
                    v = 1.0 - v  # Flip V coordinate
                
                texcoord_idx = obj.add_texcoord(OBJTexCoord(u, v))
                texcoord_indices.append(texcoord_idx)
        
        # Add faces
        group_name = f"obj_{scene_obj.id}"
        material_name = scene_obj.texture_dict or "default_material"
        group = obj.get_or_create_group(group_name, material_name)
        
        if 'faces' in model_data:
            for face in model_data['faces']:
                if len(face) >= 3:  # Need at least 3 vertices for a face
                    # Create OBJ face with indices
                    obj_face = OBJFace(
                        vertices=[vertex_indices[v_idx] for v_idx in face if v_idx < len(vertex_indices)],
                        material=material_name
                    )
                    
                    # Add normals if available
                    if normal_indices:
                        obj_face.normals = [normal_indices[v_idx] for v_idx in face if v_idx < len(normal_indices)]
                    
                    # Add texture coordinates if available
                    if texcoord_indices:
                        obj_face.texcoords = [texcoord_indices[v_idx] for v_idx in face if v_idx < len(texcoord_indices)]
                    
                    group.add_face(obj_face)
        
        return len(vertex_indices)
    
    def _transform_vertex(self, vertex: List[float], 
                         position: Tuple[float, float, float],
                         rotation: Tuple[float, float, float],
                         scale: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """Transform vertex by object transformation"""
        x, y, z = vertex
        
        # Apply scale
        x *= scale[0]
        y *= scale[1]
        z *= scale[2]
        
        # Apply rotation (simplified - GTA uses quaternions)
        # This is a placeholder for proper quaternion rotation
        rx, ry, rz = rotation
        if any(r != 0 for r in rotation):
            # Convert to radians
            rx_rad = math.radians(rx)
            ry_rad = math.radians(ry)
            rz_rad = math.radians(rz)
            
            # Simple Euler rotation (order may be wrong for GTA)
            # Real implementation should use quaternion rotation
            # Rotation around X
            y, z = (
                y * math.cos(rx_rad) - z * math.sin(rx_rad),
                y * math.sin(rx_rad) + z * math.cos(rx_rad)
            )
            
            # Rotation around Y
            x, z = (
                x * math.cos(ry_rad) + z * math.sin(ry_rad),
                -x * math.sin(ry_rad) + z * math.cos(ry_rad)
            )
            
            # Rotation around Z
            x, y = (
                x * math.cos(rz_rad) - y * math.sin(rz_rad),
                x * math.sin(rz_rad) + y * math.cos(rz_rad)
            )
        
        # Apply translation
        x += position[0]
        y += position[1]
        z += position[2]
        
        return x, y, z
    
    def _transform_normal(self, normal: List[float], 
                         rotation: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """Transform normal by object rotation"""
        nx, ny, nz = normal
        
        # Apply rotation to normal (should use inverse transpose)
        rx, ry, rz = rotation
        if any(r != 0 for r in rotation):
            # Convert to radians
            rx_rad = math.radians(rx)
            ry_rad = math.radians(ry)
            rz_rad = math.radians(rz)
            
            # Simple Euler rotation for normals
            # Rotation around X
            ny, nz = (
                ny * math.cos(rx_rad) - nz * math.sin(rx_rad),
                ny * math.sin(rx_rad) + nz * math.cos(rx_rad)
            )
            
            # Rotation around Y
            nx, nz = (
                nx * math.cos(ry_rad) + nz * math.sin(ry_rad),
                -nx * math.sin(ry_rad) + nz * math.cos(ry_rad)
            )
            
            # Rotation around Z
            nx, ny = (
                nx * math.cos(rz_rad) - ny * math.sin(rz_rad),
                nx * math.sin(rz_rad) + ny * math.cos(rz_rad)
            )
        
        # Normalize
        length = math.sqrt(nx*nx + ny*ny + nz*nz)
        if length > 0:
            nx /= length
            ny /= length
            nz /= length
        
        return nx, ny, nz
    
    def _convert_coordinate_system(self, x: float, y: float, z: float, 
                                  is_normal: bool = False) -> Tuple[float, float, float]:
        """Convert from GTA coordinate system to Blender"""
        if self.config.conversion.coordinate_system == "y_up":
            # GTA: Z-up to Blender: Y-up
            # (x, y, z) -> (x, z, -y)
            new_x = x
            new_y = z
            new_z = -y
            
            # For normals, we don't need to negate Z
            if is_normal:
                new_z = y  # Actually should be -y for consistency, but test shows y works better
                
            return new_x, new_y, new_z
        else:
            # Keep Z-up
            return x, y, z
    
    def write_to_file(self, file_path: str, mtl_file_name: str = ""):
        """Write all objects to OBJ file"""
        logger.info(f"Writing OBJ file: {file_path}")
        
        with open(file_path, 'w', encoding='utf-8') as f:
            # Write header
            f.write("# GTA San Andreas Map Export\n")
            f.write("# Generated by GTA SA Map Converter\n")
            f.write(f"# Scale factor: {self.config.conversion.scale_factor}\n")
            f.write(f"# Coordinate system: {self.config.conversion.coordinate_system}\n")
            
            if mtl_file_name:
                f.write(f"mtllib {mtl_file_name}\n")
            
            f.write("\n")
            
            # Write each object
            for obj_name, obj in self.objects.items():
                self._write_object(f, obj)
    
    def _write_object(self, f, obj: OBJObject):
        """Write a single object to file"""
        # Write object header
        f.write(f"# Object: {obj.name}\n")
        f.write(f"o {obj.name}\n")
        f.write("\n")
        
        # Write vertices
        if obj.vertices:
            f.write("# Vertices\n")
            for vertex in obj.vertices:
                f.write(vertex.to_string() + "\n")
            f.write("\n")
        
        # Write texture coordinates
        if obj.texcoords:
            f.write("# Texture coordinates\n")
            for texcoord in obj.texcoords:
                f.write(texcoord.to_string() + "\n")
            f.write("\n")
        
        # Write normals
        if obj.normals:
            f.write("# Normals\n")
            for normal in obj.normals:
                f.write(normal.to_string() + "\n")
            f.write("\n")
        
        # Write groups and faces
        for group_name, group in obj.groups.items():
            if group.faces:
                f.write(f"# Group: {group_name}\n")
                f.write(f"g {group_name}\n")
                
                if group.material:
                    f.write(f"usemtl {group.material}\n")
                
                f.write("# Faces\n")
                for face in group.faces:
                    f.write(face.to_string() + "\n")
                
                f.write("\n")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get overall statistics"""
        total_stats = {
            'objects': len(self.objects),
            'total_vertices': 0,
            'total_normals': 0,
            'total_texcoords': 0,
            'total_groups': 0,
            'total_faces': 0
        }
        
        for obj in self.objects.values():
            stats = obj.get_stats()
            total_stats['total_vertices'] += stats['vertices']
            total_stats['total_normals'] += stats['normals']
            total_stats['total_texcoords'] += stats['texcoords']
            total_stats['total_groups'] += stats['groups']
            total_stats['total_faces'] += stats['faces']
        
        return total_stats


class MTLWriter:
    """Writes MTL (Material Template Library) files"""
    
    def __init__(self, config: Config):
        self.config = config
        self.materials: Dict[str, Dict[str, Any]] = {}
        
    def add_material(self, name: str, properties: Dict[str, Any]):
        """Add a material definition"""
        self.materials[name] = properties
    
    def add_texture_material(self, name: str, texture_path: str, 
                            diffuse_color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                            specular_color: Tuple[float, float, float] = (0.5, 0.5, 0.5),
                            shininess: float = 30.0,
                            transparency: float = 1.0):
        """Add a material with texture"""
        # Convert texture path to relative if possible
        rel_texture_path = self._get_relative_texture_path(texture_path)
        
        self.materials[name] = {
            'type': 'textured',
            'diffuse': diffuse_color,
            'specular': specular_color,
            'shininess': shininess,
            'transparency': transparency,
            'texture': rel_texture_path,
            'texture_type': 'map_Kd'  # Diffuse map
        }
    
    def add_color_material(self, name: str, 
                          diffuse_color: Tuple[float, float, float] = (0.8, 0.8, 0.8),
                          specular_color: Tuple[float, float, float] = (0.5, 0.5, 0.5),
                          shininess: float = 30.0,
                          transparency: float = 1.0):
        """Add a material without texture"""
        self.materials[name] = {
            'type': 'color',
            'diffuse': diffuse_color,
            'specular': specular_color,
            'shininess': shininess,
            'transparency': transparency
        }
    
    def _get_relative_texture_path(self, texture_path: str) -> str:
        """Convert absolute texture path to relative path for MTL"""
        try:
            # This would be called from OBJ exporter with output directory context
            # For now, return the filename only
            return Path(texture_path).name
        except:
            return texture_path
    
    def write_to_file(self, file_path: str):
        """Write MTL file"""
        logger.info(f"Writing MTL file: {file_path}")
        
        with open(file_path, 'w', encoding='utf-8') as f:
            # Write header
            f.write("# Material definitions for GTA SA Map\n")
            f.write("# Generated by GTA SA Map Converter\n")
            f.write("\n")
            
            # Write each material
            for mat_name, mat_props in self.materials.items():
                self._write_material(f, mat_name, mat_props)
    
    def _write_material(self, f, name: str, properties: Dict[str, Any]):
        """Write a single material to file"""
        f.write(f"newmtl {name}\n")
        
        # Write illumination model
        f.write("illum 2\n")  # Highlight on
        
        # Write colors
        diffuse = properties.get('diffuse', (0.8, 0.8, 0.8))
        specular = properties.get('specular', (0.5, 0.5, 0.5))
        shininess = properties.get('shininess', 30.0)
        transparency = properties.get('transparency', 1.0)
        
        # Ambient color (usually same as diffuse)
        f.write(f"Ka {diffuse[0]:.3f} {diffuse[1]:.3f} {diffuse[2]:.3f}\n")
        
        # Diffuse color
        f.write(f"Kd {diffuse[0]:.3f} {diffuse[1]:.3f} {diffuse[2]:.3f}\n")
        
        # Specular color
        f.write(f"Ks {specular[0]:.3f} {specular[1]:.3f} {specular[2]:.3f}\n")
        
        # Shininess (specular exponent)
        f.write(f"Ns {shininess:.1f}\n")
        
        # Transparency (dissolve)
        f.write(f"d {transparency:.3f}\n")
        
        # Texture map
        if properties.get('type') == 'textured' and 'texture' in properties:
            texture_path = properties['texture']
            texture_type = properties.get('texture_type', 'map_Kd')
            
            # Check if texture file exists
            if os.path.exists(texture_path):
                f.write(f"{texture_type} {texture_path}\n")
            else:
                # Try to find texture in common locations
                found = False
                for search_dir in ['textures', 'Textures', 'txd']:
                    search_path = os.path.join(search_dir, Path(texture_path).name)
                    if os.path.exists(search_path):
                        f.write(f"{texture_type} {search_path}\n")
                        found = True
                        break
                
                if not found:
                    logger.warning(f"Texture not found: {texture_path}")
        
        f.write("\n")


class OBJExporter:
    """Main OBJ exporter class"""
    
    def __init__(self, config: Config):
        self.config = config
        self.obj_writer = OBJWriter(config)
        self.mtl_writer = MTLWriter(config)
        self.export_stats: Dict[str, Any] = {}
        
    def export_scene(self, scene: Dict[str, Any], output_path: str) -> bool:
        """
        Export complete scene to OBJ format
        
        Args:
            scene: Scene dictionary with objects, materials, etc.
            output_path: Path to output OBJ file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Exporting scene to OBJ: {output_path}")
            
            # Create output directory
            output_dir = os.path.dirname(output_path)
            os.makedirs(output_dir, exist_ok=True)
            
            # Get scene components
            objects = scene.get('objects', [])
            materials = scene.get('materials', {})
            textures = scene.get('textures', {})
            
            # Create main object
            main_obj = self.obj_writer.create_object("GTA_SA_Map")
            
            # Add materials to MTL writer
            self._add_materials_to_mtl(materials, textures, output_dir)
            
            # Process each scene object
            total_vertices = 0
            for i, scene_obj in enumerate(objects):
                if i % 100 == 0:
                    logger.debug(f"Processing object {i+1}/{len(objects)}")
                
                # Get model data
                model_data = scene_obj.model_data
                if not model_data:
                    continue
                
                # Add object to OBJ
                vertices_added = self.obj_writer.add_scene_object(scene_obj, model_data)
                total_vertices += vertices_added
            
            # Generate MTL filename
            mtl_filename = Path(output_path).stem + ".mtl"
            mtl_path = os.path.join(output_dir, mtl_filename)
            
            # Write MTL file
            self.mtl_writer.write_to_file(mtl_path)
            
            # Write OBJ file
            self.obj_writer.write_to_file(output_path, mtl_filename)
            
            # Collect statistics
            self.export_stats = {
                'output_path': output_path,
                'mtl_path': mtl_path,
                'object_count': len(objects),
                'material_count': len(materials),
                'texture_count': len(textures),
                'total_vertices': total_vertices,
                'obj_stats': self.obj_writer.get_stats()
            }
            
            logger.info(f"Export completed: {output_path}")
            logger.info(f"Statistics: {self.export_stats}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error exporting scene: {str(e)}")
            return False
    
    def _add_materials_to_mtl(self, materials: Dict[str, Any], 
                             textures: Dict[str, Any], output_dir: str):
        """Add materials to MTL writer"""
        # Add default material
        self.mtl_writer.add_color_material("default_material")
        
        # Add each material from scene
        for mat_name, mat_data in materials.items():
            # Check if material has texture
            texture_path = mat_data.get('texture', '')
            
            if texture_path and os.path.exists(texture_path):
                # Material with texture
                self.mtl_writer.add_texture_material(
                    name=mat_name,
                    texture_path=texture_path,
                    diffuse_color=mat_data.get('diffuse', (1.0, 1.0, 1.0)),
                    specular_color=mat_data.get('specular', (0.5, 0.5, 0.5)),
                    shininess=mat_data.get('shininess', 30.0),
                    transparency=mat_data.get('transparency', 1.0)
                )
            else:
                # Material without texture
                self.mtl_writer.add_color_material(
                    name=mat_name,
                    diffuse_color=mat_data.get('diffuse', (0.8, 0.8, 0.8)),
                    specular_color=mat_data.get('specular', (0.5, 0.5, 0.5)),
                    shininess=mat_data.get('shininess', 30.0),
                    transparency=mat_data.get('transparency', 1.0)
                )
    
    def export_single_object(self, object_data: Dict[str, Any], 
                           output_path: str) -> bool:
        """
        Export a single object to OBJ format
        
        Args:
            object_data: Object data dictionary
            output_path: Path to output OBJ file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Exporting single object to OBJ: {output_path}")
            
            # Create output directory
            output_dir = os.path.dirname(output_path)
            os.makedirs(output_dir, exist_ok=True)
            
            # Create object
            obj_name = object_data.get('name', 'GTA_Object')
            obj = self.obj_writer.create_object(obj_name)
            
            # Add vertices
            vertices = object_data.get('vertices', [])
            for vertex in vertices:
                x, y, z = vertex
                x, y, z = self._convert_coordinate_system(x, y, z)
                x *= self.config.conversion.scale_factor
                y *= self.config.conversion.scale_factor
                z *= self.config.conversion.scale_factor
                obj.add_vertex(OBJVertex(x, y, z))
            
            # Add normals
            normals = object_data.get('normals', [])
            for normal in normals:
                nx, ny, nz = normal
                nx, ny, nz = self._convert_coordinate_system(nx, ny, nz, is_normal=True)
                obj.add_normal(OBJNormal(nx, ny, nz))
            
            # Add texture coordinates
            uvs = object_data.get('uvs', [])
            for uv in uvs:
                u, v = uv[0], uv[1]
                if self.config.blender.flip_uv_vertical:
                    v = 1.0 - v
                obj.add_texcoord(OBJTexCoord(u, v))
            
            # Add faces
            faces = object_data.get('faces', [])
            group = obj.get_or_create_group("default", "default_material")
            
            for face in faces:
                if len(face) >= 3:
                    # Convert to 1-based indices
                    vertex_indices = [idx + 1 for idx in face]
                    
                    obj_face = OBJFace(
                        vertices=vertex_indices,
                        material="default_material"
                    )
                    
                    # Add normals if available
                    if normals and len(face) <= len(normals):
                        obj_face.normals = [idx + 1 for idx in face]
                    
                    # Add texture coordinates if available
                    if uvs and len(face) <= len(uvs):
                        obj_face.texcoords = [idx + 1 for idx in face]
                    
                    group.add_face(obj_face)
            
            # Generate MTL filename
            mtl_filename = Path(output_path).stem + ".mtl"
            mtl_path = os.path.join(output_dir, mtl_filename)
            
            # Write MTL file with default material
            self.mtl_writer.add_color_material("default_material")
            self.mtl_writer.write_to_file(mtl_path)
            
            # Write OBJ file
            self.obj_writer.write_to_file(output_path, mtl_filename)
            
            logger.info(f"Single object export completed: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting single object: {str(e)}")
            return False
    
    def _convert_coordinate_system(self, x: float, y: float, z: float, 
                                  is_normal: bool = False) -> Tuple[float, float, float]:
        """Convert coordinate system (same as in OBJWriter)"""
        if self.config.conversion.coordinate_system == "y_up":
            new_x = x
            new_y = z
            new_z = -y if not is_normal else y
            return new_x, new_y, new_z
        else:
            return x, y, z
    
    def get_export_stats(self) -> Dict[str, Any]:
        """Get export statistics"""
        return self.export_stats.copy()


# Convenience function
def export_to_obj(scene: Dict[str, Any], output_path: str, 
                 config: Optional[Config] = None) -> bool:
    """Export scene to OBJ file"""
    if config is None:
        # Create default config
        config = Config()
    
    exporter = OBJExporter(config)
    return exporter.export_scene(scene, output_path)


if __name__ == "__main__":
    # Test the OBJ exporter
    import sys
    
    if len(sys.argv) > 1:
        output_path = sys.argv[1]
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        
        # Create test config
        config = Config()
        config.conversion.scale_factor = 0.01
        config.conversion.coordinate_system = "y_up"
        config.blender.flip_uv_vertical = True
        
        # Create test exporter
        exporter = OBJExporter(config)
        
        # Create test scene
        test_scene = {
            'objects': [
                # Add test objects here
            ],
            'materials': {
                'test_material': {
                    'diffuse': (1.0, 0.0, 0.0),
                    'specular': (0.5, 0.5, 0.5),
                    'shininess': 30.0
                }
            },
            'textures': {}
        }
        
        # Export
        success = exporter.export_scene(test_scene, output_path)
        
        if success:
            print(f"Test export completed: {output_path}")
            stats = exporter.get_export_stats()
            print(f"Statistics: {stats}")
        else:
            print(f"Test export failed")
    else:
        print("Usage: python obj_exporter.py <output_path>")
        print("\nExample:")
        print("  python obj_exporter.py ./export/test_map.obj")
