"""
GTA SA Map Converter - Blender 2.79 Compatibility
Ensures exported OBJ files are compatible with Blender 2.79
"""

import os
import re
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class CompatibilityIssue:
    """Represents a compatibility issue"""
    severity: str  # 'ERROR', 'WARNING', 'INFO'
    category: str  # 'VERTEX', 'FACE', 'MATERIAL', 'TEXTURE', 'FILE_SIZE'
    message: str
    location: str = ""  # Object name, material name, etc.
    suggestion: str = ""
    
    def to_string(self) -> str:
        """Convert to string representation"""
        return f"[{self.severity}] {self.category}: {self.message} ({self.location})"


@dataclass
class Blender279Limits:
    """Blender 2.79 specific limits"""
    # Vertex/face limits per object
    MAX_VERTICES_PER_OBJECT: int = 65535  # 16-bit limit for some operations
    MAX_FACES_PER_OBJECT: int = 65535
    
    # Material limits
    MAX_MATERIALS_PER_OBJECT: int = 128
    MAX_MATERIAL_SLOTS: int = 128
    
    # Texture limits
    MAX_TEXTURE_SIZE: int = 4096  # Maximum texture dimension
    MAX_TEXTURE_MEMORY: int = 1024 * 1024 * 1024  # 1GB texture memory (approximate)
    
    # File size limits
    MAX_OBJ_FILE_SIZE: int = 1024 * 1024 * 1024  # 1GB
    MAX_MTL_FILE_SIZE: int = 100 * 1024 * 1024  # 100MB
    
    # String limits
    MAX_NAME_LENGTH: int = 63  # Blender object name limit
    MAX_PATH_LENGTH: int = 260  # Windows path limit
    
    # Performance considerations
    OPTIMAL_VERTICES_PER_OBJECT: int = 32768
    OPTIMAL_FACES_PER_OBJECT: int = 32768
    OPTIMAL_MATERIALS_PER_OBJECT: int = 32


class CompatibilityChecker:
    """Checks OBJ data for Blender 2.79 compatibility"""
    
    def __init__(self):
        self.limits = Blender279Limits()
        self.issues: List[CompatibilityIssue] = []
        self.fixes_applied: List[str] = []
        
    def check_obj_data(self, obj_data: Dict[str, Any]) -> List[CompatibilityIssue]:
        """
        Check OBJ data for compatibility issues
        
        Args:
            obj_data: OBJ data dictionary
            
        Returns:
            List of compatibility issues
        """
        self.issues.clear()
        
        # Check vertices
        self._check_vertices(obj_data)
        
        # Check faces
        self._check_faces(obj_data)
        
        # Check materials
        self._check_materials(obj_data)
        
        # Check textures
        self._check_textures(obj_data)
        
        # Check file structure
        self._check_file_structure(obj_data)
        
        # Check naming conventions
        self._check_naming(obj_data)
        
        return self.issues.copy()
    
    def _check_vertices(self, obj_data: Dict[str, Any]):
        """Check vertex-related issues"""
        vertices = obj_data.get('vertices', [])
        vertex_count = len(vertices)
        
        # Check vertex count per object
        if vertex_count > self.limits.MAX_VERTICES_PER_OBJECT:
            self.issues.append(CompatibilityIssue(
                severity='ERROR',
                category='VERTEX',
                message=f'Object has {vertex_count} vertices, exceeds Blender 2.79 limit of {self.limits.MAX_VERTICES_PER_OBJECT}',
                location=obj_data.get('name', 'Unknown'),
                suggestion='Split object into multiple smaller objects'
            ))
        elif vertex_count > self.limits.OPTIMAL_VERTICES_PER_OBJECT:
            self.issues.append(CompatibilityIssue(
                severity='WARNING',
                category='VERTEX',
                message=f'Object has {vertex_count} vertices, may cause performance issues in Blender 2.79',
                location=obj_data.get('name', 'Unknown'),
                suggestion=f'Consider splitting object (optimal: <{self.limits.OPTIMAL_VERTICES_PER_OBJECT} vertices)'
            ))
        
        # Check for invalid vertex coordinates
        for i, vertex in enumerate(vertices[:100]):  # Check first 100 vertices
            if len(vertex) < 3:
                self.issues.append(CompatibilityIssue(
                    severity='ERROR',
                    category='VERTEX',
                    message=f'Vertex {i} has invalid number of coordinates: {len(vertex)}',
                    location=obj_data.get('name', 'Unknown'),
                    suggestion='Ensure all vertices have at least 3 coordinates (x, y, z)'
                ))
                break
            
            # Check for extreme values that might cause precision issues
            for coord in vertex[:3]:
                if abs(coord) > 100000.0:
                    self.issues.append(CompatibilityIssue(
                        severity='WARNING',
                        category='VERTEX',
                        message=f'Vertex {i} has extreme coordinate value: {coord}',
                        location=obj_data.get('name', 'Unknown'),
                        suggestion='Apply scale factor to bring coordinates into reasonable range'
                    ))
                    break
    
    def _check_faces(self, obj_data: Dict[str, Any]):
        """Check face-related issues"""
        faces = obj_data.get('faces', [])
        face_count = len(faces)
        
        # Check face count per object
        if face_count > self.limits.MAX_FACES_PER_OBJECT:
            self.issues.append(CompatibilityIssue(
                severity='ERROR',
                category='FACE',
                message=f'Object has {face_count} faces, exceeds Blender 2.79 limit of {self.limits.MAX_FACES_PER_OBJECT}',
                location=obj_data.get('name', 'Unknown'),
                suggestion='Split object into multiple smaller objects'
            ))
        elif face_count > self.limits.OPTIMAL_FACES_PER_OBJECT:
            self.issues.append(CompatibilityIssue(
                severity='WARNING',
                category='FACE',
                message=f'Object has {face_count} faces, may cause performance issues in Blender 2.79',
                location=obj_data.get('name', 'Unknown'),
                suggestion=f'Consider splitting object (optimal: <{self.limits.OPTIMAL_FACES_PER_OBJECT} faces)'
            ))
        
        # Check for non-triangular faces
        non_tri_count = sum(1 for face in faces if len(face) != 3)
        if non_tri_count > 0:
            self.issues.append(CompatibilityIssue(
                severity='WARNING',
                category='FACE',
                message=f'Object has {non_tri_count} non-triangular faces',
                location=obj_data.get('name', 'Unknown'),
                suggestion='Triangulate all faces for better Blender 2.79 compatibility'
            ))
        
        # Check for invalid face indices
        vertex_count = len(obj_data.get('vertices', []))
        for i, face in enumerate(faces[:50]):  # Check first 50 faces
            for vertex_idx in face:
                if vertex_idx >= vertex_count:
                    self.issues.append(CompatibilityIssue(
                        severity='ERROR',
                        category='FACE',
                        message=f'Face {i} references invalid vertex index: {vertex_idx} (max: {vertex_count-1})',
                        location=obj_data.get('name', 'Unknown'),
                        suggestion='Fix vertex indices in faces'
                    ))
                    break
    
    def _check_materials(self, obj_data: Dict[str, Any]):
        """Check material-related issues"""
        materials = obj_data.get('materials', {})
        material_count = len(materials)
        
        # Check material count
        if material_count > self.limits.MAX_MATERIALS_PER_OBJECT:
            self.issues.append(CompatibilityIssue(
                severity='ERROR',
                category='MATERIAL',
                message=f'Object uses {material_count} materials, exceeds Blender 2.79 limit of {self.limits.MAX_MATERIALS_PER_OBJECT}',
                location=obj_data.get('name', 'Unknown'),
                suggestion='Merge similar materials or split object'
            ))
        elif material_count > self.limits.OPTIMAL_MATERIALS_PER_OBJECT:
            self.issues.append(CompatibilityIssue(
                severity='WARNING',
                category='MATERIAL',
                message=f'Object uses {material_count} materials, may cause performance issues in Blender 2.79',
                location=obj_data.get('name', 'Unknown'),
                suggestion=f'Consider merging materials (optimal: <{self.limits.OPTIMAL_MATERIALS_PER_OBJECT} materials)'
            ))
        
        # Check material names
        for mat_name in materials.keys():
            if len(mat_name) > self.limits.MAX_NAME_LENGTH:
                self.issues.append(CompatibilityIssue(
                    severity='WARNING',
                    category='MATERIAL',
                    message=f'Material name too long: "{mat_name[:20]}..." ({len(mat_name)} chars)',
                    location=mat_name,
                    suggestion=f'Shorten material name to {self.limits.MAX_NAME_LENGTH} characters or less'
                ))
            
            # Check for invalid characters in material names
            if not re.match(r'^[a-zA-Z0-9_\.\-]+$', mat_name):
                self.issues.append(CompatibilityIssue(
                    severity='WARNING',
                    category='MATERIAL',
                    message=f'Material name contains special characters: "{mat_name}"',
                    location=mat_name,
                    suggestion='Use only letters, numbers, underscores, dots, and hyphens'
                ))
    
    def _check_textures(self, obj_data: Dict[str, Any]):
        """Check texture-related issues"""
        textures = obj_data.get('textures', {})
        
        for tex_name, tex_info in textures.items():
            # Check texture file existence
            tex_path = tex_info.get('path', '')
            if tex_path and not os.path.exists(tex_path):
                self.issues.append(CompatibilityIssue(
                    severity='WARNING',
                    category='TEXTURE',
                    message=f'Texture file not found: {tex_path}',
                    location=tex_name,
                    suggestion='Ensure texture files are in the correct directory'
                ))
            
            # Check texture dimensions
            width = tex_info.get('width', 0)
            height = tex_info.get('height', 0)
            
            if width > self.limits.MAX_TEXTURE_SIZE or height > self.limits.MAX_TEXTURE_SIZE:
                self.issues.append(CompatibilityIssue(
                    severity='WARNING',
                    category='TEXTURE',
                    message=f'Texture dimensions ({width}x{height}) exceed Blender 2.79 recommended maximum ({self.limits.MAX_TEXTURE_SIZE}x{self.limits.MAX_TEXTURE_SIZE})',
                    location=tex_name,
                    suggestion='Resize texture to smaller dimensions'
                ))
            
            # Check for power-of-two textures (recommended for older Blender)
            if width > 0 and height > 0:
                if not (width & (width - 1) == 0) or not (height & (height - 1) == 0):
                    self.issues.append(CompatibilityIssue(
                        severity='INFO',
                        category='TEXTURE',
                        message=f'Texture dimensions ({width}x{height}) are not power of two',
                        location=tex_name,
                        suggestion='Consider resizing to power of two dimensions (e.g., 256, 512, 1024) for better compatibility'
                    ))
            
            # Check texture format
            tex_format = tex_info.get('format', '').lower()
            if tex_format not in ['png', 'jpg', 'jpeg', 'tga', 'bmp']:
                self.issues.append(CompatibilityIssue(
                    severity='WARNING',
                    category='TEXTURE',
                    message=f'Unsupported texture format: {tex_format}',
                    location=tex_name,
                    suggestion='Convert to PNG, JPEG, TGA, or BMP format'
                ))
    
    def _check_file_structure(self, obj_data: Dict[str, Any]):
        """Check file structure issues"""
        # Estimate file sizes
        vertex_count = len(obj_data.get('vertices', []))
        face_count = len(obj_data.get('faces', []))
        material_count = len(obj_data.get('materials', {}))
        texture_count = len(obj_data.get('textures', {}))
        
        # Rough size estimation
        obj_size_estimate = vertex_count * 30 + face_count * 20 + material_count * 100
        mtl_size_estimate = material_count * 200 + texture_count * 50
        
        if obj_size_estimate > self.limits.MAX_OBJ_FILE_SIZE:
            self.issues.append(CompatibilityIssue(
                severity='WARNING',
                category='FILE_SIZE',
                message=f'Estimated OBJ file size ({obj_size_estimate:,} bytes) may be too large',
                location='OBJ File',
                suggestion='Split export into multiple files or reduce geometry'
            ))
        
        if mtl_size_estimate > self.limits.MAX_MTL_FILE_SIZE:
            self.issues.append(CompatibilityIssue(
                severity='WARNING',
                category='FILE_SIZE',
                message=f'Estimated MTL file size ({mtl_size_estimate:,} bytes) may be too large',
                location='MTL File',
                suggestion='Reduce number of materials or use fewer textures'
            ))
    
    def _check_naming(self, obj_data: Dict[str, Any]):
        """Check naming convention issues"""
        obj_name = obj_data.get('name', '')
        
        # Check object name length
        if len(obj_name) > self.limits.MAX_NAME_LENGTH:
            self.issues.append(CompatibilityIssue(
                severity='WARNING',
                category='NAMING',
                message=f'Object name too long: "{obj_name[:20]}..." ({len(obj_name)} chars)',
                location=obj_name,
                suggestion=f'Shorten object name to {self.limits.MAX_NAME_LENGTH} characters or less'
            ))
        
        # Check for invalid characters in object name
        if obj_name and not re.match(r'^[a-zA-Z0-9_\.\-]+$', obj_name):
            self.issues.append(CompatibilityIssue(
                severity='WARNING',
                category='NAMING',
                message=f'Object name contains special characters: "{obj_name}"',
                location=obj_name,
                suggestion='Use only letters, numbers, underscores, dots, and hyphens'
            ))
        
        # Check group names if present
        groups = obj_data.get('groups', [])
        for group in groups:
            group_name = group.get('name', '')
            if len(group_name) > self.limits.MAX_NAME_LENGTH:
                self.issues.append(CompatibilityIssue(
                    severity='INFO',
                    category='NAMING',
                    message=f'Group name too long: "{group_name[:20]}..."',
                    location=group_name,
                    suggestion=f'Shorten group name to {self.limits.MAX_NAME_LENGTH} characters'
                ))
    
    def get_issue_summary(self) -> Dict[str, int]:
        """Get summary of issues by severity"""
        summary = {
            'ERROR': 0,
            'WARNING': 0,
            'INFO': 0,
            'TOTAL': len(self.issues)
        }
        
        for issue in self.issues:
            summary[issue.severity] += 1
        
        return summary
    
    def has_errors(self) -> bool:
        """Check if there are any ERROR severity issues"""
        return any(issue.severity == 'ERROR' for issue in self.issues)
    
    def has_warnings(self) -> bool:
        """Check if there are any WARNING severity issues"""
        return any(issue.severity == 'WARNING' for issue in self.issues)


class Blender279Compat:
    """Applies fixes for Blender 2.79 compatibility"""
    
    def __init__(self, config):
        self.config = config
        self.checker = CompatibilityChecker()
        self.fixes_applied: List[str] = []
        
    def fix_obj_data(self, obj_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply fixes to OBJ data for Blender 2.79 compatibility
        
        Args:
            obj_data: Original OBJ data
            
        Returns:
            Fixed OBJ data
        """
        self.fixes_applied.clear()
        fixed_data = obj_data.copy()
        
        # Check for issues first
        issues = self.checker.check_obj_data(fixed_data)
        
        # Apply fixes based on issues
        if issues:
            logger.info(f"Found {len(issues)} compatibility issues")
            
            # Fix vertex count if too high
            if self._needs_vertex_split(fixed_data):
                fixed_data = self._split_large_object(fixed_data)
            
            # Fix face count if too high
            if self._needs_face_reduction(fixed_data):
                fixed_data = self._reduce_face_count(fixed_data)
            
            # Fix material count if too high
            if self._needs_material_reduction(fixed_data):
                fixed_data = self._reduce_material_count(fixed_data)
            
            # Fix non-triangular faces
            if self._has_non_triangular_faces(fixed_data):
                fixed_data = self._triangulate_faces(fixed_data)
            
            # Fix naming issues
            fixed_data = self._fix_naming_issues(fixed_data)
            
            # Fix coordinate system if needed
            fixed_data = self._fix_coordinate_system(fixed_data)
        
        return fixed_data
    
    def _needs_vertex_split(self, obj_data: Dict[str, Any]) -> bool:
        """Check if object needs to be split due to vertex count"""
        vertices = obj_data.get('vertices', [])
        return len(vertices) > self.checker.limits.MAX_VERTICES_PER_OBJECT
    
    def _needs_face_reduction(self, obj_data: Dict[str, Any]) -> bool:
        """Check if object needs face reduction"""
        faces = obj_data.get('faces', [])
        return len(faces) > self.checker.limits.MAX_FACES_PER_OBJECT
    
    def _needs_material_reduction(self, obj_data: Dict[str, Any]) -> bool:
        """Check if object needs material reduction"""
        materials = obj_data.get('materials', {})
        return len(materials) > self.checker.limits.MAX_MATERIALS_PER_OBJECT
    
    def _has_non_triangular_faces(self, obj_data: Dict[str, Any]) -> bool:
        """Check if object has non-triangular faces"""
        faces = obj_data.get('faces', [])
        return any(len(face) != 3 for face in faces)
    
    def _split_large_object(self, obj_data: Dict[str, Any]) -> Dict[str, Any]:
        """Split large object into multiple smaller objects"""
        logger.info("Splitting large object for Blender 2.79 compatibility")
        
        vertices = obj_data.get('vertices', [])
        faces = obj_data.get('faces', [])
        materials = obj_data.get('materials', {})
        
        if not vertices or not faces:
            return obj_data
        
        # For now, just log the need for splitting
        # Actual splitting would be complex and context-dependent
        self.fixes_applied.append("Marked for splitting (vertex count too high)")
        
        return obj_data
    
    def _reduce_face_count(self, obj_data: Dict[str, Any]) -> Dict[str, Any]:
        """Reduce face count through simplification"""
        logger.info("Reducing face count for Blender 2.79 compatibility")
        
        faces = obj_data.get('faces', [])
        if len(faces) <= self.checker.limits.MAX_FACES_PER_OBJECT:
            return obj_data
        
        # Simple reduction: remove every nth face
        # In production, would use proper mesh simplification
        reduction_factor = 2
        reduced_faces = faces[::reduction_factor]
        
        if len(reduced_faces) <= self.checker.limits.MAX_FACES_PER_OBJECT:
            obj_data['faces'] = reduced_faces
            self.fixes_applied.append(f"Reduced faces from {len(faces)} to {len(reduced_faces)}")
        
        return obj_data
    
    def _reduce_material_count(self, obj_data: Dict[str, Any]) -> Dict[str, Any]:
        """Reduce material count by merging similar materials"""
        logger.info("Reducing material count for Blender 2.79 compatibility")
        
        materials = obj_data.get('materials', {})
        if len(materials) <= self.checker.limits.MAX_MATERIALS_PER_OBJECT:
            return obj_data
        
        # Simple reduction: keep only first N materials
        # In production, would merge similar materials
        max_materials = self.checker.limits.OPTIMAL_MATERIALS_PER_OBJECT
        
        material_items = list(materials.items())
        if len(material_items) > max_materials:
            reduced_materials = dict(material_items[:max_materials])
            obj_data['materials'] = reduced_materials
            self.fixes_applied.append(f"Reduced materials from {len(materials)} to {len(reduced_materials)}")
        
        return obj_data
    
    def _triangulate_faces(self, obj_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert all faces to triangles"""
        logger.info("Triangulating faces for Blender 2.79 compatibility")
        
        faces = obj_data.get('faces', [])
        triangulated_faces = []
        
        for face in faces:
            if len(face) == 3:
                # Already a triangle
                triangulated_faces.append(face)
            elif len(face) > 3:
                # Convert polygon to triangles using simple fan triangulation
                # Assumes convex polygon
                for i in range(1, len(face) - 1):
                    triangulated_faces.append([face[0], face[i], face[i + 1]])
            else:
                # Invalid face (less than 3 vertices)
                logger.warning(f"Found invalid face with {len(face)} vertices")
        
        if len(triangulated_faces) != len(faces):
            obj_data['faces'] = triangulated_faces
            self.fixes_applied.append(f"Triangulated {len(faces) - len(triangulated_faces)} non-triangular faces")
        
        return obj_data
    
    def _fix_naming_issues(self, obj_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fix naming issues in OBJ data"""
        # Fix object name
        obj_name = obj_data.get('name', '')
        if obj_name:
            fixed_name = self._sanitize_name(obj_name)
            if fixed_name != obj_name:
                obj_data['name'] = fixed_name
                self.fixes_applied.append(f"Fixed object name: '{obj_name}' -> '{fixed_name}'")
        
        # Fix material names
        materials = obj_data.get('materials', {})
        fixed_materials = {}
        for mat_name, mat_data in materials.items():
            fixed_name = self._sanitize_name(mat_name)
            fixed_materials[fixed_name] = mat_data
            if fixed_name != mat_name:
                self.fixes_applied.append(f"Fixed material name: '{mat_name}' -> '{fixed_name}'")
        
        if fixed_materials != materials:
            obj_data['materials'] = fixed_materials
        
        return obj_data
    
    def _sanitize_name(self, name: str) -> str:
        """Sanitize name for Blender compatibility"""
        # Remove invalid characters
        sanitized = re.sub(r'[^\w\.\-]', '_', name)
        
        # Limit length
        max_len = self.checker.limits.MAX_NAME_LENGTH
        if len(sanitized) > max_len:
            sanitized = sanitized[:max_len]
        
        # Ensure it starts with a letter
        if sanitized and not sanitized[0].isalpha():
            sanitized = 'obj_' + sanitized
        
        return sanitized
    
    def _fix_coordinate_system(self, obj_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fix coordinate system for Blender"""
        if self.config.conversion.coordinate_system != "y_up":
            return obj_data
        
        vertices = obj_data.get('vertices', [])
        normals = obj_data.get('normals', [])
        
        # Convert vertices from GTA Z-up to Blender Y-up
        fixed_vertices = []
        for vertex in vertices:
            if len(vertex) >= 3:
                # GTA: (x, y, z) -> Blender: (x, z, -y)
                x, y, z = vertex[0], vertex[1], vertex[2]
                fixed_vertices.append([x, z, -y])
            else:
                fixed_vertices.append(vertex)
        
        # Convert normals
        fixed_normals = []
        for normal in normals:
            if len(normal) >= 3:
                nx, ny, nz = normal[0], normal[1], normal[2]
                fixed_normals.append([nx, nz, -ny])
            else:
                fixed_normals.append(normal)
        
        if fixed_vertices != vertices:
            obj_data['vertices'] = fixed_vertices
            self.fixes_applied.append("Converted coordinate system from Z-up to Y-up")
        
        if fixed_normals != normals:
            obj_data['normals'] = fixed_normals
        
        return obj_data
    
    def prepare_for_export(self, scene_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare scene data for Blender 2.79 export
        
        Args:
            scene_data: Complete scene data
            
        Returns:
            Prepared scene data
        """
        logger.info("Preparing scene for Blender 2.79 export")
        
        prepared_scene = scene_data.copy()
        
        # Check and fix each object
        objects = prepared_scene.get('objects', [])
        fixed_objects = []
        
        for obj in objects:
            # Convert SceneObject to dictionary for compatibility checking
            obj_dict = self._scene_object_to_dict(obj)
            
            # Apply fixes
            fixed_obj_dict = self.fix_obj_data(obj_dict)
            
            # Convert back to SceneObject if needed
            fixed_objects.append(self._dict_to_scene_object(fixed_obj_dict, obj))
        
        prepared_scene['objects'] = fixed_objects
        
        # Log fixes applied
        if self.fixes_applied:
            logger.info(f"Applied {len(self.fixes_applied)} fixes: {', '.join(self.fixes_applied)}")
        
        return prepared_scene
    
    def _scene_object_to_dict(self, scene_obj) -> Dict[str, Any]:
        """Convert SceneObject to dictionary for compatibility checking"""
        # This is a simplified conversion
        # In the actual implementation, you'd extract the relevant data
        return {
            'name': getattr(scene_obj, 'model_name', 'Unknown'),
            'vertices': getattr(scene_obj, 'model_data', {}).get('vertices', []),
            'faces': getattr(scene_obj, 'model_data', {}).get('faces', []),
            'normals': getattr(scene_obj, 'model_data', {}).get('normals', []),
            'materials': {},
            'textures': {},
            'groups': []
        }
    
    def _dict_to_scene_object(self, obj_dict: Dict[str, Any], original_obj) -> Any:
        """Convert fixed dictionary back to SceneObject"""
        # Update the original object's model data
        if hasattr(original_obj, 'model_data'):
            original_obj.model_data['vertices'] = obj_dict.get('vertices', [])
            original_obj.model_data['faces'] = obj_dict.get('faces', [])
            original_obj.model_data['normals'] = obj_dict.get('normals', [])
        
        return original_obj
    
    def validate_export(self, obj_file_path: str, mtl_file_path: str = "") -> bool:
        """
        Validate exported files for Blender 2.79 compatibility
        
        Args:
            obj_file_path: Path to OBJ file
            mtl_file_path: Path to MTL file (optional)
            
        Returns:
            True if files appear compatible, False otherwise
        """
        logger.info(f"Validating export: {obj_file_path}")
        
        validation_passed = True
        
        # Check OBJ file size
        try:
            obj_size = os.path.getsize(obj_file_path)
            if obj_size > self.checker.limits.MAX_OBJ_FILE_SIZE:
                logger.warning(f"OBJ file size ({obj_size:,} bytes) exceeds recommended limit")
                validation_passed = False
        except OSError:
            logger.warning("Could not check OBJ file size")
        
        # Check MTL file if provided
        if mtl_file_path and os.path.exists(mtl_file_path):
            try:
                mtl_size = os.path.getsize(mtl_file_path)
                if mtl_size > self.checker.limits.MAX_MTL_FILE_SIZE:
                    logger.warning(f"MTL file size ({mtl_size:,} bytes) exceeds recommended limit")
                    validation_passed = False
            except OSError:
                logger.warning("Could not check MTL file size")
            
            # Check MTL file content
            try:
                with open(mtl_file_path, 'r', encoding='utf-8') as f:
                    mtl_content = f.read()
                
                # Count materials in MTL file
                material_count = mtl_content.count('newmtl')
                if material_count > self.checker.limits.MAX_MATERIALS_PER_OBJECT:
                    logger.warning(f"MTL file contains {material_count} materials, exceeds recommended limit")
                    validation_passed = False
            except Exception as e:
                logger.warning(f"Could not analyze MTL file: {e}")
        
        # Quick check of OBJ file structure
        try:
            with open(obj_file_path, 'r', encoding='utf-8') as f:
                # Read first few lines
                lines = []
                for _ in range(20):
                    line = f.readline()
                    if not line:
                        break
                    lines.append(line)
                
                # Check for OBJ header
                if not any(line.startswith('#') for line in lines[:5]):
                    logger.warning("OBJ file missing standard header comment")
                
                # Check for vertex definitions
                if not any(line.startswith('v ') for line in lines):
                    logger.warning("OBJ file appears to have no vertex definitions")
                    validation_passed = False
        except Exception as e:
            logger.warning(f"Could not analyze OBJ file: {e}")
            validation_passed = False
        
        if validation_passed:
            logger.info("Export validation passed")
        else:
            logger.warning("Export validation found potential issues")
        
        return validation_passed
    
    def get_compatibility_report(self) -> Dict[str, Any]:
        """Get compatibility report"""
        return {
            'fixes_applied': self.fixes_applied.copy(),
            'issues_found': len(self.checker.issues),
            'issue_summary': self.checker.get_issue_summary(),
            'has_errors': self.checker.has_errors(),
            'has_warnings': self.checker.has_warnings()
        }


# Convenience functions
def check_compatibility(obj_data: Dict[str, Any]) -> List[CompatibilityIssue]:
    """Check OBJ data for Blender 2.79 compatibility"""
    checker = CompatibilityChecker()
    return checker.check_obj_data(obj_data)


def fix_for_blender279(obj_data: Dict[str, Any], config) -> Dict[str, Any]:
    """Fix OBJ data for Blender 2.79 compatibility"""
    compat = Blender279Compat(config)
    return compat.fix_obj_data(obj_data)


if __name__ == "__main__":
    # Test the compatibility checker
    import sys
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create test config
    class TestConfig:
        def __init__(self):
            self.conversion = type('obj', (), {
                'scale_factor': 0.01,
                'coordinate_system': 'y_up'
            })()
    
    config = TestConfig()
    
    # Create test OBJ data
    test_obj_data = {
        'name': 'Test_Object_With_Very_Long_Name_That_Exceeds_Blender_Limits_And_Has_Special_Characters@#$',
        'vertices': [[0, 0, 0]] * 70000,  # Too many vertices
        'faces': [[0, 1, 2, 3]] * 50000,  # Non-triangular faces
        'materials': {f'material_{i}': {} for i in range(150)},  # Too many materials
        'textures': {
            'test_texture': {
                'path': '/very/long/path/that/might/cause/issues/with/windows/limit/test_texture.png',
                'width': 8192,  # Too large
                'height': 8192,
                'format': 'png'
            }
        }
    }
    
    print("Testing Blender 2.79 Compatibility Checker")
    print("=" * 50)
    
    # Check compatibility
    checker = CompatibilityChecker()
    issues = checker.check_obj_data(test_obj_data)
    
    print(f"Found {len(issues)} issues:")
    for issue in issues:
        print(f"  {issue.to_string()}")
    
    print("\nIssue Summary:")
    summary = checker.get_issue_summary()
    for severity, count in summary.items():
        if severity != 'TOTAL':
            print(f"  {severity}: {count}")
    
    print("\n" + "=" * 50)
    print("\nTesting Compatibility Fixer")
    
    # Apply fixes
    compat = Blender279Compat(config)
    fixed_data = compat.fix_obj_data(test_obj_data)
    
    print(f"Applied {len(compat.fixes_applied)} fixes:")
    for fix in compat.fixes_applied:
        print(f"  - {fix}")
    
    print("\nCompatibility Report:")
    report = compat.get_compatibility_report()
    for key, value in report.items():
        if key != 'issue_summary':
            print(f"  {key}: {value}")
