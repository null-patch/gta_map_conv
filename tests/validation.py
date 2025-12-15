"""
GTA SA Map Converter - Validation Tests
Validates exported files and conversion results
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import hashlib

logger = logging.getLogger(__name__)

class MapValidator:
    """Validates exported map files and conversion results"""
    
    def __init__(self, config):
        self.config = config
        self.required_obj_sections = ['v', 'vt', 'vn', 'f', 'g', 'usemtl']
        self.min_vertex_count = 100  # Expected minimum vertices for a valid map
        self.min_face_count = 100    # Expected minimum faces for a valid map

    def validate_exported_obj(self, obj_path: Path) -> Tuple[bool, Dict[str, any]]:
        """
        Validate an exported OBJ file meets all requirements
        
        Args:
            obj_path: Path to the OBJ file
            
        Returns:
            Tuple of (is_valid, validation_results)
        """
        results = {
            'file_exists': False,
            'has_vertices': False,
            'has_faces': False,
            'has_materials': False,
            'vertex_count': 0,
            'face_count': 0,
            'material_count': 0,
            'section_counts': {},
            'errors': []
        }

        if not obj_path.exists():
            results['errors'].append(f"OBJ file not found: {obj_path}")
            return False, results

        results['file_exists'] = True

        try:
            with open(obj_path, 'r') as f:
                obj_content = f.readlines()

            # Parse OBJ file
            section_counts = {section: 0 for section in self.required_obj_sections}
            materials = set()

            for line in obj_content:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                parts = line.split()
                if not parts:
                    continue

                section = parts[0]
                if section in section_counts:
                    section_counts[section] += 1

                if section == 'usemtl' and len(parts) > 1:
                    materials.add(parts[1])

            # Update results
            results['section_counts'] = section_counts
            results['vertex_count'] = section_counts['v']
            results['face_count'] = section_counts['f']
            results['material_count'] = len(materials)
            results['has_vertices'] = section_counts['v'] >= self.min_vertex_count
            results['has_faces'] = section_counts['f'] >= self.min_face_count
            results['has_materials'] = len(materials) > 0

            # Check for errors
            if not results['has_vertices']:
                results['errors'].append(f"Insufficient vertices (found {section_counts['v']}, expected >= {self.min_vertex_count})")
            if not results['has_faces']:
                results['errors'].append(f"Insufficient faces (found {section_counts['f']}, expected >= {self.min_face_count})")
            if not results['has_materials']:
                results['errors'].append("No materials found")

            is_valid = all([
                results['file_exists'],
                results['has_vertices'],
                results['has_faces'],
                results['has_materials']
            ])

            return is_valid, results

        except Exception as e:
            results['errors'].append(f"Error parsing OBJ file: {str(e)}")
            return False, results

    def validate_texture_references(self, obj_path: Path, texture_dir: Path) -> Tuple[bool, Dict[str, any]]:
        """
        Validate that all referenced textures exist in the texture directory
        
        Args:
            obj_path: Path to the OBJ file
            texture_dir: Path to the texture directory
            
        Returns:
            Tuple of (is_valid, validation_results)
        """
        results = {
            'textures_referenced': 0,
            'textures_found': 0,
            'missing_textures': [],
            'errors': []
        }

        if not obj_path.exists():
            results['errors'].append(f"OBJ file not found: {obj_path}")
            return False, results

        if not texture_dir.exists():
            results['errors'].append(f"Texture directory not found: {texture_dir}")
            return False, results

        try:
            with open(obj_path, 'r') as f:
                obj_content = f.readlines()

            referenced_textures = set()

            for line in obj_content:
                line = line.strip()
                if line.startswith('usemtl'):
                    parts = line.split()
                    if len(parts) > 1:
                        referenced_textures.add(parts[1])

            results['textures_referenced'] = len(referenced_textures)

            # Check for texture files
            for tex_name in referenced_textures:
                # Look for common texture extensions
                found = False
                for ext in ['.png', '.jpg', '.jpeg', '.tga', '.bmp']:
                    tex_path = texture_dir / f"{tex_name}{ext}"
                    if tex_path.exists():
                        found = True
                        results['textures_found'] += 1
                        break

                if not found:
                    results['missing_textures'].append(tex_name)

            is_valid = (
                results['textures_referenced'] > 0 and
                len(results['missing_textures']) == 0
            )

            return is_valid, results

        except Exception as e:
            results['errors'].append(f"Error validating textures: {str(e)}")
            return False, results

    def validate_map_integrity(self, obj_path: Path, reference_data: Dict) -> Tuple[bool, Dict[str, any]]:
        """
        Validate the converted map matches expected properties from reference data
        
        Args:
            obj_path: Path to the exported OBJ file
            reference_data: Dictionary containing reference data (vertex count, bounds, etc.)
            
        Returns:
            Tuple of (is_valid, validation_results)
        """
        results = {
            'bounds_match': False,
            'vertex_count_match': False,
            'material_match': False,
            'errors': []
        }

        # First validate basic OBJ structure
        obj_valid, obj_results = self.validate_exported_obj(obj_path)
        if not obj_valid:
            results['errors'].extend(obj_results['errors'])
            return False, results

        try:
            # Calculate bounds from OBJ file
            vertices = []
            for line in obj_path.read_text().splitlines():
                if line.startswith('v '):
                    parts = line.split()
                    if len(parts) >= 4:
                        vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])

            if not vertices:
                results['errors'].append("No vertices found in OBJ file")
                return False, results

            vertices = np.array(vertices)
            min_bounds = vertices.min(axis=0)
            max_bounds = vertices.max(axis=0)

            # Check bounds against reference
            bounds_match = (
                np.allclose(min_bounds, reference_data.get('min_bounds', []), atol=0.1) and
                np.allclose(max_bounds, reference_data.get('max_bounds', []), atol=0.1)
            )
            results['bounds_match'] = bounds_match

            # Check vertex count
            vertex_count_match = (
                abs(len(vertices) - reference_data.get('vertex_count', 0)) / 
                max(1, reference_data.get('vertex_count', 1)) < 0.1  # Within 10%
            )
            results['vertex_count_match'] = vertex_count_match

            # Check material count
            material_match = (
                obj_results['material_count'] >= reference_data.get('material_count', 0)
            )
            results['material_match'] = material_match

            is_valid = all([
                bounds_match,
                vertex_count_match,
                material_match
            ])

            return is_valid, results

        except Exception as e:
            results['errors'].append(f"Error validating map integrity: {str(e)}")
            return False, results

    def generate_file_checksum(self, file_path: Path) -> Optional[str]:
        """Generate MD5 checksum for a file"""
        try:
            with open(file_path, 'rb') as f:
                file_hash = hashlib.md5()
                while chunk := f.read(8192):
                    file_hash.update(chunk)
            return file_hash.hexdigest()
        except Exception as e:
            logger.error(f"Error generating checksum: {str(e)}")
            return None

    def validate_file_against_reference(self, file_path: Path, reference_checksum: str) -> bool:
        """
        Validate a file matches the reference checksum
        
        Args:
            file_path: Path to file to validate
            reference_checksum: Expected MD5 checksum
            
        Returns:
            True if checksums match
        """
        if not file_path.exists():
            return False

        current_checksum = self.generate_file_checksum(file_path)
        return current_checksum == reference_checksum
