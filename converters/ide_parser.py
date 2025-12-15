"""
GTA SA Map Converter - IDE File Parser
Parses GTA San Andreas .ide files for object definitions
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class IDEObject:
    """Represents an object definition from IDE file"""
    # Basic properties
    id: int
    model_name: str
    texture_dictionary: str
    flags: int
    draw_distance: float
    lods: List[str] = field(default_factory=list)
    
    # Extended properties (from IDE format)
    time_on: int = 0
    time_off: int = 24
    obj_count: int = 1
    is_tunnel: bool = False
    is_tunnel_transition: bool = False
    is_underwater: bool = False
    is_night_only: bool = False
    is_alpha: bool = False
    is_alpha2: bool = False
    
    # Additional data
    section_type: str = "objs"  # objs, tobj, hier, anim, etc.
    comment: str = ""
    file_path: str = ""
    
    def get_flags_info(self) -> Dict[str, bool]:
        """Extract flag information"""
        flags_dict = {}
        
        # Common flags from GTA SA
        if self.flags & 0x1:  # Is road
            flags_dict['is_road'] = True
        if self.flags & 0x4:  # Is underwater
            flags_dict['is_underwater'] = True
        if self.flags & 0x8:  # Is night only
            flags_dict['is_night_only'] = True
        if self.flags & 0x10:  # Alpha transparency 1
            flags_dict['is_alpha'] = True
        if self.flags & 0x20:  # Alpha transparency 2
            flags_dict['is_alpha2'] = True
        if self.flags & 0x40:  # Is interior
            flags_dict['is_interior'] = True
        if self.flags & 0x80:  # Is tunnel
            flags_dict['is_tunnel'] = True
        if self.flags & 0x100:  # Is tunnel transition
            flags_dict['is_tunnel_transition'] = True
        if self.flags & 0x200:  # Is wrecker can tow
            flags_dict['wrecker_can_tow'] = True
        if self.flags & 0x400:  # Is nightlight
            flags_dict['is_nightlight'] = True
        if self.flags & 0x800:  # Has high detail
            flags_dict['has_high_detail'] = True
        if self.flags & 0x1000:  # Is stair
            flags_dict['is_stair'] = True
        if self.flags & 0x2000:  # Is sky
            flags_dict['is_sky'] = True
        if self.flags & 0x4000:  # Is weapon object
            flags_dict['is_weapon'] = True
        if self.flags & 0x8000:  # Is vegetation
            flags_dict['is_vegetation'] = True
        if self.flags & 0x10000:  # Is glass
            flags_dict['is_glass'] = True
        if self.flags & 0x20000:  # Is garage door
            flags_dict['is_garage_door'] = True
        if self.flags & 0x40000:  # Is damageable
            flags_dict['is_damageable'] = True
        if self.flags & 0x80000:  # Is tree
            flags_dict['is_tree'] = True
        if self.flags & 0x100000:  # Is palm
            flags_dict['is_palm'] = True
        if self.flags & 0x200000:  # Is wreck
            flags_dict['is_wreck'] = True
        if self.flags & 0x400000:  # Is explosion
            flags_dict['is_explosion'] = True
        if self.flags & 0x800000:  # Is flammable
            flags_dict['is_flammable'] = True
        if self.flags & 0x1000000:  # Is shadow
            flags_dict['casts_shadow'] = True
        if self.flags & 0x2000000:  # Is shadow
            flags_dict['casts_shadow2'] = True
        if self.flags & 0x4000000:  # Is breakable glass
            flags_dict['is_breakable_glass'] = True
        if self.flags & 0x8000000:  # Is breakable glass sound
            flags_dict['breakable_glass_sound'] = True
        if self.flags & 0x10000000:  # Is breakable
            flags_dict['is_breakable'] = True
        if self.flags & 0x20000000:  # Is breakable
            flags_dict['is_breakable2'] = True
        
        return flags_dict
    
    def has_lods(self) -> bool:
        """Check if object has LOD models"""
        return len(self.lods) > 0
    
    def get_primary_model(self) -> str:
        """Get primary model name"""
        return self.model_name
    
    def get_lod_models(self) -> List[str]:
        """Get LOD model names"""
        return self.lods


@dataclass
class IDEAnimation:
    """Represents an animation object from IDE file"""
    id: int
    model_name: str
    texture_dictionary: str
    anim_count: int
    draw_distance: float
    flags: int
    
    # Animation specific
    anim_type: str = ""
    time_on: int = 0
    time_off: int = 24
    anim_file: str = ""
    
    section_type: str = "anim"


@dataclass 
class IDEHierarchical:
    """Represents a hierarchical object (skinned model) from IDE file"""
    id: int
    model_name: str
    texture_dictionary: str
    mesh_count: int
    draw_distance: float
    flags: int
    
    section_type: str = "hier"


@dataclass
class IDETimedObject:
    """Represents a timed object from IDE file"""
    id: int
    model_name: str
    texture_dictionary: str
    time_on: int
    time_off: int
    draw_distance: float
    flags: int
    
    section_type: str = "tobj"


@dataclass
class IDE2DFX:
    """Represents a 2D effect from IDE file"""
    id: int
    effect_type: int
    pos_x: float
    pos_y: float
    pos_z: float
    color_r: int = 255
    color_g: int = 255
    color_b: int = 255
    color_a: int = 255
    corona_size: float = 1.0
    inner_angle: float = 0.0
    outer_angle: float = 45.0
    
    section_type: str = "2dfx"


class IDEParser:
    """Parser for GTA San Andreas IDE files"""
    
    # IDE file section markers
    SECTION_MARKERS = {
        'objs': 'objs',
        'tobj': 'tobj',
        'anim': 'anim',
        'hier': 'hier',
        '2dfx': '2dfx',
        'cars': 'cars',
        'peds': 'peds',
        'path': 'path',
        'weap': 'weap',
        'end': 'end'
    }
    
    def __init__(self):
        self.objects: Dict[int, IDEObject] = {}
        self.animations: Dict[int, IDEAnimation] = {}
        self.timed_objects: Dict[int, IDETimedObject] = {}
        self.hierarchical: Dict[int, IDEHierarchical] = {}
        self.effects_2dfx: Dict[int, IDE2DFX] = {}
        self.section_order: List[str] = []
        self.current_section: Optional[str] = None
        self.comments: Dict[str, List[str]] = {}
        
    def parse_file(self, file_path: str) -> Dict[str, Any]:
        """
        Parse an IDE file and extract all object definitions
        
        Args:
            file_path: Path to the IDE file
            
        Returns:
            Dictionary containing all parsed data
        """
        if not os.path.exists(file_path):
            logger.error(f"IDE file not found: {file_path}")
            return {}
            
        logger.info(f"Parsing IDE file: {file_path}")
        
        # Reset state
        self.objects.clear()
        self.animations.clear()
        self.timed_objects.clear()
        self.hierarchical.clear()
        self.effects_2dfx.clear()
        self.section_order.clear()
        self.current_section = None
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                
            self._parse_lines(lines, file_path)
            
            logger.info(f"Parsed {len(self.objects)} objects from {file_path}")
            
            return {
                'objects': self.objects,
                'animations': self.animations,
                'timed_objects': self.timed_objects,
                'hierarchical': self.hierarchical,
                'effects': self.effects_2dfx,
                'section_order': self.section_order,
                'file_path': file_path
            }
            
        except Exception as e:
            logger.error(f"Error parsing IDE file {file_path}: {str(e)}")
            return {}
    
    def _parse_lines(self, lines: List[str], file_path: str):
        """Parse all lines from IDE file"""
        line_number = 0
        current_comments = []
        
        for line in lines:
            line_number += 1
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
                
            # Handle comments
            if line.startswith('#'):
                current_comments.append(line[1:].strip())
                continue
                
            # Check for section markers
            section_match = self._check_section_marker(line)
            if section_match:
                section_name = section_match.lower()
                self.current_section = section_name
                self.section_order.append(section_name)
                
                # Store any accumulated comments for this section
                if current_comments:
                    self.comments[section_name] = current_comments.copy()
                    current_comments.clear()
                    
                continue
                
            # Parse data line based on current section
            if self.current_section:
                self._parse_data_line(line, line_number, file_path, current_comments)
                
                # Clear comments after using them
                current_comments.clear()
    
    def _check_section_marker(self, line: str) -> Optional[str]:
        """Check if line is a section marker"""
        # Remove trailing colon if present
        clean_line = line.rstrip(':').lower()
        
        # Check if it's a known section
        for marker in self.SECTION_MARKERS.values():
            if clean_line == marker:
                return marker
                
        # Check for "end" marker
        if clean_line == "end":
            return "end"
            
        return None
    
    def _parse_data_line(self, line: str, line_number: int, file_path: str, comments: List[str]):
        """Parse a data line based on current section"""
        # Split by comma, but handle quoted strings
        parts = self._split_ide_line(line)
        
        if not parts:
            return
            
        try:
            if self.current_section == 'objs':
                self._parse_objs_line(parts, line_number, file_path, comments)
            elif self.current_section == 'tobj':
                self._parse_tobj_line(parts, line_number, file_path, comments)
            elif self.current_section == 'anim':
                self._parse_anim_line(parts, line_number, file_path, comments)
            elif self.current_section == 'hier':
                self._parse_hier_line(parts, line_number, file_path, comments)
            elif self.current_section == '2dfx':
                self._parse_2dfx_line(parts, line_number, file_path, comments)
            # Note: Other sections (cars, peds, weap, path) can be added as needed
                
        except (ValueError, IndexError) as e:
            logger.warning(f"Error parsing line {line_number} in {file_path}: {str(e)}")
            logger.warning(f"Line content: {line}")
    
    def _split_ide_line(self, line: str) -> List[str]:
        """Split IDE line by comma, handling whitespace"""
        parts = []
        current = ""
        in_quotes = False
        
        for char in line:
            if char == '"':
                in_quotes = not in_quotes
            elif char == ',' and not in_quotes:
                parts.append(current.strip())
                current = ""
            else:
                current += char
                
        # Add last part
        if current:
            parts.append(current.strip())
            
        return parts
    
    def _parse_objs_line(self, parts: List[str], line_number: int, file_path: str, comments: List[str]):
        """Parse an objs section line"""
        # Basic objs format: ID, ModelName, TexDict, DrawDist, Flags
        # Extended format may include more fields
        
        if len(parts) < 5:
            logger.warning(f"Invalid objs line (too few parts): {parts}")
            return
            
        try:
            obj_id = int(parts[0])
            model_name = parts[1].strip()
            texture_dict = parts[2].strip()
            
            # Parse draw distance
            draw_distance = float(parts[3])
            
            # Parse flags (hex or decimal)
            if parts[4].startswith('0x'):
                flags = int(parts[4], 16)
            else:
                flags = int(parts[4])
            
            # Create object
            obj = IDEObject(
                id=obj_id,
                model_name=model_name,
                texture_dictionary=texture_dict,
                draw_distance=draw_distance,
                flags=flags,
                section_type='objs',
                comment=' '.join(comments) if comments else '',
                file_path=file_path
            )
            
            # Parse additional fields if present
            if len(parts) >= 6:
                obj.time_on = self._parse_time(parts[5])
            if len(parts) >= 7:
                obj.time_off = self._parse_time(parts[6])
            if len(parts) >= 8:
                obj.obj_count = int(parts[7])
                
            # Parse LOD models (if any)
            lod_start = 8
            while lod_start < len(parts):
                lod_model = parts[lod_start].strip()
                if lod_model and lod_model.lower() != 'end':
                    obj.lods.append(lod_model)
                lod_start += 1
            
            # Store object
            self.objects[obj_id] = obj
            
        except ValueError as e:
            logger.warning(f"Error parsing objs line {line_number}: {str(e)}")
    
    def _parse_tobj_line(self, parts: List[str], line_number: int, file_path: str, comments: List[str]):
        """Parse a tobj (timed object) section line"""
        if len(parts) < 8:
            logger.warning(f"Invalid tobj line (too few parts): {parts}")
            return
            
        try:
            obj_id = int(parts[0])
            model_name = parts[1].strip()
            texture_dict = parts[2].strip()
            time_on = self._parse_time(parts[3])
            time_off = self._parse_time(parts[4])
            draw_distance = float(parts[5])
            
            # Parse flags
            if parts[6].startswith('0x'):
                flags = int(parts[6], 16)
            else:
                flags = int(parts[6])
            
            # Create timed object
            tobj = IDETimedObject(
                id=obj_id,
                model_name=model_name,
                texture_dictionary=texture_dict,
                time_on=time_on,
                time_off=time_off,
                draw_distance=draw_distance,
                flags=flags,
                section_type='tobj'
            )
            
            # Additional properties if present
            if len(parts) >= 8:
                tobj.obj_count = int(parts[7])
            
            self.timed_objects[obj_id] = tobj
            
        except ValueError as e:
            logger.warning(f"Error parsing tobj line {line_number}: {str(e)}")
    
    def _parse_anim_line(self, parts: List[str], line_number: int, file_path: str, comments: List[str]):
        """Parse an anim (animation) section line"""
        if len(parts) < 6:
            logger.warning(f"Invalid anim line (too few parts): {parts}")
            return
            
        try:
            obj_id = int(parts[0])
            model_name = parts[1].strip()
            texture_dict = parts[2].strip()
            anim_count = int(parts[3])
            draw_distance = float(parts[4])
            
            # Parse flags
            if parts[5].startswith('0x'):
                flags = int(parts[5], 16)
            else:
                flags = int(parts[5])
            
            anim = IDEAnimation(
                id=obj_id,
                model_name=model_name,
                texture_dictionary=texture_dict,
                anim_count=anim_count,
                draw_distance=draw_distance,
                flags=flags,
                section_type='anim'
            )
            
            # Parse additional fields
            if len(parts) >= 7:
                anim.time_on = self._parse_time(parts[6])
            if len(parts) >= 8:
                anim.time_off = self._parse_time(parts[7])
            if len(parts) >= 9:
                anim.anim_file = parts[8].strip()
            
            self.animations[obj_id] = anim
            
        except ValueError as e:
            logger.warning(f"Error parsing anim line {line_number}: {str(e)}")
    
    def _parse_hier_line(self, parts: List[str], line_number: int, file_path: str, comments: List[str]):
        """Parse a hier (hierarchical/skinned) section line"""
        if len(parts) < 6:
            logger.warning(f"Invalid hier line (too few parts): {parts}")
            return
            
        try:
            obj_id = int(parts[0])
            model_name = parts[1].strip()
            texture_dict = parts[2].strip()
            mesh_count = int(parts[3])
            draw_distance = float(parts[4])
            
            # Parse flags
            if parts[5].startswith('0x'):
                flags = int(parts[5], 16)
            else:
                flags = int(parts[5])
            
            hier = IDEHierarchical(
                id=obj_id,
                model_name=model_name,
                texture_dictionary=texture_dict,
                mesh_count=mesh_count,
                draw_distance=draw_distance,
                flags=flags,
                section_type='hier'
            )
            
            self.hierarchical[obj_id] = hier
            
        except ValueError as e:
            logger.warning(f"Error parsing hier line {line_number}: {str(e)}")
    
    def _parse_2dfx_line(self, parts: List[str], line_number: int, file_path: str, comments: List[str]):
        """Parse a 2dfx section line"""
        if len(parts) < 6:
            logger.warning(f"Invalid 2dfx line (too few parts): {parts}")
            return
            
        try:
            obj_id = int(parts[0])
            effect_type = int(parts[1])
            pos_x = float(parts[2])
            pos_y = float(parts[3])
            pos_z = float(parts[4])
            
            effect = IDE2DFX(
                id=obj_id,
                effect_type=effect_type,
                pos_x=pos_x,
                pos_y=pos_y,
                pos_z=pos_z,
                section_type='2dfx'
            )
            
            # Parse additional fields based on effect type
            if effect_type == 0:  # Light
                if len(parts) >= 10:
                    effect.color_r = int(parts[5])
                    effect.color_g = int(parts[6])
                    effect.color_b = int(parts[7])
                    effect.color_a = int(parts[8])
                    effect.corona_size = float(parts[9])
                if len(parts) >= 12:
                    effect.inner_angle = float(parts[10])
                    effect.outer_angle = float(parts[11])
            
            self.effects_2dfx[obj_id] = effect
            
        except ValueError as e:
            logger.warning(f"Error parsing 2dfx line {line_number}: {str(e)}")
    
    def _parse_time(self, time_str: str) -> int:
        """Parse time string (24h format) to integer hours"""
        if not time_str:
            return 0
            
        # Handle various time formats
        time_str = time_str.strip().lower()
        
        if time_str == 'midnight':
            return 0
        elif time_str == 'noon':
            return 12
            
        # Try to parse as integer
        try:
            return int(time_str)
        except ValueError:
            # Try to parse as "HH:MM" or similar
            if ':' in time_str:
                hours = time_str.split(':')[0]
                try:
                    return int(hours)
                except ValueError:
                    pass
                    
        return 0
    
    def get_object_by_id(self, obj_id: int) -> Optional[IDEObject]:
        """Get object by ID"""
        return self.objects.get(obj_id)
    
    def get_object_by_name(self, model_name: str) -> Optional[IDEObject]:
        """Get object by model name"""
        for obj in self.objects.values():
            if obj.model_name.lower() == model_name.lower():
                return obj
        return None
    
    def get_objects_with_flag(self, flag_mask: int) -> List[IDEObject]:
        """Get objects with specific flag(s) set"""
        result = []
        for obj in self.objects.values():
            if obj.flags & flag_mask:
                result.append(obj)
        return result
    
    def get_objects_by_type(self, obj_type: str) -> List[IDEObject]:
        """Get objects by type (road, building, vegetation, etc.)"""
        result = []
        
        for obj in self.objects.values():
            flags_info = obj.get_flags_info()
            
            if obj_type == 'road' and flags_info.get('is_road', False):
                result.append(obj)
            elif obj_type == 'building' and not flags_info.get('is_road', False):
                # Simple heuristic: not a road = likely building
                result.append(obj)
            elif obj_type == 'vegetation' and flags_info.get('is_vegetation', False):
                result.append(obj)
            elif obj_type == 'tree' and flags_info.get('is_tree', False):
                result.append(obj)
            elif obj_type == 'glass' and flags_info.get('is_glass', False):
                result.append(obj)
                
        return result
    
    def merge_with(self, other_parser: 'IDEParser'):
        """Merge another parser's data into this one"""
        # Merge objects (later definitions override earlier ones)
        self.objects.update(other_parser.objects)
        
        # Merge other collections
        self.animations.update(other_parser.animations)
        self.timed_objects.update(other_parser.timed_objects)
        self.hierarchical.update(other_parser.hierarchical)
        self.effects_2dfx.update(other_parser.effects_2dfx)
        
        # Merge section order
        self.section_order.extend(other_parser.section_order)
        
        # Merge comments
        for section, comments in other_parser.comments.items():
            if section in self.comments:
                self.comments[section].extend(comments)
            else:
                self.comments[section] = comments.copy()
    
    def get_stats(self) -> Dict[str, int]:
        """Get statistics about parsed data"""
        return {
            'total_objects': len(self.objects),
            'timed_objects': len(self.timed_objects),
            'animations': len(self.animations),
            'hierarchical': len(self.hierarchical),
            '2dfx_effects': len(self.effects_2dfx),
            'sections': len(self.section_order)
        }
    
    def export_to_dict(self) -> Dict[str, Any]:
        """Export parsed data to dictionary"""
        return {
            'objects': {str(k): vars(v) for k, v in self.objects.items()},
            'animations': {str(k): vars(v) for k, v in self.animations.items()},
            'timed_objects': {str(k): vars(v) for k, v in self.timed_objects.items()},
            'hierarchical': {str(k): vars(v) for k, v in self.hierarchical.items()},
            'effects': {str(k): vars(v) for k, v in self.effects_2dfx.items()},
            'section_order': self.section_order,
            'comments': self.comments
        }


class IDEManager:
    """Manager for handling multiple IDE files"""
    
    def __init__(self):
        self.parsers: Dict[str, IDEParser] = {}
        self.master_parser = IDEParser()
        self.file_paths: List[str] = []
        
    def parse_directory(self, directory_path: str, recursive: bool = True) -> bool:
        """
        Parse all IDE files in a directory
        
        Args:
            directory_path: Path to directory containing IDE files
            recursive: Whether to search subdirectories
            
        Returns:
            True if successful, False otherwise
        """
        if not os.path.exists(directory_path):
            logger.error(f"Directory not found: {directory_path}")
            return False
            
        try:
            ide_files = []
            
            if recursive:
                for root, dirs, files in os.walk(directory_path):
                    for file in files:
                        if file.lower().endswith('.ide'):
                            ide_files.append(os.path.join(root, file))
            else:
                for file in os.listdir(directory_path):
                    if file.lower().endswith('.ide'):
                        ide_files.append(os.path.join(directory_path, file))
            
            logger.info(f"Found {len(ide_files)} IDE files in {directory_path}")
            
            # Parse each file
            for ide_file in ide_files:
                parser = IDEParser()
                result = parser.parse_file(ide_file)
                
                if result:
                    self.parsers[ide_file] = parser
                    self.master_parser.merge_with(parser)
                    self.file_paths.append(ide_file)
                    
                    logger.debug(f"Successfully parsed: {ide_file}")
            
            logger.info(f"Total objects parsed: {len(self.master_parser.objects)}")
            return True
            
        except Exception as e:
            logger.error(f"Error parsing IDE directory {directory_path}: {str(e)}")
            return False
    
    def get_all_objects(self) -> Dict[int, IDEObject]:
        """Get all objects from all parsed IDE files"""
        return self.master_parser.objects.copy()
    
    def get_object_by_id(self, obj_id: int) -> Optional[IDEObject]:
        """Get object by ID from all parsed files"""
        return self.master_parser.get_object_by_id(obj_id)
    
    def get_object_by_name(self, model_name: str) -> Optional[IDEObject]:
        """Get object by model name from all parsed files"""
        return self.master_parser.get_object_by_name(model_name)
    
    def find_object_file(self, obj_id: int) -> Optional[str]:
        """Find which IDE file contains a specific object"""
        for file_path, parser in self.parsers.items():
            if obj_id in parser.objects:
                return file_path
        return None
    
    def get_ide_files_for_object(self, obj_id: int) -> List[str]:
        """Get all IDE files that contain an object (in case of duplicates)"""
        files = []
        for file_path, parser in self.parsers.items():
            if obj_id in parser.objects:
                files.append(file_path)
        return files
    
    def clear(self):
        """Clear all parsed data"""
        self.parsers.clear()
        self.master_parser = IDEParser()
        self.file_paths.clear()


# Convenience function
def parse_ide_file(file_path: str) -> Dict[str, Any]:
    """Parse a single IDE file"""
    parser = IDEParser()
    return parser.parse_file(file_path)


if __name__ == "__main__":
    # Test the IDE parser
    import sys
    
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        
        # Parse file
        parser = IDEParser()
        result = parser.parse_file(test_file)
        
        if result:
            print(f"Successfully parsed {test_file}")
            stats = parser.get_stats()
            print(f"Statistics: {stats}")
            
            # Print first few objects
            print("\nFirst 5 objects:")
            for i, (obj_id, obj) in enumerate(list(parser.objects.items())[:5]):
                print(f"  {obj_id}: {obj.model_name} (Tex: {obj.texture_dictionary})")
                
        else:
            print(f"Failed to parse {test_file}")
    else:
        print("Usage: python ide_parser.py <ide_file_path>")
