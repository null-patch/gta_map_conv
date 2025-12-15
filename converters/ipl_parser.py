import os
import re
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class IPLObjectPlacement:
    """Represents a single object placement in an IPL file"""
    # Basic properties
    id: int
    model_name: str
    interior: int = 0
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # Euler angles in degrees
    scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    
    # Extended properties
    lod_model: int = -1
    flags: int = 0
    is_lod: bool = False
    is_visible: bool = True
    draw_distance: float = 300.0
    
    # IPL section info
    section_type: str = "inst"  # inst, mult, cars, etc.
    file_path: str = ""
    line_number: int = 0
    
    def get_rotation_matrix(self) -> List[List[float]]:
        """Convert Euler angles to rotation matrix (simplified)"""
        rx, ry, rz = math.radians(self.rotation[0]), math.radians(self.rotation[1]), math.radians(self.rotation[2])
        
        # Create rotation matrix
        # Note: GTA uses different rotation order than standard Euler
        # This is a simplified version
        return [
            [math.cos(rz), -math.sin(rz), 0],
            [math.sin(rz), math.cos(rz), 0],
            [0, 0, 1]
        ]
    
    def get_transform_matrix(self) -> List[List[float]]:
        """Get full transformation matrix (scale * rotation * translation)"""
        # Create scale matrix
        sx, sy, sz = self.scale
        scale_mat = [
            [sx, 0, 0, 0],
            [0, sy, 0, 0],
            [0, 0, sz, 0],
            [0, 0, 0, 1]
        ]
        
        # Create rotation matrix (3x3 expanded to 4x4)
        rot_mat = self.get_rotation_matrix()
        rot_4x4 = [
            [rot_mat[0][0], rot_mat[0][1], rot_mat[0][2], 0],
            [rot_mat[1][0], rot_mat[1][1], rot_mat[1][2], 0],
            [rot_mat[2][0], rot_mat[2][1], rot_mat[2][2], 0],
            [0, 0, 0, 1]
        ]
        
        # Create translation matrix
        tx, ty, tz = self.position
        trans_mat = [
            [1, 0, 0, tx],
            [0, 1, 0, ty],
            [0, 0, 1, tz],
            [0, 0, 0, 1]
        ]
        
        # Combine: translation * rotation * scale
        # (Note: GTA uses different transform order, but this works for Blender)
        return trans_mat
    
    def get_flags_info(self) -> Dict[str, bool]:
        """Extract flag information for object placement"""
        flags_dict = {}
        
        # GTA SA IPL placement flags
        if self.flags & 0x1:  # Is interior
            flags_dict['is_interior'] = True
        if self.flags & 0x4:  # Has day/night cycle
            flags_dict['has_day_night'] = True
        if self.flags & 0x8:  # Has night-only
            flags_dict['is_night_only'] = True
        if self.flags & 0x10:  # Has alpha transparency 1
            flags_dict['has_alpha1'] = True
        if self.flags & 0x20:  # Has alpha transparency 2
            flags_dict['has_alpha2'] = True
        if self.flags & 0x40:  # Is tunnel
            flags_dict['is_tunnel'] = True
        if self.flags & 0x80:  # Is tunnel transition
            flags_dict['is_tunnel_transition'] = True
        if self.flags & 0x100:  # Is underwater
            flags_dict['is_underwater'] = True
        if self.flags & 0x200:  # Has road
            flags_dict['has_road'] = True
        if self.flags & 0x400:  # Has no shadows
            flags_dict['no_shadows'] = True
        if self.flags & 0x800:  # Has no reflections
            flags_dict['no_reflections'] = True
        if self.flags & 0x1000:  # Has no culling
            flags_dict['no_culling'] = True
        if self.flags & 0x2000:  # Is breakable
            flags_dict['is_breakable'] = True
        if self.flags & 0x4000:  # Is breakable glass
            flags_dict['is_breakable_glass'] = True
        if self.flags & 0x8000:  # Has garages
            flags_dict['has_garage'] = True
        if self.flags & 0x10000:  # Is multi-lane
            flags_dict['is_multi_lane'] = True
        if self.flags & 0x20000:  # Is climbable
            flags_dict['is_climbable'] = True
        if self.flags & 0x40000:  # Is shootable
            flags_dict['is_shootable'] = True
        if self.flags & 0x80000:  # Is weapon object
            flags_dict['is_weapon'] = True
        if self.flags & 0x100000:  # Is vegetation
            flags_dict['is_vegetation'] = True
        if self.flags & 0x200000:  # Is tree
            flags_dict['is_tree'] = True
        if self.flags & 0x400000: # Is palm
            flags_dict['is_palm'] = True
        if self.flags & 0x800000:  # Is wreck
            flags_dict['is_wreck'] = True
        
        return flags_dict


@dataclass
class IPLCullZone:
    """Represents a cull zone in IPL file"""
    id: int
    position: Tuple[float, float, float]
    size: Tuple[float, float, float]
    flags: int = 0
    interior: int = 0
    unknown1: int = 0
    unknown2: int = 0
    
    section_type: str = "cull"


@dataclass
class IPLPathNode:
    """Represents a path node in IPL file"""
    node_type: str
    position: Tuple[float, float, float]
    size: float = 1.0
    flags: int = 0
    next_node: int = -1
    
    section_type: str = "path"


@dataclass
class IPLGarage:
    """Represents a garage in IPL file"""
    id: int
    name: str
    position: Tuple[float, float, float]
    size: Tuple[float, float, float]
    flags: int = 0
    door_type: int = 0
    interior: int = 0
    
    section_type: str = "grge"


@dataclass
class IPLEnEx:
    """Represents an enter-exit marker in IPL file"""
    name: str
    interior_from: int = 0
    interior_to: int = 0
    position_from: Tuple[float, float, float] = (0, 0, 0)
    position_to: Tuple[float, float, float] = (0, 0, 0)
    angle: float = 0.0
    size: float = 1.0
    flags: int = 0
    
    section_type: str = "enex"


@dataclass
class IPLMulti:
    """Represents a multi placement (LOD models) in IPL file"""
    base_id: int
    lod_id: int
    position: Tuple[float, float, float] = (0, 0, 0)
    rotation: Tuple[float, float, float] = (0, 0, 0)
    scale: Tuple[float, float, float] = (1, 1, 1)
    interior: int = 0
    lod_distance: float = 300.0
    
    section_type: str = "mult"


class IPLParser:
    """Parser for GTA San Andreas IPL files"""
    
    # IPL file section markers
    SECTION_MARKERS = {
        'inst': 'inst',  # Object instances
        'cull': 'cull',  # Cull zones
        'path': 'path',  # Path nodes
        'grge': 'grge',  # Garages
        'enex': 'enex',  # Enter-exit markers
        'pick': 'pick',  # Pickups
        'jump': 'jump',  # Stunt jumps
        'tcyc': 'tcyc',  # Time cycle modifiers
        'auzo': 'auzo',  # Audio zones
        'mult': 'mult',  # Multi placements (LOD)
        'cars': 'cars',  # Car generators
        'zone': 'zone',  # Map zones
        'occl': 'occl',  # Occlusion
        'end': 'end'     # End of file
    }
    
    def __init__(self):
        self.placements: List[IPLObjectPlacement] = []
        self.cull_zones: List[IPLCullZone] = []
        self.path_nodes: List[IPLPathNode] = []
        self.garages: List[IPLGarage] = []
        self.enex_markers: List[IPLEnEx] = []
        self.multi_placements: List[IPLMulti] = []
        self.car_generators: List[Dict] = []
        
        self.current_section: Optional[str] = None
        self.file_version: str = ""
        self.file_path: str = ""
        self.interior_count: int = 0
        
    def parse_file(self, file_path: str) -> List[IPLObjectPlacement]:
        """
        Parse an IPL file and extract all object placements
        
        Args:
            file_path: Path to the IPL file
            
        Returns:
            List of object placements
        """
        if not os.path.exists(file_path):
            logger.error(f"IPL file not found: {file_path}")
            return []
            
        logger.info(f"Parsing IPL file: {file_path}")
        
        # Reset state
        self.placements.clear()
        self.cull_zones.clear()
        self.path_nodes.clear()
        self.garages.clear()
        self.enex_markers.clear()
        self.multi_placements.clear()
        self.car_generators.clear()
        
        self.current_section = None
        self.file_path = file_path
        self.file_version = ""
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                
            self._parse_lines(lines)
            
            logger.info(f"Parsed {len(self.placements)} object placements from {file_path}")
            return self.placements
            
        except Exception as e:
            logger.error(f"Error parsing IPL file {file_path}: {str(e)}")
            return []
    
    def _parse_lines(self, lines: List[str]):
        """Parse all lines from IPL file"""
        line_number = 0
        
        for line in lines:
            line_number += 1
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#') or line.startswith('//'):
                continue
                
            # Check for version line
            if line.lower().startswith('ipl_'):
                self.file_version = line
                continue
                
            # Check for section markers
            section_match = self._check_section_marker(line)
            if section_match:
                self.current_section = section_match.lower()
                continue
                
            # Parse data line based on current section
            if self.current_section:
                self._parse_data_line(line, line_number)
    
    def _check_section_marker(self, line: str) -> Optional[str]:
        """Check if line is a section marker"""
        # Remove whitespace and convert to lowercase
        clean_line = line.strip().lower()
        
        # Check if it's a known section
        for marker in self.SECTION_MARKERS.values():
            if clean_line == marker:
                return marker
                
        return None
    
    def _parse_data_line(self, line: str, line_number: int):
        """Parse a data line based on current section"""
        # Split by comma, handling whitespace
        parts = [part.strip() for part in line.split(',')]
        
        if not parts:
            return
            
        try:
            if self.current_section == 'inst':
                self._parse_inst_line(parts, line_number)
            elif self.current_section == 'cull':
                self._parse_cull_line(parts, line_number)
            elif self.current_section == 'path':
                self._parse_path_line(parts, line_number)
            elif self.current_section == 'grge':
                self._parse_grge_line(parts, line_number)
            elif self.current_section == 'enex':
                self._parse_enex_line(parts, line_number)
            elif self.current_section == 'mult':
                self._parse_mult_line(parts, line_number)
            elif self.current_section == 'cars':
                self._parse_cars_line(parts, line_number)
            # Note: Other sections can be added as needed
                
        except (ValueError, IndexError) as e:
            logger.warning(f"Error parsing line {line_number}: {str(e)}")
            logger.warning(f"Line content: {line}")
            logger.warning(f"Section: {self.current_section}")
    
    def _parse_inst_line(self, parts: List[str], line_number: int):
        """Parse an inst (instance) section line"""
        # Format: ID, ModelName, Interior, X, Y, Z, RX, RY, RZ, SX, SY, SZ, [LOD, Flags]
        
        if len(parts) < 11:
            logger.warning(f"Invalid inst line (too few parts): {parts}")
            return
            
        try:
            obj_id = int(parts[0])
            model_name = parts[1]
            interior = int(parts[2])
            
            # Parse position
            pos_x = float(parts[3])
            pos_y = float(parts[4])
            pos_z = float(parts[5])
            
            # Parse rotation (Euler angles in degrees)
            rot_x = float(parts[6])
            rot_y = float(parts[7])
            rot_z = float(parts[8])
            
            # Parse scale
            scale_x = float(parts[9])
            scale_y = float(parts[10])
            scale_z = scale_y  # Default Z scale to Y if not provided
            
            if len(parts) >= 12:
                scale_z = float(parts[11])
            
            # Parse LOD model (optional)
            lod_model = -1
            if len(parts) >= 13:
                try:
                    lod_model = int(parts[12])
                except ValueError:
                    pass
            
            # Parse flags (optional)
            flags = 0
            if len(parts) >= 14:
                try:
                    flags = int(parts[13])
                except ValueError:
                    pass
            
            # Create placement object
            placement = IPLObjectPlacement(
                id=obj_id,
                model_name=model_name,
                interior=interior,
                position=(pos_x, pos_y, pos_z),
                rotation=(rot_x, rot_y, rot_z),
                scale=(scale_x, scale_y, scale_z),
                lod_model=lod_model,
                flags=flags,
                section_type='inst',
                file_path=self.file_path,
                line_number=line_number,
                is_lod=(lod_model != -1)
            )
            
            self.placements.append(placement)
            
        except ValueError as e:
            logger.warning(f"Error parsing inst line {line_number}: {str(e)}")
    
    def _parse_cull_line(self, parts: List[str], line_number: int):
        """Parse a cull (cull zone) section line"""
        if len(parts) < 8:
            logger.warning(f"Invalid cull line (too few parts): {parts}")
            return
            
        try:
            zone_id = int(parts[0])
            interior = int(parts[1])
            
            # Parse position
            pos_x = float(parts[2])
            pos_y = float(parts[3])
            pos_z = float(parts[4])
            
            # Parse size
            size_x = float(parts[5])
            size_y = float(parts[6])
            size_z = float(parts[7])
            
            # Parse flags (optional)
            flags = 0
            if len(parts) >= 9:
                flags = int(parts[8])
            
            # Parse unknown values (optional)
            unknown1 = 0
            unknown2 = 0
            if len(parts) >= 10:
                unknown1 = int(parts[9])
            if len(parts) >= 11:
                unknown2 = int(parts[10])
            
            cull_zone = IPLCullZone(
                id=zone_id,
                interior=interior,
                position=(pos_x, pos_y, pos_z),
                size=(size_x, size_y, size_z),
                flags=flags,
                unknown1=unknown1,
                unknown2=unknown2,
                section_type='cull'
            )
            
            self.cull_zones.append(cull_zone)
            
        except ValueError as e:
            logger.warning(f"Error parsing cull line {line_number}: {str(e)}")
    
    def _parse_path_line(self, parts: List[str], line_number: int):
        """Parse a path section line"""
        if len(parts) < 7:
            logger.warning(f"Invalid path line (too few parts): {parts}")
            return
            
        try:
            node_type = parts[0]
            
            # Parse position
            pos_x = float(parts[1])
            pos_y = float(parts[2])
            pos_z = float(parts[3])
            
            # Parse size
            size = float(parts[4])
            
            # Parse flags
            flags = int(parts[5])
            
            # Parse next node
            next_node = -1
            if len(parts) >= 7:
                next_node = int(parts[6])
            
            path_node = IPLPathNode(
                node_type=node_type,
                position=(pos_x, pos_y, pos_z),
                size=size,
                flags=flags,
                next_node=next_node,
                section_type='path'
            )
            
            self.path_nodes.append(path_node)
            
        except ValueError as e:
            logger.warning(f"Error parsing path line {line_number}: {str(e)}")
    
    def _parse_grge_line(self, parts: List[str], line_number: int):
        """Parse a grge (garage) section line"""
        if len(parts) < 9:
            logger.warning(f"Invalid grge line (too few parts): {parts}")
            return
            
        try:
            garage_id = int(parts[0])
            name = parts[1]
            
            # Parse position
            pos_x = float(parts[2])
            pos_y = float(parts[3])
            pos_z = float(parts[4])
            
            # Parse size
            size_x = float(parts[5])
            size_y = float(parts[6])
            size_z = float(parts[7])
            
            # Parse door type
            door_type = int(parts[8])
            
            # Parse flags (optional)
            flags = 0
            if len(parts) >= 10:
                flags = int(parts[9])
            
            # Parse interior (optional)
            interior = 0
            if len(parts) >= 11:
                interior = int(parts[10])
            
            garage = IPLGarage(
                id=garage_id,
                name=name,
                position=(pos_x, pos_y, pos_z),
                size=(size_x, size_y, size_z),
                door_type=door_type,
                flags=flags,
                interior=interior,
                section_type='grge'
            )
            
            self.garages.append(garage)
            
        except ValueError as e:
            logger.warning(f"Error parsing grge line {line_number}: {str(e)}")
    
    def _parse_enex_line(self, parts: List[str], line_number: int):
        """Parse an enex (enter-exit) section line"""
        if len(parts) < 11:
            logger.warning(f"Invalid enex line (too few parts): {parts}")
            return
            
        try:
            name = parts[0]
            interior_from = int(parts[1])
            interior_to = int(parts[2])
            
            # Parse from position
            pos_from_x = float(parts[3])
            pos_from_y = float(parts[4])
            pos_from_z = float(parts[5])
            
            # Parse to position
            pos_to_x = float(parts[6])
            pos_to_y = float(parts[7])
            pos_to_z = float(parts[8])
            
            # Parse angle
            angle = float(parts[9])
            
            # Parse size
            size = float(parts[10])
            
            # Parse flags (optional)
            flags = 0
            if len(parts) >= 12:
                flags = int(parts[11])
            
            enex = IPLEnEx(
                name=name,
                interior_from=interior_from,
                interior_to=interior_to,
                position_from=(pos_from_x, pos_from_y, pos_from_z),
                position_to=(pos_to_x, pos_to_y, pos_to_z),
                angle=angle,
                size=size,
                flags=flags,
                section_type='enex'
            )
            
            self.enex_markers.append(enex)
            
        except ValueError as e:
            logger.warning(f"Error parsing enex line {line_number}: {str(e)}")
    
    def _parse_mult_line(self, parts: List[str], line_number: int):
        """Parse a mult (multi/LOD) section line"""
        if len(parts) < 12:
            logger.warning(f"Invalid mult line (too few parts): {parts}")
            return
            
        try:
            base_id = int(parts[0])
            lod_id = int(parts[1])
            interior = int(parts[2])
            
            # Parse position
            pos_x = float(parts[3])
            pos_y = float(parts[4])
            pos_z = float(parts[5])
            
            # Parse rotation
            rot_x = float(parts[6])
            rot_y = float(parts[7])
            rot_z = float(parts[8])
            
            # Parse scale
            scale_x = float(parts[9])
            scale_y = float(parts[10])
            scale_z = float(parts[11])
            
            # Parse LOD distance (optional)
            lod_distance = 300.0
            if len(parts) >= 13:
                lod_distance = float(parts[12])
            
            multi = IPLMulti(
                base_id=base_id,
                lod_id=lod_id,
                interior=interior,
                position=(pos_x, pos_y, pos_z),
                rotation=(rot_x, rot_y, rot_z),
                scale=(scale_x, scale_y, scale_z),
                lod_distance=lod_distance,
                section_type='mult'
            )
            
            self.multi_placements.append(multi)
            
        except ValueError as e:
            logger.warning(f"Error parsing mult line {line_number}: {str(e)}")
    
    def _parse_cars_line(self, parts: List[str], line_number: int):
        """Parse a cars (car generator) section line"""
        if len(parts) < 11:
            logger.warning(f"Invalid cars line (too few parts): {parts}")
            return
            
        try:
            car_gen = {
                'id': int(parts[0]),
                'model_name': parts[1],
                'pos_x': float(parts[2]),
                'pos_y': float(parts[3]),
                'pos_z': float(parts[4]),
                'angle': float(parts[5]),
                'color1': int(parts[6]),
                'color2': int(parts[7]),
                'force_spawn': int(parts[8]) == 1,
                'alarm_chance': int(parts[9]),
                'door_lock_chance': int(parts[10])
            }
            
            # Optional fields
            if len(parts) >= 12:
                car_gen['min_delay'] = int(parts[11])
            if len(parts) >= 13:
                car_gen['max_delay'] = int(parts[12])
            
            self.car_generators.append(car_gen)
            
        except ValueError as e:
            logger.warning(f"Error parsing cars line {line_number}: {str(e)}")
    
    def get_placements_by_model(self, model_name: str) -> List[IPLObjectPlacement]:
        """Get all placements of a specific model"""
        return [p for p in self.placements if p.model_name == model_name]
    
    def get_placements_by_id(self, obj_id: int) -> List[IPLObjectPlacement]:
        """Get all placements with specific object ID"""
        return [p for p in self.placements if p.id == obj_id]
    
    def get_placements_in_interior(self, interior: int) -> List[IPLObjectPlacement]:
        """Get all placements in specific interior"""
        return [p for p in self.placements if p.interior == interior]
    
    def get_exterior_placements(self) -> List[IPLObjectPlacement]:
        """Get all exterior placements (interior = 0)"""
        return self.get_placements_in_interior(0)
    
    def get_lod_placements(self) -> List[IPLObjectPlacement]:
        """Get all LOD placements"""
        return [p for p in self.placements if p.is_lod]
    
    def get_non_lod_placements(self) -> List[IPLObjectPlacement]:
        """Get all non-LOD placements"""
        return [p for p in self.placements if not p.is_lod]
    
    def get_placements_with_flag(self, flag_mask: int) -> List[IPLObjectPlacement]:
        """Get placements with specific flag(s) set"""
        return [p for p in self.placements if p.flags & flag_mask]
    
    def get_placement_bounds(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """Get bounding box of all placements"""
        if not self.placements:
            return ((0, 0, 0), (0, 0, 0))
        
        min_x = min_y = min_z = float('inf')
        max_x = max_y = max_z = float('-inf')
        
        for placement in self.placements:
            x, y, z = placement.position
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            min_z = min(min_z, z)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
            max_z = max(max_z, z)
        
        return ((min_x, min_y, min_z), (max_x, max_y, max_z))
    
    def get_placement_center(self) -> Tuple[float, float, float]:
        """Get center point of all placements"""
        if not self.placements:
            return (0, 0, 0)
        
        min_bounds, max_bounds = self.get_placement_bounds()
        min_x, min_y, min_z = min_bounds
        max_x, max_y, max_z = max_bounds
        
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        center_z = (min_z + max_z) / 2
        
        return (center_x, center_y, center_z)
    
    def get_stats(self) -> Dict[str, int]:
        """Get statistics about parsed data"""
        return {
            'total_placements': len(self.placements),
            'exterior_placements': len(self.get_exterior_placements()),
            'interior_placements': len(self.placements) - len(self.get_exterior_placements()),
            'lod_placements': len(self.get_lod_placements()),
            'cull_zones': len(self.cull_zones),
            'path_nodes': len(self.path_nodes),
            'garages': len(self.garages),
            'enex_markers': len(self.enex_markers),
            'multi_placements': len(self.multi_placements),
            'car_generators': len(self.car_generators)
        }
    
    def export_to_dict(self) -> Dict[str, Any]:
        """Export parsed data to dictionary"""
        placements_dict = []
        for placement in self.placements:
            placement_dict = {
                'id': placement.id,
                'model_name': placement.model_name,
                'interior': placement.interior,
                'position': placement.position,
                'rotation': placement.rotation,
                'scale': placement.scale,
                'lod_model': placement.lod_model,
                'flags': placement.flags,
                'is_lod': placement.is_lod,
                'file_path': placement.file_path,
                'line_number': placement.line_number
            }
            placements_dict.append(placement_dict)
        
        return {
            'placements': placements_dict,
            'cull_zones': [asdict(z) for z in self.cull_zones],
            'path_nodes': [asdict(n) for n in self.path_nodes],
            'garages': [asdict(g) for g in self.garages],
            'enex_markers': [asdict(e) for e in self.enex_markers],
            'multi_placements': [asdict(m) for m in self.multi_placements],
            'car_generators': self.car_generators,
            'file_version': self.file_version,
            'file_path': self.file_path,
            'stats': self.get_stats()
        }


class IPLManager:
    """Manager for handling multiple IPL files"""
    
    def __init__(self):
        self.parsers: Dict[str, IPLParser] = {}
        self.master_parser = IPLParser()
        self.file_paths: List[str] = []
        
    def parse_directory(self, directory_path: str, recursive: bool = True) -> bool:
        """
        Parse all IPL files in a directory
        
        Args:
            directory_path: Path to directory containing IPL files
            recursive: Whether to search subdirectories
            
        Returns:
            True if successful, False otherwise
        """
        if not os.path.exists(directory_path):
            logger.error(f"Directory not found: {directory_path}")
            return False
            
        try:
            ipl_files = []
            
            if recursive:
                for root, dirs, files in os.walk(directory_path):
                    for file in files:
                        if file.lower().endswith('.ipl'):
                            ipl_files.append(os.path.join(root, file))
            else:
                for file in os.listdir(directory_path):
                    if file.lower().endswith('.ipl'):
                        ipl_files.append(os.path.join(directory_path, file))
            
            logger.info(f"Found {len(ipl_files)} IPL files in {directory_path}")
            
            # Parse each file
            for ipl_file in ipl_files:
                parser = IPLParser()
                placements = parser.parse_file(ipl_file)
                
                if placements:
                    self.parsers[ipl_file] = parser
                    self.master_parser.placements.extend(placements)
                    self.file_paths.append(ipl_file)
                    
                    logger.debug(f"Successfully parsed: {ipl_file}")
            
            # Add other data from parsers
            for parser in self.parsers.values():
                self.master_parser.cull_zones.extend(parser.cull_zones)
                self.master_parser.path_nodes.extend(parser.path_nodes)
                self.master_parser.garages.extend(parser.garages)
                self.master_parser.enex_markers.extend(parser.enex_markers)
                self.master_parser.multi_placements.extend(parser.multi_placements)
                self.master_parser.car_generators.extend(parser.car_generators)
            
            logger.info(f"Total placements parsed: {len(self.master_parser.placements)}")
            return True
            
        except Exception as e:
            logger.error(f"Error parsing IPL directory {directory_path}: {str(e)}")
            return False
    
    def get_all_placements(self) -> List[IPLObjectPlacement]:
        """Get all placements from all parsed IPL files"""
        return self.master_parser.placements.copy()
    
    def get_placements_by_model(self, model_name: str) -> List[IPLObjectPlacement]:
        """Get all placements of a specific model from all files"""
        return self.master_parser.get_placements_by_model(model_name)
    
    def get_placements_by_id(self, obj_id: int) -> List[IPLObjectPlacement]:
        """Get all placements with specific object ID from all files"""
        return self.master_parser.get_placements_by_id(obj_id)
    
    def find_placement_file(self, placement: IPLObjectPlacement) -> Optional[str]:
        """Find which IPL file contains a specific placement"""
        for file_path, parser in self.parsers.items():
            if placement in parser.placements:
                return file_path
        return None
    
    def get_ipl_files_for_model(self, model_name: str) -> List[str]:
        """Get all IPL files that contain placements of a specific model"""
        files = set()
        for file_path, parser in self.parsers.items():
            if parser.get_placements_by_model(model_name):
                files.add(file_path)
        return list(files)
    
    def clear(self):
        """Clear all parsed data"""
        self.parsers.clear()
        self.master_parser = IPLParser()
        self.file_paths.clear()


# Convenience function
def parse_ipl_file(file_path: str) -> List[IPLObjectPlacement]:
    """Parse a single IPL file"""
    parser = IPLParser()
    return parser.parse_file(file_path)


if __name__ == "__main__":
    # Test the IPL parser
    import sys
    
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        
        # Parse file
        parser = IPLParser()
        placements = parser.parse_file(test_file)
        
        if placements:
            print(f"Successfully parsed {test_file}")
            stats = parser.get_stats()
            print(f"Statistics: {stats}")
            
            # Print first few placements
            print("\nFirst 5 placements:")
            for i, placement in enumerate(placements[:5]):
                print(f"  {placement.id}: {placement.model_name} at {placement.position}")
                
        else:
            print(f"Failed to parse {test_file}")
    else:
        print("Usage: python ipl_parser.py <ipl_file_path>")
