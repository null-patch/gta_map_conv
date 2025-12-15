"""
GTA SA Map Converter - DFF File Converter
Converts GTA San Andreas .dff models to usable geometry data
Note: DFF (RenderWare) format is complex and proprietary. This module provides:
1. A simplified DFF parser for basic geometry
2. Integration with external tools for full conversion
3. Fallback methods for different scenarios
"""

import os
import struct
import zlib
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, BinaryIO
from dataclasses import dataclass, field
import logging
import subprocess
import tempfile

logger = logging.getLogger(__name__)


@dataclass
class DFFGeometry:
    """Represents geometry data from DFF file"""
    # Vertex data
    vertices: List[List[float]] = field(default_factory=list)  # [[x, y, z], ...]
    normals: List[List[float]] = field(default_factory=list)   # [[nx, ny, nz], ...]
    colors: List[List[float]] = field(default_factory=list)    # [[r, g, b, a], ...]
    uvs: List[List[float]] = field(default_factory=list)       # [[u, v], ...]
    
    # Face data
    faces: List[List[int]] = field(default_factory=list)       # [[v1, v2, v3], ...]
    face_materials: List[int] = field(default_factory=list)    # Material index for each face
    face_normals: List[List[int]] = field(default_factory=list)  # Normal indices per face
    
    # Material data
    materials: List[Dict[str, Any]] = field(default_factory=list)
    texture_names: List[str] = field(default_factory=list)
    
    # Bounding box
    bounding_box: Tuple[Tuple[float, float, float], Tuple[float, float, float]] = (
        (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    )
    
    # Flags and properties
    has_normals: bool = False
    has_colors: bool = False
    has_uvs: bool = False
    is_triangulated: bool = True
    is_textured: bool = False
    vertex_count: int = 0
    face_count: int = 0
    
    def calculate_bounds(self):
        """Calculate bounding box from vertices"""
        if not self.vertices:
            return
            
        vertices_array = np.array(self.vertices)
        min_vals = vertices_array.min(axis=0)
        max_vals = vertices_array.max(axis=0)
        
        self.bounding_box = (
            (float(min_vals[0]), float(min_vals[1]), float(min_vals[2])),
            (float(max_vals[0]), float(max_vals[1]), float(max_vals[2]))
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get geometry statistics"""
        return {
            'vertex_count': len(self.vertices),
            'face_count': len(self.faces),
            'normal_count': len(self.normals),
            'uv_count': len(self.uvs),
            'material_count': len(self.materials),
            'has_normals': self.has_normals,
            'has_colors': self.has_colors,
            'has_uvs': self.has_uvs,
            'is_textured': self.is_textured,
            'bounds': self.bounding_box
        }


@dataclass
class DFFModel:
    """Complete DFF model with frame hierarchy and geometry"""
    name: str = ""
    geometries: List[DFFGeometry] = field(default_factory=list)
    frame_hierarchy: Dict[str, Any] = field(default_factory=dict)
    materials: List[Dict[str, Any]] = field(default_factory=list)
    textures: List[Dict[str, Any]] = field(default_factory=list)
    
    # Model properties
    version: int = 0
    flags: int = 0
    is_vehicle: bool = False
    is_pedestrian: bool = False
    is_weapon: bool = False
    has_alpha: bool = False
    has_collision: bool = False
    
    def merge_geometries(self) -> DFFGeometry:
        """Merge all geometries into one"""
        if not self.geometries:
            return DFFGeometry()
            
        if len(self.geometries) == 1:
            return self.geometries[0]
            
        # Merge multiple geometries
        merged = DFFGeometry()
        vertex_offset = 0
        
        for geometry in self.geometries:
            # Add vertices
            merged.vertices.extend(geometry.vertices)
            merged.normals.extend(geometry.normals)
            merged.colors.extend(geometry.colors)
            merged.uvs.extend(geometry.uvs)
            
            # Add faces with offset
            for face in geometry.faces:
                merged.faces.append([v + vertex_offset for v in face])
                
            # Add face materials
            merged.face_materials.extend(geometry.face_materials)
            
            # Update vertex offset
            vertex_offset += len(geometry.vertices)
            
            # Merge materials
            for material in geometry.materials:
                if material not in merged.materials:
                    merged.materials.append(material)
                    
            # Merge texture names
            for tex_name in geometry.texture_names:
                if tex_name not in merged.texture_names:
                    merged.texture_names.append(tex_name)
                    
        # Update flags
        merged.has_normals = any(g.has_normals for g in self.geometries)
        merged.has_colors = any(g.has_colors for g in self.geometries)
        merged.has_uvs = any(g.has_uvs for g in self.geometries)
        merged.is_textured = any(g.is_textured for g in self.geometries)
        merged.vertex_count = len(merged.vertices)
        merged.face_count = len(merged.faces)
        
        # Calculate new bounds
        merged.calculate_bounds()
        
        return merged


class DFFHeader:
    """DFF file header structure"""
    def __init__(self):
        self.magic = b''  # Should be 0x00 0x01 0x00 0x00
        self.version = 0
        self.chunk_count = 0


class DFFChunk:
    """DFF chunk structure"""
    def __init__(self, chunk_type: int = 0, chunk_size: int = 0, data: bytes = b''):
        self.type = chunk_type
        self.size = chunk_size
        self.data = data
        self.children: List['DFFChunk'] = []
        
    @classmethod
    def parse(cls, data: bytes, offset: int = 0) -> Tuple['DFFChunk', int]:
        """Parse a chunk from binary data"""
        # Read chunk header
        if offset + 12 > len(data):
            raise ValueError("Insufficient data for chunk header")
            
        chunk_type, chunk_size, library_id = struct.unpack('<III', data[offset:offset+12])
        
        # Create chunk
        chunk = cls(chunk_type, chunk_size)
        
        # Move to chunk data (skip header)
        offset += 12
        
        # Read chunk data
        chunk.data = data[offset:offset+chunk_size]
        
        return chunk, offset + chunk_size


# DFF Chunk Types (RenderWare)
CHUNK_TYPES = {
    0x01: 'STRUCT',          # 1
    0x0F: 'GEOMETRY',        # 15
    0x14: 'ATOMIC',          # 20
    0x15: 'TEXTURENATIVE',   # 21
    0x16: 'TEXTUREDICTIONARY',  # 22
    0x17: 'ANIMDATABASE',    # 23
    0x18: 'IMAGE',           # 24
    0x19: 'SKINANIMATION',   # 25
    0x1A: 'GEOMETRYLIST',    # 26
    0x1B: 'HANIMANIMATION',  # 27
    0x1C: 'TEAM',            # 28
    0x1D: 'CROWD',           # 29
    0x1E: 'DELTAMORPHANIMATION',  # 30
    0x1F: 'RIGHTTORENDER',   # 31
}


class DFFParser:
    """Parser for GTA DFF (RenderWare) files"""
    
    def __init__(self):
        self.model = DFFModel()
        self.current_offset = 0
        self.warnings: List[str] = []
        self.errors: List[str] = []
        
    def parse_file(self, file_path: str) -> Optional[DFFModel]:
        """
        Parse a DFF file
        
        Args:
            file_path: Path to DFF file
            
        Returns:
            DFFModel if successful, None otherwise
        """
        if not os.path.exists(file_path):
            logger.error(f"DFF file not found: {file_path}")
            return None
            
        logger.info(f"Parsing DFF file: {file_path}")
        
        try:
            with open(file_path, 'rb') as f:
                data = f.read()
                
            # Parse the file
            return self._parse_data(data, os.path.basename(file_path))
            
        except Exception as e:
            logger.error(f"Error parsing DFF file {file_path}: {str(e)}")
            self.errors.append(str(e))
            return None
    
    def _parse_data(self, data: bytes, filename: str) -> Optional[DFFModel]:
        """Parse DFF binary data"""
        self.model.name = Path(filename).stem
        
        try:
            # Check minimum size
            if len(data) < 12:
                raise ValueError("File too small to be a DFF file")
                
            # Parse chunks recursively
            self._parse_chunks(data, 0, len(data))
            
            if not self.model.geometries:
                self.warnings.append("No geometry found in DFF file")
                
            logger.info(f"Parsed DFF: {self.model.name} with {len(self.model.geometries)} geometries")
            return self.model
            
        except Exception as e:
            logger.error(f"Error parsing DFF data: {str(e)}")
            self.errors.append(str(e))
            return None
    
    def _parse_chunks(self, data: bytes, offset: int, end_offset: int):
        """Recursively parse chunks"""
        while offset < end_offset:
            try:
                chunk, offset = DFFChunk.parse(data, offset)
                self._process_chunk(chunk)
            except Exception as e:
                logger.warning(f"Error parsing chunk at offset {offset}: {e}")
                break
    
    def _process_chunk(self, chunk: DFFChunk):
        """Process a parsed chunk"""
        try:
            if chunk.type == 0x01:  # STRUCT
                self._parse_struct_chunk(chunk)
            elif chunk.type == 0x0F:  # GEOMETRY
                self._parse_geometry_chunk(chunk)
            elif chunk.type == 0x14:  # ATOMIC
                self._parse_atomic_chunk(chunk)
            elif chunk.type == 0x16:  # TEXTUREDICTIONARY
                self._parse_texture_dict_chunk(chunk)
            elif chunk.type == 0x1A:  # GEOMETRYLIST
                self._parse_geometry_list_chunk(chunk)
            else:
                # Skip unknown chunks
                pass
                
        except Exception as e:
            logger.warning(f"Error processing chunk type 0x{chunk.type:02X}: {e}")
    
    def _parse_struct_chunk(self, chunk: DFFChunk):
        """Parse STRUCT chunk (contains frame information)"""
        # STRUCT chunk contains frame hierarchy data
        # This is simplified - actual parsing is complex
        pass
    
    def _parse_geometry_chunk(self, chunk: DFFChunk):
        """Parse GEOMETRY chunk"""
        try:
            geometry = DFFGeometry()
            
            # Parse geometry data
            # Format is complex - this is a simplified version
            offset = 0
            data = chunk.data
            
            if len(data) < 64:
                self.warnings.append("Geometry chunk too small")
                return
                
            # Read vertex count and face count
            # Note: These offsets are approximations
            vertex_count = struct.unpack('<H', data[24:26])[0]
            face_count = struct.unpack('<H', data[26:28])[0]
            
            if vertex_count == 0 or face_count == 0:
                self.warnings.append("Empty geometry")
                return
                
            # Parse vertices (simplified - actual format varies)
            vertex_offset = 64  # Starting offset for vertices
            
            for i in range(vertex_count):
                if vertex_offset + 12 > len(data):
                    break
                    
                x, y, z = struct.unpack('<fff', data[vertex_offset:vertex_offset+12])
                geometry.vertices.append([x, y, z])
                vertex_offset += 12
                
            # Parse faces (triangles)
            # Faces are stored as triangle strips/lists
            # This is a simplified version
            face_offset = vertex_offset
            
            for i in range(face_count):
                if face_offset + 6 > len(data):
                    break
                    
                # Read 16-bit indices
                v1, v2, v3 = struct.unpack('<HHH', data[face_offset:face_offset+6])
                geometry.faces.append([v1, v2, v3])
                face_offset += 6
                
            # Parse normals (if present)
            # Check for normal data after faces
            if face_offset + (vertex_count * 12) <= len(data):
                geometry.has_normals = True
                for i in range(vertex_count):
                    nx, ny, nz = struct.unpack('<fff', data[face_offset:face_offset+12])
                    geometry.normals.append([nx, ny, nz])
                    face_offset += 12
                    
            # Parse UVs (if present)
            # UVs often come after normals
            if face_offset + (vertex_count * 8) <= len(data):
                geometry.has_uvs = True
                for i in range(vertex_count):
                    u, v = struct.unpack('<ff', data[face_offset:face_offset+8])
                    geometry.uvs.append([u, v])
                    face_offset += 8
                    
            # Calculate bounds and update counts
            geometry.calculate_bounds()
            geometry.vertex_count = len(geometry.vertices)
            geometry.face_count = len(geometry.faces)
            
            self.model.geometries.append(geometry)
            
        except Exception as e:
            logger.warning(f"Error parsing geometry: {e}")
            self.warnings.append(f"Geometry parsing error: {e}")
    
    def _parse_atomic_chunk(self, chunk: DFFChunk):
        """Parse ATOMIC chunk (links geometry to frames)"""
        # Atomic chunks link geometry to frame hierarchy
        # This is simplified
        pass
    
    def _parse_texture_dict_chunk(self, chunk: DFFChunk):
        """Parse TEXTUREDICTIONARY chunk"""
        try:
            data = chunk.data
            offset = 0
            
            # Parse texture count
            texture_count = struct.unpack('<H', data[offset:offset+2])[0]
            offset += 2
            
            for i in range(texture_count):
                # Parse texture name (null-terminated string)
                name_end = data.find(b'\x00', offset)
                if name_end == -1:
                    break
                    
                tex_name = data[offset:name_end].decode('ascii', errors='ignore')
                self.model.textures.append({'name': tex_name})
                offset = name_end + 1
                
        except Exception as e:
            logger.warning(f"Error parsing texture dictionary: {e}")
    
    def _parse_geometry_list_chunk(self, chunk: DFFChunk):
        """Parse GEOMETRYLIST chunk"""
        # Geometry list contains multiple geometries
        # This chunk would be parsed similarly to individual geometry chunks
        pass
    
    def get_warnings(self) -> List[str]:
        """Get parsing warnings"""
        return self.warnings
        
    def get_errors(self) -> List[str]:
        """Get parsing errors"""
        return self.errors


class DFFConverter:
    """Main DFF converter class with multiple conversion strategies"""
    
    def __init__(self, scale_factor: float = 0.01, use_external_tool: bool = True):
        self.scale_factor = scale_factor
        self.use_external_tool = use_external_tool
        self.external_tools = self._detect_external_tools()
        
    def _detect_external_tools(self) -> Dict[str, str]:
        """Detect available external DFF conversion tools"""
        tools = {}
        
        # Check for common DFF tools
        tool_paths = [
            '/usr/bin/rwanalyze',          # RWAnalyze
            '/usr/local/bin/rwanalyze',
            '/usr/bin/dff2obj',            # dff2obj
            '/usr/local/bin/dff2obj',
            '/opt/gta_tools/rwanalyze',
            Path.home() / '.local/bin/rwanalyze',
        ]
        
        for tool_path in tool_paths:
            if os.path.exists(tool_path):
                tools['rwanalyze'] = str(tool_path)
                break
                
        # Check for Python libraries
        try:
            import pydff
            tools['pydff'] = 'pydff'
        except ImportError:
            pass
            
        try:
            import gta_tools
            tools['gta_tools'] = 'gta_tools'
        except ImportError:
            pass
            
        return tools
    
    def convert(self, dff_path: str) -> Optional[Dict[str, Any]]:
        """
        Convert DFF file to geometry data
        
        Args:
            dff_path: Path to DFF file
            
        Returns:
            Dictionary with geometry data or None
        """
        logger.info(f"Converting DFF: {dff_path}")
        
        # Try different conversion methods
        methods = [
            self._convert_with_external_tool,
            self._convert_with_parser,
            self._convert_with_simplified_parser
        ]
        
        for method in methods:
            try:
                result = method(dff_path)
                if result:
                    # Apply scale factor
                    self._apply_scale(result)
                    return result
            except Exception as e:
                logger.warning(f"Conversion method {method.__name__} failed: {e}")
                continue
                
        logger.error(f"All conversion methods failed for {dff_path}")
        return None
    
    def _convert_with_external_tool(self, dff_path: str) -> Optional[Dict[str, Any]]:
        """Convert using external tool"""
        if not self.use_external_tool or not self.external_tools:
            return None
            
        try:
            if 'rwanalyze' in self.external_tools:
                return self._convert_with_rwanalyze(dff_path)
            elif 'pydff' in self.external_tools:
                return self._convert_with_pydff(dff_path)
                
        except Exception as e:
            logger.warning(f"External tool conversion failed: {e}")
            
        return None
    
    def _convert_with_rwanalyze(self, dff_path: str) -> Optional[Dict[str, Any]]:
        """Convert using RWAnalyze tool"""
        # Create temporary output file
        with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as tmp:
            output_path = tmp.name
            
        try:
            # Run RWAnalyze to convert DFF to OBJ
            cmd = [self.external_tools['rwanalyze'], '-o', output_path, dff_path]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30  # Timeout after 30 seconds
            )
            
            if result.returncode == 0:
                # Parse the OBJ file
                return self._parse_obj_file(output_path)
            else:
                logger.warning(f"RWAnalyze failed: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            logger.warning("RWAnalyze timed out")
        except Exception as e:
            logger.warning(f"RWAnalyze error: {e}")
        finally:
            # Clean up temporary file
            try:
                os.unlink(output_path)
            except:
                pass
                
        return None
    
    def _convert_with_pydff(self, dff_path: str) -> Optional[Dict[str, Any]]:
        """Convert using pydff Python library"""
        try:
            import pydff
            
            # Load DFF file
            dff = pydff.DFF(dff_path)
            
            # Extract geometry
            geometry = {
                'vertices': [],
                'faces': [],
                'normals': [],
                'uvs': [],
                'materials': [],
                'textures': []
            }
            
            # Process each geometry in the DFF
            for geom in dff.geometries:
                # Get vertices
                for vertex in geom.vertices:
                    geometry['vertices'].append([vertex.x, vertex.y, vertex.z])
                    
                # Get faces
                for face in geom.faces:
                    geometry['faces'].append([face.v1, face.v2, face.v3])
                    
                # Get normals
                if hasattr(geom, 'normals'):
                    for normal in geom.normals:
                        geometry['normals'].append([normal.x, normal.y, normal.z])
                        
                # Get UVs
                if hasattr(geom, 'tex_coords'):
                    for uv in geom.tex_coords:
                        geometry['uvs'].append([uv.u, uv.v])
                        
                # Get materials
                if hasattr(geom, 'materials'):
                    for material in geom.materials:
                        geometry['materials'].append({
                            'name': getattr(material, 'name', ''),
                            'color': getattr(material, 'color', [1, 1, 1, 1]),
                            'texture': getattr(material, 'texture', '')
                        })
                        
            return geometry
            
        except ImportError:
            logger.warning("pydff not installed")
        except Exception as e:
            logger.warning(f"pydff conversion error: {e}")
            
        return None
    
    def _convert_with_parser(self, dff_path: str) -> Optional[Dict[str, Any]]:
        """Convert using full DFF parser"""
        parser = DFFParser()
        model = parser.parse_file(dff_path)
        
        if not model:
            return None
            
        # Merge geometries
        geometry = model.merge_geometries()
        
        # Convert to dictionary
        result = {
            'vertices': geometry.vertices,
            'faces': geometry.faces,
            'normals': geometry.normals,
            'uvs': geometry.uvs,
            'colors': geometry.colors,
            'materials': geometry.materials,
            'texture_names': geometry.texture_names,
            'bounds': geometry.bounding_box,
            'stats': geometry.get_stats()
        }
        
        return result
    
    def _convert_with_simplified_parser(self, dff_path: str) -> Optional[Dict[str, Any]]:
        """Convert using simplified DFF parser (for basic DFFs)"""
        try:
            with open(dff_path, 'rb') as f:
                data = f.read()
                
            # Try to find geometry data using simple pattern matching
            # This is a fallback method that might work for some DFFs
            
            geometry = {
                'vertices': [],
                'faces': [],
                'normals': [],
                'uvs': [],
                'materials': [],
                'texture_names': []
            }
            
            # Look for vertex data patterns
            # Search for sequences that look like vertex positions
            offset = 0
            max_vertices = 10000  # Safety limit
            
            while offset < len(data) - 12 and len(geometry['vertices']) < max_vertices:
                try:
                    # Try to read 3 floats
                    x, y, z = struct.unpack('<fff', data[offset:offset+12])
                    
                    # Check if these look like valid vertex coordinates
                    # GTA coordinates are typically in reasonable ranges
                    if -10000.0 < x < 10000.0 and -10000.0 < y < 10000.0 and -10000.0 < z < 10000.0:
                        geometry['vertices'].append([x, y, z])
                        offset += 12
                    else:
                        offset += 4  # Move forward
                        
                except struct.error:
                    offset += 1
                    
            # Look for face indices (triangles)
            offset = 0
            max_faces = 10000
            
            while offset < len(data) - 6 and len(geometry['faces']) < max_faces:
                try:
                    # Try to read 3 unsigned shorts
                    v1, v2, v3 = struct.unpack('<HHH', data[offset:offset+6])
                    
                    # Check if indices are valid
                    max_index = len(geometry['vertices'])
                    if v1 < max_index and v2 < max_index and v3 < max_index:
                        geometry['faces'].append([v1, v2, v3])
                        offset += 6
                    else:
                        offset += 2
                        
                except struct.error:
                    offset += 1
            
            if geometry['vertices'] and geometry['faces']:
                logger.info(f"Extracted {len(geometry['vertices'])} vertices and {len(geometry['faces'])} faces")
                return geometry
                
        except Exception as e:
            logger.warning(f"Simplified parser failed: {e}")
            
        return None
    
    def _parse_obj_file(self, obj_path: str) -> Dict[str, Any]:
        """Parse OBJ file to extract geometry"""
        geometry = {
            'vertices': [],
            'faces': [],
            'normals': [],
            'uvs': [],
            'materials': [],
            'texture_names': []
        }
        
        try:
            with open(obj_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                        
                    parts = line.split()
                    if not parts:
                        continue
                        
                    if parts[0] == 'v':  # Vertex
                        if len(parts) >= 4:
                            geometry['vertices'].append([
                                float(parts[1]),
                                float(parts[2]),
                                float(parts[3])
                            ])
                    elif parts[0] == 'vn':  # Normal
                        if len(parts) >= 4:
                            geometry['normals'].append([
                                float(parts[1]),
                                float(parts[2]),
                                float(parts[3])
                            ])
                    elif parts[0] == 'vt':  # UV
                        if len(parts) >= 3:
                            geometry['uvs'].append([
                                float(parts[1]),
                                float(parts[2])
                            ])
                    elif parts[0] == 'f':  # Face
                        # Parse face indices (can be v/vt/vn or just v)
                        face_vertices = []
                        for part in parts[1:]:
                            if '/' in part:
                                # Handle v/vt/vn format
                                indices = part.split('/')
                                face_vertices.append(int(indices[0]) - 1)  # OBJ is 1-indexed
                            else:
                                face_vertices.append(int(part) - 1)
                                
                        # Triangulate if needed
                        if len(face_vertices) >= 3:
                            # Simple triangulation for convex polygons
                            for i in range(1, len(face_vertices) - 1):
                                geometry['faces'].append([
                                    face_vertices[0],
                                    face_vertices[i],
                                    face_vertices[i + 1]
                                ])
                    elif parts[0] == 'usemtl':  # Material
                        if len(parts) >= 2:
                            geometry['materials'].append({
                                'name': parts[1],
                                'color': [1, 1, 1, 1]
                            })
                            
        except Exception as e:
            logger.warning(f"Error parsing OBJ file: {e}")
            
        return geometry
    
    def _apply_scale(self, geometry: Dict[str, Any]):
        """Apply scale factor to geometry"""
        if not geometry or 'vertices' not in geometry:
            return
            
        # Scale vertices
        for i, vertex in enumerate(geometry['vertices']):
            geometry['vertices'][i] = [
                vertex[0] * self.scale_factor,
                vertex[1] * self.scale_factor,
                vertex[2] * self.scale_factor
            ]
            
        # Scale bounds if present
        if 'bounds' in geometry:
            min_bounds, max_bounds = geometry['bounds']
            geometry['bounds'] = (
                (
                    min_bounds[0] * self.scale_factor,
                    min_bounds[1] * self.scale_factor,
                    min_bounds[2] * self.scale_factor
                ),
                (
                    max_bounds[0] * self.scale_factor,
                    max_bounds[1] * self.scale_factor,
                    max_bounds[2] * self.scale_factor
                )
            )


class DFFManager:
    """Manager for handling multiple DFF files"""
    
    def __init__(self, scale_factor: float = 0.01):
        self.converter = DFFConverter(scale_factor)
        self.converted_models: Dict[str, Dict[str, Any]] = {}
        self.conversion_stats: Dict[str, Dict] = {}
        
    def convert_files(self, dff_files: List[str], max_workers: int = 4) -> Dict[str, Dict[str, Any]]:
        """
        Convert multiple DFF files
        
        Args:
            dff_files: List of DFF file paths
            max_workers: Maximum number of parallel workers
            
        Returns:
            Dictionary mapping model names to geometry data
        """
        import concurrent.futures
        
        self.converted_models.clear()
        
        logger.info(f"Converting {len(dff_files)} DFF files")
        
        # Use ThreadPoolExecutor for parallel conversion
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit conversion tasks
            future_to_file = {
                executor.submit(self._convert_single_file, dff_file): dff_file
                for dff_file in dff_files
            }
            
            # Process completed tasks
            for future in concurrent.futures.as_completed(future_to_file):
                dff_file = future_to_file[future]
                try:
                    result = future.result()
                    if result:
                        model_name, geometry = result
                        self.converted_models[model_name] = geometry
                        logger.debug(f"Converted: {model_name}")
                except Exception as e:
                    logger.error(f"Failed to convert {dff_file}: {e}")
                    
        logger.info(f"Successfully converted {len(self.converted_models)} out of {len(dff_files)} files")
        return self.converted_models.copy()
    
    def _convert_single_file(self, dff_path: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Convert a single DFF file"""
        try:
            geometry = self.converter.convert(dff_path)
            if geometry:
                model_name = Path(dff_path).stem
                return model_name, geometry
        except Exception as e:
            logger.error(f"Error converting {dff_path}: {e}")
            
        return None
    
    def get_model(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get converted model by name"""
        return self.converted_models.get(model_name)
    
    def list_models(self) -> List[str]:
        """List all converted model names"""
        return list(self.converted_models.keys())
    
    def get_model_stats(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a model"""
        model = self.get_model(model_name)
        if model and 'stats' in model:
            return model['stats']
        return None
    
    def clear(self):
        """Clear all converted models"""
        self.converted_models.clear()


# Convenience functions
def convert_dff_file(dff_path: str, scale_factor: float = 0.01) -> Optional[Dict[str, Any]]:
    """Convert a single DFF file"""
    converter = DFFConverter(scale_factor)
    return converter.convert(dff_path)


if __name__ == "__main__":
    # Test the DFF converter
    import sys
    
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        
        # Convert file
        converter = DFFConverter()
        result = converter.convert(test_file)
        
        if result:
            print(f"Successfully converted {test_file}")
            stats = result.get('stats', {})
            print(f"Statistics: {stats}")
            
            if 'vertices' in result:
                print(f"Vertices: {len(result['vertices'])}")
            if 'faces' in result:
                print(f"Faces: {len(result['faces'])}")
        else:
            print(f"Failed to convert {test_file}")
    else:
        print("Usage: python dff_converter.py <dff_file_path>")
        print("\nNote: DFF conversion requires either:")
        print("  1. External tools like RWAnalyze")
        print("  2. Python libraries like pydff")
        print("  3. Manual implementation of DFF parser")
