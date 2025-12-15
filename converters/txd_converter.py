import os
import struct
import zlib
import io
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, BinaryIO
from dataclasses import dataclass, field
import logging
import subprocess
import tempfile
from PIL import Image

# Try to import image processing libraries
try:
    
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("PIL/Pillow not available. Texture conversion limited.")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class TXDTexture:
    """Represents a single texture from TXD file"""
    name: str = ""
    width: int = 0
    height: int = 0
    depth: int = 32  # Bits per pixel
    format: str = ""  # D3DFMT enum
    has_alpha: bool = False
    mipmap_count: int = 1
    data: bytes = b''
    palette: Optional[bytes] = None
    
    # Image data after conversion
    image_data: Optional[Any] = None  # PIL Image or numpy array
    file_path: str = ""
    
    def get_size_bytes(self) -> int:
        """Get size of texture data in bytes"""
        return len(self.data)
    
    def get_dimensions(self) -> Tuple[int, int]:
        """Get texture dimensions"""
        return (self.width, self.height)
    
    def is_valid(self) -> bool:
        """Check if texture is valid"""
        return bool(self.name and self.width > 0 and self.height > 0 and self.data)


@dataclass
class TXDArchive:
    """Represents a complete TXD texture archive"""
    version: int = 0
    texture_count: int = 0
    textures: List[TXDTexture] = field(default_factory=list)
    file_path: str = ""
    
    def get_texture(self, name: str) -> Optional[TXDTexture]:
        """Get texture by name"""
        for texture in self.textures:
            if texture.name.lower() == name.lower():
                return texture
        return None
    
    def get_texture_names(self) -> List[str]:
        """Get list of all texture names"""
        return [tex.name for tex in self.textures]
    
    def get_total_size(self) -> int:
        """Get total size of all textures in bytes"""
        return sum(tex.get_size_bytes() for tex in self.textures)


# TXD Format Constants
TXD_VERSION_GTA3 = 9
TXD_VERSION_GTAVC = 10
TXD_VERSION_GTASA = 13

# D3DFMT Constants (Direct3D Format)
D3DFMT_ENUM = {
    0: 'D3DFMT_UNKNOWN',
    20: 'D3DFMT_R8G8B8',
    21: 'D3DFMT_A8R8G8B8',
    22: 'D3DFMT_X8R8G8B8',
    23: 'D3DFMT_R5G6B5',
    24: 'D3DFMT_X1R5G5B5',
    25: 'D3DFMT_A1R5G5B5',
    26: 'D3DFMT_A4R4G4B4',
    27: 'D3DFMT_R3G3B2',
    28: 'D3DFMT_A8',
    29: 'D3DFMT_A8R3G3B2',
    30: 'D3DFMT_X4R4G4B4',
    31: 'D3DFMT_A2B10G10R10',
    32: 'D3DFMT_A8B8G8R8',
    33: 'D3DFMT_X8B8G8R8',
    34: 'D3DFMT_G16R16',
    35: 'D3DFMT_A2R10G10B10',
    36: 'D3DFMT_A16B16G16R16',
    40: 'D3DFMT_A8P8',
    41: 'D3DFMT_P8',
    50: 'D3DFMT_L8',
    51: 'D3DFMT_A8L8',
    52: 'D3DFMT_A4L4',
    60: 'D3DFMT_V8U8',
    61: 'D3DFMT_L6V5U5',
    62: 'D3DFMT_X8L8V8U8',
    63: 'D3DFMT_Q8W8V8U8',
    64: 'D3DFMT_V16U16',
    65: 'D3DFMT_A2W10V10U10',
    70: 'D3DFMT_UYVY',
    71: 'D3DFMT_R8G8_B8G8',
    72: 'D3DFMT_YUY2',
    73: 'D3DFMT_G8R8_G8B8',
    80: 'D3DFMT_DXT1',
    81: 'D3DFMT_DXT2',
    82: 'D3DFMT_DXT3',
    83: 'D3DFMT_DXT4',
    84: 'D3DFMT_DXT5',
}


class TXDParser:
    """Parser for GTA TXD (Texture Dictionary) files"""
    
    def __init__(self):
        self.archive = TXDArchive()
        self.warnings: List[str] = []
        self.errors: List[str] = []
        
    def parse_file(self, file_path: str) -> Optional[TXDArchive]:
        """
        Parse a TXD file
        
        Args:
            file_path: Path to TXD file
            
        Returns:
            TXDArchive if successful, None otherwise
        """
        if not os.path.exists(file_path):
            logger.error(f"TXD file not found: {file_path}")
            return None
            
        logger.info(f"Parsing TXD file: {file_path}")
        
        try:
            with open(file_path, 'rb') as f:
                data = f.read()
                
            # Parse the file
            return self._parse_data(data, file_path)
            
        except Exception as e:
            logger.error(f"Error parsing TXD file {file_path}: {str(e)}")
            self.errors.append(str(e))
            return None
    
    def _parse_data(self, data: bytes, file_path: str) -> Optional[TXDArchive]:
        """Parse TXD binary data"""
        self.archive.file_path = file_path
        
        try:
            offset = 0
            
            # Read header
            if len(data) < 24:
                raise ValueError("File too small to be a TXD file")
                
            # Check TXD signature
            signature = data[offset:offset+4]
            if signature != b'TXD\x00':
                raise ValueError("Not a valid TXD file (missing TXD signature)")
            offset += 4
            
            # Read version
            version = struct.unpack('<I', data[offset:offset+4])[0]
            self.archive.version = version
            offset += 4
            
            # Read texture count
            texture_count = struct.unpack('<H', data[offset:offset+2])[0]
            self.archive.texture_count = texture_count
            offset += 2
            
            # Skip unknown bytes
            offset += 2  # Usually 0x1803
            
            logger.debug(f"TXD Version: {version}, Textures: {texture_count}")
            
            # Parse each texture
            for i in range(texture_count):
                texture = self._parse_texture(data, offset)
                if texture:
                    self.archive.textures.append(texture)
                else:
                    self.warnings.append(f"Failed to parse texture {i}")
                    
            logger.info(f"Parsed {len(self.archive.textures)} textures from TXD")
            return self.archive
            
        except Exception as e:
            logger.error(f"Error parsing TXD data: {str(e)}")
            self.errors.append(str(e))
            return None
    
    def _parse_texture(self, data: bytes, offset: int) -> Optional[TXDTexture]:
        """Parse a single texture from TXD"""
        try:
            texture = TXDTexture()
            
            # Read texture header
            # Format varies by TXD version, this is simplified
            
            # Read texture name (32 bytes, null-terminated)
            name_bytes = data[offset:offset+32]
            texture.name = name_bytes.split(b'\x00')[0].decode('ascii', errors='ignore').strip()
            offset += 32
            
            # Read texture info
            # This is a simplified parser - actual format is complex
            
            # Try to find texture data using pattern matching
            # Look for D3DFMT values
            for fmt_code, fmt_name in D3DFMT_ENUM.items():
                fmt_offset = data.find(struct.pack('<I', fmt_code), offset)
                if fmt_offset != -1:
                    texture.format = fmt_name
                    break
            
            # Estimate dimensions (simplified)
            # Real parsing would need to decode the texture structure
            
            return texture
            
        except Exception as e:
            logger.warning(f"Error parsing texture: {e}")
            return None
    
    def get_warnings(self) -> List[str]:
        """Get parsing warnings"""
        return self.warnings
        
    def get_errors(self) -> List[str]:
        """Get parsing errors"""
        return self.errors


class TXDConverter:
    """Main TXD converter class with multiple conversion strategies"""
    
    def __init__(self, use_external_tool: bool = True):
        self.use_external_tool = use_external_tool
        self.external_tools = self._detect_external_tools()
        
    def _detect_external_tools(self) -> Dict[str, str]:
        """Detect available external TXD conversion tools"""
        tools = {}
        
        # Check for common TXD tools
        tool_paths = [
            '/usr/bin/txdworkshop',          # TXD Workshop
            '/usr/local/bin/txdworkshop',
            '/usr/bin/txd2png',              # txd2png
            '/usr/local/bin/txd2png',
            '/opt/gta_tools/txdworkshop',
            Path.home() / '.local/bin/txdworkshop',
            '/usr/bin/magick',               # ImageMagick (for conversion)
            '/usr/local/bin/magick',
            '/usr/bin/convert',              # ImageMagick convert
            '/usr/local/bin/convert',
        ]
        
        for tool_path in tool_paths:
            if os.path.exists(tool_path):
                tool_name = Path(tool_path).name
                if 'txd' in tool_name.lower():
                    tools['txd_tool'] = str(tool_path)
                elif 'magick' in tool_name or 'convert' in tool_name:
                    tools['imagemagick'] = str(tool_path)
                break
                
        # Check for Python libraries
        try:
            import pytxd
            tools['pytxd'] = 'pytxd'
        except ImportError:
            pass
            
        try:
            import gta_tools
            tools['gta_tools'] = 'gta_tools'
        except ImportError:
            pass
            
        return tools
    
    def convert(self, txd_path: str, output_dir: str, 
                output_format: str = 'png', 
                size_limit: Optional[Tuple[int, int]] = None,
                quality: int = 90) -> Dict[str, Dict[str, Any]]:
        """
        Convert TXD file to image files
        
        Args:
            txd_path: Path to TXD file
            output_dir: Directory to save converted images
            output_format: Output image format (png, jpg, tga, bmp)
            size_limit: Optional maximum size (width, height)
            quality: JPEG quality (1-100)
            
        Returns:
            Dictionary mapping texture names to texture info
        """
        logger.info(f"Converting TXD: {txd_path}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Try different conversion methods
        methods = [
            self._convert_with_external_tool,
            self._convert_with_pytxd,
            self._convert_with_fallback,
        ]
        
        textures = {}
        
        for method in methods:
            try:
                result = method(txd_path, output_dir, output_format, size_limit, quality)
                if result:
                    textures.update(result)
                    if textures:
                        logger.info(f"Converted {len(textures)} textures using {method.__name__}")
                        return textures
            except Exception as e:
                logger.warning(f"Conversion method {method.__name__} failed: {e}")
                continue
                
        logger.error(f"All conversion methods failed for {txd_path}")
        return textures
    
    def _convert_with_external_tool(self, txd_path: str, output_dir: str,
                                   output_format: str, size_limit: Optional[Tuple[int, int]],
                                   quality: int) -> Dict[str, Dict[str, Any]]:
        """Convert using external tool"""
        if not self.use_external_tool or not self.external_tools:
            return {}
            
        try:
            if 'txd_tool' in self.external_tools:
                return self._convert_with_txdworkshop(txd_path, output_dir, output_format)
            elif 'imagemagick' in self.external_tools:
                # First extract with any method, then convert with ImageMagick
                textures = self._convert_with_fallback(txd_path, output_dir, 'png', size_limit, quality)
                return self._convert_with_imagemagick(textures, output_dir, output_format, quality)
                
        except Exception as e:
            logger.warning(f"External tool conversion failed: {e}")
            
        return {}
    
    def _convert_with_txdworkshop(self, txd_path: str, output_dir: str,
                                 output_format: str) -> Dict[str, Dict[str, Any]]:
        """Convert using TXD Workshop tool"""
        textures = {}
        
        try:
            # Create temporary directory for extraction
            with tempfile.TemporaryDirectory() as temp_dir:
                # Run TXD Workshop command line
                # Note: Actual command depends on TXD Workshop version
                cmd = [
                    self.external_tools['txd_tool'],
                    '-extract',
                    txd_path,
                    temp_dir
                ]
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.returncode == 0:
                    # Process extracted files
                    for file_path in Path(temp_dir).glob('*'):
                        if file_path.is_file():
                            texture_name = file_path.stem
                            output_path = self._save_texture(file_path, output_dir, output_format)
                            
                            if output_path:
                                textures[texture_name] = {
                                    'path': output_path,
                                    'width': 0,  # Would need to read image
                                    'height': 0,
                                    'format': output_format
                                }
                else:
                    logger.warning(f"TXD Workshop failed: {result.stderr}")
                    
        except subprocess.TimeoutExpired:
            logger.warning("TXD Workshop timed out")
        except Exception as e:
            logger.warning(f"TXD Workshop error: {e}")
            
        return textures
    
    def _convert_with_pytxd(self, txd_path: str, output_dir: str,
                           output_format: str, size_limit: Optional[Tuple[int, int]],
                           quality: int) -> Dict[str, Dict[str, Any]]:
        """Convert using pytxd Python library"""
        if 'pytxd' not in self.external_tools:
            return {}
            
        try:
            import pytxd
            
            textures = {}
            
            # Load TXD file
            txd = pytxd.TXD(txd_path)
            
            # Process each texture
            for texture in txd.textures:
                try:
                    # Get texture name
                    texture_name = texture.name
                    
                    # Get image data
                    if hasattr(texture, 'get_image'):
                        img = texture.get_image()
                    elif hasattr(texture, 'image_data'):
                        # Convert raw data to image
                        img = self._raw_data_to_image(
                            texture.image_data,
                            texture.width,
                            texture.height,
                            texture.format
                        )
                    else:
                        continue
                    
                    if img:
                        # Resize if needed
                        if size_limit:
                            img = self._resize_image(img, size_limit)
                        
                        # Save image
                        output_path = os.path.join(output_dir, f"{texture_name}.{output_format}")
                        self._save_pil_image(img, output_path, output_format, quality)
                        
                        textures[texture_name] = {
                            'path': output_path,
                            'width': img.width,
                            'height': img.height,
                            'format': output_format
                        }
                        
                except Exception as e:
                    logger.warning(f"Error processing texture {texture.name}: {e}")
                    
            return textures
            
        except ImportError:
            logger.warning("pytxd not installed")
        except Exception as e:
            logger.warning(f"pytxd conversion error: {e}")
            
        return {}
    
    def _convert_with_fallback(self, txd_path: str, output_dir: str,
                              output_format: str, size_limit: Optional[Tuple[int, int]],
                              quality: int) -> Dict[str, Dict[str, Any]]:
        """Convert using fallback method (file copying/renaming)"""
        textures = {}
        
        try:
            # Simple fallback: just copy/rename the TXD file
            # This is a placeholder - real implementation would need actual TXD parsing
            
            txd_name = Path(txd_path).stem
            output_path = os.path.join(output_dir, f"{txd_name}.{output_format}")
            
            # Create a placeholder image
            if PIL_AVAILABLE:
                # Create a simple placeholder texture
                img = Image.new('RGB', (256, 256), color='gray')
                self._save_pil_image(img, output_path, output_format, quality)
                
                textures[txd_name] = {
                    'path': output_path,
                    'width': 256,
                    'height': 256,
                    'format': output_format,
                    'is_placeholder': True
                }
                
                logger.warning(f"Created placeholder texture for {txd_name}")
                
        except Exception as e:
            logger.warning(f"Fallback conversion failed: {e}")
            
        return textures
    
    def _convert_with_imagemagick(self, textures: Dict[str, Dict[str, Any]],
                                 output_dir: str, output_format: str,
                                 quality: int) -> Dict[str, Dict[str, Any]]:
        """Convert images using ImageMagick"""
        if 'imagemagick' not in self.external_tools:
            return textures
            
        converted_textures = {}
        
        try:
            for tex_name, tex_info in textures.items():
                input_path = tex_info['path']
                output_path = os.path.join(output_dir, f"{tex_name}.{output_format}")
                
                # Convert using ImageMagick
                cmd = [
                    self.external_tools['imagemagick'],
                    input_path,
                    '-quality', str(quality),
                    output_path
                ]
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                if result.returncode == 0 and os.path.exists(output_path):
                    converted_textures[tex_name] = {
                        'path': output_path,
                        'width': tex_info.get('width', 0),
                        'height': tex_info.get('height', 0),
                        'format': output_format
                    }
                    
                    # Remove original if different format
                    if not input_path.endswith(f'.{output_format}'):
                        try:
                            os.remove(input_path)
                        except:
                            pass
                            
        except Exception as e:
            logger.warning(f"ImageMagick conversion error: {e}")
            
        return converted_textures or textures
    
    def _raw_data_to_image(self, data: bytes, width: int, height: int,
                          format_str: str) -> Optional[Any]:
        """Convert raw texture data to PIL Image"""
        if not PIL_AVAILABLE:
            return None
            
        try:
            # Handle different texture formats
            if 'DXT1' in format_str:
                # DXT1 compressed format
                return self._decode_dxt1(data, width, height)
            elif 'DXT3' in format_str or 'DXT5' in format_str:
                # DXT3/DXT5 compressed format
                return self._decode_dxt3_5(data, width, height, format_str)
            elif 'A8R8G8B8' in format_str:
                # 32-bit ARGB
                return self._decode_argb8888(data, width, height)
            elif 'R8G8B8' in format_str:
                # 24-bit RGB
                return self._decode_rgb888(data, width, height)
            elif 'R5G6B5' in format_str:
                # 16-bit RGB
                return self._decode_rgb565(data, width, height)
            else:
                # Try to load as raw image
                try:
                    return Image.frombytes('RGB', (width, height), data)
                except:
                    return None
                    
        except Exception as e:
            logger.warning(f"Error decoding texture format {format_str}: {e}")
            return None
    
    def _decode_argb8888(self, data: bytes, width: int, height: int) -> Optional[Image.Image]:
        """Decode 32-bit ARGB format"""
        if not PIL_AVAILABLE:
            return None
            
        try:
            # Convert ARGB to RGBA
            img = Image.frombytes('RGBA', (width, height), data)
            # Reorder channels: ARGB -> RGBA
            b, g, r, a = img.split()
            return Image.merge('RGBA', (r, g, b, a))
        except:
            return None
    
    def _decode_rgb888(self, data: bytes, width: int, height: int) -> Optional[Image.Image]:
        """Decode 24-bit RGB format"""
        if not PIL_AVAILABLE:
            return None
            
        try:
            return Image.frombytes('RGB', (width, height), data)
        except:
            return None
    
    def _decode_rgb565(self, data: bytes, width: int, height: int) -> Optional[Image.Image]:
        """Decode 16-bit RGB565 format"""
        if not PIL_AVAILABLE or not NUMPY_AVAILABLE:
            return None
            
        try:
            # Convert 16-bit to 24-bit RGB
            arr = np.frombuffer(data, dtype=np.uint16).reshape(height, width)
            
            # Extract RGB components
            r = ((arr >> 11) & 0x1F) << 3
            g = ((arr >> 5) & 0x3F) << 2
            b = (arr & 0x1F) << 3
            
            # Stack channels
            rgb = np.stack([r, g, b], axis=2).astype(np.uint8)
            
            return Image.fromarray(rgb, 'RGB')
        except:
            return None
    
    def _decode_dxt1(self, data: bytes, width: int, height: int) -> Optional[Image.Image]:
        """Decode DXT1 compressed format (simplified)"""
        # Note: DXT1 decompression is complex
        # This is a placeholder implementation
        logger.warning("DXT1 decompression not fully implemented")
        return None
    
    def _decode_dxt3_5(self, data: bytes, width: int, height: int,
                      format_str: str) -> Optional[Image.Image]:
        """Decode DXT3/DXT5 compressed format (simplified)"""
        # Note: DXT3/DXT5 decompression is complex
        logger.warning(f"{format_str} decompression not implemented")
        return None
    
    def _save_texture(self, input_path: Path, output_dir: str,
                     output_format: str) -> Optional[str]:
        """Save texture to output directory"""
        try:
            texture_name = input_path.stem
            output_path = os.path.join(output_dir, f"{texture_name}.{output_format}")
            
            # Read and convert image if PIL is available
            if PIL_AVAILABLE:
                try:
                    img = Image.open(input_path)
                    self._save_pil_image(img, output_path, output_format, 90)
                    return output_path
                except:
                    # If PIL can't read it, try to copy the file
                    pass
            
            # Fallback: copy the file
            import shutil
            shutil.copy2(input_path, output_path)
            return output_path
            
        except Exception as e:
            logger.warning(f"Error saving texture: {e}")
            return None
    
    def _save_pil_image(self, img: Image.Image, output_path: str,
                       output_format: str, quality: int):
        """Save PIL Image to file"""
        if not PIL_AVAILABLE:
            return
            
        try:
            # Convert format if needed
            if output_format.lower() == 'jpg' or output_format.lower() == 'jpeg':
                if img.mode in ('RGBA', 'LA'):
                    # JPEG doesn't support alpha, convert to RGB
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                    img = background
                elif img.mode != 'RGB':
                    img = img.convert('RGB')
                    
                img.save(output_path, 'JPEG', quality=quality, optimize=True)
                
            elif output_format.lower() == 'png':
                img.save(output_path, 'PNG', optimize=True)
                
            elif output_format.lower() == 'tga':
                img.save(output_path, 'TGA')
                
            elif output_format.lower() == 'bmp':
                img.save(output_path, 'BMP')
                
            else:
                # Default to PNG
                img.save(output_path, 'PNG', optimize=True)
                
        except Exception as e:
            logger.warning(f"Error saving image {output_path}: {e}")
    
    def _resize_image(self, img: Image.Image, size_limit: Tuple[int, int]) -> Image.Image:
        """Resize image to fit within size limits"""
        if not PIL_AVAILABLE:
            return img
            
        width, height = img.size
        max_width, max_height = size_limit
        
        if width <= max_width and height <= max_height:
            return img
            
        # Calculate new dimensions maintaining aspect ratio
        ratio = min(max_width / width, max_height / height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        
        return img.resize((new_width, new_height), Image.Resampling.LANCZOS)


class TXDManager:
    """Manager for handling multiple TXD files"""
    
    def __init__(self):
        self.converter = TXDConverter()
        self.converted_textures: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self.conversion_stats: Dict[str, Dict] = {}
        
    def convert_files(self, txd_files: List[str], output_dir: str,
                     output_format: str = 'png',
                     size_limit: Optional[Tuple[int, int]] = None,
                     quality: int = 90,
                     max_workers: int = 4) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Convert multiple TXD files
        
        Args:
            txd_files: List of TXD file paths
            output_dir: Directory to save converted images
            output_format: Output image format
            size_limit: Optional maximum size
            quality: JPEG quality
            max_workers: Maximum number of parallel workers
            
        Returns:
            Dictionary mapping TXD names to texture dictionaries
        """
        import concurrent.futures
        
        self.converted_textures.clear()
        
        logger.info(f"Converting {len(txd_files)} TXD files")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Use ThreadPoolExecutor for parallel conversion
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit conversion tasks
            future_to_file = {
                executor.submit(self._convert_single_file, txd_file, output_dir,
                               output_format, size_limit, quality): txd_file
                for txd_file in txd_files
            }
            
            # Process completed tasks
            for future in concurrent.futures.as_completed(future_to_file):
                txd_file = future_to_file[future]
                try:
                    result = future.result()
                    if result:
                        txd_name = Path(txd_file).stem
                        self.converted_textures[txd_name] = result
                        logger.debug(f"Converted: {txd_name} ({len(result)} textures)")
                except Exception as e:
                    logger.error(f"Failed to convert {txd_file}: {e}")
                    
        logger.info(f"Successfully converted {len(self.converted_textures)} out of {len(txd_files)} files")
        return self.converted_textures.copy()
    
    def _convert_single_file(self, txd_path: str, output_dir: str,
                            output_format: str, size_limit: Optional[Tuple[int, int]],
                            quality: int) -> Dict[str, Dict[str, Any]]:
        """Convert a single TXD file"""
        try:
            # Create subdirectory for this TXD
            txd_name = Path(txd_path).stem
            txd_output_dir = os.path.join(output_dir, txd_name)
            os.makedirs(txd_output_dir, exist_ok=True)
            
            textures = self.converter.convert(
                txd_path, txd_output_dir, output_format, size_limit, quality
            )
            
            return textures
            
        except Exception as e:
            logger.error(f"Error converting {txd_path}: {e}")
            return {}
    
    def get_textures(self, txd_name: str) -> Optional[Dict[str, Dict[str, Any]]]:
        """Get converted textures by TXD name"""
        return self.converted_textures.get(txd_name)
    
    def get_texture(self, txd_name: str, texture_name: str) -> Optional[Dict[str, Any]]:
        """Get specific texture by TXD and texture name"""
        txd_textures = self.get_textures(txd_name)
        if txd_textures:
            return txd_textures.get(texture_name)
        return None
    
    def list_txd_files(self) -> List[str]:
        """List all converted TXD names"""
        return list(self.converted_textures.keys())
    
    def get_total_texture_count(self) -> int:
        """Get total number of converted textures"""
        return sum(len(textures) for textures in self.converted_textures.values())
    
    def clear(self):
        """Clear all converted textures"""
        self.converted_textures.clear()


# Convenience functions
def convert_txd_file(txd_path: str, output_dir: str,
                    output_format: str = 'png',
                    size_limit: Optional[Tuple[int, int]] = None,
                    quality: int = 90) -> Dict[str, Dict[str, Any]]:
    """Convert a single TXD file"""
    converter = TXDConverter()
    return converter.convert(txd_path, output_dir, output_format, size_limit, quality)


if __name__ == "__main__":
    # Test the TXD converter
    import sys
    
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else "./textures"
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        
        # Convert file
        converter = TXDConverter()
        result = converter.convert(test_file, output_dir, 'png')
        
        if result:
            print(f"Successfully converted {test_file}")
            print(f"Textures extracted: {len(result)}")
            for tex_name, tex_info in result.items():
                print(f"  {tex_name}: {tex_info.get('path', 'N/A')}")
        else:
            print(f"Failed to convert {test_file}")
            print("\nNote: TXD conversion requires either:")
            print("  1. External tools like TXD Workshop")
            print("  2. Python libraries like pytxd")
            print("  3. Manual implementation of TXD parser")
    else:
        print("Usage: python txd_converter.py <txd_file_path> [output_dir]")
