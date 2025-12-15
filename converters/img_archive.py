"""
GTA SA Map Converter - IMG Archive Handler
Handles GTA3.IMG archive extraction and file management
"""

import os
import struct
import zlib
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, BinaryIO
from dataclasses import dataclass, field
import logging
import subprocess
import tempfile
import shutil

logger = logging.getLogger(__name__)


@dataclass
class IMGEntry:
    """Represents a file entry in IMG archive"""
    offset: int = 0
    size: int = 0
    compressed_size: int = 0
    name: str = ""
    is_compressed: bool = False
    is_encrypted: bool = False
    checksum: int = 0
    file_type: str = ""  # dff, txd, col, etc.
    
    def get_extension(self) -> str:
        """Get file extension from name"""
        return Path(self.name).suffix.lower()
    
    def is_model_file(self) -> bool:
        """Check if file is a model file"""
        return self.get_extension() == '.dff'
    
    def is_texture_file(self) -> bool:
        """Check if file is a texture file"""
        return self.get_extension() == '.txd'
    
    def is_collision_file(self) -> bool:
        """Check if file is a collision file"""
        return self.get_extension() == '.col'


@dataclass
class IMGArchive:
    """Represents a complete IMG archive"""
    file_path: str = ""
    version: int = 0
    entry_count: int = 0
    entries: List[IMGEntry] = field(default_factory=list)
    toc_offset: int = 0
    data_offset: int = 0
    
    def get_entry_by_name(self, name: str) -> Optional[IMGEntry]:
        """Get entry by filename"""
        for entry in self.entries:
            if entry.name.lower() == name.lower():
                return entry
        return None
    
    def get_entries_by_type(self, file_type: str) -> List[IMGEntry]:
        """Get entries by file extension"""
        return [entry for entry in self.entries if entry.get_extension() == f'.{file_type}']
    
    def get_model_files(self) -> List[IMGEntry]:
        """Get all model files"""
        return self.get_entries_by_type('dff')
    
    def get_texture_files(self) -> List[IMGEntry]:
        """Get all texture files"""
        return self.get_entries_by_type('txd')
    
    def get_collision_files(self) -> List[IMGEntry]:
        """Get all collision files"""
        return self.get_entries_by_type('col')
    
    def get_total_size(self) -> int:
        """Get total size of all files"""
        return sum(entry.size for entry in self.entries)
    
    def get_compressed_size(self) -> int:
        """Get total compressed size"""
        return sum(entry.compressed_size for entry in self.entries if entry.is_compressed)


class IMGParser:
    """Parser for GTA3.IMG archives"""
    
    def __init__(self):
        self.archive = IMGArchive()
        self.warnings: List[str] = []
        self.errors: List[str] = []
        
    def parse_file(self, file_path: str) -> Optional[IMGArchive]:
        """
        Parse an IMG archive file
        
        Args:
            file_path: Path to IMG file
            
        Returns:
            IMGArchive if successful, None otherwise
        """
        if not os.path.exists(file_path):
            logger.error(f"IMG file not found: {file_path}")
            return None
            
        logger.info(f"Parsing IMG archive: {file_path}")
        
        try:
            with open(file_path, 'rb') as f:
                # Read first few bytes to check format
                header = f.read(4)
                
                if header == b'VER2':
                    return self._parse_ver2_format(f, file_path)
                elif header == b'IMG2':
                    return self._parse_img2_format(f, file_path)
                else:
                    # Try to parse as standard GTA3.IMG
                    return self._parse_standard_format(f, file_path)
                    
        except Exception as e:
            logger.error(f"Error parsing IMG file {file_path}: {str(e)}")
            self.errors.append(str(e))
            return None
    
    def _parse_standard_format(self, f: BinaryIO, file_path: str) -> Optional[IMGArchive]:
        """Parse standard GTA3.IMG format"""
        try:
            self.archive.file_path = file_path
            
            # Get file size
            f.seek(0, 2)
            file_size = f.tell()
            f.seek(0)
            
            # Standard IMG has TOC at the end
            # Read last 8 bytes to get TOC offset
            f.seek(-8, 2)
            toc_offset, entry_count = struct.unpack('<II', f.read(8))
            
            self.archive.toc_offset = toc_offset
            self.archive.entry_count = entry_count
            
            # Validate TOC offset
            if toc_offset >= file_size:
                raise ValueError(f"Invalid TOC offset: {toc_offset}")
            
            # Read TOC entries
            f.seek(toc_offset)
            entries = []
            
            for i in range(entry_count):
                # Read entry (16 bytes: offset, size, name[24])
                entry_data = f.read(32)
                if len(entry_data) < 32:
                    break
                    
                offset, size, name_bytes = struct.unpack('<II24s', entry_data)
                
                # Extract filename (null-terminated)
                name = name_bytes.split(b'\x00')[0].decode('ascii', errors='ignore')
                
                entry = IMGEntry(
                    offset=offset * 2048,  # IMG uses 2048-byte sectors
                    size=size * 2048,
                    compressed_size=size * 2048,
                    name=name,
                    is_compressed=False,
                    file_type=Path(name).suffix.lower()[1:] if Path(name).suffix else ""
                )
                
                entries.append(entry)
            
            self.archive.entries = entries
            logger.info(f"Parsed {len(entries)} entries from IMG archive")
            return self.archive
            
        except Exception as e:
            logger.error(f"Error parsing standard IMG format: {e}")
            self.errors.append(str(e))
            return None
    
    def _parse_ver2_format(self, f: BinaryIO, file_path: str) -> Optional[IMGArchive]:
        """Parse VER2 format (compressed IMG)"""
        try:
            self.archive.file_path = file_path
            self.archive.version = 2
            
            # Read header
            f.seek(0)
            header = f.read(12)
            magic, version, entry_count = struct.unpack('<4sII', header)
            
            self.archive.entry_count = entry_count
            
            # Read TOC
            entries = []
            for i in range(entry_count):
                # Read entry (40 bytes)
                entry_data = f.read(40)
                if len(entry_data) < 40:
                    break
                
                # Parse entry structure
                offset, size, compressed_size, name_bytes = struct.unpack('<III28s', entry_data)
                
                # Extract filename
                name = name_bytes.split(b'\x00')[0].decode('ascii', errors='ignore')
                
                entry = IMGEntry(
                    offset=offset,
                    size=size,
                    compressed_size=compressed_size,
                    name=name,
                    is_compressed=(compressed_size != size),
                    file_type=Path(name).suffix.lower()[1:] if Path(name).suffix else ""
                )
                
                entries.append(entry)
            
            self.archive.entries = entries
            logger.info(f"Parsed {len(entries)} entries from VER2 IMG archive")
            return self.archive
            
        except Exception as e:
            logger.error(f"Error parsing VER2 format: {e}")
            self.errors.append(str(e))
            return None
    
    def _parse_img2_format(self, f: BinaryIO, file_path: str) -> Optional[IMGArchive]:
        """Parse IMG2 format (GTA IV/V style)"""
        # Simplified parser for IMG2 format
        logger.warning("IMG2 format parsing is limited")
        
        try:
            self.archive.file_path = file_path
            self.archive.version = 2
            
            # Read basic info
            f.seek(0)
            header = f.read(16)
            magic, version, entry_count, toc_offset = struct.unpack('<4sIII', header)
            
            self.archive.entry_count = entry_count
            self.archive.toc_offset = toc_offset
            
            # Read TOC
            f.seek(toc_offset)
            entries = []
            
            for i in range(entry_count):
                # Simplified entry reading
                entry_data = f.read(32)
                if len(entry_data) < 32:
                    break
                
                # Parse entry (simplified)
                offset, size, name_bytes = struct.unpack('<II24s', entry_data[:32])
                
                name = name_bytes.split(b'\x00')[0].decode('ascii', errors='ignore')
                
                entry = IMGEntry(
                    offset=offset,
                    size=size,
                    compressed_size=size,
                    name=name,
                    is_compressed=False,
                    file_type=Path(name).suffix.lower()[1:] if Path(name).suffix else ""
                )
                
                entries.append(entry)
            
            self.archive.entries = entries
            logger.info(f"Parsed {len(entries)} entries from IMG2 archive")
            return self.archive
            
        except Exception as e:
            logger.error(f"Error parsing IMG2 format: {e}")
            self.errors.append(str(e))
            return None
    
    def get_warnings(self) -> List[str]:
        """Get parsing warnings"""
        return self.warnings
        
    def get_errors(self) -> List[str]:
        """Get parsing errors"""
        return self.errors


class IMGExtractor:
    """Extracts files from IMG archives using multiple methods"""
    
    def __init__(self, use_external_tool: bool = True):
        self.use_external_tool = use_external_tool
        self.external_tools = self._detect_external_tools()
        self.parser = IMGParser()
        
    def _detect_external_tools(self) -> Dict[str, str]:
        """Detect available external IMG extraction tools"""
        tools = {}
        
        # Check for common IMG tools
        tool_paths = [
            '/usr/bin/imgfactory',          # IMG Factory
            '/usr/local/bin/imgfactory',
            '/usr/bin/imgtool',             # IMG Tool
            '/usr/local/bin/imgtool',
            '/opt/gta_tools/imgfactory',
            Path.home() / '.local/bin/imgfactory',
            '/usr/bin/alci_img',            # ALCI's IMG Editor
            '/usr/local/bin/alci_img',
        ]
        
        for tool_path in tool_paths:
            if os.path.exists(tool_path):
                tool_name = Path(tool_path).name
                if 'img' in tool_name.lower():
                    tools['img_tool'] = str(tool_path)
                    break
                
        # Check for Python libraries
        try:
            import pygtair
            tools['pygtair'] = 'pygtair'
        except ImportError:
            pass
            
        try:
            import gta_tools
            tools['gta_tools'] = 'gta_tools'
        except ImportError:
            pass
            
        return tools
    
    def extract(self, img_path: str, output_dir: str, 
                file_filter: Optional[List[str]] = None) -> Dict[str, str]:
        """
        Extract files from IMG archive
        
        Args:
            img_path: Path to IMG file
            output_dir: Directory to extract files to
            file_filter: Optional list of file extensions to extract (e.g., ['.dff', '.txd'])
            
        Returns:
            Dictionary mapping filenames to extracted file paths
        """
        logger.info(f"Extracting from IMG: {img_path}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Try different extraction methods
        methods = [
            self._extract_with_external_tool,
            self._extract_with_parser,
            self._extract_with_fallback,
        ]
        
        extracted_files = {}
        
        for method in methods:
            try:
                result = method(img_path, output_dir, file_filter)
                if result:
                    extracted_files.update(result)
                    if extracted_files:
                        logger.info(f"Extracted {len(extracted_files)} files using {method.__name__}")
                        return extracted_files
            except Exception as e:
                logger.warning(f"Extraction method {method.__name__} failed: {e}")
                continue
                
        logger.error(f"All extraction methods failed for {img_path}")
        return extracted_files
    
    def _extract_with_external_tool(self, img_path: str, output_dir: str,
                                   file_filter: Optional[List[str]]) -> Dict[str, str]:
        """Extract using external tool"""
        if not self.use_external_tool or not self.external_tools:
            return {}
            
        try:
            if 'img_tool' in self.external_tools:
                return self._extract_with_imgfactory(img_path, output_dir, file_filter)
            elif 'pygtair' in self.external_tools:
                return self._extract_with_pygtair(img_path, output_dir, file_filter)
                
        except Exception as e:
            logger.warning(f"External tool extraction failed: {e}")
            
        return {}
    
    def _extract_with_imgfactory(self, img_path: str, output_dir: str,
                                file_filter: Optional[List[str]]) -> Dict[str, str]:
        """Extract using IMG Factory tool"""
        extracted_files = {}
        
        try:
            # Create temporary directory for extraction
            with tempfile.TemporaryDirectory() as temp_dir:
                # Run IMG Factory command line
                cmd = [
                    self.external_tools['img_tool'],
                    '-extract',
                    img_path,
                    temp_dir
                ]
                
                # Add filter if specified
                if file_filter:
                    cmd.extend(['-filter', ','.join(file_filter)])
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=60  # Extraction can take time
                )
                
                if result.returncode == 0:
                    # Process extracted files
                    for file_path in Path(temp_dir).rglob('*'):
                        if file_path.is_file():
                            # Copy to output directory
                            rel_path = file_path.relative_to(temp_dir)
                            output_path = os.path.join(output_dir, rel_path)
                            os.makedirs(os.path.dirname(output_path), exist_ok=True)
                            shutil.copy2(file_path, output_path)
                            
                            extracted_files[rel_path.name] = str(output_path)
                else:
                    logger.warning(f"IMG Factory failed: {result.stderr}")
                    
        except subprocess.TimeoutExpired:
            logger.warning("IMG Factory timed out")
        except Exception as e:
            logger.warning(f"IMG Factory error: {e}")
            
        return extracted_files
    
    def _extract_with_pygtair(self, img_path: str, output_dir: str,
                             file_filter: Optional[List[str]]) -> Dict[str, str]:
        """Extract using pygtair Python library"""
        if 'pygtair' not in self.external_tools:
            return {}
            
        try:
            import pygtair
            
            extracted_files = {}
            
            # Open IMG archive
            archive = pygtair.IMGArchive(img_path)
            
            # Extract each entry
            for entry in archive.entries:
                # Apply filter if specified
                if file_filter:
                    ext = Path(entry.name).suffix.lower()
                    if ext not in file_filter:
                        continue
                
                try:
                    # Read file data
                    data = entry.read()
                    
                    # Create output path
                    output_path = os.path.join(output_dir, entry.name)
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    
                    # Write file
                    with open(output_path, 'wb') as f:
                        f.write(data)
                    
                    extracted_files[entry.name] = output_path
                    
                except Exception as e:
                    logger.warning(f"Error extracting {entry.name}: {e}")
                    
            return extracted_files
            
        except ImportError:
            logger.warning("pygtair not installed")
        except Exception as e:
            logger.warning(f"pygtair extraction error: {e}")
            
        return {}
    
    def _extract_with_parser(self, img_path: str, output_dir: str,
                            file_filter: Optional[List[str]]) -> Dict[str, str]:
        """Extract using internal parser"""
        extracted_files = {}
        
        try:
            # Parse IMG archive
            archive = self.parser.parse_file(img_path)
            if not archive:
                return {}
            
            with open(img_path, 'rb') as f:
                for entry in archive.entries:
                    # Apply filter if specified
                    if file_filter:
                        ext = Path(entry.name).suffix.lower()
                        if ext not in file_filter:
                            continue
                    
                    try:
                        # Seek to file data
                        f.seek(entry.offset)
                        
                        # Read data
                        if entry.is_compressed:
                            # Read compressed data
                            compressed_data = f.read(entry.compressed_size)
                            # Decompress (simplified - actual compression varies)
                            data = self._decompress_data(compressed_data, entry.size)
                        else:
                            # Read uncompressed data
                            data = f.read(entry.size)
                        
                        # Create output path
                        output_path = os.path.join(output_dir, entry.name)
                        os.makedirs(os.path.dirname(output_path), exist_ok=True)
                        
                        # Write file
                        with open(output_path, 'wb') as out_f:
                            out_f.write(data)
                        
                        extracted_files[entry.name] = output_path
                        
                        logger.debug(f"Extracted: {entry.name} ({len(data)} bytes)")
                        
                    except Exception as e:
                        logger.warning(f"Error extracting {entry.name}: {e}")
                        
            return extracted_files
            
        except Exception as e:
            logger.warning(f"Parser extraction failed: {e}")
            return {}
    
    def _extract_with_fallback(self, img_path: str, output_dir: str,
                              file_filter: Optional[List[str]]) -> Dict[str, str]:
        """Extract using fallback method (file copying)"""
        extracted_files = {}
        
        try:
            # Simple fallback: treat IMG as a container and try to extract known file types
            # This is a placeholder for more sophisticated fallback
            
            # For now, just copy the IMG file itself
            img_name = Path(img_path).name
            output_path = os.path.join(output_dir, img_name)
            shutil.copy2(img_path, output_path)
            
            extracted_files[img_name] = output_path
            logger.warning(f"Used fallback extraction for {img_path}")
            
        except Exception as e:
            logger.warning(f"Fallback extraction failed: {e}")
            
        return extracted_files
    
    def _decompress_data(self, compressed_data: bytes, expected_size: int) -> bytes:
        """Decompress IMG file data"""
        try:
            # Try zlib decompression
            return zlib.decompress(compressed_data)
        except zlib.error:
            # Try other decompression methods
            pass
            
        # If decompression fails, return raw data
        logger.warning("Decompression failed, returning raw data")
        return compressed_data
    
    def extract_specific_files(self, img_path: str, output_dir: str,
                              file_names: List[str]) -> Dict[str, str]:
        """
        Extract specific files from IMG archive
        
        Args:
            img_path: Path to IMG file
            output_dir: Directory to extract files to
            file_names: List of specific filenames to extract
            
        Returns:
            Dictionary mapping filenames to extracted file paths
        """
        extracted_files = {}
        
        try:
            # Parse archive to get file list
            archive = self.parser.parse_file(img_path)
            if not archive:
                return {}
            
            # Convert to lowercase for case-insensitive matching
            target_files = {name.lower() for name in file_names}
            
            with open(img_path, 'rb') as f:
                for entry in archive.entries:
                    if entry.name.lower() in target_files:
                        try:
                            # Extract the file
                            f.seek(entry.offset)
                            data = f.read(entry.size)
                            
                            output_path = os.path.join(output_dir, entry.name)
                            os.makedirs(os.path.dirname(output_path), exist_ok=True)
                            
                            with open(output_path, 'wb') as out_f:
                                out_f.write(data)
                            
                            extracted_files[entry.name] = output_path
                            logger.info(f"Extracted specific file: {entry.name}")
                            
                        except Exception as e:
                            logger.warning(f"Error extracting {entry.name}: {e}")
            
            return extracted_files
            
        except Exception as e:
            logger.error(f"Error extracting specific files: {e}")
            return {}
    
    def list_files(self, img_path: str) -> List[str]:
        """
        List all files in IMG archive
        
        Args:
            img_path: Path to IMG file
            
        Returns:
            List of filenames in archive
        """
        try:
            archive = self.parser.parse_file(img_path)
            if archive:
                return [entry.name for entry in archive.entries]
        except Exception as e:
            logger.error(f"Error listing files: {e}")
            
        return []
    
    def get_file_info(self, img_path: str, file_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific file in IMG archive
        
        Args:
            img_path: Path to IMG file
            file_name: Name of file to get info for
            
        Returns:
            Dictionary with file information or None
        """
        try:
            archive = self.parser.parse_file(img_path)
            if archive:
                entry = archive.get_entry_by_name(file_name)
                if entry:
                    return {
                        'name': entry.name,
                        'size': entry.size,
                        'compressed_size': entry.compressed_size,
                        'is_compressed': entry.is_compressed,
                        'file_type': entry.file_type,
                        'offset': entry.offset
                    }
        except Exception as e:
            logger.error(f"Error getting file info: {e}")
            
        return None


class IMGManager:
    """Manager for handling multiple IMG archives"""
    
    def __init__(self, use_external_tool: bool = True):
        self.extractor = IMGExtractor(use_external_tool)
        self.extracted_files: Dict[str, Dict[str, str]] = {}  # img_name -> {filename: path}
        self.parsed_archives: Dict[str, IMGArchive] = {}
        
    def extract_archives(self, img_files: List[str], output_dir: str,
                        file_filter: Optional[List[str]] = None,
                        max_workers: int = 4) -> Dict[str, Dict[str, str]]:
        """
        Extract multiple IMG archives
        
        Args:
            img_files: List of IMG file paths
            output_dir: Directory to extract files to
            file_filter: Optional list of file extensions to extract
            max_workers: Maximum number of parallel workers
            
        Returns:
            Dictionary mapping IMG names to extracted file dictionaries
        """
        import concurrent.futures
        
        self.extracted_files.clear()
        
        logger.info(f"Extracting {len(img_files)} IMG archives")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Use ThreadPoolExecutor for parallel extraction
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit extraction tasks
            future_to_file = {
                executor.submit(self._extract_single_archive, img_file, output_dir, file_filter): img_file
                for img_file in img_files
            }
            
            # Process completed tasks
            for future in concurrent.futures.as_completed(future_to_file):
                img_file = future_to_file[future]
                try:
                    result = future.result()
                    if result:
                        img_name = Path(img_file).stem
                        self.extracted_files[img_name] = result
                        logger.debug(f"Extracted: {img_name} ({len(result)} files)")
                except Exception as e:
                    logger.error(f"Failed to extract {img_file}: {e}")
                    
        logger.info(f"Successfully extracted {len(self.extracted_files)} out of {len(img_files)} archives")
        return self.extracted_files.copy()
    
    def _extract_single_archive(self, img_path: str, output_dir: str,
                               file_filter: Optional[List[str]]) -> Dict[str, str]:
        """Extract a single IMG archive"""
        try:
            # Create subdirectory for this IMG
            img_name = Path(img_path).stem
            img_output_dir = os.path.join(output_dir, img_name)
            
            files = self.extractor.extract(img_path, img_output_dir, file_filter)
            return files
            
        except Exception as e:
            logger.error(f"Error extracting {img_path}: {e}")
            return {}
    
    def get_extracted_files(self, img_name: str) -> Optional[Dict[str, str]]:
        """Get extracted files by IMG name"""
        return self.extracted_files.get(img_name)
    
    def get_all_extracted_files(self) -> Dict[str, str]:
        """Get all extracted files flattened"""
        all_files = {}
        for img_files in self.extracted_files.values():
            all_files.update(img_files)
        return all_files
    
    def find_file(self, filename: str) -> Optional[str]:
        """Find an extracted file by name"""
        for img_files in self.extracted_files.values():
            if filename in img_files:
                return img_files[filename]
        return None
    
    def list_archives(self) -> List[str]:
        """List all processed IMG archive names"""
        return list(self.extracted_files.keys())
    
    def get_total_file_count(self) -> int:
        """Get total number of extracted files"""
        return sum(len(files) for files in self.extracted_files.values())
    
    def clear(self):
        """Clear all extracted files"""
        self.extracted_files.clear()
        self.parsed_archives.clear()


# Convenience functions
def extract_img_file(img_path: str, output_dir: str,
                    file_filter: Optional[List[str]] = None) -> Dict[str, str]:
    """Extract a single IMG file"""
    extractor = IMGExtractor()
    return extractor.extract(img_path, output_dir, file_filter)


def list_img_files(img_path: str) -> List[str]:
    """List files in IMG archive"""
    extractor = IMGExtractor()
    return extractor.list_files(img_path)


if __name__ == "__main__":
    # Test the IMG extractor
    import sys
    
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else "./extracted"
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        
        # Extract file
        extractor = IMGExtractor()
        result = extractor.extract(test_file, output_dir, ['.dff', '.txd'])
        
        if result:
            print(f"Successfully extracted from {test_file}")
            print(f"Files extracted: {len(result)}")
            for filename, filepath in list(result.items())[:10]:  # Show first 10
                print(f"  {filename} -> {filepath}")
            if len(result) > 10:
                print(f"  ... and {len(result) - 10} more files")
        else:
            print(f"Failed to extract {test_file}")
            print("\nNote: IMG extraction requires either:")
            print("  1. External tools like IMG Factory")
            print("  2. Python libraries like pygtair")
            print("  3. Manual implementation of IMG parser")
    else:
        print("Usage: python img_archive.py <img_file_path> [output_dir]")
