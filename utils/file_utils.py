import os
import sys
import shutil
import hashlib
import zipfile
import tarfile
import tempfile
import fnmatch
from pathlib import Path, PurePath
from typing import Dict, List, Optional, Tuple, Any, Set, BinaryIO, Union
from dataclasses import dataclass, field
import logging
import json
import pickle
import time
import stat
import mimetypes
import subprocess

logger = logging.getLogger(__name__)


def list_files(directory: str, extensions: Optional[List[str]] = None, recursive: bool = True) -> List[str]:
    """List files in a directory with optional extension filter"""
    matched_files = []
    extensions = [e.lower() for e in extensions] if extensions else None

    for root, _, files in os.walk(directory):
        for file in files:
            if not extensions or any(file.lower().endswith(ext) for ext in extensions):
                matched_files.append(os.path.join(root, file))
        if not recursive:
            break

    return matched_files


def ensure_dir(path: str):
    """Ensure a directory exists"""
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


@dataclass
class FileInfo:
    """Information about a file"""
    path: str
    size: int = 0
    modified_time: float = 0.0
    created_time: float = 0.0
    is_file: bool = True
    is_dir: bool = False
    is_symlink: bool = False
    permissions: int = 0
    owner: str = ""
    group: str = ""
    mime_type: str = ""
    extension: str = ""
    hash_md5: str = ""
    hash_sha256: str = ""
    
    def get_human_size(self) -> str:
        """Get human-readable file size"""
        return self._bytes_to_human(self.size)
    
    def get_age_days(self) -> float:
        """Get file age in days"""
        current_time = time.time()
        return (current_time - self.modified_time) / 86400
    
    def is_recent(self, days: int = 7) -> bool:
        """Check if file was modified recently"""
        return self.get_age_days() <= days
    
    def is_large(self, threshold_mb: int = 100) -> bool:
        """Check if file is large"""
        return self.size > threshold_mb * 1024 * 1024
    
    def is_empty(self) -> bool:
        """Check if file is empty"""
        return self.size == 0
    
    @staticmethod
    def _bytes_to_human(size: int) -> str:
        """Convert bytes to human-readable format"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size < 1024.0:
                return f"{size:.2f} {unit}"
            size /= 1024.0
        return f"{size:.2f} PB"


@dataclass
class DirectoryInfo:
    """Information about a directory"""
    path: str
    file_count: int = 0
    dir_count: int = 0
    total_size: int = 0
    modified_time: float = 0.0
    permissions: int = 0
    
    def get_human_size(self) -> str:
        """Get human-readable total size"""
        return FileInfo._bytes_to_human(self.total_size)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get directory statistics"""
        return {
            'path': self.path,
            'file_count': self.file_count,
            'dir_count': self.dir_count,
            'total_size': self.total_size,
            'human_size': self.get_human_size(),
            'modified_time': self.modified_time
        }


class FileValidator:
    """Validates files for various conditions"""
    
    # Common GTA file extensions
    GTA_EXTENSIONS = {
        '.ide': 'IDE Definition File',
        '.ipl': 'IPL Placement File',
        '.dff': 'DFF Model File',
        '.txd': 'TXD Texture Dictionary',
        '.img': 'IMG Archive',
        '.col': 'Collision File',
        '.dat': 'Data File',
        '.ifp': 'Animation File',
        '.scm': 'Script File',
        '.gxt': 'Text File',
        '.cut': 'Cutscene File'
    }
    
    # Valid texture extensions
    TEXTURE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tga', '.bmp', '.tiff', '.gif'}
    
    # Valid model extensions
    MODEL_EXTENSIONS = {'.obj', '.fbx', '.dae', '.blend', '.3ds', '.max'}
    
    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
        
    def validate_gta_file(self, file_path: str) -> Tuple[bool, List[str]]:
        """
        Validate a GTA file
        
        Args:
            file_path: Path to file
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        self.errors.clear()
        self.warnings.clear()
        
        # Check file exists
        if not os.path.exists(file_path):
            self.errors.append(f"File does not exist: {file_path}")
            return False, self.errors + self.warnings
        
        # Check file extension
        ext = Path(file_path).suffix.lower()
        if ext not in self.GTA_EXTENSIONS:
            self.warnings.append(f"Unusual extension for GTA file: {ext}")
        
        # Check file size
        try:
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                self.errors.append(f"File is empty: {file_path}")
            elif file_size > 1024 * 1024 * 1024:  # 1GB
                self.warnings.append(f"File is very large: {FileInfo._bytes_to_human(file_size)}")
        except OSError:
            self.errors.append(f"Cannot access file: {file_path}")
        
        # Check file permissions
        try:
            if not os.access(file_path, os.R_OK):
                self.errors.append(f"Cannot read file (permission denied): {file_path}")
        except OSError:
            self.errors.append(f"Cannot check file permissions: {file_path}")
        
        # File-specific validation
        if ext == '.dff':
            self._validate_dff_file(file_path)
        elif ext == '.txd':
            self._validate_txd_file(file_path)
        elif ext == '.img':
            self._validate_img_file(file_path)
        elif ext == '.ide':
            self._validate_ide_file(file_path)
        elif ext == '.ipl':
            self._validate_ipl_file(file_path)
        
        return len(self.errors) == 0, self.errors + self.warnings
    
    def _validate_dff_file(self, file_path: str):
        """Validate DFF file"""
        try:
            with open(file_path, 'rb') as f:
                # Check DFF header
                header = f.read(4)
                if header != b'\x10\x00\x00\x00':
                    self.warnings.append("DFF file has unexpected header")
                
                # Check file size consistency
                f.seek(0, 2)
                file_size = f.tell()
                if file_size < 100:  # Minimum reasonable DFF size
                    self.warnings.append("DFF file is unusually small")
        except Exception as e:
            self.errors.append(f"Cannot read DFF file: {str(e)}")
    
    def _validate_txd_file(self, file_path: str):
        """Validate TXD file"""
        try:
            with open(file_path, 'rb') as f:
                # Check TXD signature
                signature = f.read(4)
                if signature != b'TXD\x00':
                    self.warnings.append("TXD file has unexpected signature")
        except Exception as e:
            self.errors.append(f"Cannot read TXD file: {str(e)}")
    
    def _validate_img_file(self, file_path: str):
        """Validate IMG file"""
        try:
            file_size = os.path.getsize(file_path)
            if file_size < 2048:  # Minimum IMG size
                self.warnings.append("IMG file is unusually small")
        except Exception as e:
            self.errors.append(f"Cannot read IMG file: {str(e)}")
    
    def _validate_ide_file(self, file_path: str):
        """Validate IDE file"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(1024)  # Read first 1KB
                if not content.strip():
                    self.errors.append("IDE file is empty")
                elif 'objs' not in content.lower() and 'tobj' not in content.lower():
                    self.warnings.append("IDE file doesn't contain expected sections")
        except Exception as e:
            self.errors.append(f"Cannot read IDE file: {str(e)}")
    
    def _validate_ipl_file(self, file_path: str):
        """Validate IPL file"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(1024)
                if not content.strip():
                    self.errors.append("IPL file is empty")
                elif 'inst' not in content.lower():
                    self.warnings.append("IPL file doesn't contain instance section")
        except Exception as e:
            self.errors.append(f"Cannot read IPL file: {str(e)}")
    
    def validate_texture_file(self, file_path: str) -> Tuple[bool, List[str]]:
        """Validate texture file"""
        self.errors.clear()
        self.warnings.clear()
        
        if not os.path.exists(file_path):
            self.errors.append(f"Texture file not found: {file_path}")
            return False, self.errors + self.warnings
        
        ext = Path(file_path).suffix.lower()
        if ext not in self.TEXTURE_EXTENSIONS:
            self.errors.append(f"Unsupported texture format: {ext}")
        
        # Try to open with PIL if available
        try:
            from PIL import Image
            with Image.open(file_path) as img:
                width, height = img.size
                if width == 0 or height == 0:
                    self.errors.append(f"Invalid texture dimensions: {width}x{height}")
                elif width > 8192 or height > 8192:
                    self.warnings.append(f"Texture is very large: {width}x{height}")
        except ImportError:
            # PIL not available, skip image validation
            pass
        except Exception as e:
            self.warnings.append(f"Cannot validate texture: {str(e)}")
        
        return len(self.errors) == 0, self.errors + self.warnings
    
    def validate_directory(self, dir_path: str, 
                          required_files: Optional[List[str]] = None) -> Tuple[bool, List[str]]:
        """
        Validate a directory
        
        Args:
            dir_path: Path to directory
            required_files: List of required file patterns
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        self.errors.clear()
        self.warnings.clear()
        
        if not os.path.exists(dir_path):
            self.errors.append(f"Directory does not exist: {dir_path}")
            return False, self.errors + self.warnings
        
        if not os.path.isdir(dir_path):
            self.errors.append(f"Path is not a directory: {dir_path}")
            return False, self.errors + self.warnings
        
        # Check directory permissions
        if not os.access(dir_path, os.R_OK):
            self.errors.append(f"Cannot read directory (permission denied): {dir_path}")
        
        # Check for required files
        if required_files:
            missing_files = []
            for pattern in required_files:
                found = False
                for root, dirs, files in os.walk(dir_path):
                    for file in files:
                        if fnmatch.fnmatch(file, pattern):
                            found = True
                            break
                    if found:
                        break
                if not found:
                    missing_files.append(pattern)
            
            if missing_files:
                self.errors.append(f"Missing required files: {', '.join(missing_files)}")
        
        return len(self.errors) == 0, self.errors + self.warnings


class FileOperations:
    """File operations utilities"""
    
    def __init__(self):
        self.validator = FileValidator()
    
    def get_file_info(self, file_path: str) -> Optional[FileInfo]:
        """Get detailed information about a file"""
        try:
            stat_info = os.stat(file_path)
            
            # Get file extension
            ext = Path(file_path).suffix.lower()
            
            # Get MIME type
            mime_type, _ = mimetypes.guess_type(file_path)
            
            # Get owner/group (platform dependent)
            owner = group = ""
            try:
                import pwd
                import grp
                owner = pwd.getpwuid(stat_info.st_uid).pw_name
                group = grp.getgrgid(stat_info.st_gid).gr_name
            except (ImportError, KeyError):
                pass
            
            return FileInfo(
                path=file_path,
                size=stat_info.st_size,
                modified_time=stat_info.st_mtime,
                created_time=stat_info.st_ctime,
                is_file=os.path.isfile(file_path),
                is_dir=os.path.isdir(file_path),
                is_symlink=os.path.islink(file_path),
                permissions=stat_info.st_mode,
                owner=owner,
                group=group,
                mime_type=mime_type or "",
                extension=ext
            )
        except OSError as e:
            logger.error(f"Cannot get file info for {file_path}: {e}")
            return None
    
    def get_directory_info(self, dir_path: str, recursive: bool = True) -> Optional[DirectoryInfo]:
        """Get information about a directory"""
        try:
            if not os.path.isdir(dir_path):
                return None
            
            stat_info = os.stat(dir_path)
            dir_info = DirectoryInfo(
                path=dir_path,
                permissions=stat_info.st_mode,
                modified_time=stat_info.st_mtime
            )
            
            # Count files and directories
            if recursive:
                for root, dirs, files in os.walk(dir_path):
                    dir_info.dir_count += len(dirs)
                    dir_info.file_count += len(files)
                    
                    # Calculate total size
                    for file in files:
                        file_path = os.path.join(root, file)
                        try:
                            dir_info.total_size += os.path.getsize(file_path)
                        except OSError:
                            pass
            else:
                items = os.listdir(dir_path)
                for item in items:
                    item_path = os.path.join(dir_path, item)
                    if os.path.isdir(item_path):
                        dir_info.dir_count += 1
                    else:
                        dir_info.file_count += 1
                        try:
                            dir_info.total_size += os.path.getsize(item_path)
                        except OSError:
                            pass
            
            return dir_info
        except OSError as e:
            logger.error(f"Cannot get directory info for {dir_path}: {e}")
            return None
    
    def calculate_hash(self, file_path: str, algorithm: str = 'md5',
                      chunk_size: int = 8192) -> Optional[str]:
        """
        Calculate file hash
        
        Args:
            file_path: Path to file
            algorithm: Hash algorithm (md5, sha1, sha256, sha512)
            chunk_size: Chunk size for reading
            
        Returns:
            Hash string or None
        """
        try:
            hash_func = hashlib.new(algorithm)
            
            with open(file_path, 'rb') as f:
                while chunk := f.read(chunk_size):
                    hash_func.update(chunk)
            
            return hash_func.hexdigest()
        except Exception as e:
            logger.error(f"Cannot calculate hash for {file_path}: {e}")
            return None
    
    def find_files(self, directory: str, patterns: List[str],
                  recursive: bool = True) -> List[str]:
        """
        Find files matching patterns
        
        Args:
            directory: Directory to search
            patterns: List of file patterns (e.g., ['*.txt', '*.md'])
            recursive: Whether to search recursively
            
        Returns:
            List of matching file paths
        """
        matches = []
        
        try:
            if recursive:
                for root, dirs, files in os.walk(directory):
                    for pattern in patterns:
                        for file in fnmatch.filter(files, pattern):
                            matches.append(os.path.join(root, file))
            else:
                for item in os.listdir(directory):
                    item_path = os.path.join(directory, item)
                    if os.path.isfile(item_path):
                        for pattern in patterns:
                            if fnmatch.fnmatch(item, pattern):
                                matches.append(item_path)
                                break
        except OSError as e:
            logger.error(f"Cannot search directory {directory}: {e}")
        
        return matches
    
    def find_gta_files(self, directory: str, recursive: bool = True) -> Dict[str, List[str]]:
        """
        Find GTA files in directory
        
        Args:
            directory: Directory to search
            recursive: Whether to search recursively
            
        Returns:
            Dictionary of file types to file paths
        """
        gta_files = {ext: [] for ext in FileValidator.GTA_EXTENSIONS.keys()}
        
        try:
            if recursive:
                for root, dirs, files in os.walk(directory):
                    for file in files:
                        ext = Path(file).suffix.lower()
                        if ext in gta_files:
                            gta_files[ext].append(os.path.join(root, file))
            else:
                for item in os.listdir(directory):
                    item_path = os.path.join(directory, item)
                    if os.path.isfile(item_path):
                        ext = Path(item).suffix.lower()
                        if ext in gta_files:
                            gta_files[ext].append(item_path)
        except OSError as e:
            logger.error(f"Cannot search for GTA files in {directory}: {e}")
        
        return gta_files
    
    def copy_file(self, src: str, dst: str, overwrite: bool = True,
                 preserve_metadata: bool = True) -> bool:
        """
        Copy a file
        
        Args:
            src: Source file path
            dst: Destination file path
            overwrite: Whether to overwrite existing file
            preserve_metadata: Whether to preserve file metadata
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if source exists
            if not os.path.exists(src):
                logger.error(f"Source file does not exist: {src}")
                return False
            
            # Check if destination exists
            if os.path.exists(dst) and not overwrite:
                logger.error(f"Destination file exists and overwrite is disabled: {dst}")
                return False
            
            # Create destination directory if needed
            dst_dir = os.path.dirname(dst)
            if dst_dir and not os.path.exists(dst_dir):
                os.makedirs(dst_dir, exist_ok=True)
            
            # Copy file
            if preserve_metadata:
                shutil.copy2(src, dst)
            else:
                shutil.copy(src, dst)
            
            logger.debug(f"Copied {src} to {dst}")
            return True
            
        except Exception as e:
            logger.error(f"Cannot copy file {src} to {dst}: {e}")
            return False
    
    def move_file(self, src: str, dst: str, overwrite: bool = True) -> bool:
        try:
            # Check if source exists
            if not os.path.exists(src):
                logger.error(f"Source file does not exist: {src}")
                return False
            
            # Check if destination exists
            if os.path.exists(dst) and not overwrite:
                logger.error(f"Destination file exists and overwrite is disabled: {dst}")
                return False
            
            # Create destination directory if needed
            dst_dir = os.path.dirname(dst)
            if dst_dir and not os.path.exists(dst_dir):
                os.makedirs(dst_dir, exist_ok=True)
            
            # Move file
            shutil.move(src, dst)
            
            logger.debug(f"Moved {src} to {dst}")
            return True
            
        except Exception as e:
            logger.error(f"Cannot move file {src} to {dst}: {e}")
            return False
    
    def delete_file(self, file_path: str, force: bool = False) -> bool:

        try:
            if not os.path.exists(file_path):
                if not force:
                    logger.warning(f"File does not exist: {file_path}")
                return True
            
            os.remove(file_path)
            logger.debug(f"Deleted file: {file_path}")
            return True
            
        except Exception as e:
            if force:
                logger.warning(f"Cannot delete file {file_path}: {e}")
                return False
            else:
                logger.error(f"Cannot delete file {file_path}: {e}")
                return False
    
    def create_directory(self, dir_path: str, parents: bool = True,
                        exist_ok: bool = True) -> bool:

        try:
            os.makedirs(dir_path, exist_ok=exist_ok)
            logger.debug(f"Created directory: {dir_path}")
            return True
        except Exception as e:
            logger.error(f"Cannot create directory {dir_path}: {e}")
            return False
    
    def delete_directory(self, dir_path: str, recursive: bool = True,
                        force: bool = False) -> bool:

        try:
            if not os.path.exists(dir_path):
                if not force:
                    logger.warning(f"Directory does not exist: {dir_path}")
                return True
            
            if recursive:
                shutil.rmtree(dir_path)
            else:
                os.rmdir(dir_path)
            
            logger.debug(f"Deleted directory: {dir_path}")
            return True
            
        except Exception as e:
            if force:
                logger.warning(f"Cannot delete directory {dir_path}: {e}")
                return False
            else:
                logger.error(f"Cannot delete directory {dir_path}: {e}")
                return False
    
    def read_text_file(self, file_path: str, encoding: str = 'utf-8') -> Optional[str]:
        """
        Read text file
        
        Args:
            file_path: Path to file
            encoding: File encoding
            
        Returns:
            File content or None
        """
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        except Exception as e:
            logger.error(f"Cannot read text file {file_path}: {e}")
            return None
    
    def write_text_file(self, file_path: str, content: str,
                       encoding: str = 'utf-8', overwrite: bool = True) -> bool:
        
        try:
            if os.path.exists(file_path) and not overwrite:
                logger.error(f"File exists and overwrite is disabled: {file_path}")
                return False
            
            # Create directory if needed
            dir_path = os.path.dirname(file_path)
            if dir_path and not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
            
            with open(file_path, 'w', encoding=encoding) as f:
                f.write(content)
            
            logger.debug(f"Wrote text file: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Cannot write text file {file_path}: {e}")
            return False
    
    def read_binary_file(self, file_path: str) -> Optional[bytes]:

        try:
            with open(file_path, 'rb') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Cannot read binary file {file_path}: {e}")
            return None
    
    def write_binary_file(self, file_path: str, data: bytes,
                         overwrite: bool = True) -> bool:

        try:
            if os.path.exists(file_path) and not overwrite:
                logger.error(f"File exists and overwrite is disabled: {file_path}")
                return False
            
            # Create directory if needed
            dir_path = os.path.dirname(file_path)
            if dir_path and not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
            
            with open(file_path, 'wb') as f:
                f.write(data)
            
            logger.debug(f"Wrote binary file: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Cannot write binary file {file_path}: {e}")
            return False
    
    def read_json_file(self, file_path: str) -> Optional[Any]:

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Cannot read JSON file {file_path}: {e}")
            return None
    
    def write_json_file(self, file_path: str, data: Any,
                       indent: int = 2, overwrite: bool = True) -> bool:

        try:
            if os.path.exists(file_path) and not overwrite:
                logger.error(f"File exists and overwrite is disabled: {file_path}")
                return False
            
            # Create directory if needed
            dir_path = os.path.dirname(file_path)
            if dir_path and not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=indent, default=str)
            
            logger.debug(f"Wrote JSON file: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Cannot write JSON file {file_path}: {e}")
            return False
    
    def create_temp_file(self, suffix: str = "", prefix: str = "tmp",
                        dir: Optional[str] = None, delete: bool = True) -> Optional[str]:

        try:
            with tempfile.NamedTemporaryFile(
                suffix=suffix,
                prefix=prefix,
                dir=dir,
                delete=delete
            ) as tmp:
                return tmp.name
        except Exception as e:
            logger.error(f"Cannot create temporary file: {e}")
            return None
    
    def create_temp_directory(self, suffix: str = "", prefix: str = "tmp",
                             dir: Optional[str] = None) -> Optional[str]:

        try:
            return tempfile.mkdtemp(suffix=suffix, prefix=prefix, dir=dir)
        except Exception as e:
            logger.error(f"Cannot create temporary directory: {e}")
            return None
    
    def extract_archive(self, archive_path: str, extract_dir: str,
                       format: Optional[str] = None) -> bool:

        try:
            # Create extraction directory
            self.create_directory(extract_dir)
            
            # Determine archive format
            if format is None:
                if archive_path.lower().endswith('.zip'):
                    format = 'zip'
                elif archive_path.lower().endswith('.tar.gz') or archive_path.lower().endswith('.tgz'):
                    format = 'gztar'
                elif archive_path.lower().endswith('.tar.bz2') or archive_path.lower().endswith('.tbz2'):
                    format = 'bztar'
                elif archive_path.lower().endswith('.tar'):
                    format = 'tar'
                else:
                    # Try to auto-detect
                    format = 'auto'
            
            # Extract archive
            shutil.unpack_archive(archive_path, extract_dir, format)
            
            logger.debug(f"Extracted {archive_path} to {extract_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Cannot extract archive {archive_path}: {e}")
            return False
    
    def create_archive(self, source_dir: str, archive_path: str,
                      format: str = 'zip') -> bool:

        try:
            # Create parent directory if needed
            archive_dir = os.path.dirname(archive_path)
            if archive_dir and not os.path.exists(archive_dir):
                os.makedirs(archive_dir, exist_ok=True)
            
            # Create archive
            shutil.make_archive(
                os.path.splitext(archive_path)[0],
                format,
                source_dir
            )
            
            logger.debug(f"Created archive {archive_path} from {source_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Cannot create archive {archive_path}: {e}")
            return False
    
    def compare_files(self, file1: str, file2: str,
                     compare_size: bool = True,
                     compare_hash: bool = True,
                     algorithm: str = 'md5') -> Tuple[bool, Dict[str, Any]]:
        result = {
            'are_equal': False,
            'file1_exists': os.path.exists(file1),
            'file2_exists': os.path.exists(file2),
            'size_equal': None,
            'hash_equal': None,
            'file1_size': 0,
            'file2_size': 0,
            'file1_hash': '',
            'file2_hash': ''
        }
        
        # Check if both files exist
        if not result['file1_exists'] or not result['file2_exists']:
            return False, result
        
        # Get file sizes
        try:
            result['file1_size'] = os.path.getsize(file1)
            result['file2_size'] = os.path.getsize(file2)
            
            if compare_size:
                result['size_equal'] = (result['file1_size'] == result['file2_size'])
        except OSError:
            result['size_equal'] = False
        
        # Calculate file hashes
        if compare_hash:
            hash1 = self.calculate_hash(file1, algorithm)
            hash2 = self.calculate_hash(file2, algorithm)
            
            if hash1 and hash2:
                result['file1_hash'] = hash1
                result['file2_hash'] = hash2
                result['hash_equal'] = (hash1 == hash2)
            else:
                result['hash_equal'] = False
        
        # Determine if files are equal
        if compare_size and compare_hash:
            result['are_equal'] = (result['size_equal'] and result['hash_equal'])
        elif compare_size:
            result['are_equal'] = result['size_equal']
        elif compare_hash:
            result['are_equal'] = result['hash_equal']
        else:
            # If neither size nor hash comparison is requested,
            # files are considered equal if both exist
            result['are_equal'] = True
        
        return result['are_equal'], result


# Global file operations instance
_global_file_ops = None

def get_file_operations() -> FileOperations:
    """Get global file operations instance"""
    global _global_file_ops
    if _global_file_ops is None:
        _global_file_ops = FileOperations()
    return _global_file_ops


# Convenience functions
def get_file_hash(file_path: str, algorithm: str = 'md5') -> Optional[str]:
    """Get file hash"""
    ops = get_file_operations()
    return ops.calculate_hash(file_path, algorithm)

def find_gta_files(directory: str, recursive: bool = True) -> Dict[str, List[str]]:
    """Find GTA files in directory"""
    ops = get_file_operations()
    return ops.find_gta_files(directory, recursive)

def validate_gta_file(file_path: str) -> Tuple[bool, List[str]]:
    """Validate GTA file"""
    validator = FileValidator()
    return validator.validate_gta_file(file_path)

def copy_file_safe(src: str, dst: str, overwrite: bool = True) -> bool:
    """Safely copy a file"""
    ops = get_file_operations()
    return ops.copy_file(src, dst, overwrite)

def read_json_safe(file_path: str) -> Optional[Any]:
    """Safely read JSON file"""
    ops = get_file_operations()
    return ops.read_json_file(file_path)

def write_json_safe(file_path: str, data: Any, indent: int = 2) -> bool:
    """Safely write JSON file"""
    ops = get_file_operations()
    return ops.write_json_file(file_path, data, indent)


if __name__ == "__main__":
    # Test the file utilities
    import sys
    
    # Setup logging
    logging.basicConfig(level=logging.DEBUG)
    
    print("Testing File Utilities")
    print("=" * 50)
    
    # Create test directory
    test_dir = "./test_file_utils"
    ops = get_file_operations()
    
    if ops.create_directory(test_dir):
        print(f"Created test directory: {test_dir}")
        
        # Create test files
        test_file1 = os.path.join(test_dir, "test1.txt")
        test_file2 = os.path.join(test_dir, "test2.txt")
        
        ops.write_text_file(test_file1, "Hello, World!")
        ops.write_text_file(test_file2, "Hello, World!")
        
        # Test file info
        info = ops.get_file_info(test_file1)
        if info:
            print(f"\nFile Info for {test_file1}:")
            print(f"  Size: {info.get_human_size()}")
            print(f"  Modified: {time.ctime(info.modified_time)}")
            print(f"  Extension: {info.extension}")
        
        # Test directory info
        dir_info = ops.get_directory_info(test_dir)
        if dir_info:
            print(f"\nDirectory Info for {test_dir}:")
            print(f"  Files: {dir_info.file_count}")
            print(f"  Dirs: {dir_info.dir_count}")
            print(f"  Total Size: {dir_info.get_human_size()}")
        
        # Test file hash
        hash1 = ops.calculate_hash(test_file1)
        hash2 = ops.calculate_hash(test_file2)
        print(f"\nFile Hashes:")
        print(f"  {test_file1}: {hash1}")
        print(f"  {test_file2}: {hash2}")
        
        # Test file comparison
