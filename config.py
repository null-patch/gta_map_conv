import json
import os
import sys
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any
import hashlib
import pickle

@dataclass
class PathConfig:
    """Configuration for file paths"""
    img_dir: str = ""
    maps_dir: str = ""
    output_dir: str = ""
    temp_dir: str = "/tmp/gta_sa_converter"
    log_dir: str = "logs"
    
    def validate(self) -> List[str]:
        """Validate all paths, return list of errors"""
        errors = []
        
        if self.img_dir and not os.path.exists(self.img_dir):
            errors.append(f"IMG directory does not exist: {self.img_dir}")
            
        if self.maps_dir and not os.path.exists(self.maps_dir):
            errors.append(f"Maps directory does not exist: {self.maps_dir}")
            
        # Create output directory if it doesn't exist
        if self.output_dir:
            try:
                os.makedirs(self.output_dir, exist_ok=True)
            except Exception as e:
                errors.append(f"Cannot create output directory: {str(e)}")
                
        # Create temp directory if needed
        try:
            os.makedirs(self.temp_dir, exist_ok=True)
        except Exception as e:
            errors.append(f"Cannot create temp directory: {str(e)}")
            
        return errors
    
    def get_img_files(self) -> List[str]:
        """Get list of IMG files in img_dir (full paths)"""
        if not self.img_dir or not os.path.exists(self.img_dir):
            return []
        
        return [
            os.path.join(self.img_dir, f)
            for f in os.listdir(self.img_dir)
            if f.lower().endswith(".img")
        ]
                
    def get_ide_files(self) -> List[str]:
        """Get list of IDE files in maps_dir"""
        if not self.maps_dir or not os.path.exists(self.maps_dir):
            return []
            
        ide_files = []
        for root, dirs, files in os.walk(self.maps_dir):
            for file in files:
                if file.lower().endswith('.ide'):
                    ide_files.append(os.path.join(root, file))
                    
        return ide_files
        
    def get_ipl_files(self) -> List[str]:
        """Get list of IPL files in maps_dir"""
        if not self.maps_dir or not os.path.exists(self.maps_dir):
            return []
            
        ipl_files = []
        for root, dirs, files in os.walk(self.maps_dir):
            for file in files:
                if file.lower().endswith('.ipl'):
                    ipl_files.append(os.path.join(root, file))
                    
        return ipl_files

@dataclass
class ConversionConfig:
    """Configuration for conversion settings"""
    # Scale and units
    scale_factor: float = 0.01
    coordinate_system: str = "y_up"  # "y_up" or "z_up"
    units: str = "meters"
    
    # Export options
    export_textures: bool = True
    texture_format: str = "PNG"
    texture_quality: int = 90  # 1-100
    texture_resize: str = "original"  # "original", "512", "1024", "2048"
    
    # Mesh options
    combine_meshes: bool = True
    generate_lods: bool = True
    generate_normals: bool = True
    generate_uvs: bool = True
    
    # Optimization
    remove_duplicate_vertices: bool = True
    merge_close_vertices: bool = True
    merge_distance: float = 0.001
    triangulate_faces: bool = True
    
    # Object grouping
    group_by_ipl: bool = True
    group_by_ide: bool = False
    group_by_material: bool = False
    
    def validate(self) -> List[str]:
        """Validate conversion settings"""
        errors = []
        
        if self.scale_factor <= 0:
            errors.append("Scale factor must be greater than 0")
            
        if self.coordinate_system not in ["y_up", "z_up"]:
            errors.append("Coordinate system must be 'y_up' or 'z_up'")
            
        if self.texture_format.upper() not in ["PNG", "JPEG", "JPG", "TGA", "BMP"]:
            errors.append(f"Unsupported texture format: {self.texture_format}")
            
        if not 1 <= self.texture_quality <= 100:
            errors.append("Texture quality must be between 1 and 100")
            
        if self.texture_resize not in ["original", "512", "1024", "2048"]:
            errors.append("Invalid texture resize option")
            
        if self.merge_distance < 0:
            errors.append("Merge distance cannot be negative")
            
        return errors
    
    def get_texture_extension(self) -> str:
        """Get file extension for texture format"""
        fmt = self.texture_format.lower()
        if fmt == "jpeg":
            return "jpg"
        return fmt
    
    def get_texture_size(self) -> Optional[tuple]:
        """Get texture size as tuple or None for original"""
        if self.texture_resize == "original":
            return None
        elif self.texture_resize == "512":
            return (512, 512)
        elif self.texture_resize == "1024":
            return (1024, 1024)
        elif self.texture_resize == "2048":
            return (2048, 2048)
        return None


@dataclass
class PerformanceConfig:
    """Configuration for performance settings"""
    keep_temp_files: bool = False
    # Threading
    use_multithreading: bool = True
    max_threads: int = 4
    thread_pool_size: int = 8
    
    # Memory management
    max_memory_mb: int = 4096  # 4GB
    use_disk_cache: bool = True
    cache_dir: str = "/tmp/gta_sa_cache"
    
    # Batch processing
    batch_size: int = 100
    use_incremental_saving: bool = True
    
    # Progress updates
    update_interval_ms: int = 100
    log_level: str = "INFO"  # "DEBUG", "INFO", "WARNING", "ERROR"
    
    def validate(self) -> List[str]:
        """Validate performance settings"""
        errors = []
        
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        
        if self.max_threads < 1:
            errors.append("Maximum threads must be at least 1")
        elif self.max_threads > cpu_count * 2:
            errors.append(f"Maximum threads ({self.max_threads}) exceeds reasonable limit ({cpu_count * 2})")
            
        if self.max_memory_mb < 256:
            errors.append("Maximum memory must be at least 256MB")
            
        if self.batch_size < 1:
            errors.append("Batch size must be at least 1")
            
        if self.update_interval_ms < 10:
            errors.append("Update interval must be at least 10ms")
            
        if self.log_level not in ["DEBUG", "INFO", "WARNING", "ERROR"]:
            errors.append("Invalid log level")
            
        return errors
    
    def get_optimal_thread_count(self) -> int:
        """Get optimal thread count based on system"""
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        
        if not self.use_multithreading:
            return 1
            
        return min(self.max_threads, max(1, cpu_count - 1))


@dataclass
class BlenderConfig:
    """Configuration for Blender-specific settings"""
    # Blender version compatibility
    blender_version: str = "2.79"
    apply_scale_fix: bool = True
    flip_normals: bool = False
    flip_uv_vertical: bool = True
    
    # Material settings
    create_materials: bool = True
    use_node_materials: bool = False  # Blender 2.79 uses simple materials
    material_prefix: str = "gta_"
    
    # Object naming
    use_object_ids: bool = True
    object_prefix: str = "obj_"
    group_prefix: str = "grp_"
    
    # Export settings
    export_selected_only: bool = False
    export_animation: bool = False
    export_custom_properties: bool = False
    
    def validate(self) -> List[str]:
        """Validate Blender settings"""
        errors = []
        
        # Check Blender version format
        try:
            major, minor = map(int, self.blender_version.split('.'))
            if major != 2 or minor != 79:
                errors.append("Only Blender 2.79 is officially supported")
        except ValueError:
            errors.append("Invalid Blender version format")
            
        return errors


@dataclass
class UIConfig:
    """Configuration for UI settings"""
    # Window state
    window_width: int = 900
    window_height: int = 750
    window_maximized: bool = False
    window_position_x: int = 100
    window_position_y: int = 100
    
    # UI appearance
    theme: str = "dark"  # "dark", "light", "system"
    font_size: int = 10
    show_tooltips: bool = True
    show_status_bar: bool = True
    
    # Recent files
    recent_projects: List[str] = field(default_factory=list)
    max_recent_projects: int = 10
    
    # Last used directories
    last_img_dir: str = ""
    last_maps_dir: str = ""
    last_output_dir: str = ""
    
    def validate(self) -> List[str]:
        """Validate UI settings"""
        errors = []
        
        if self.window_width < 400:
            errors.append("Window width too small")
        if self.window_height < 300:
            errors.append("Window height too small")
        if self.font_size < 8 or self.font_size > 24:
            errors.append("Font size out of range")
        if self.theme not in ["dark", "light", "system"]:
            errors.append("Invalid theme")
            
        return errors
    
    def add_recent_project(self, project_path: str):
        """Add project to recent projects list"""
        if project_path in self.recent_projects:
            self.recent_projects.remove(project_path)
            
        self.recent_projects.insert(0, project_path)
        
        # Limit list size
        self.recent_projects = self.recent_projects[:self.max_recent_projects]
    
    def get_valid_recent_projects(self) -> List[str]:
        """Get list of recent projects that still exist"""
        valid = []
        for project in self.recent_projects:
            if os.path.exists(project):
                valid.append(project)
        return valid


class Config:
    """Main configuration manager"""
    
    def __init__(self):
        # Initialize sub-configs
        self.paths = PathConfig()
        self.conversion = ConversionConfig()
        self.performance = PerformanceConfig()
        self.blender = BlenderConfig()
        self.ui = UIConfig()
        
        # Configuration directories
        self.config_dir = Path.home() / ".config" / "gta_map_converter"
        self.project_dir = Path.home() / "GTA_SA_Projects"
        
        # Default config file
        self.config_file = self.config_dir / "settings.json"
        
        # Current project
        self.current_project: Optional[str] = None
        self.project_modified: bool = False
        
        # Initialize
        self._create_directories()
        self.load()
        
    def _create_directories(self):
        """Create necessary directories"""
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.project_dir.mkdir(parents=True, exist_ok=True)
        
        # Create cache directory
        cache_dir = Path(self.performance.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        
    def validate_all(self) -> Dict[str, List[str]]:
        """Validate all configuration sections"""
        errors = {
            'paths': self.paths.validate(),
            'conversion': self.conversion.validate(),
            'performance': self.performance.validate(),
            'blender': self.blender.validate(),
            'ui': self.ui.validate()
        }
        
        return errors
    
    def has_errors(self) -> bool:
        """Check if any configuration errors exist"""
        errors = self.validate_all()
        return any(errors.values())
    
    def get_error_summary(self) -> str:
        """Get formatted error summary"""
        errors = self.validate_all()
        
        if not self.has_errors():
            return "No configuration errors"
            
        summary = "Configuration Errors:\n"
        for section, section_errors in errors.items():
            if section_errors:
                summary += f"\n{section.upper()}:\n"
                for error in section_errors:
                    summary += f"  • {error}\n"
                    
        return summary
    
    def load(self, config_path: Optional[str] = None):
        """Load configuration from file"""
        path = config_path or self.config_file
        
        if not os.path.exists(path):
            self._set_defaults()
            return
            
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                
            # Load each section
            self._load_section(data, 'paths', self.paths)
            self._load_section(data, 'conversion', self.conversion)
            self._load_section(data, 'performance', self.performance)
            self._load_section(data, 'blender', self.blender)
            self._load_section(data, 'ui', self.ui)
            
            # Load project info
            self.current_project = data.get('current_project')
            self.project_modified = data.get('project_modified', False)
            
        except Exception as e:
            print(f"Error loading config: {e}")
            self._set_defaults()
    
    def _load_section(self, data: dict, section_name: str, section_obj):
        """Load a configuration section"""
        if section_name in data:
            section_data = data[section_name]
            
            # Update the dataclass fields
            for key, value in section_data.items():
                if hasattr(section_obj, key):
                    # Handle special cases
                    if key == 'recent_projects' and isinstance(value, list):
                        # Ensure recent projects are unique
                        section_obj.recent_projects = list(dict.fromkeys(value))
                    else:
                        setattr(section_obj, key, value)
    
    def save(self, config_path: Optional[str] = None):
        """Save configuration to file"""
        path = config_path or self.config_file
        
        try:
            # Prepare data dictionary
            data = {
                'paths': asdict(self.paths),
                'conversion': asdict(self.conversion),
                'performance': asdict(self.performance),
                'blender': asdict(self.blender),
                'ui': asdict(self.ui),
                'current_project': self.current_project,
                'project_modified': self.project_modified
            }
            
            # Ensure config directory exists
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Write to file
            with open(path, 'w') as f:
                json.dump(data, f, indent=4)
                
        except Exception as e:
            print(f"Error saving config: {e}")
            
    def _set_defaults(self):
        """Set default configuration values"""
        # Set default output directory
        if not self.paths.output_dir:
            default_output = Path.home() / "GTA_SA_Exports"
            default_output.mkdir(exist_ok=True)
            self.paths.output_dir = str(default_output)
            
        # Set default temp directory
        self.paths.temp_dir = "/tmp/gta_sa_converter"
        
        # Set default cache directory
        self.performance.cache_dir = "/tmp/gta_sa_cache"
        
    def load_project(self, project_path: str):
        """Load a project file"""
        try:
            with open(project_path, 'r') as f:
                data = json.load(f)
                
            # Update configuration from project
            self._load_section(data, 'paths', self.paths)
            self._load_section(data, 'conversion', self.conversion)
            self._load_section(data, 'performance', self.performance)
            self._load_section(data, 'blender', self.blender)
            
            # Set current project
            self.current_project = project_path
            self.project_modified = False
            
            # Add to recent projects
            self.ui.add_recent_project(project_path)
            
            # Save updated config
            self.save()
            
            return True
            
        except Exception as e:
            print(f"Error loading project: {e}")
            return False
    
    def save_project(self, project_path: Optional[str] = None):
        """Save current configuration as a project file"""
        path = project_path or self.current_project
        
        if not path:
            raise ValueError("No project path specified")
            
        try:
            # Prepare project data (exclude UI settings)
            project_data = {
                'paths': asdict(self.paths),
                'conversion': asdict(self.conversion),
                'performance': asdict(self.performance),
                'blender': asdict(self.blender),
                'project_info': {
                    'name': os.path.basename(path),
                    'created': self._get_timestamp(),
                    'modified': self._get_timestamp()
                }
            }
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Write project file
            with open(path, 'w') as f:
                json.dump(project_data, f, indent=4)
                
            # Update current project
            self.current_project = path
            self.project_modified = False
            
            # Add to recent projects
            self.ui.add_recent_project(path)
            
            # Save config
            self.save()
            
            return True
            
        except Exception as e:
            print(f"Error saving project: {e}")
            return False
    
    def create_project(self, name: str, description: str = "") -> str:
        """Create a new project"""
        # Generate project filename
        timestamp = self._get_timestamp().replace(':', '-').replace(' ', '_')
        filename = f"{name}_{timestamp}.gtaproject"
        project_path = str(self.project_dir / filename)
        
        # Create project data
        project_data = {
            'project_info': {
                'name': name,
                'description': description,
                'created': self._get_timestamp(),
                'modified': self._get_timestamp(),
                'author': os.getlogin()
            },
            'paths': asdict(self.paths),
            'conversion': asdict(self.conversion),
            'performance': asdict(self.performance),
            'blender': asdict(self.blender)
        }
        
        # Save project
        with open(project_path, 'w') as f:
            json.dump(project_data, f, indent=4)
            
        # Update config
        self.current_project = project_path
        self.ui.add_recent_project(project_path)
        self.save()
        
        return project_path
    
    def get_project_info(self) -> Dict[str, Any]:
        """Get information about current project"""
        if not self.current_project or not os.path.exists(self.current_project):
            return {}
            
        try:
            with open(self.current_project, 'r') as f:
                data = json.load(f)
                
            return data.get('project_info', {})
        except:
            return {}
    
    def mark_modified(self):
        """Mark project as modified"""
        self.project_modified = True
    
    def is_modified(self) -> bool:
        """Check if project has been modified"""
        return self.project_modified
    
    def get_cache_key(self, data_type: str, identifier: str) -> str:
        """Generate cache key for data"""
        # Create hash of relevant configuration
        config_hash = hashlib.md5()
        
        # Include conversion settings that affect the output
        config_data = {
            'scale': self.conversion.scale_factor,
            'coord_system': self.conversion.coordinate_system,
            'texture_format': self.conversion.texture_format,
            'texture_size': self.conversion.texture_resize
        }
        
        config_hash.update(json.dumps(config_data, sort_keys=True).encode())
        config_digest = config_hash.hexdigest()[:8]
        
        # Create cache key
        return f"{data_type}_{identifier}_{config_digest}"
    
    def get_cache_path(self, cache_key: str) -> Path:
        """Get path for cached data"""
        cache_dir = Path(self.performance.cache_dir)
        return cache_dir / f"{cache_key}.cache"
    
    def save_cache(self, cache_key: str, data: Any):
        """Save data to cache"""
        if not self.performance.use_disk_cache:
            return
            
        cache_path = self.get_cache_path(cache_key)
        
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"Warning: Failed to cache data: {e}")
    
    def load_cache(self, cache_key: str) -> Optional[Any]:
        """Load data from cache"""
        if not self.performance.use_disk_cache:
            return None
            
        cache_path = self.get_cache_path(cache_key)
        
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Warning: Failed to load cached data: {e}")
                
        return None
    
    def clear_cache(self):
        """Clear all cached data"""
        cache_dir = Path(self.performance.cache_dir)
        
        if cache_dir.exists():
            import shutil
            try:
                shutil.rmtree(cache_dir)
                cache_dir.mkdir(parents=True, exist_ok=True)
                return True
            except Exception as e:
                print(f"Error clearing cache: {e}")
                return False
                
        return True
    
    def _get_timestamp(self) -> str:
        """Get current timestamp as string"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'paths': asdict(self.paths),
            'conversion': asdict(self.conversion),
            'performance': asdict(self.performance),
            'blender': asdict(self.blender),
            'ui': asdict(self.ui),
            'current_project': self.current_project,
            'project_modified': self.project_modified
        }
    
    def from_dict(self, data: Dict[str, Any]):
        """Load configuration from dictionary"""
        self._load_section(data, 'paths', self.paths)
        self._load_section(data, 'conversion', self.conversion)
        self._load_section(data, 'performance', self.performance)
        self._load_section(data, 'blender', self.blender)
        self._load_section(data, 'ui', self.ui)
        
        if 'current_project' in data:
            self.current_project = data['current_project']
        if 'project_modified' in data:
            self.project_modified = data['project_modified']
    
    def reset_to_defaults(self):
        """Reset configuration to defaults"""
        self.paths = PathConfig()
        self.conversion = ConversionConfig()
        self.performance = PerformanceConfig()
        self.blender = BlenderConfig()
        self.ui = UIConfig()
        self.current_project = None
        self.project_modified = False
        
        self._set_defaults()


# Singleton instance
_config_instance = None

def get_config() -> Config:
    """Get global configuration instance"""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance


if __name__ == "__main__":
    # Test the configuration system
    config = Config()
    
    print("Configuration loaded successfully")
    print(f"Config file: {config.config_file}")
    print(f"Has errors: {config.has_errors()}")
    
    if config.has_errors():
        print(config.get_error_summary())
