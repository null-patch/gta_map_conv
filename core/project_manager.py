"""
GTA SA Map Converter - Project Manager
Handles saving, loading, and managing conversion projects
"""

import json
import os
import shutil
import pickle
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
import zipfile
import logging

# Local imports
try:
    from config import Config
    from utils.file_utils import FileUtils
    from utils.logger import Logger
except ImportError:
    # Dummy classes for development
    class Config:
        def __init__(self): pass
    class FileUtils:
        def __init__(self): pass
        @staticmethod
        def get_file_hash(path): return ""
    class Logger:
        def __init__(self): pass

logger = logging.getLogger(__name__)


@dataclass
class ProjectInfo:
    """Information about a conversion project"""
    name: str
    description: str = ""
    author: str = ""
    created_date: str = ""
    modified_date: str = ""
    version: str = "1.0.0"
    
    # Project state
    is_complete: bool = False
    is_archived: bool = False
    conversion_stage: str = "not_started"  # not_started, parsing, converting, exporting, complete
    
    # Statistics
    total_files: int = 0
    processed_files: int = 0
    total_objects: int = 0
    processed_objects: int = 0
    
    # Performance
    estimated_size_mb: float = 0.0
    actual_size_mb: float = 0.0
    processing_time_seconds: float = 0.0
    
    # Tags and categories
    tags: List[str] = field(default_factory=list)
    category: str = "map"
    subcategory: str = "full_map"  # full_map, city_area, building, interior, etc.
    
    def get_age_days(self) -> float:
        """Get project age in days"""
        if not self.created_date:
            return 0.0
        try:
            created = datetime.fromisoformat(self.created_date.replace('Z', '+00:00'))
            now = datetime.now()
            return (now - created).total_seconds() / 86400
        except:
            return 0.0
    
    def is_recent(self, days: int = 7) -> bool:
        """Check if project was created recently"""
        return self.get_age_days() <= days
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProjectInfo':
        """Create from dictionary"""
        return cls(**data)


@dataclass
class ProjectState:
    """Current state of a conversion project"""
    # File paths
    ide_files: List[str] = field(default_factory=list)
    ipl_files: List[str] = field(default_factory=list)
    img_archives: List[str] = field(default_factory=list)
    output_directory: str = ""
    
    # Processing state
    parsed_objects: Dict[int, Dict] = field(default_factory=dict)
    parsed_placements: List[Dict] = field(default_factory=list)
    extracted_files: Dict[str, str] = field(default_factory=dict)  # filename -> path
    converted_models: Dict[str, Dict] = field(default_factory=dict)
    converted_textures: Dict[str, Dict] = field(default_factory=dict)
    
    # Cache information
    cache_entries: Dict[str, str] = field(default_factory=dict)  # cache_key -> hash
    cache_valid: bool = True
    
    # Progress
    current_step: str = ""
    step_progress: float = 0.0
    total_progress: float = 0.0
    
    # Errors and warnings
    errors: List[Dict] = field(default_factory=list)
    warnings: List[Dict] = field(default_factory=list)
    
    def update_progress(self, step: str, step_progress: float, total_progress: float):
        """Update progress state"""
        self.current_step = step
        self.step_progress = step_progress
        self.total_progress = total_progress
    
    def add_error(self, error_type: str, message: str, file: str = "", line: int = 0):
        """Add an error to the state"""
        self.errors.append({
            'type': error_type,
            'message': message,
            'file': file,
            'line': line,
            'timestamp': datetime.now().isoformat()
        })
    
    def add_warning(self, warning_type: str, message: str, file: str = "", line: int = 0):
        """Add a warning to the state"""
        self.warnings.append({
            'type': warning_type,
            'message': message,
            'file': file,
            'line': line,
            'timestamp': datetime.now().isoformat()
        })
    
    def clear_errors(self):
        """Clear all errors and warnings"""
        self.errors.clear()
        self.warnings.clear()
    
    def get_error_count(self) -> int:
        """Get total error count"""
        return len(self.errors)
    
    def get_warning_count(self) -> int:
        """Get total warning count"""
        return len(self.warnings)
    
    def is_healthy(self) -> bool:
        """Check if project state is healthy (no critical errors)"""
        critical_errors = [e for e in self.errors if e['type'] in ['critical', 'fatal']]
        return len(critical_errors) == 0


class ProjectManager:
    """Manages conversion projects including saving, loading, and organization"""
    
    # Project file extensions
    PROJECT_EXTENSION = ".gtaproject"
    ARCHIVE_EXTENSION = ".gtazip"
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.file_utils = FileUtils()
        self.logger = Logger()
        
        # Project directories
        self.projects_dir = Path.home() / "GTA_SA_Projects"
        self.templates_dir = self.projects_dir / "templates"
        self.backup_dir = self.projects_dir / "backups"
        
        # Current project
        self.current_project: Optional[ProjectInfo] = None
        self.current_state: Optional[ProjectState] = None
        self.project_path: Optional[Path] = None
        
        # Project cache
        self.project_cache: Dict[str, ProjectInfo] = {}
        self.project_states: Dict[str, ProjectState] = {}
        
        # Initialize directories
        self._initialize_directories()
        
        # Load project cache
        self._load_project_cache()
    
    def _initialize_directories(self):
        """Initialize project directories"""
        self.projects_dir.mkdir(parents=True, exist_ok=True)
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_project_cache(self):
        """Load project cache from disk"""
        cache_file = self.projects_dir / "project_cache.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                
                for project_id, project_data in cache_data.items():
                    try:
                        project_info = ProjectInfo.from_dict(project_data)
                        self.project_cache[project_id] = project_info
                    except Exception as e:
                        logger.warning(f"Failed to load project {project_id}: {e}")
                        
            except Exception as e:
                logger.error(f"Failed to load project cache: {e}")
    
    def _save_project_cache(self):
        """Save project cache to disk"""
        cache_file = self.projects_dir / "project_cache.json"
        
        try:
            cache_data = {
                project_id: project_info.to_dict()
                for project_id, project_info in self.project_cache.items()
            }
            
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save project cache: {e}")
    
    def create_project(self, name: str, description: str = "", 
                      category: str = "map", tags: List[str] = None) -> str:
        """
        Create a new project
        
        Args:
            name: Project name
            description: Project description
            category: Project category
            tags: List of tags
            
        Returns:
            Project ID
        """
        # Generate project ID
        project_id = self._generate_project_id(name)
        
        # Create project info
        now = datetime.now().isoformat()
        project_info = ProjectInfo(
            name=name,
            description=description,
            created_date=now,
            modified_date=now,
            category=category,
            tags=tags or []
        )
        
        # Create project directory
        project_dir = self.projects_dir / project_id
        project_dir.mkdir(exist_ok=True)
        
        # Create project state
        project_state = ProjectState()
        
        # Save project
        self._save_project(project_id, project_info, project_state)
        
        # Update cache
        self.project_cache[project_id] = project_info
        self.project_states[project_id] = project_state
        
        # Set as current project
        self.current_project = project_info
        self.current_state = project_state
        self.project_path = project_dir / f"{name}{self.PROJECT_EXTENSION}"
        
        self.logger.info(f"Created new project: {name} ({project_id})")
        return project_id
    
    def _generate_project_id(self, name: str) -> str:
        """Generate unique project ID"""
        # Create base ID from name
        base_id = name.lower().replace(' ', '_').replace('-', '_')
        base_id = ''.join(c for c in base_id if c.isalnum() or c == '_')
        
        # Add timestamp for uniqueness
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        project_id = f"{base_id}_{timestamp}"
        
        return project_id
    
    def load_project(self, project_id_or_path: str) -> bool:
        """
        Load a project by ID or path
        
        Args:
            project_id_or_path: Project ID or file path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            project_path = Path(project_id_or_path)
            
            if project_path.is_file():
                # Load from file path
                return self._load_project_from_file(project_path)
            else:
                # Load from project ID
                return self._load_project_from_id(project_id_or_path)
                
        except Exception as e:
            self.logger.error(f"Failed to load project {project_id_or_path}: {e}")
            return False
    
    def _load_project_from_id(self, project_id: str) -> bool:
        """Load project by ID"""
        # Find project file
        project_dir = self.projects_dir / project_id
        project_files = list(project_dir.glob(f"*{self.PROJECT_EXTENSION}"))
        
        if not project_files:
            self.logger.error(f"No project file found for ID: {project_id}")
            return False
        
        return self._load_project_from_file(project_files[0])
    
    def _load_project_from_file(self, project_file: Path) -> bool:
        """Load project from file"""
        try:
            # Read project file
            with open(project_file, 'r') as f:
                project_data = json.load(f)
            
            # Extract project info
            info_data = project_data.get('info', {})
            project_info = ProjectInfo.from_dict(info_data)
            
            # Extract project state
            state_data = project_data.get('state', {})
            project_state = ProjectState()
            
            # Update state fields (simplified - would need proper deserialization)
            for key, value in state_data.items():
                if hasattr(project_state, key):
                    setattr(project_state, key, value)
            
            # Update current project
            self.current_project = project_info
            self.current_state = project_state
            self.project_path = project_file
            
            # Update cache
            project_id = project_file.parent.name
            self.project_cache[project_id] = project_info
            self.project_states[project_id] = project_state
            
            self.logger.info(f"Loaded project: {project_info.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load project file {project_file}: {e}")
            return False
    
    def save_current_project(self) -> bool:
        """Save current project"""
        if not self.current_project or not self.current_state:
            self.logger.error("No current project to save")
            return False
        
        # Update modification date
        self.current_project.modified_date = datetime.now().isoformat()
        
        # Save to file
        if self.project_path:
            return self._save_project_to_file(self.project_path)
        else:
            # Create new file
            project_id = self._generate_project_id(self.current_project.name)
            project_dir = self.projects_dir / project_id
            project_dir.mkdir(exist_ok=True)
            
            project_file = project_dir / f"{self.current_project.name}{self.PROJECT_EXTENSION}"
            self.project_path = project_file
            
            return self._save_project_to_file(project_file)
    
    def _save_project(self, project_id: str, project_info: ProjectInfo, project_state: ProjectState):
        """Save project data"""
        project_dir = self.projects_dir / project_id
        project_dir.mkdir(exist_ok=True)
        
        project_file = project_dir / f"{project_info.name}{self.PROJECT_EXTENSION}"
        
        # Prepare project data
        project_data = {
            'info': project_info.to_dict(),
            'state': asdict(project_state),
            'config': self.config.to_dict() if hasattr(self.config, 'to_dict') else {},
            'metadata': {
                'version': '1.0.0',
                'saved_at': datetime.now().isoformat(),
                'project_id': project_id
            }
        }
        
        # Save to file
        with open(project_file, 'w') as f:
            json.dump(project_data, f, indent=2, default=str)
        
        # Update cache
        self.project_cache[project_id] = project_info
        self.project_states[project_id] = project_state
        
        self.logger.info(f"Saved project: {project_info.name}")
    
    def _save_project_to_file(self, project_file: Path) -> bool:
        """Save current project to file"""
        try:
            # Prepare project data
            project_data = {
                'info': self.current_project.to_dict(),
                'state': asdict(self.current_state),
                'config': self.config.to_dict() if hasattr(self.config, 'to_dict') else {},
                'metadata': {
                    'version': '1.0.0',
                    'saved_at': datetime.now().isoformat(),
                    'project_id': project_file.parent.name
                }
            }
            
            # Save to file
            with open(project_file, 'w') as f:
                json.dump(project_data, f, indent=2, default=str)
            
            self.logger.info(f"Saved project to: {project_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save project: {e}")
            return False
    
    def delete_project(self, project_id: str, keep_backup: bool = True) -> bool:
        """
        Delete a project
        
        Args:
            project_id: Project ID to delete
            keep_backup: Whether to keep a backup
            
        Returns:
            True if successful, False otherwise
        """
        try:
            project_dir = self.projects_dir / project_id
            
            if not project_dir.exists():
                self.logger.warning(f"Project directory not found: {project_id}")
                return False
            
            # Create backup if requested
            if keep_backup:
                backup_path = self.backup_dir / f"{project_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                shutil.copytree(project_dir, backup_path)
                self.logger.info(f"Created backup at: {backup_path}")
            
            # Remove from cache
            if project_id in self.project_cache:
                del self.project_cache[project_id]
            if project_id in self.project_states:
                del self.project_states[project_id]
            
            # Delete directory
            shutil.rmtree(project_dir)
            
            # Save updated cache
            self._save_project_cache()
            
            self.logger.info(f"Deleted project: {project_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete project {project_id}: {e}")
            return False
    
    def list_projects(self, filter_tags: List[str] = None, 
                     filter_category: str = None) -> List[ProjectInfo]:
        """
        List all projects with optional filtering
        
        Args:
            filter_tags: List of tags to filter by
            filter_category: Category to filter by
            
        Returns:
            List of project info
        """
        projects = []
        
        # Scan project directories
        for project_dir in self.projects_dir.iterdir():
            if project_dir.is_dir() and project_dir.name not in ['templates', 'backups']:
                # Look for project file
                project_files = list(project_dir.glob(f"*{self.PROJECT_EXTENSION}"))
                
                if project_files:
                    try:
                        # Load project info
                        with open(project_files[0], 'r') as f:
                            project_data = json.load(f)
                        
                        project_info = ProjectInfo.from_dict(project_data.get('info', {}))
                        
                        # Apply filters
                        if filter_category and project_info.category != filter_category:
                            continue
                            
                        if filter_tags and not any(tag in project_info.tags for tag in filter_tags):
                            continue
                        
                        projects.append(project_info)
                        
                    except Exception as e:
                        logger.warning(f"Failed to read project in {project_dir}: {e}")
        
        # Sort by modification date (newest first)
        projects.sort(key=lambda p: p.modified_date, reverse=True)
        
        return projects
    
    def get_project_stats(self, project_id: str) -> Dict[str, Any]:
        """
        Get statistics for a project
        
        Args:
            project_id: Project ID
            
        Returns:
            Project statistics
        """
        if project_id not in self.project_cache:
            return {}
        
        project_info = self.project_cache[project_id]
        project_state = self.project_states.get(project_id, ProjectState())
        
        stats = {
            'project_info': project_info.to_dict(),
            'file_counts': {
                'ide_files': len(project_state.ide_files),
                'ipl_files': len(project_state.ipl_files),
                'img_archives': len(project_state.img_archives),
                'extracted_files': len(project_state.extracted_files),
                'converted_models': len(project_state.converted_models),
                'converted_textures': len(project_state.converted_textures)
            },
            'progress': {
                'current_step': project_state.current_step,
                'step_progress': project_state.step_progress,
                'total_progress': project_state.total_progress
            },
            'issues': {
                'errors': project_state.get_error_count(),
                'warnings': project_state.get_warning_count()
            },
            'size_estimate': project_info.estimated_size_mb,
            'processing_time': project_info.processing_time_seconds
        }
        
        return stats
    
    def create_project_archive(self, project_id: str, 
                              include_source_files: bool = False) -> Optional[Path]:
        """
        Create a compressed archive of the project
        
        Args:
            project_id: Project ID
            include_source_files: Whether to include source GTA files
            
        Returns:
            Path to archive file or None
        """
        try:
            project_dir = self.projects_dir / project_id
            
            if not project_dir.exists():
                self.logger.error(f"Project directory not found: {project_id}")
                return None
            
            # Create archive filename
            archive_name = f"{project_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{self.ARCHIVE_EXTENSION}"
            archive_path = self.projects_dir / archive_name
            
            # Create zip archive
            with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Add project files
                for file_path in project_dir.rglob('*'):
                    if file_path.is_file():
                        arcname = file_path.relative_to(project_dir)
                        zipf.write(file_path, arcname)
            
            # Calculate archive size
            archive_size_mb = archive_path.stat().st_size / (1024 * 1024)
            
            self.logger.info(f"Created project archive: {archive_path} ({archive_size_mb:.2f} MB)")
            return archive_path
            
        except Exception as e:
            self.logger.error(f"Failed to create project archive: {e}")
            return None
    
    def restore_project_from_archive(self, archive_path: Path) -> Optional[str]:
        """
        Restore a project from an archive
        
        Args:
            archive_path: Path to archive file
            
        Returns:
            Project ID or None
        """
        try:
            # Extract archive to temporary directory
            temp_dir = self.projects_dir / "temp_restore"
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            temp_dir.mkdir()
            
            # Extract files
            with zipfile.ZipFile(archive_path, 'r') as zipf:
                zipf.extractall(temp_dir)
            
            # Find project directory in archive
            project_dirs = list(temp_dir.iterdir())
            if not project_dirs:
                self.logger.error("No project found in archive")
                return None
            
            # Use first directory as project
            extracted_dir = project_dirs[0]
            project_id = extracted_dir.name
            
            # Check if project already exists
            target_dir = self.projects_dir / project_id
            if target_dir.exists():
                # Create backup of existing project
                backup_dir = self.backup_dir / f"{project_id}_before_restore"
                if backup_dir.exists():
                    shutil.rmtree(backup_dir)
                shutil.copytree(target_dir, backup_dir)
                
                # Remove existing project
                shutil.rmtree(target_dir)
            
            # Move to projects directory
            shutil.move(str(extracted_dir), str(target_dir))
            
            # Clean up temp directory
            shutil.rmtree(temp_dir)
            
            # Load the restored project
            if self.load_project(project_id):
                self.logger.info(f"Restored project from archive: {project_id}")
                return project_id
            else:
                self.logger.error(f"Failed to load restored project: {project_id}")
                return None
            
        except Exception as e:
            self.logger.error(f"Failed to restore project from archive: {e}")
            return None
    
    def create_project_template(self, project_id: str, template_name: str, 
                               description: str = "") -> bool:
        """
        Create a template from an existing project
        
        Args:
            project_id: Source project ID
            template_name: Template name
            description: Template description
            
        Returns:
            True if successful, False otherwise
        """
        try:
            source_dir = self.projects_dir / project_id
            
            if not source_dir.exists():
                self.logger.error(f"Source project not found: {project_id}")
                return False
            
            # Create template directory
            template_id = template_name.lower().replace(' ', '_')
            template_dir = self.templates_dir / template_id
            template_dir.mkdir(exist_ok=True)
            
            # Copy project configuration (excluding data files)
            for file_path in source_dir.glob(f"*{self.PROJECT_EXTENSION}"):
                if file_path.is_file():
                    shutil.copy2(file_path, template_dir)
            
            # Create template info file
            template_info = {
                'name': template_name,
                'description': description,
                'source_project': project_id,
                'created_at': datetime.now().isoformat(),
                'version': '1.0.0'
            }
            
            info_file = template_dir / "template_info.json"
            with open(info_file, 'w') as f:
                json.dump(template_info, f, indent=2)
            
            self.logger.info(f"Created template: {template_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create template: {e}")
            return False
    
    def create_project_from_template(self, template_id: str, project_name: str, 
                                    description: str = "") -> Optional[str]:
        """
        Create a new project from a template
        
        Args:
            template_id: Template ID
            project_name: New project name
            description: Project description
            
        Returns:
            New project ID or None
        """
        try:
            template_dir = self.templates_dir / template_id
            
            if not template_dir.exists():
                self.logger.error(f"Template not found: {template_id}")
                return None
            
            # Create new project
            project_id = self.create_project(project_name, description)
            
            if not project_id:
                return None
            
            project_dir = self.projects_dir / project_id
            
            # Copy template files to new project
            for file_path in template_dir.glob("*"):
                if file_path.is_file() and file_path.name != "template_info.json":
                    shutil.copy2(file_path, project_dir)
            
            self.logger.info(f"Created project from template {template_id}: {project_name}")
            return project_id
            
        except Exception as e:
            self.logger.error(f"Failed to create project from template: {e}")
            return None
    
    def cleanup_old_backups(self, max_age_days: int = 30, 
                           max_backups: int = 10) -> int:
        """
        Clean up old backup files
        
        Args:
            max_age_days: Maximum age in days
            max_backups: Maximum number of backups to keep
            
        Returns:
            Number of backups deleted
        """
        try:
            if not self.backup_dir.exists():
                return 0
            
            deleted_count = 0
            now = datetime.now()
            
            # Get all backup directories
            backups = []
            for backup_dir in self.backup_dir.iterdir():
                if backup_dir.is_dir():
                    # Try to parse creation time from directory name
                    try:
                        parts = backup_dir.name.split('_')
                        if len(parts) >= 2:
                            timestamp_str = parts[-2] + '_' + parts[-1]
                            created_time = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                            age_days = (now - created_time).days
                            
                            backups.append((backup_dir, created_time, age_days))
                    except:
                        # If can't parse timestamp, use directory modification time
                        mtime = datetime.fromtimestamp(backup_dir.stat().st_mtime)
                        age_days = (now - mtime).days
                        backups.append((backup_dir, mtime, age_days))
            
            # Sort by creation time (oldest first)
            backups.sort(key=lambda x: x[1])
            
            # Delete old backups
            for backup_dir, created_time, age_days in backups:
                if age_days > max_age_days or len(backups) - deleted_count > max_backups:
                    try:
                        shutil.rmtree(backup_dir)
                        deleted_count += 1
                        self.logger.info(f"Deleted old backup: {backup_dir.name}")
                    except Exception as e:
                        self.logger.warning(f"Failed to delete backup {backup_dir.name}: {e}")
            
            return deleted_count
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup backups: {e}")
            return 0
    
    def validate_project(self, project_id: str) -> Dict[str, Any]:
        """
        Validate project integrity
        
        Args:
            project_id: Project ID
            
        Returns:
            Validation results
        """
        results = {
            'valid': True,
            'issues': [],
            'missing_files': [],
            'corrupted_files': []
        }
        
        try:
            project_dir = self.projects_dir / project_id
            
            if not project_dir.exists():
                results['valid'] = False
                results['issues'].append("Project directory not found")
                return results
            
            # Check for project file
            project_files = list(project_dir.glob(f"*{self.PROJECT_EXTENSION}"))
            if not project_files:
                results['valid'] = False
                results['issues'].append("Project file not found")
                return results
            
            # Validate project file
            project_file = project_files[0]
            try:
                with open(project_file, 'r') as f:
                    project_data = json.load(f)
                
                # Check required fields
                required_fields = ['info', 'state', 'metadata']
                for field in required_fields:
                    if field not in project_data:
                        results['valid'] = False
                        results['issues'].append(f"Missing required field: {field}")
                
            except Exception as e:
                results['valid'] = False
                results['issues'].append(f"Invalid project file: {str(e)}")
            
            # Check referenced files
            if project_id in self.project_states:
                project_state = self.project_states[project_id]
                
                # Check extracted files
                for filename, filepath in project_state.extracted_files.items():
                    if not Path(filepath).exists():
                        results['missing_files'].append(filename)
                        results['valid'] = False
            
            # Check cache integrity
            cache_file = project_dir / "cache_data.pkl"
            if cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        pickle.load(f)
                except Exception as e:
                    results['corrupted_files'].append("cache_data.pkl")
                    results['valid'] = False
            
        except Exception as e:
            results['valid'] = False
            results['issues'].append(f"Validation error: {str(e)}")
        
        return results


class ProjectBrowser:
    """GUI component for browsing and managing projects"""
    
    def __init__(self, project_manager: ProjectManager):
        self.project_manager = project_manager
        self.projects = []
        self.filtered_projects = []
        self.current_filter = ""
        self.current_category = ""
        self.current_tags = []
    
    def refresh_projects(self):
        """Refresh project list"""
        self.projects = self.project_manager.list_projects()
        self.filtered_projects = self.projects.copy()
    
    def filter_projects(self, search_text: str = "", category: str = "", 
                       tags: List[str] = None):
        """Filter projects based on criteria"""
        self.current_filter = search_text.lower()
        self.current_category = category
        self.current_tags = tags or []
        
        self.filtered_projects = [
            p for p in self.projects
            if self._matches_filter(p)
        ]
    
    def _matches_filter(self, project: ProjectInfo) -> bool:
        """Check if project matches current filter"""
        # Text filter
        if self.current_filter:
            if (self.current_filter not in project.name.lower() and
                self.current_filter not in project.description.lower()):
                return False
        
        # Category filter
        if self.current_category and project.category != self.current_category:
            return False
        
        # Tags filter
        if self.current_tags:
            if not any(tag in project.tags for tag in self.current_tags):
                return False
        
        return True
    
    def get_project_categories(self) -> List[str]:
        """Get list of all project categories"""
        categories = set()
        for project in self.projects:
            categories.add(project.category)
        return sorted(categories)
    
    def get_project_tags(self) -> List[str]:
        """Get list of all project tags"""
        tags = set()
        for project in self.projects:
            tags.update(project.tags)
        return sorted(tags)


if __name__ == "__main__":
    # Test the project manager
    print("Testing Project Manager...")
    
    # Create config
    config = Config()
    
    # Create project manager
    manager = ProjectManager(config)
    
    # Create a test project
    project_id = manager.create_project(
        name="Test Project",
        description="A test project for development",
        category="test",
        tags=["test", "development"]
    )
    
    print(f"Created project: {project_id}")
    
    # List projects
    projects = manager.list_projects()
    print(f"Total projects: {len(projects)}")
    
    # Get project stats
    stats = manager.get_project_stats(project_id)
    print(f"Project stats: {stats}")
    
    # Create archive
    archive_path = manager.create_project_archive(project_id)
    if archive_path:
        print(f"Created archive: {archive_path}")
    
    # Cleanup
    manager.delete_project(project_id, keep_backup=False)
    print("Test completed")
