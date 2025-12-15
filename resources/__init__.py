import os
import json
from pathlib import Path
from typing import Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)

class ResourceManager:
    """Manages application resources"""
    
    def __init__(self, resource_dir: Optional[str] = None):
        """
        Initialize resource manager
        
        Args:
            resource_dir: Path to resources directory (defaults to package resources)
        """
        self.resource_dir = self._get_resource_dir(resource_dir)
        self._icons: Dict[str, str] = {}
        self._styles: Dict[str, str] = {}
        self._loaded = False
        
    def _get_resource_dir(self, resource_dir: Optional[str]) -> Path:
        """Determine the resource directory path"""
        if resource_dir:
            return Path(resource_dir)
        
        # Default to package resources
        package_dir = Path(__file__).parent
        return package_dir

    def load_resources(self) -> bool:
        """Load all resources from the resource directory"""
        if self._loaded:
            return True
            
        try:
            # Load icons
            icons_dir = self.resource_dir / "icons"
            if icons_dir.exists():
                for icon_file in icons_dir.glob("*.*"):
                    self._icons[icon_file.stem] = str(icon_file)
            
            # Load styles
            styles_dir = self.resource_dir / "styles"
            if styles_dir.exists():
                for style_file in styles_dir.glob("*.qss"):
                    self._styles[style_file.stem] = style_file.read_text(encoding="utf-8")
            
            self._loaded = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to load resources: {e}")
            return False
    
    def get_icon(self, name: str) -> Optional[str]:
        """Get path to an icon by name"""
        return self._icons.get(name)
    
    def get_style(self, name: str) -> Optional[str]:
        """Get style sheet string by name"""
        return self._styles.get(name)
    
    def get_resource(self, relative_path: str) -> Optional[Path]:
        """Get any resource by relative path"""
        path = self.resource_dir / relative_path
        return path if path.exists() else None
    
    def load_json(self, relative_path: str) -> Optional[Dict[str, Any]]:
        """Load a JSON resource file"""
        try:
            path = self.resource_dir / relative_path
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load JSON resource {relative_path}: {e}")
        return None


# ✅ Global instance accessor pattern

_global_resource_manager = None

def get_resource_manager() -> ResourceManager:
    """Get the global resource manager instance"""
    global _global_resource_manager
    if _global_resource_manager is None:
        _global_resource_manager = ResourceManager()
        _global_resource_manager.load_resources()
    return _global_resource_manager

def get_icon(name: str) -> Optional[str]:
    """Get icon path by name (convenience function)"""
    return get_resource_manager().get_icon(name)

def get_style(name: str) -> Optional[str]:
    """Get style by name (convenience function)"""
    return get_resource_manager().get_style(name)

def load_json_resource(relative_path: str) -> Optional[Dict[str, Any]]:
    """Load JSON resource (convenience function)"""
    return get_resource_manager().load_json(relative_path)

# Default exports
__all__ = [
    'ResourceManager',
    'get_resource_manager',
    'get_icon',
    'get_style',
    'load_json_resource'
]
