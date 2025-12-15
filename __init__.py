"""
GTA SA Map Converter
Convert GTA San Andreas map files to Blender 2.79 compatible OBJ format

This package provides tools for parsing, converting, and exporting
GTA San Andreas game files for use in Blender 2.79.

Modules:
    main: Main application entry point
    gui: PyQt5-based user interface
    config: Configuration management
    core: Core conversion logic and pipeline
    converters: File format parsers (IDE, IPL, DFF, TXD, IMG)
    blender: Blender-specific export tools
    utils: Utility functions and helpers
"""

import os
import sys
from pathlib import Path

# Add the parent directory to sys.path for development
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Version information
__version__ = "1.0.0"
__author__ = "GTA Modding Community"
__license__ = "MIT"
__description__ = "GTA San Andreas Map to Blender Converter for Linux Mint"

# Import key classes for easier access
try:
    # Main application
    from .main import main, ConversionThread
    
    # GUI
    from .gui import MainWindow, DirectorySelector, ProgressWidget, LogWidget
    
    # Configuration
    from .config import Config, get_config
    
    # Core components
    from .core.conversion_pipeline import ConversionPipeline, ConversionStats
    
    # Converters
    from .converters.ide_parser import IDEParser, IDEManager, IDEObject
    from .converters.ipl_parser import IPLParser, IPLManager
    from .converters.dff_converter import DFFConverter, DFFManager
    from .converters.txd_converter import TXDConverter, TXDManager
    from .converters.img_archive import IMGExtractor
    
    # Blender export
    from .blender.obj_exporter import OBJExporter
    from .blender.material_builder import MaterialBuilder
    
    # Utilities
    from .utils.progress_tracker import ProgressTracker
    from .utils.logger import Logger
    from .utils.file_utils import FileUtils
    from .utils.error_handler import ErrorHandler
    from .utils.performance import PerformanceOptimizer
    
except ImportError as e:
    print(f"Warning: Could not import all modules: {e}")
    # Define dummy classes for missing imports
    class Dummy:
        def __init__(self, *args, **kwargs): pass
        def __getattr__(self, name): return lambda *args, **kwargs: None
    
    # Create dummy replacements for missing imports
    for name in [
        'main', 'ConversionThread', 'MainWindow', 'DirectorySelector',
        'ProgressWidget', 'LogWidget', 'Config', 'get_config',
        'ConversionPipeline', 'ConversionStats', 'IDEParser', 'IDEManager',
        'IDEObject', 'IPLParser', 'IPLManager', 'DFFConverter', 'DFFManager',
        'TXDConverter', 'TXDManager', 'IMGExtractor', 'OBJExporter',
        'MaterialBuilder', 'ProgressTracker', 'Logger', 'FileUtils',
        'ErrorHandler', 'PerformanceOptimizer'
    ]:
        if name not in locals():
            locals()[name] = Dummy

# Package metadata
__all__ = [
    # Main application
    'main', 'ConversionThread',
    
    # GUI
    'MainWindow', 'DirectorySelector', 'ProgressWidget', 'LogWidget',
    
    # Configuration
    'Config', 'get_config',
    
    # Core
    'ConversionPipeline', 'ConversionStats',
    
    # Converters
    'IDEParser', 'IDEManager', 'IDEObject',
    'IPLParser', 'IPLManager',
    'DFFConverter', 'DFFManager',
    'TXDConverter', 'TXDManager',
    'IMGExtractor',
    
    # Blender
    'OBJExporter', 'MaterialBuilder',
    
    # Utilities
    'ProgressTracker', 'Logger', 'FileUtils', 'ErrorHandler', 
    'PerformanceOptimizer',
]


def get_version() -> str:
    """Return the package version."""
    return __version__


def get_author() -> str:
    """Return the package author."""
    return __author__


def get_description() -> str:
    """Return the package description."""
    return __description__


def check_dependencies() -> dict:
    """
    Check if all required dependencies are installed.
    
    Returns:
        Dictionary with dependency status
    """
    dependencies = {
        'PyQt5': False,
        'numpy': False,
        'PIL': False,
    }
    
    missing = []
    
    try:
        import PyQt5
        dependencies['PyQt5'] = True
    except ImportError:
        missing.append('PyQt5 (sudo apt-get install python3-pyqt5)')
    
    try:
        import numpy
        dependencies['numpy'] = True
    except ImportError:
        missing.append('numpy (pip install numpy)')
    
    try:
        from PIL import Image
        dependencies['PIL'] = True
    except ImportError:
        missing.append('Pillow (pip install Pillow)')
    
    result = {
        'dependencies': dependencies,
        'all_met': all(dependencies.values()),
        'missing': missing
    }
    
    return result


def setup_logging(level: str = "INFO"):
    """
    Set up logging for the application.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    import logging
    
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Basic logging configuration
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('gta_converter.log')
        ]
    )
    
    return logging.getLogger(__name__)


def run_cli(img_dir: str = None, maps_dir: str = None, output_dir: str = None):
    """
    Run the converter from command line interface.
    
    Args:
        img_dir: Path to GTA_SA_map directory
        maps_dir: Path to maps directory
        output_dir: Path to output directory
    """
    import argparse
    
    parser = argparse.ArgumentParser(description=__description__)
    parser.add_argument('--img-dir', help='Path to GTA_SA_map directory')
    parser.add_argument('--maps-dir', help='Path to maps directory')
    parser.add_argument('--output-dir', help='Path to output directory')
    parser.add_argument('--version', action='version', version=__version__)
    
    args = parser.parse_args()
    
    # Use provided arguments or defaults
    img_dir = img_dir or args.img_dir
    maps_dir = maps_dir or args.maps_dir
    output_dir = output_dir or args.output_dir
    
    if not all([img_dir, maps_dir]):
        parser.print_help()
        print("\nError: --img-dir and --maps-dir are required for CLI mode")
        sys.exit(1)
    
    # Run the converter
    from .config import Config
    from .core.conversion_pipeline import ConversionPipeline
    
    config = Config()
    config.paths.img_dir = img_dir
    config.paths.maps_dir = maps_dir
    config.paths.output_dir = output_dir or os.path.join(os.getcwd(), 'output')
    
    print(f"Starting conversion:")
    print(f"  IMG dir: {img_dir}")
    print(f"  Maps dir: {maps_dir}")
    print(f"  Output dir: {config.paths.output_dir}")
    
    pipeline = ConversionPipeline(config)
    
    # Set up simple logging
    def log_callback(message, level="INFO"):
        print(f"[{level}] {message}")
    
    pipeline.set_log_callback(log_callback)
    
    try:
        success, output_path, message = pipeline.convert()
        
        if success:
            print(f"\n✓ SUCCESS: {message}")
            print(f"Output file: {output_path}")
            sys.exit(0)
        else:
            print(f"\n✗ FAILED: {message}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\nConversion cancelled by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    # If run directly, check dependencies and provide info
    deps = check_dependencies()
    
    print(f"{__description__} v{__version__}")
    print(f"Author: {__author__}")
    print(f"License: {__license__}")
    print()
    
    if deps['all_met']:
        print("✓ All dependencies are installed")
    else:
        print("✗ Missing dependencies:")
        for dep in deps['missing']:
            print(f"  - {dep}")
        print()
        print("Install missing dependencies and try again.")
        sys.exit(1)
    
    print("\nUsage:")
    print("  GUI mode: python -m gta_map_converter.main")
    print("  CLI mode: python -m gta_map_converter --help")
