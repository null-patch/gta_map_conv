import os
import sys
import time
import traceback
from core.models import SceneObject
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import threading
from queue import Queue, Empty

# Local imports
try:
    from converters.ide_parser import IDEManager, IDEParser
    from converters.ipl_parser import IPLManager, IPLParser
    from converters.dff_converter import DFFConverter, DFFManager
    from converters.txd_converter import TXDConverter, TXDManager
    from converters.img_archive import IMGExtractor
    from blender.obj_exporter import OBJExporter
    from blender.material_builder import MaterialBuilder
    from utils.progress_tracker import ProgressTracker
    from utils.logger import Logger
    from utils.file_utils import list_files, ensure_dir, get_file_operations
    from utils.error_handler import ErrorHandler
    from config import Config
except ImportError as e:
    print(f"Import error in conversion_pipeline: {e}")
    # Create dummy classes for testing
    class Dummy:
        def __init__(self, *args, **kwargs): pass
        def __getattr__(self, name): return lambda *args, **kwargs: None
    IDEManager = IPLParser = DFFConverter = TXDConverter = IMGExtractor = Dummy
    OBJExporter = MaterialBuilder = ProgressTracker = Logger = FileUtils = ErrorHandler = Dummy
    Config = Dummy

logger = logging.getLogger(__name__)


@dataclass
class ConversionStats:
    """Statistics for conversion process"""
    start_time: float = 0.0
    end_time: float = 0.0
    total_files_processed: int = 0
    total_vertices: int = 0
    total_faces: int = 0
    total_textures: int = 0
    total_objects_placed: int = 0
    errors_encountered: int = 0
    warnings_encountered: int = 0
    
    # File type counts
    ide_files: int = 0
    ipl_files: int = 0
    dff_files: int = 0
    txd_files: int = 0
    img_archives: int = 0
    
    # Performance metrics
    extraction_time: float = 0.0
    parsing_time: float = 0.0
    conversion_time: float = 0.0
    export_time: float = 0.0
    
    def get_duration(self) -> float:
        """Get total conversion duration in seconds"""
        if self.end_time and self.start_time:
            return self.end_time - self.start_time
        return 0.0
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics"""
        return {
            'total_time': self.get_duration(),
            'total_files': self.total_files_processed,
            'total_vertices': self.total_vertices,
            'total_faces': self.total_faces,
            'total_textures': self.total_textures,
            'total_objects': self.total_objects_placed,
            'errors': self.errors_encountered,
            'warnings': self.warnings_encountered
        }

    def get_transform_matrix(self) -> List[List[float]]:
        """Get transformation matrix for this object"""
        # Simplified transformation - actual GTA uses quaternions
        # This would need proper quaternion to matrix conversion
        return [
            [self.scale[0], 0, 0, self.position[0]],
            [0, self.scale[1], 0, self.position[1]],
            [0, 0, self.scale[2], self.position[2]],
            [0, 0, 0, 1]
        ]


class ConversionPipeline:
    """Main pipeline orchestrating the entire conversion process"""
    
    def __init__(self, config: Config):
        self.config = config
        self.stats = ConversionStats()
        
        # Managers
        self.ide_manager = IDEManager()
        self.ipl_manager = IPLManager()
        self.dff_manager = DFFManager()
        self.txd_manager = TXDManager()
        
        # Extractors/Converters
        self.img_extractor = IMGExtractor()
        self.dff_converter = DFFConverter(config.conversion.scale_factor)
        self.txd_converter = TXDConverter()
        self.obj_exporter = OBJExporter(config)
        self.material_builder = MaterialBuilder(config)
        
        # Utilities
        self.progress_tracker = ProgressTracker()
        self.file_utils = get_file_operations()
        self.error_handler = ErrorHandler()
        
        # State
        self.is_running = False
        self.cancel_requested = False
        self.current_stage = ""
        
        # Callbacks
        self.progress_callback: Optional[Callable] = None
        self.status_callback: Optional[Callable] = None
        self.log_callback: Optional[Callable] = None
        
        # Results
        self.extracted_files: Dict[str, str] = {}
        self.converted_models: Dict[str, Dict] = {}
        self.converted_textures: Dict[str, Dict] = {}
        self.scene_objects: List[SceneObject] = []
        self.materials: Dict[str, Dict] = {}
        
        # Threading
        self.thread_pool = None
        self.max_workers = min(4, os.cpu_count() or 1)
        
    def set_progress_callback(self, callback: Callable):
        """Set progress update callback"""
        self.progress_callback = callback
        
    def set_status_callback(self, callback: Callable):
        """Set status update callback"""
        self.status_callback = callback
        
    def set_log_callback(self, callback: Callable):
        """Set log callback"""
        self.log_callback = callback
        
    def _update_progress(self, task: str, percent: int, sub_task: str = "", sub_percent: int = 0):
        """Update progress through callback"""
        if self.progress_callback:
            # Format progress message
            if sub_task:
                message = f"{task} - {sub_task}"
            else:
                message = task
                
            # Call with structured data
            if callable(self.progress_callback):
                self.progress_callback(message, percent, sub_task, sub_percent)
                
    def _update_status(self, status: str):
        """Update status through callback"""
        if self.status_callback:
            self.status_callback(status)
            
    def _log_message(self, message: str, level: str = "INFO"):
        """Log message through callback"""
        if self.log_callback:
            self.log_callback(message, level)
        else:
            # Default logging
            print(f"[{level}] {message}")
            
    def _check_cancel(self) -> bool:
        """Check if cancellation was requested"""
        return self.cancel_requested
            
    def convert(self) -> Tuple[bool, str, str]:
        """
        Main conversion entry point
        
        Returns:
            Tuple of (success, output_path, message)
        """
        self.is_running = True
        self.cancel_requested = False
        self.stats.start_time = time.time()
        
        try:
            # Validate configuration
            self._update_status("Validating configuration...")
            if not self._validate_config():
                return False, "", "Configuration validation failed"
                
            if self._check_cancel():
                return False, "", "Conversion cancelled"
                
            # Step 1: Parse IDE files
            self._update_status("Parsing IDE files...")
            ide_data = self._parse_ide_files()
            if not ide_data and not self.cancel_requested:
                self._log_message("No IDE files found or parsed", "WARNING")
                
            if self._check_cancel():
                return False, "", "Conversion cancelled"
                
            # Step 2: Parse IPL files
            self._update_status("Parsing IPL files...")
            ipl_data = self._parse_ipl_files()
            if not ipl_data and not self.cancel_requested:
                self._log_message("No IPL files found or parsed", "WARNING")
                
            if self._check_cancel():
                return False, "", "Conversion cancelled"
                
            # Step 3: Extract IMG archives
            self._update_status("Extracting IMG archives...")
            extracted_files = self._extract_img_archives()
            if not extracted_files and not self.cancel_requested:
                self._log_message("No files extracted from IMG archives", "WARNING")
                
            if self._check_cancel():
                return False, "", "Conversion cancelled"
                
            # Step 4: Process DFF files
            self._update_status("Converting DFF models...")
            models = self._convert_dff_files(extracted_files)
            if not models and not self.cancel_requested:
                self._log_message("No DFF models converted", "WARNING")
                
            if self._check_cancel():
                return False, "", "Conversion cancelled"
                
            # Step 5: Process TXD files
            self._update_status("Processing textures...")
            textures = self._convert_txd_files(extracted_files)
            if not textures and self.config.conversion.export_textures:
                self._log_message("No textures processed", "WARNING")
                
            if self._check_cancel():
                return False, "", "Conversion cancelled"
                
            # Step 6: Build scene
            self._update_status("Building scene...")
            scene = self._build_scene(ide_data, ipl_data, models, textures)
            
            if self._check_cancel():
                return False, "", "Conversion cancelled"
                
            # Step 7: Export to OBJ
            self._update_status("Exporting to OBJ...")
            output_path = self._export_to_obj(scene)
            
            # Update statistics
            self.stats.end_time = time.time()
            self.stats.total_files_processed = (
                self.stats.ide_files + 
                self.stats.ipl_files + 
                self.stats.dff_files + 
                self.stats.txd_files + 
                self.stats.img_archives
            )
            
            success = os.path.exists(output_path)
            
            if success:
                message = f"Conversion completed successfully in {self.stats.get_duration():.2f} seconds"
                self._log_message(message, "SUCCESS")
                return True, output_path, message
            else:
                message = "Failed to create output file"
                self._log_message(message, "ERROR")
                return False, "", message
                
        except Exception as e:
            error_msg = f"Conversion failed with error: {str(e)}"
            self._log_message(error_msg, "ERROR")
            traceback.print_exc()
            return False, "", error_msg
            
        finally:
            self.is_running = False
            self._cleanup_temp_files()
            
    def _validate_config(self) -> bool:
        """Validate configuration and paths"""
        errors = []
        
        # Check directories
        if not os.path.exists(self.config.paths.img_dir):
            errors.append(f"IMG directory does not exist: {self.config.paths.img_dir}")
            
        if not os.path.exists(self.config.paths.maps_dir):
            errors.append(f"Maps directory does not exist: {self.config.paths.maps_dir}")
            
        # Check for required files
        img_files = self.config.paths.get_img_files()
        if not img_files:
            errors.append("No .img files found in IMG directory")
            
        ide_files = self.config.paths.get_ide_files()
        if not ide_files:
            errors.append("No .ide files found in maps directory")
            
        # Create output directory
        try:
            os.makedirs(self.config.paths.output_dir, exist_ok=True)
        except Exception as e:
            errors.append(f"Cannot create output directory: {str(e)}")
            
        # Check for errors
        if errors:
            for error in errors:
                self._log_message(error, "ERROR")
            return False
            
        self._log_message(f"Found {len(img_files)} IMG files and {len(ide_files)} IDE files", "INFO")
        return True
        
    def _parse_ide_files(self) -> Dict[int, Any]:
        """Parse all IDE files"""
        start_time = time.time()
        
        try:
            ide_files = self.config.paths.get_ide_files()
            self.stats.ide_files = len(ide_files)
            
            if not ide_files:
                self._log_message("No IDE files found", "WARNING")
                return {}
                
            self._update_progress("Parsing IDE files", 10, f"0/{len(ide_files)}", 0)
            
            # Parse each IDE file
            for i, ide_file in enumerate(ide_files):
                if self._check_cancel():
                    return {}
                    
                self._update_progress(
                    "Parsing IDE files", 
                    10 + int((i / len(ide_files)) * 10),
                    f"{i+1}/{len(ide_files)}: {os.path.basename(ide_file)}",
                    int((i / len(ide_files)) * 100)
                )
                
                try:
                    success = self.ide_manager.parse_file(ide_file)
                    if success:
                        self._log_message(f"Parsed IDE file: {os.path.basename(ide_file)}", "INFO")
                    else:
                        self._log_message(f"Failed to parse IDE file: {ide_file}", "WARNING")
                        self.stats.warnings_encountered += 1
                        
                except Exception as e:
                    self._log_message(f"Error parsing IDE file {ide_file}: {str(e)}", "ERROR")
                    self.stats.errors_encountered += 1
                    
            self._log_message(f"Parsed {len(self.ide_manager.get_all_objects())} objects from {len(ide_files)} IDE files", "INFO")
            
            self.stats.parsing_time += time.time() - start_time
            return self.ide_manager.get_all_objects()
            
        except Exception as e:
            self._log_message(f"Error in IDE parsing: {str(e)}", "ERROR")
            self.stats.errors_encountered += 1
            return {}
            
    def _parse_ipl_files(self) -> List[Dict[str, Any]]:
        """Parse all IPL files"""
        start_time = time.time()
        
        try:
            ipl_files = self.config.paths.get_ipl_files()
            self.stats.ipl_files = len(ipl_files)
            
            if not ipl_files:
                self._log_message("No IPL files found", "WARNING")
                return []
                
            self._update_progress("Parsing IPL files", 30, f"0/{len(ipl_files)}", 0)
            
            # Parse each IPL file
            all_placements = []
            for i, ipl_file in enumerate(ipl_files):
                if self._check_cancel():
                    return []
                    
                self._update_progress(
                    "Parsing IPL files",
                    30 + int((i / len(ipl_files)) * 10),
                    f"{i+1}/{len(ipl_files)}: {os.path.basename(ipl_file)}",
                    int((i / len(ipl_files)) * 100)
                )
                
                try:
                    placements = self.ipl_manager.parse_file(ipl_file)
                    if placements:
                        all_placements.extend(placements)
                        self._log_message(f"Parsed {len(placements)} placements from {os.path.basename(ipl_file)}", "INFO")
                    else:
                        self._log_message(f"No placements found in IPL file: {ipl_file}", "WARNING")
                        self.stats.warnings_encountered += 1
                        
                except Exception as e:
                    self._log_message(f"Error parsing IPL file {ipl_file}: {str(e)}", "ERROR")
                    self.stats.errors_encountered += 1
                    
            self._log_message(f"Total placements parsed: {len(all_placements)}", "INFO")
            
            self.stats.parsing_time += time.time() - start_time
            return all_placements
            
        except Exception as e:
            self._log_message(f"Error in IPL parsing: {str(e)}", "ERROR")
            self.stats.errors_encountered += 1
            return []
            
    def _extract_img_archives(self) -> Dict[str, str]:
        """Extract files from IMG archives"""
        start_time = time.time()
        
        try:
            img_files = self.config.paths.get_img_files()
            self.stats.img_archives = len(img_files)
            
            if not img_files:
                self._log_message("No IMG files found", "WARNING")
                return {}
                
            self._update_progress("Extracting IMG archives", 50, f"0/{len(img_files)}", 0)
            
            # Create temp directory
            temp_dir = Path(self.config.paths.temp_dir)
            temp_dir.mkdir(exist_ok=True)
            
            all_extracted = {}
            
            for i, img_file in enumerate(img_files):
                if self._check_cancel():
                    return {}
                    
                self._update_progress(
                    "Extracting IMG archives",
                    50 + int((i / len(img_files)) * 10),
                    f"{i+1}/{len(img_files)}: {os.path.basename(img_file)}",
                    int((i / len(img_files)) * 100)
                )
                
                try:
                    extracted = self.img_extractor.extract(img_file, str(temp_dir))
                    if extracted:
                        all_extracted.update(extracted)
                        self._log_message(f"Extracted {len(extracted)} files from {os.path.basename(img_file)}", "INFO")
                    else:
                        self._log_message(f"No files extracted from {img_file}", "WARNING")
                        self.stats.warnings_encountered += 1
                        
                except Exception as e:
                    self._log_message(f"Error extracting IMG archive {img_file}: {str(e)}", "ERROR")
                    self.stats.errors_encountered += 1
                    
            self._log_message(f"Total files extracted: {len(all_extracted)}", "INFO")
            
            self.stats.extraction_time = time.time() - start_time
            self.extracted_files = all_extracted
            return all_extracted
            
        except Exception as e:
            self._log_message(f"Error in IMG extraction: {str(e)}", "ERROR")
            self.stats.errors_encountered += 1
            return {}
            
    def _convert_dff_files(self, extracted_files: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
        """Convert DFF files to geometry data"""
        start_time = time.time()
        
        try:
            # Find all DFF files
            dff_files = []
            for file_path in extracted_files.values():
                if file_path.lower().endswith('.dff'):
                    dff_files.append(file_path)
                    
            self.stats.dff_files = len(dff_files)
            
            if not dff_files:
                self._log_message("No DFF files found", "WARNING")
                return {}
                
            self._update_progress("Converting DFF models", 70, f"0/{len(dff_files)}", 0)
            
            all_models = {}
            
            # Use thread pool for parallel processing
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit conversion tasks
                future_to_file = {
                    executor.submit(self._convert_single_dff, dff_file): dff_file
                    for dff_file in dff_files
                }
                
                # Process completed tasks
                completed = 0
                for future in as_completed(future_to_file):
                    if self._check_cancel():
                        executor.shutdown(wait=False)
                        return {}
                        
                    dff_file = future_to_file[future]
                    try:
                        model_data = future.result()
                        if model_data:
                            model_name = Path(dff_file).stem
                            all_models[model_name] = model_data
                            
                            # Update stats
                            vertices = len(model_data.get('vertices', []))
                            faces = len(model_data.get('faces', []))
                            self.stats.total_vertices += vertices
                            self.stats.total_faces += faces
                            
                        completed += 1
                        self._update_progress(
                            "Converting DFF models",
                            70 + int((completed / len(dff_files)) * 10),
                            f"{completed}/{len(dff_files)}: {Path(dff_file).name}",
                            int((completed / len(dff_files)) * 100)
                        )
                        
                    except Exception as e:
                        self._log_message(f"Error converting DFF file {dff_file}: {str(e)}", "ERROR")
                        self.stats.errors_encountered += 1
                        
            self._log_message(f"Converted {len(all_models)} DFF models", "INFO")
            
            self.stats.conversion_time += time.time() - start_time
            self.converted_models = all_models
            return all_models
            
        except Exception as e:
            self._log_message(f"Error in DFF conversion: {str(e)}", "ERROR")
            self.stats.errors_encountered += 1
            return {}
            
    def _convert_single_dff(self, dff_path: str) -> Optional[Dict[str, Any]]:
        """Convert a single DFF file"""
        try:
            model_data = self.dff_converter.convert(dff_path)
            if model_data:
                return model_data
        except Exception as e:
            self._log_message(f"Error converting {dff_path}: {str(e)}", "ERROR")
        return None
        
    def _convert_txd_files(self, extracted_files: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
        """Convert TXD files to textures"""
        start_time = time.time()
        
        try:
            # Find all TXD files
            txd_files = []
            for file_path in extracted_files.values():
                if file_path.lower().endswith('.txd'):
                    txd_files.append(file_path)
                    
            self.stats.txd_files = len(txd_files)
            
            if not txd_files or not self.config.conversion.export_textures:
                if not self.config.conversion.export_textures:
                    self._log_message("Texture export disabled", "INFO")
                return {}
                
            self._update_progress("Converting textures", 85, f"0/{len(txd_files)}", 0)
            
            all_textures = {}
            
            # Use thread pool for parallel processing
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit conversion tasks
                future_to_file = {
                    executor.submit(self._convert_single_txd, txd_file): txd_file
                    for txd_file in txd_files
                }
                
                # Process completed tasks
                completed = 0
                for future in as_completed(future_to_file):
                    if self._check_cancel():
                        executor.shutdown(wait=False)
                        return {}
                        
                    txd_file = future_to_file[future]
                    try:
                        texture_data = future.result()
                        if texture_data:
                            for tex_name, tex_info in texture_data.items():
                                all_textures[tex_name] = tex_info
                                self.stats.total_textures += 1
                                
                        completed += 1
                        self._update_progress(
                            "Converting textures",
                            85 + int((completed / len(txd_files)) * 5),
                            f"{completed}/{len(txd_files)}: {Path(txd_file).name}",
                            int((completed / len(txd_files)) * 100)
                        )
                        
                    except Exception as e:
                        self._log_message(f"Error converting TXD file {txd_file}: {str(e)}", "ERROR")
                        self.stats.errors_encountered += 1
                        
            self._log_message(f"Converted {len(all_textures)} textures", "INFO")
            
            self.stats.conversion_time += time.time() - start_time
            self.converted_textures = all_textures
            return all_textures
            
        except Exception as e:
            self._log_message(f"Error in TXD conversion: {str(e)}", "ERROR")
            self.stats.errors_encountered += 1
            return {}
            
    def _convert_single_txd(self, txd_path: str) -> Optional[Dict[str, Dict]]:
        """Convert a single TXD file"""
        try:
            texture_format = self.config.conversion.get_texture_extension()
            texture_size = self.config.conversion.get_texture_size()
            
            textures = self.txd_converter.convert(
                txd_path,
                self.config.paths.output_dir,
                texture_format,
                texture_size,
                self.config.conversion.texture_quality
            )
            return textures
        except Exception as e:
            self._log_message(f"Error converting {txd_path}: {str(e)}", "ERROR")
        return None
        
    def _build_scene(self, ide_data: Dict, ipl_data: List, models: Dict, textures: Dict) -> Dict[str, Any]:
        """Build complete scene from all components"""
        start_time = time.time()
        
        try:
            self._update_progress("Building scene", 95, "Processing objects", 0)
            
            # Create materials from textures
            materials = self.material_builder.create_materials(textures)
            self.materials = materials
            
            # Build scene objects from placements
            scene_objects = []
            placed_count = 0
            skipped_count = 0
            
            for i, placement in enumerate(ipl_data):
                if self._check_cancel():
                    break
                    
                # Update progress every 100 placements
                if i % 100 == 0:
                    self._update_progress(
                        "Building scene",
                        95 + int((i / max(1, len(ipl_data))) * 2),
                        f"Placements: {i}/{len(ipl_data)}",
                        int((i / max(1, len(ipl_data))) * 100)
                    )
                
                try:
                    # Get object ID from placement
                    obj_id = placement.get('id')
                    if not obj_id:
                        continue
                        
                    # Find IDE object definition
                    ide_obj = ide_data.get(obj_id)
                    if not ide_obj:
                        # Try to find by name
                        model_name = placement.get('model_name')
                        if model_name:
                            # Search for object with matching name
                            for obj in ide_data.values():
                                if hasattr(obj, 'model_name') and obj.model_name == model_name:
                                    ide_obj = obj
                                    break
                    
                    if not ide_obj:
                        skipped_count += 1
                        continue
                        
                    # Get model data
                    model_name = getattr(ide_obj, 'model_name', f"object_{obj_id}")
                    model_data = models.get(model_name)
                    
                    if not model_data:
                        # Model not found in converted DFFs
                        skipped_count += 1
                        continue
                        
                    # Create scene object
                    scene_obj = SceneObject(
                        id=obj_id,
                        model_name=model_name,
                        model_data=model_data,
                        position=placement.get('position', (0, 0, 0)),
                        rotation=placement.get('rotation', (0, 0, 0)),
                        scale=placement.get('scale', (1, 1, 1)),
                        flags=placement.get('flags', 0),
                        draw_distance=getattr(ide_obj, 'draw_distance', 300.0),
                        texture_dict=getattr(ide_obj, 'texture_dictionary', ''),
                        lod_level=placement.get('lod', 0),
                        parent_id=placement.get('parent_id', -1),
                        time_on=getattr(ide_obj, 'time_on', 0),
                        time_off=getattr(ide_obj, 'time_off', 24),
                        interior=placement.get('interior', 0)
                    )
                    
                    scene_objects.append(scene_obj)
                    placed_count += 1
                    
                except Exception as e:
                    self._log_message(f"Error processing placement {i}: {str(e)}", "ERROR")
                    self.stats.errors_encountered += 1
                    
            self.stats.total_objects_placed = placed_count
            
            # Build scene structure
            scene = {
                'name': 'GTA_SA_Map',
                'objects': scene_objects,
                'materials': materials,
                'textures': textures,
                'models': models,
                'settings': {
                    'scale_factor': self.config.conversion.scale_factor,
                    'coordinate_system': self.config.conversion.coordinate_system,
                    'units': self.config.conversion.units
                }
            }
            
            self._log_message(f"Scene built: {placed_count} objects placed, {skipped_count} skipped", "INFO")
            
            self.scene_objects = scene_objects
            return scene
            
        except Exception as e:
            self._log_message(f"Error building scene: {str(e)}", "ERROR")
            self.stats.errors_encountered += 1
            return {}
            
        finally:
            self.stats.conversion_time += time.time() - start_time
    
    def _export_to_obj(self, scene: Dict[str, Any]) -> str:
        start_time = time.time()

        try:
            self._update_progress("Exporting to OBJ", 98, "Writing files", 0)

            # Generate output filename
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_filename = f"gta_sa_map_{timestamp}.obj"
            output_path = os.path.join(self.config.paths.output_dir, output_filename)

            # Ensure output directory exists and is writable
            try:
                os.makedirs(self.config.paths.output_dir, exist_ok=True)
            except Exception as e:
                msg = f"Cannot create output directory '{self.config.paths.output_dir}': {e}"
                self._log_message(msg, "ERROR")
                raise RuntimeError(msg)

            # Use exporter API. It should return True on success.
            try:
                # Some exporter implementations expose export_scene or export; support both.
                exporter = self.obj_exporter
                exported_ok = False
                if hasattr(exporter, "export"):
                    exported_ok = exporter.export(scene, output_path)
                elif hasattr(exporter, "export_scene"):
                    exported_ok = exporter.export_scene(scene, output_path)
                else:
                    msg = "OBJ exporter has no export/export_scene method"
                    self._log_message(msg, "ERROR")
                    raise RuntimeError(msg)
            except Exception as e:
                # Log and re-raise with context so ConversionThread can emit the error
                err_msg = f"Error exporting OBJ: {e}"
                self._log_message(err_msg, "ERROR")
                raise

            if not exported_ok:
                msg = "OBJ exporter returned failure (export returned False)"
                self._log_message(msg, "ERROR")
                raise RuntimeError(msg)

            # Confirm file exists
            if not os.path.exists(output_path):
                msg = f"OBJ exporter reported success but file not found: {output_path}"
                self._log_message(msg, "ERROR")
                raise RuntimeError(msg)

            self._log_message(f"Exported to: {output_path}", "SUCCESS")

            self.stats.export_time = time.time() - start_time
            return output_path

        except Exception:
            # Ensure caller sees the exception; conversion() will handle and report it.
            raise
            
    def _cleanup_temp_files(self):
        """Clean up temporary files"""
        try:
            if not self.config.performance.keep_temp_files:
                temp_dir = Path(self.config.paths.temp_dir)
                if temp_dir.exists():
                    import shutil
                    shutil.rmtree(temp_dir)
                    self._log_message("Cleaned up temporary files", "INFO")
        except Exception as e:
            self._log_message(f"Error cleaning temp files: {str(e)}", "WARNING")
            
    def cancel(self):
        """Cancel the conversion process"""
        self.cancel_requested = True
        self._log_message("Cancellation requested", "INFO")
        
    def get_stats(self) -> ConversionStats:
        """Get conversion statistics"""
        return self.stats
        
    def get_progress(self) -> Dict[str, Any]:
        """Get current progress information"""
        return {
            'current_stage': self.current_stage,
            'is_running': self.is_running,
            'stats': self.stats.get_summary()
        }


class BatchProcessor:
    """Process conversions in batch mode"""
    
    def __init__(self, config: Config):
        self.config = config
        self.pipelines: List[ConversionPipeline] = []
        self.results: List[Dict[str, Any]] = []
        
    def add_conversion(self, img_dir: str, maps_dir: str, output_dir: str) -> ConversionPipeline:
        """Add a conversion job"""
        # Create a copy of config with new paths
        pipeline = ConversionPipeline(self.config)
        
        # Update paths (this is a simplification - would need proper config copy)
        pipeline.config.paths.img_dir = img_dir
        pipeline.config.paths.maps_dir = maps_dir
        pipeline.config.paths.output_dir = output_dir
        
        self.pipelines.append(pipeline)
        return pipeline
        
    def run_all(self, parallel: bool = False) -> List[Dict[str, Any]]:
        """Run all conversion jobs"""
        self.results.clear()
        
        if parallel:
            return self._run_parallel()
        else:
            return self._run_sequential()
            
    def _run_sequential(self) -> List[Dict[str, Any]]:
        """Run conversions sequentially"""
        for i, pipeline in enumerate(self.pipelines):
            print(f"Processing conversion {i+1}/{len(self.pipelines)}")
            
            try:
                success, output_path, message = pipeline.convert()
                
                result = {
                    'index': i,
                    'success': success,
                    'output_path': output_path,
                    'message': message,
                    'stats': pipeline.get_stats().get_summary()
                }
                self.results.append(result)
                
            except Exception as e:
                result = {
                    'index': i,
                    'success': False,
                    'output_path': '',
                    'message': str(e),
                    'stats': {}
                }
                self.results.append(result)
                
        return self.results
        
    def _run_parallel(self) -> List[Dict[str, Any]]:
        """Run conversions in parallel"""
        # Note: Parallel processing of GTA conversions is complex due to
        # memory usage and file I/O. This is a simplified version.
        with ProcessPoolExecutor(max_workers=min(2, len(self.pipelines))) as executor:
            future_to_pipeline = {
                executor.submit(self._run_single_conversion, i, pipeline): (i, pipeline)
                for i, pipeline in enumerate(self.pipelines)
            }
            
            for future in as_completed(future_to_pipeline):
                i, pipeline = future_to_pipeline[future]
                try:
                    success, output_path, message = future.result()
                    
                    result = {
                        'index': i,
                        'success': success,
                        'output_path': output_path,
                        'message': message
                    }
                    self.results.append(result)
                    
                except Exception as e:
                    result = {
                        'index': i,
                        'success': False,
                        'output_path': '',
                        'message': str(e)
                    }
                    self.results.append(result)
                    
        return self.results
        
    def _run_single_conversion(self, index: int, pipeline: ConversionPipeline) -> Tuple[bool, str, str]:
        """Run a single conversion in separate process"""
        return pipeline.convert()
        
    def get_summary(self) -> Dict[str, Any]:
        """Get batch processing summary"""
        total = len(self.results)
        successful = sum(1 for r in self.results if r['success'])
        failed = total - successful
        
        total_time = sum(r.get('stats', {}).get('total_time', 0) for r in self.results)
        
        return {
            'total_conversions': total,
            'successful': successful,
            'failed': failed,
            'total_time': total_time,
            'results': self.results
        }


if __name__ == "__main__":
    # Test the conversion pipeline
    import sys
    
    if len(sys.argv) > 2:
        img_dir = sys.argv[1]
        maps_dir = sys.argv[2]
        output_dir = sys.argv[3] if len(sys.argv) > 3 else "output"
        
        print(f"Testing conversion pipeline:")
        print(f"  IMG dir: {img_dir}")
        print(f"  Maps dir: {maps_dir}")
        print(f"  Output dir: {output_dir}")
        
        # Create config
        config = Config()
        config.paths.img_dir = img_dir
        config.paths.maps_dir = maps_dir
        config.paths.output_dir = output_dir
        
        # Run conversion
        pipeline = ConversionPipeline(config)
        
        # Set up simple callbacks
        def progress_callback(task, percent, sub_task="", sub_percent=0):
            if sub_task:
                print(f"[{percent}%] {task} - {sub_task}")
            else:
                print(f"[{percent}%] {task}")
                
        def log_callback(message, level="INFO"):
            print(f"[{level}] {message}")
            
        pipeline.set_progress_callback(progress_callback)
        pipeline.set_log_callback(log_callback)
        
        success, output_path, message = pipeline.convert()
        
        if success:
            print(f"\n✓ SUCCESS: {message}")
            print(f"Output: {output_path}")
        else:
            print(f"\n✗ FAILED: {message}")
            
    else:
        print("Usage: python conversion_pipeline.py <img_dir> <maps_dir> [output_dir]")
        print("\nExample:")
        print("  python conversion_pipeline.py /path/to/Gta_SA_map /path/to/maps ./export")
