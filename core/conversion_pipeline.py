import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

from core.models import SceneObject  # dataclass with scene object fields

# Local imports
try:
    from converters.ide_parser import IDEManager, IDEParser
    from converters.ipl_parser import IPLManager, IPLParser
    from converters.dff_converter import DFFConverter, DFFManager
    from converters.txd_converter import TXDConverter, TXDManager
    from blender.obj_exporter import OBJExporter
    from blender.material_builder import MaterialBuilder
    from utils.progress_tracker import ProgressTracker
    from utils.logger import Logger
    from utils.file_utils import get_file_operations
    from utils.error_handler import ErrorHandler
    from config import Config
except ImportError as e:
    print(f"Import error in conversion_pipeline: {e}")
    # Dummy types for import-time errors
    class Dummy:
        def __init__(self, *args, **kwargs): ...
        def __getattr__(self, name): return lambda *args, **kwargs: None
    IDEManager = IPLParser = DFFConverter = TXDConverter = Dummy
    OBJExporter = MaterialBuilder = ProgressTracker = Logger = ErrorHandler = Dummy
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

    ide_files: int = 0
    ipl_files: int = 0
    dff_files: int = 0
    txd_files: int = 0
    img_archives: int = 0

    extraction_time: float = 0.0
    parsing_time: float = 0.0
    conversion_time: float = 0.0
    export_time: float = 0.0

    def get_duration(self) -> float:
        if self.end_time and self.start_time:
            return self.end_time - self.start_time
        return 0.0

    def get_summary(self) -> Dict[str, Any]:
        return {
            "total_time": self.get_duration(),
            "total_files": self.total_files_processed,
            "total_vertices": self.total_vertices,
            "total_faces": self.total_faces,
            "total_textures": self.total_textures,
            "total_objects": self.total_objects_placed,
            "errors": self.errors_encountered,
            "warnings": self.warnings_encountered,
        }


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

        # Converters / exporters
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

        self.thread_pool = None
        self.max_workers = min(4, os.cpu_count() or 1)

    # ------------------------------------------------------------------ callbacks / logging

    def set_progress_callback(self, callback: Callable):
        self.progress_callback = callback

    def set_status_callback(self, callback: Callable):
        self.status_callback = callback

    def set_log_callback(self, callback: Callable):
        self.log_callback = callback

    def _update_progress(
        self,
        task: str,
        percent: int,
        sub_task: str = "",
        sub_percent: int = 0,
    ):
        if self.progress_callback and callable(self.progress_callback):
            message = f"{task} - {sub_task}" if sub_task else task
            self.progress_callback(message, percent, sub_task, sub_percent)

    def _update_status(self, status: str):
        if self.status_callback:
            self.status_callback(status)

    def _log_message(self, message: str, level: str = "INFO"):
        if self.log_callback:
            self.log_callback(message, level)
        else:
            print(f"[{level}] {message}")

    def _check_cancel(self) -> bool:
        return self.cancel_requested

    # ------------------------------------------------------------------ main convert

    def convert(self) -> Tuple[bool, str, str]:
        self.is_running = True
        self.cancel_requested = False
        self.stats.start_time = time.time()

        try:
            self._update_status("Validating configuration...")
            if not self._validate_config():
                return False, "", "Configuration validation failed"

            if self._check_cancel():
                return False, "", "Conversion cancelled"

            # 1) IDE
            self._update_status("Parsing IDE files...")
            ide_data = self._parse_ide_files()
            if not ide_data and not self.cancel_requested:
                self._log_message("No IDE files found or parsed", "WARNING")

            if self._check_cancel():
                return False, "", "Conversion cancelled"

            # 2) IPL
            self._update_status("Parsing IPL files...")
            ipl_data = self._parse_ipl_files()
            if not ipl_data and not self.cancel_requested:
                self._log_message("No IPL files found or parsed", "WARNING")

            if self._check_cancel():
                return False, "", "Conversion cancelled"

            # 3) Collect DFF/TXD from extracted folder
            self._update_status("Collecting DFF/TXD files...")
            extracted_files = self._extract_img_archives()
            if not extracted_files and not self.cancel_requested:
                self._log_message("No DFF/TXD files found in source folder", "WARNING")

            if self._check_cancel():
                return False, "", "Conversion cancelled"

            # 4) DFF
            self._update_status("Converting DFF models...")
            models = self._convert_dff_files(extracted_files)
            if not models and not self.cancel_requested:
                self._log_message("No DFF models converted", "WARNING")

            if self._check_cancel():
                return False, "", "Conversion cancelled"

            # 5) TXD
            self._update_status("Processing textures...")
            textures = self._convert_txd_files(extracted_files)
            if not textures and self.config.conversion.export_textures:
                self._log_message("No textures processed", "WARNING")

            if self._check_cancel():
                return False, "", "Conversion cancelled"

            # 6) Scene
            self._update_status("Building scene...")
            scene = self._build_scene(ide_data, ipl_data, models, textures)

            if self._check_cancel():
                return False, "", "Conversion cancelled"

            # 7) Export
            self._update_status("Exporting to OBJ...")
            output_path = self._export_to_obj(scene)

            self.stats.end_time = time.time()
            self.stats.total_files_processed = (
                self.stats.ide_files
                + self.stats.ipl_files
                + self.stats.dff_files
                + self.stats.txd_files
                + self.stats.img_archives
            )

            success = os.path.exists(output_path)
            if success:
                message = f"Conversion completed successfully in {self.stats.get_duration():.2f} seconds"
                self._log_message(message, "SUCCESS")
                return True, output_path, message

            message = "Failed to create output file"
            self._log_message(message, "ERROR")
            return False, "", message

        except Exception as e:
            error_msg = f"Conversion failed with error: {e}"
            self._log_message(error_msg, "ERROR")
            traceback.print_exc()
            return False, "", error_msg

        finally:
            self.is_running = False
            self._cleanup_temp_files()

    # ------------------------------------------------------------------ validation & file collection

    def _validate_config(self) -> bool:
        """Validate configuration and paths (no IMG archives, use extracted folder)."""
        errors: List[str] = []

        # Directories
        if not os.path.exists(self.config.paths.img_dir):
            errors.append(
                f"Model/texture source directory does not exist: {self.config.paths.img_dir}"
            )

        if not os.path.exists(self.config.paths.maps_dir):
            errors.append(
                f"Maps directory does not exist: {self.config.paths.maps_dir}"
            )

        # IDE files
        ide_files = self.config.paths.get_ide_files()
        if not ide_files:
            errors.append("No .ide files found in maps directory")

        # At least some DFF/TXD
        source_dir = self.config.paths.img_dir
        dff_count = 0
        txd_count = 0
        if os.path.exists(source_dir):
            for root, _, files in os.walk(source_dir):
                for f in files:
                    ext = os.path.splitext(f)[1].lower()
                    if ext == ".dff":
                        dff_count += 1
                    elif ext == ".txd":
                        txd_count += 1

        if dff_count == 0:
            errors.append(f"No .dff files found in source folder: {source_dir}")
        if self.config.conversion.export_textures and txd_count == 0:
            self._log_message(
                f"No .txd files found in source folder: {source_dir} (textures may be missing)",
                "WARNING",
            )

        # Output dir
        try:
            os.makedirs(self.config.paths.output_dir, exist_ok=True)
        except Exception as e:
            errors.append(f"Cannot create output directory: {e}")

        if errors:
            for error in errors:
                self._log_message(error, "ERROR")
            return False

        self._log_message(
            f"Source folder: {source_dir} | DFF: {dff_count}, TXD: {txd_count} | "
            f"IDE files: {len(ide_files)}",
            "INFO",
        )
        return True

    def _extract_img_archives(self) -> Dict[str, str]:
        """
        Collect already-extracted DFF / TXD files from a folder.

        Instead of reading GTA3.IMG, we assume self.config.paths.img_dir
        points to a directory that already contains unpacked .dff/.txd files.
        """
        start_time = time.time()
        source_dir = self.config.paths.img_dir
        extracted: Dict[str, str] = {}

        if not os.path.exists(source_dir):
            self._log_message(
                f"Source directory does not exist: {source_dir}", "ERROR"
            )
            return {}

        self._log_message(
            f"Scanning source folder for DFF/TXD: {source_dir}", "INFO"
        )

        exts = {".dff", ".txd"}
        for root, _, files in os.walk(source_dir):
            for fname in files:
                ext = os.path.splitext(fname)[1].lower()
                if ext in exts:
                    full_path = os.path.join(root, fname)
                    extracted[fname.lower()] = full_path

        dff_count = len([f for f in extracted if f.endswith(".dff")])
        txd_count = len([f for f in extracted if f.endswith(".txd")])

        self._log_message(
            f"Found {dff_count} DFF and {txd_count} TXD files in source folder",
            "INFO",
        )

        self.stats.extraction_time = time.time() - start_time
        self.extracted_files = extracted
        self.stats.img_archives = 0
        return extracted

    # ------------------------------------------------------------------ IDE/IPL parsing

    def _parse_ide_files(self) -> Dict[int, Any]:
        start_time = time.time()
        try:
            ide_files = self.config.paths.get_ide_files()
            self.stats.ide_files = len(ide_files)

            if not ide_files:
                self._log_message("No IDE files found", "WARNING")
                return {}

            self._update_progress(
                "Parsing IDE files", 10, f"0/{len(ide_files)}", 0
            )

            for i, ide_file in enumerate(ide_files):
                if self._check_cancel():
                    return {}

                self._update_progress(
                    "Parsing IDE files",
                    10 + int((i / len(ide_files)) * 10),
                    f"{i+1}/{len(ide_files)}: {os.path.basename(ide_file)}",
                    int((i / len(ide_files)) * 100),
                )

                try:
                    success = self.ide_manager.parse_file(ide_file)
                    if success:
                        self._log_message(
                            f"Parsed IDE file: {os.path.basename(ide_file)}", "INFO"
                        )
                    else:
                        self._log_message(
                            f"Failed to parse IDE file: {ide_file}", "WARNING"
                        )
                        self.stats.warnings_encountered += 1
                except Exception as e:
                    self._log_message(
                        f"Error parsing IDE file {ide_file}: {e}", "ERROR"
                    )
                    self.stats.errors_encountered += 1

            self._log_message(
                f"Parsed {len(self.ide_manager.get_all_objects())} objects from {len(ide_files)} IDE files",
                "INFO",
            )

            self.stats.parsing_time += time.time() - start_time
            return self.ide_manager.get_all_objects()

        except Exception as e:
            self._log_message(f"Error in IDE parsing: {e}", "ERROR")
            self.stats.errors_encountered += 1
            return {}

    def _parse_ipl_files(self) -> List[Dict[str, Any]]:
        start_time = time.time()
        try:
            ipl_files = self.config.paths.get_ipl_files()
            self.stats.ipl_files = len(ipl_files)

            if not ipl_files:
                self._log_message("No IPL files found", "WARNING")
                return []

            self._update_progress(
                "Parsing IPL files", 30, f"0/{len(ipl_files)}", 0
            )

            all_placements: List[Dict[str, Any]] = []
            for i, ipl_file in enumerate(ipl_files):
                if self._check_cancel():
                    return []

                self._update_progress(
                    "Parsing IPL files",
                    30 + int((i / len(ipl_files)) * 10),
                    f"{i+1}/{len(ipl_files)}: {os.path.basename(ipl_file)}",
                    int((i / len(ipl_files)) * 100),
                )

                try:
                    placements = self.ipl_manager.parse_file(ipl_file)
                    if placements:
                        all_placements.extend(placements)
                        self._log_message(
                            f"Parsed {len(placements)} placements from {os.path.basename(ipl_file)}",
                            "INFO",
                        )
                    else:
                        self._log_message(
                            f"No placements found in IPL file: {ipl_file}", "WARNING"
                        )
                        self.stats.warnings_encountered += 1
                except Exception as e:
                    self._log_message(
                        f"Error parsing IPL file {ipl_file}: {e}", "ERROR"
                    )
                    self.stats.errors_encountered += 1

            self._log_message(
                f"Total placements parsed: {len(all_placements)}", "INFO"
            )

            self.stats.parsing_time += time.time() - start_time
            return all_placements

        except Exception as e:
            self._log_message(f"Error in IPL parsing: {e}", "ERROR")
            self.stats.errors_encountered += 1
            return []

    # ------------------------------------------------------------------ DFF / TXD conversion (unchanged except using extracted_files)

    def _convert_dff_files(
        self, extracted_files: Dict[str, str]
    ) -> Dict[str, Dict[str, Any]]:
        start_time = time.time()
        try:
            dff_files: List[str] = [
                p for p in extracted_files.values() if p.lower().endswith(".dff")
            ]
            self.stats.dff_files = len(dff_files)

            if not dff_files:
                self._log_message("No DFF files found", "WARNING")
                return {}

            self._update_progress(
                "Converting DFF models", 70, f"0/{len(dff_files)}", 0
            )

            all_models: Dict[str, Dict[str, Any]] = {}

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_file = {
                    executor.submit(self._convert_single_dff, dff_file): dff_file
                    for dff_file in dff_files
                }

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

                            vertices = len(model_data.get("vertices", []))
                            faces = len(model_data.get("faces", []))
                            self.stats.total_vertices += vertices
                            self.stats.total_faces += faces

                        completed += 1
                        self._update_progress(
                            "Converting DFF models",
                            70 + int((completed / len(dff_files)) * 10),
                            f"{completed}/{len(dff_files)}: {Path(dff_file).name}",
                            int((completed / len(dff_files)) * 100),
                        )

                    except Exception as e:
                        self._log_message(
                            f"Error converting DFF file {dff_file}: {e}", "ERROR"
                        )
                        self.stats.errors_encountered += 1

            self._log_message(f"Converted {len(all_models)} DFF models", "INFO")

            self.stats.conversion_time += time.time() - start_time
            self.converted_models = all_models
            return all_models

        except Exception as e:
            self._log_message(f"Error in DFF conversion: {e}", "ERROR")
            self.stats.errors_encountered += 1
            return {}

    def _convert_single_dff(self, dff_path: str) -> Optional[Dict[str, Any]]:
        try:
            return self.dff_converter.convert(dff_path)
        except Exception as e:
            self._log_message(f"Error converting {dff_path}: {e}", "ERROR")
            return None

    def _convert_txd_files(
        self, extracted_files: Dict[str, str]
    ) -> Dict[str, Dict[str, Any]]:
        start_time = time.time()
        try:
            txd_files: List[str] = [
                p for p in extracted_files.values() if p.lower().endswith(".txd")
            ]
            self.stats.txd_files = len(txd_files)

            if not txd_files or not self.config.conversion.export_textures:
                if not self.config.conversion.export_textures:
                    self._log_message("Texture export disabled", "INFO")
                return {}

            self._update_progress(
                "Converting textures", 85, f"0/{len(txd_files)}", 0
            )

            all_textures: Dict[str, Dict[str, Any]] = {}

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_file = {
                    executor.submit(self._convert_single_txd, txd_file): txd_file
                    for txd_file in txd_files
                }

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
                            int((completed / len(txd_files)) * 100),
                        )

                    except Exception as e:
                        self._log_message(
                            f"Error converting TXD file {txd_file}: {e}", "ERROR"
                        )
                        self.stats.errors_encountered += 1

            self._log_message(f"Converted {len(all_textures)} textures", "INFO")

            self.stats.conversion_time += time.time() - start_time
            self.converted_textures = all_textures
            return all_textures

        except Exception as e:
            self._log_message(f"Error in TXD conversion: {e}", "ERROR")
            self.stats.errors_encountered += 1
            return {}

    def _convert_single_txd(self, txd_path: str) -> Optional[Dict[str, Dict]]:
        try:
            texture_format = self.config.conversion.get_texture_extension()
            texture_size = self.config.conversion.get_texture_size()
            return self.txd_converter.convert(
                txd_path,
                self.config.paths.output_dir,
                texture_format,
                texture_size,
                self.config.conversion.texture_quality,
            )
        except Exception as e:
            self._log_message(f"Error converting {txd_path}: {e}", "ERROR")
            return None

    # ------------------------------------------------------------------ scene building & export

    def _build_scene(
        self,
        ide_data: Dict[int, Any],
        ipl_data: List[Dict[str, Any]],
        models: Dict[str, Any],
        textures: Dict[str, Any],
    ) -> Dict[str, Any]:
        start_time = time.time()
        try:
            self._update_progress(
                "Building scene", 95, "Processing objects", 0
            )

            materials = self.material_builder.create_materials(textures)
            self.materials = materials

            scene_objects: List[SceneObject] = []
            placed = 0
            skipped = 0

            for i, placement in enumerate(ipl_data):
                if self._check_cancel():
                    break

                if i % 100 == 0:
                    self._update_progress(
                        "Building scene",
                        95 + int((i / max(1, len(ipl_data))) * 2),
                        f"Placements: {i}/{len(ipl_data)}",
                        int((i / max(1, len(ipl_data))) * 100),
                    )

                try:
                    obj_id = placement.get("id")
                    if obj_id is None:
                        continue

                    ide_obj = ide_data.get(obj_id)
                    if not ide_obj:
                        model_name = placement.get("model_name")
                        if model_name:
                            for obj in ide_data.values():
                                if getattr(obj, "model_name", "").lower() == model_name.lower():
                                    ide_obj = obj
                                    break

                    if not ide_obj:
                        skipped += 1
                        continue

                    model_name = getattr(ide_obj, "model_name", f"object_{obj_id}")
                    model_data = models.get(model_name)
                    if not model_data:
                        skipped += 1
                        continue

                    scene_obj = SceneObject(
                        id=obj_id,
                        model_name=model_name,
                        model_data=model_data,
                        position=placement.get("position", (0, 0, 0)),
                        rotation=placement.get("rotation", (0, 0, 0)),
                        scale=placement.get("scale", (1, 1, 1)),
                        flags=placement.get("flags", 0),
                        draw_distance=getattr(ide_obj, "draw_distance", 300.0),
                        texture_dict=getattr(ide_obj, "texture_dictionary", ""),
                        lod_level=placement.get("lod", 0),
                        parent_id=placement.get("parent_id", -1),
                        time_on=getattr(ide_obj, "time_on", 0),
                        time_off=getattr(ide_obj, "time_off", 24),
                        interior=placement.get("interior", 0),
                    )

                    scene_objects.append(scene_obj)
                    placed += 1

                except Exception as e:
                    self._log_message(
                        f"Error processing placement {i}: {e}", "ERROR"
                    )
                    self.stats.errors_encountered += 1

            self.stats.total_objects_placed = placed

            scene = {
                "name": "GTA_SA_Map",
                "objects": scene_objects,
                "materials": materials,
                "textures": textures,
                "models": models,
                "settings": {
                    "scale_factor": self.config.conversion.scale_factor,
                    "coordinate_system": self.config.conversion.coordinate_system,
                    "units": self.config.conversion.units,
                },
            }

            self._log_message(
                f"Scene built: {placed} objects placed, {skipped} skipped", "INFO"
            )
            self.scene_objects = scene_objects
            return scene

        except Exception as e:
            self._log_message(f"Error building scene: {e}", "ERROR")
            self.stats.errors_encountered += 1
            return {}

        finally:
            self.stats.conversion_time += time.time() - start_time

    def _export_to_obj(self, scene: Dict[str, Any]) -> str:
        start_time = time.time()
        try:
            self._update_progress("Exporting to OBJ", 98, "Writing files", 0)

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_filename = f"gta_sa_map_{timestamp}.obj"
            output_path = os.path.join(self.config.paths.output_dir, output_filename)

            try:
                os.makedirs(self.config.paths.output_dir, exist_ok=True)
            except Exception as e:
                msg = f"Cannot create output directory '{self.config.paths.output_dir}': {e}"
                self._log_message(msg, "ERROR")
                raise RuntimeError(msg)

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

            if not exported_ok:
                msg = "OBJ exporter returned failure (export returned False)"
                self._log_message(msg, "ERROR")
                raise RuntimeError(msg)

            if not os.path.exists(output_path):
                msg = f"OBJ exporter reported success but file not found: {output_path}"
                self._log_message(msg, "ERROR")
                raise RuntimeError(msg)

            self._log_message(f"Exported to: {output_path}", "SUCCESS")
            self.stats.export_time = time.time() - start_time
            return output_path

        except Exception:
            raise

    # ------------------------------------------------------------------ misc

    def _cleanup_temp_files(self):
        try:
            if not self.config.performance.keep_temp_files:
                temp_dir = Path(self.config.paths.temp_dir)
                if temp_dir.exists():
                    import shutil
                    shutil.rmtree(temp_dir)
                    self._log_message("Cleaned up temporary files", "INFO")
        except Exception as e:
            self._log_message(f"Error cleaning temp files: {e}", "WARNING")

    def cancel(self):
        self.cancel_requested = True
        self._log_message("Cancellation requested", "INFO")

    def get_stats(self) -> ConversionStats:
        return self.stats

    def get_progress(self) -> Dict[str, Any]:
        return {
            "current_stage": self.current_stage,
            "is_running": self.is_running,
            "stats": self.stats.get_summary(),
        }


class BatchProcessor:
    """Process conversions in batch mode"""

    def __init__(self, config: Config):
        self.config = config
        self.pipelines: List[ConversionPipeline] = []
        self.results: List[Dict[str, Any]] = []

    def add_conversion(
        self, img_dir: str, maps_dir: str, output_dir: str
    ) -> ConversionPipeline:
        pipeline = ConversionPipeline(self.config)
        pipeline.config.paths.img_dir = img_dir
        pipeline.config.paths.maps_dir = maps_dir
        pipeline.config.paths.output_dir = output_dir
        self.pipelines.append(pipeline)
        return pipeline

    def run_all(self, parallel: bool = False) -> List[Dict[str, Any]]:
        self.results.clear()
        if parallel:
            return self._run_parallel()
        return self._run_sequential()

    def _run_sequential(self) -> List[Dict[str, Any]]:
        for i, pipeline in enumerate(self.pipelines):
            print(f"Processing conversion {i + 1}/{len(self.pipelines)}")
            try:
                success, output_path, message = pipeline.convert()
                result = {
                    "index": i,
                    "success": success,
                    "output_path": output_path,
                    "message": message,
                    "stats": pipeline.get_stats().get_summary(),
                }
                self.results.append(result)
            except Exception as e:
                result = {
                    "index": i,
                    "success": False,
                    "output_path": "",
                    "message": str(e),
                    "stats": {},
                }
                self.results.append(result)
        return self.results

    def _run_parallel(self) -> List[Dict[str, Any]]:
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
                        "index": i,
                        "success": success,
                        "output_path": output_path,
                        "message": message,
                    }
                    self.results.append(result)
                except Exception as e:
                    result = {
                        "index": i,
                        "success": False,
                        "output_path": "",
                        "message": str(e),
                    }
                    self.results.append(result)
        return self.results

    def _run_single_conversion(
        self, index: int, pipeline: ConversionPipeline
    ) -> Tuple[bool, str, str]:
        return pipeline.convert()

    def get_summary(self) -> Dict[str, Any]:
        total = len(self.results)
        successful = sum(1 for r in self.results if r["success"])
        failed = total - successful
        total_time = sum(r.get("stats", {}).get("total_time", 0) for r in self.results)
        return {
            "total_conversions": total,
            "successful": successful,
            "failed": failed,
            "total_time": total_time,
            "results": self.results,
        }