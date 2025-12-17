import os
import sys
import subprocess
import logging
from pathlib import Path
from typing import List, Tuple

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

from config import Config
from utils.logger import get_logger
from utils.progress_tracker import ProgressTracker


class ClickableLineEdit(QLineEdit):
    clicked = pyqtSignal()

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        if event.button() == Qt.LeftButton:
            self.clicked.emit()


class DirectorySelector(QWidget):
    def __init__(self, label_text: str, default_path: str = "", dialog_title: str = "Select Directory", parent=None):
        super().__init__(parent)
        self.dialog_title = dialog_title
        self.init_ui(label_text, default_path)

    def init_ui(self, label_text: str, default_path: str):
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        label = QLabel(label_text)
        label.setMinimumWidth(120)
        layout.addWidget(label)

        self.path_edit = ClickableLineEdit()
        self.path_edit.setText(default_path)
        self.path_edit.setPlaceholderText("Click to browse or enter path...")
        self.path_edit.clicked.connect(self.browse_directory)
        layout.addWidget(self.path_edit)

        self.browse_btn = QPushButton("...")
        self.browse_btn.setFixedWidth(40)
        self.browse_btn.clicked.connect(self.browse_directory)
        layout.addWidget(self.browse_btn)

        self.open_btn = QPushButton("📂")
        self.open_btn.setFixedWidth(30)
        self.open_btn.setToolTip("Open directory")
        self.open_btn.clicked.connect(self.open_directory)
        layout.addWidget(self.open_btn)

        self.setLayout(layout)

    def browse_directory(self):
        path = QFileDialog.getExistingDirectory(self, self.dialog_title)
        if path:
            self.path_edit.setText(path)
            self.path_edit.setStyleSheet("")

    def open_directory(self):
        path = self.get_path()
        if os.path.exists(path):
            try:
                if sys.platform.startswith("linux"):
                    subprocess.run(['xdg-open', path], check=False)
                elif sys.platform == "darwin":
                    subprocess.run(['open', path], check=False)
                elif sys.platform.startswith("win"):
                    os.startfile(path)
            except Exception:
                # ignore errors opening folder, but log them
                logging.exception("Failed to open directory: %s", path)

    def get_path(self) -> str:
        return self.path_edit.text().strip()

    def set_path(self, path: str):
        self.path_edit.setText(path)

    def set_valid(self, valid: bool):
        style = "border: 2px solid #27ae60;" if valid else "border: 2px solid #e74c3c;"
        self.path_edit.setStyleSheet(style)

    def is_valid(self) -> bool:
        path = self.get_path()
        return os.path.exists(path) if path else False


class MainWindow(QMainWindow):
    conversion_started = pyqtSignal()
    conversion_stopped = pyqtSignal()

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.logger = get_logger("GUI")
        self.progress_tracker = ProgressTracker()
        self.conversion_thread = None
        self.is_converting = False
        self._last_validation_errors: List[str] = []

        self.init_ui()
        self.load_settings()
        self.setup_connections()

    def init_ui(self):
        self.setWindowTitle("GTA San Andreas Map Converter for Blender 2.79")
        self.setGeometry(100, 100, 900, 600)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        main_widget = QWidget()
        self.main_layout = QVBoxLayout(main_widget)

        self.create_directory_section()
        self.create_button_section()

        scroll.setWidget(main_widget)
        self.setCentralWidget(scroll)

    def create_directory_section(self):
        section = QGroupBox("Directories")
        layout = QGridLayout()

        self.img_dir_selector = DirectorySelector(
            "GTA_SA_map:", self.config.paths.img_dir, "Select GTA_SA map directory"
        )
        layout.addWidget(self.img_dir_selector, 0, 0, 1, 2)

        self.maps_dir_selector = DirectorySelector(
            "Maps folder:", self.config.paths.maps_dir, "Select maps directory"
        )
        layout.addWidget(self.maps_dir_selector, 1, 0, 1, 2)

        self.output_dir_selector = DirectorySelector(
            "Output folder:", self.config.paths.output_dir, "Select output directory"
        )
        layout.addWidget(self.output_dir_selector, 2, 0, 1, 2)

        self.validate_btn = QPushButton("Validate Directories")
        self.validate_btn.clicked.connect(self.validate_directories)
        layout.addWidget(self.validate_btn, 3, 0, 1, 2)

        section.setLayout(layout)
        self.main_layout.addWidget(section)

    def create_button_section(self):
        section = QWidget()
        layout = QHBoxLayout()

        self.convert_btn = QPushButton("🚀 START CONVERSION")
        self.convert_btn.setMinimumHeight(40)
        self.convert_btn.clicked.connect(self.start_conversion)
        layout.addWidget(self.convert_btn)

        self.cancel_btn = QPushButton("✗ CANCEL")
        self.cancel_btn.setMinimumHeight(40)
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel_conversion)
        layout.addWidget(self.cancel_btn)

        section.setLayout(layout)
        self.main_layout.addWidget(section)

    def setup_connections(self):
        # keep UI responsive to user edits
        self.img_dir_selector.path_edit.textChanged.connect(
            lambda: self.validate_directory(self.img_dir_selector)
        )
        self.maps_dir_selector.path_edit.textChanged.connect(
            lambda: self.validate_directory(self.maps_dir_selector)
        )
        self.output_dir_selector.path_edit.textChanged.connect(
            lambda: self.validate_directory(self.output_dir_selector)
        )

    def load_settings(self):
        try:
            self.img_dir_selector.set_path(self.config.paths.img_dir)
            self.maps_dir_selector.set_path(self.config.paths.maps_dir)
            self.output_dir_selector.set_path(self.config.paths.output_dir)

            # validate initial values
            self.validate_directory(self.img_dir_selector)
            self.validate_directory(self.maps_dir_selector)
            self.validate_directory(self.output_dir_selector)
        except Exception:
            self.logger.exception("Failed to load settings")

    def update_config(self):
        self.config.paths.img_dir = self.img_dir_selector.get_path()
        self.config.paths.maps_dir = self.maps_dir_selector.get_path()
        self.config.paths.output_dir = self.output_dir_selector.get_path()
        try:
            self.config.save()
        except Exception:
            self.logger.exception("Failed to save config")

    def validate_directory(self, selector: DirectorySelector) -> bool:
        valid = selector.is_valid()
        selector.set_valid(valid)
        return valid

    def validate_directories(self):
        valid_img = self.validate_directory(self.img_dir_selector)
        valid_maps = self.validate_directory(self.maps_dir_selector)
        valid_out = self.validate_directory(self.output_dir_selector)

        if valid_img and valid_maps and valid_out:
            QMessageBox.information(self, "Validation", "All directories are valid!")
        else:
            msgs = []
            if not valid_img:
                msgs.append("GTA_SA_map directory does not exist or is invalid.")
            if not valid_maps:
                msgs.append("Maps directory does not exist or is invalid.")
            if not valid_out:
                msgs.append("Output directory does not exist or is invalid.")
            QMessageBox.warning(self, "Validation", "\n".join(msgs))

    def start_conversion(self):
        """
        Invoked when the START CONVERSION button is pressed.
        This method performs input validation, updates config, updates UI state,
        and emits conversion_started. It logs and displays helpful messages so the
        user knows what's happening.
        """
        self.logger.info("start_conversion() called")
        print("DEBUG: start_conversion() called")

        valid, errors = self.validate_inputs()
        print(f"DEBUG: validate_inputs returned {valid} errors={errors}")
        self.logger.info("validate_inputs returned %s errors=%s", valid, errors)

        if not valid:
            # Show a clear message with all validation errors
            QMessageBox.warning(self, "Conversion - Invalid Inputs", "\n".join(errors))
            self.logger.warning("Conversion aborted due to invalid inputs: %s", errors)
            return

        # update config before starting
        self.update_config()

        # update UI state
        self.convert_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.is_converting = True

        # Inform user conversion started
        QMessageBox.information(self, "Conversion", "Conversion started!")
        self.logger.info("Conversion started - emitting conversion_started")

        # Emit signal and catch any exceptions thrown by connected slots
        try:
            self.conversion_started.emit()
        except Exception:
            self.logger.exception("Exception occurred while emitting conversion_started")
            # Re-enable controls so user can try again
            self.is_converting = False
            self.convert_btn.setEnabled(True)
            self.cancel_btn.setEnabled(False)
            QMessageBox.critical(self, "Conversion", "An unexpected error occurred when starting conversion. See logs for details.")

    def cancel_conversion(self):
        self.is_converting = False
        # If there's a long running thread, conversion_stopped is expected to be handled by it
        try:
            self.conversion_stopped.emit()
        except Exception:
            self.logger.exception("Exception while emitting conversion_stopped")

        self.convert_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        QMessageBox.information(self, "Conversion", "Conversion cancelled.")
        self.logger.info("Conversion cancelled by user")

    def validate_inputs(self) -> Tuple[bool, List[str]]:
        """
        Validate user inputs. Returns (is_valid, errors_list).
        Validation includes checking that directories exist and that
        required files are present where reasonable.
        """
        errors: List[str] = []

        # check directories exist
        if not self.img_dir_selector.is_valid():
            errors.append("GTA_SA_map directory does not exist.")
        if not self.maps_dir_selector.is_valid():
            errors.append("Maps directory does not exist.")
        if not self.output_dir_selector.is_valid():
            # If output path doesn't exist, try to create it. If that fails, report error.
            out_path = self.output_dir_selector.get_path()
            if out_path:
                try:
                    Path(out_path).mkdir(parents=True, exist_ok=True)
                    # re-check validity after attempting to create
                    if not self.output_dir_selector.is_valid():
                        errors.append("Output directory is invalid or not writable.")
                except Exception:
                    errors.append("Output directory does not exist and could not be created.")
            else:
                errors.append("Output directory is not set.")

        # optional: check maps folder contains files (helpful to catch empty selection)
        maps_path = self.maps_dir_selector.get_path()
        if maps_path and os.path.exists(maps_path):
            try:
                # check that there is at least one file - avoid specifying extensions in case user has custom names
                if not any(os.path.isfile(os.path.join(maps_path, f)) for f in os.listdir(maps_path)):
                    errors.append("Maps directory appears empty. Please ensure it contains map files.")
            except Exception:
                self.logger.exception("Error while inspecting maps directory")
                errors.append("Could not read Maps directory to verify files (permission error?).")

        # If there are errors, store them for UI reference and return False
        self._last_validation_errors = errors
        is_valid = len(errors) == 0
        return is_valid, errors
