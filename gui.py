import os
import sys
import subprocess
from pathlib import Path
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
    def __init__(self, label_text, default_path="", dialog_title="Select Directory", parent=None):
        super().__init__(parent)
        self.dialog_title = dialog_title
        self.init_ui(label_text, default_path)

    def init_ui(self, label_text, default_path):
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
            subprocess.run(['xdg-open', path])

    def get_path(self):
        return self.path_edit.text().strip()

    def set_path(self, path):
        self.path_edit.setText(path)

    def set_valid(self, valid):
        style = "border: 2px solid #27ae60;" if valid else "border: 2px solid #e74c3c;"
        self.path_edit.setStyleSheet(style)

    def is_valid(self):
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
            "GTA_SA_map:", self.config.paths.img_dir, "Select GTA_SA_map directory"
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
        self.img_dir_selector.path_edit.textChanged.connect(
            lambda: self.validate_directory(self.img_dir_selector)
        )
        self.maps_dir_selector.path_edit.textChanged.connect(
            lambda: self.validate_directory(self.maps_dir_selector)
        )

    def load_settings(self):
        self.img_dir_selector.set_path(self.config.paths.img_dir)
        self.maps_dir_selector.set_path(self.config.paths.maps_dir)
        self.output_dir_selector.set_path(self.config.paths.output_dir)

        self.validate_directory(self.img_dir_selector)
        self.validate_directory(self.maps_dir_selector)

    def update_config(self):
        self.config.paths.img_dir = self.img_dir_selector.get_path()
        self.config.paths.maps_dir = self.maps_dir_selector.get_path()
        self.config.paths.output_dir = self.output_dir_selector.get_path()
        self.config.save()

    def validate_directory(self, selector) -> bool:
        valid = selector.is_valid()
        selector.set_valid(valid)
        return valid

    def validate_directories(self):
        valid_img = self.validate_directory(self.img_dir_selector)
        valid_maps = self.validate_directory(self.maps_dir_selector)

        if valid_img and valid_maps:
            QMessageBox.information(self, "Validation", "All directories are valid!")
        else:
            QMessageBox.warning(self, "Validation", "Some directories are invalid.")

    def start_conversion(self):
        if not self.validate_inputs():
            return

        self.update_config()
        self.convert_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.is_converting = True

        QMessageBox.information(self, "Conversion", "Conversion started!")
        self.conversion_started.emit()

    def cancel_conversion(self):
        self.is_converting = False
        self.convert_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        QMessageBox.information(self, "Conversion", "Conversion cancelled.")
        self.conversion_stopped.emit()

    def validate_inputs(self) -> bool:
        errors = []

        if not self.img_dir_selector.is_valid():
            errors.append("GTA_SA_map directory does not exist.")
        if not self.maps_dir_selector.is_valid():
            errors.append("Maps directory does not exist.")

        if self.img_dir_selector.is_valid():
            img_dir = self.img_dir_selector.get_path()
            img_files = [f for f in os.listdir(img_dir) if f.lower().endswith('.img')]
            if not img_files:
                errors.append("No .img files found in the GTA_SA_map directory.")

        if errors:
            QMessageBox.critical(self, "Input Errors", "\n".join(errors))
            return False

        return True

    def closeEvent(self, event):
        if self.is_converting:
            reply = QMessageBox.question(
                self,
                "Quit?",
                "A conversion is running. Are you sure you want to exit?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                event.ignore()
                return

        self.update_config()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    config = Config()
    window = MainWindow(config)
    window.show()
    sys.exit(app.exec_())
