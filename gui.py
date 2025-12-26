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


class QtLogHandler(logging.Handler):
    def __init__(self, widget):
        super().__init__()
        self.widget = widget

    def emit(self, record):
        msg = self.format(record)
        level = record.levelname
        color = {
            "INFO": "#5A9CF8",
            "WARNING": "#F1C40F",
            "ERROR": "#E74C3C",
            "CRITICAL": "#E74C3C",
            "SUCCESS": "#2ECC71",
        }.get(level, "#E1E1E6")
        self.widget.moveCursor(QTextCursor.End)
        self.widget.insertHtml(f'<span style="color:{color}">{msg}</span><br>')
        self.widget.moveCursor(QTextCursor.End)


class ClickableLineEdit(QLineEdit):
    clicked = pyqtSignal()

    def mousePressEvent(self, e):
        super().mousePressEvent(e)
        if e.button() == Qt.LeftButton:
            self.clicked.emit()


class DirectorySelector(QWidget):
    def __init__(self, label, path="", title="", parent=None):
        super().__init__(parent)
        self.dialog_title = title
        l = QHBoxLayout(self)
        l.setContentsMargins(0, 0, 0, 0)

        lbl = QLabel(label)
        lbl.setMinimumWidth(120)
        l.addWidget(lbl)

        self.edit = ClickableLineEdit()
        self.edit.setText(path)
        self.edit.clicked.connect(self.browse)
        l.addWidget(self.edit)

        self.browse_btn = QPushButton()
        self.browse_btn.setIcon(self.style().standardIcon(QStyle.SP_DirOpenIcon))
        self.browse_btn.setFixedWidth(36)
        self.browse_btn.clicked.connect(self.browse)
        l.addWidget(self.browse_btn)

        self.open_btn = QPushButton()
        self.open_btn.setIcon(self.style().standardIcon(QStyle.SP_DesktopIcon))
        self.open_btn.setFixedWidth(36)
        self.open_btn.clicked.connect(self.open)
        l.addWidget(self.open_btn)

    def browse(self):
        p = QFileDialog.getExistingDirectory(self, self.dialog_title)
        if p:
            self.edit.setText(p)
            self.edit.setStyleSheet("")

    def open(self):
        p = self.edit.text().strip()
        if os.path.exists(p):
            if sys.platform.startswith("linux"):
                subprocess.run(["xdg-open", p], check=False)
            elif sys.platform == "darwin":
                subprocess.run(["open", p], check=False)
            elif sys.platform.startswith("win"):
                os.startfile(p)

    def is_valid(self):
        p = self.edit.text().strip()
        return os.path.exists(p) if p else False

    def set_valid(self, v):
        self.edit.setStyleSheet("border:2px solid #27ae60;" if v else "border:2px solid #e74c3c;")

    def path(self):
        return self.edit.text().strip()


class MainWindow(QMainWindow):
    conversion_started = pyqtSignal()
    conversion_stopped = pyqtSignal()

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.logger = get_logger("GUI")
        self.progress = ProgressTracker()
        self.is_converting = False
        self.init_ui()
        self.load_settings()
        self.apply_theme()
        self.hook_logging()
        self.hook_progress()

    def init_ui(self):
        self.setWindowTitle("GTA San Andreas Map Converter")
        self.resize(920, 680)

        central = QWidget()
        main = QVBoxLayout(central)
        main.setContentsMargins(12, 12, 12, 12)
        main.setSpacing(10)

        header = QWidget()
        header.setFixedHeight(36)
        hl = QHBoxLayout(header)
        hl.setContentsMargins(6, 0, 6, 0)

        title = QLabel("GTA SA Map Converter")
        title.setStyleSheet("font-weight:600;")
        hl.addWidget(title)
        hl.addStretch()

        self.theme_toggle = QCheckBox()
        self.theme_toggle.setChecked(self.config.ui.theme == "dark")
        self.theme_toggle.toggled.connect(self.toggle_theme)
        hl.addWidget(self.theme_toggle)

        main.addWidget(header)

        dirs = QGroupBox("Directories")
        gl = QGridLayout(dirs)

        self.img_dir = DirectorySelector("GTA_SA_map:", self.config.paths.img_dir, "Select GTA map")
        self.maps_dir = DirectorySelector("Maps folder:", self.config.paths.maps_dir, "Select maps folder")
        self.out_dir = DirectorySelector("Output folder:", self.config.paths.output_dir, "Select output folder")

        gl.addWidget(self.img_dir, 0, 0, 1, 2)
        gl.addWidget(self.maps_dir, 1, 0, 1, 2)
        gl.addWidget(self.out_dir, 2, 0, 1, 2)

        self.validate_btn = QPushButton("Validate")
        self.validate_btn.clicked.connect(self.validate_all)
        gl.addWidget(self.validate_btn, 3, 0, 1, 2)

        main.addWidget(dirs)

        actions = QWidget()
        al = QHBoxLayout(actions)

        self.start_btn = QPushButton("Start Conversion")
        self.start_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.start_btn.clicked.connect(self.start_conversion)
        al.addWidget(self.start_btn)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setIcon(self.style().standardIcon(QStyle.SP_BrowserStop))
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel_conversion)
        al.addWidget(self.cancel_btn)

        al.addStretch()
        main.addWidget(actions)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        main.addWidget(self.progress_bar)

        log_box = QGroupBox("Log")
        ll = QVBoxLayout(log_box)
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setFont(QFont("Consolas", 10))
        ll.addWidget(self.log_view)
        main.addWidget(log_box)

        self.setCentralWidget(central)

    def hook_logging(self):
        h = QtLogHandler(self.log_view)
        h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logging.getLogger().addHandler(h)

    def hook_progress(self):
        self.progress.add_callback(self.on_progress)

    def on_progress(self, percent, message=None):
        self.progress_bar.setValue(percent)
        if message:
            self.logger.info(message)

    def toggle_theme(self, enabled):
        self.config.ui.theme = "dark" if enabled else "light"
        self.apply_theme()
        self.config.save()

    def apply_theme(self):
        app = QApplication.instance()
        if self.config.ui.theme == "dark":
            p = Path("resources/styles/dark.qss")
            if p.exists():
                app.setStyleSheet(p.read_text())
        else:
            app.setStyleSheet("")

    def validate_all(self):
        v1 = self.img_dir.is_valid()
        v2 = self.maps_dir.is_valid()
        v3 = self.out_dir.is_valid()
        self.img_dir.set_valid(v1)
        self.maps_dir.set_valid(v2)
        self.out_dir.set_valid(v3)
        if v1 and v2 and v3:
            QMessageBox.information(self, "Validation", "All directories are valid")
        else:
            QMessageBox.warning(self, "Validation", "One or more directories are invalid")

    def start_conversion(self):
        if self.is_converting:
            return
        self.is_converting = True
        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.conversion_started.emit()
        self.logger.info("Conversion started")

    def cancel_conversion(self):
        if not self.is_converting:
            return
        self.is_converting = False
        self.cancel_btn.setEnabled(False)
        self.start_btn.setEnabled(True)
        self.conversion_stopped.emit()
        self.logger.warning("Conversion cancelled")

    def load_settings(self):
        pass

    def closeEvent(self, e):
        if self.is_converting:
            r = QMessageBox.question(self, "Exit", "Conversion running. Quit?")
            if r != QMessageBox.Yes:
                e.ignore()
                return
            self.conversion_stopped.emit()
        self.config.save()
        e.accept()
