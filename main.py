import sys
import os
import traceback
import logging
from pathlib import Path
from typing import Optional

try:
    from PyQt5.QtWidgets import QApplication, QMessageBox
    from PyQt5.QtCore import Qt, QThread, pyqtSignal
    PYQT5_AVAILABLE = True
except ImportError:
    print("ERROR: PyQt5 not found!")
    print("Install with: sudo apt-get install python3-pyqt5")
    PYQT5_AVAILABLE = False
    sys.exit(1)

# Import local modules
from resources import get_style
from gui import MainWindow
from config import Config


class ConversionThread(QThread):
    """Background thread for running conversion process"""

    # Signals for communication with GUI
    # Updated to accept 4 args to match ConversionPipeline._update_progress calls:
    # message: str, percent: int, sub_task: str, sub_percent: int
    progress_update = pyqtSignal(str, int, str, int)
    status_update = pyqtSignal(str)
    error_signal = pyqtSignal(str, str)  # error_type, message
    completion_signal = pyqtSignal(bool, str, str)  # success, output_path, message

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.is_running = True
        self.current_task = ""

    def run(self):
        """Main conversion process"""
        try:
            if not self.is_running:
                return

            # Import here to avoid circular imports
            try:
                from core.conversion_pipeline import ConversionPipeline
            except ImportError as e:
                self.error_signal.emit(
                    "Module Error",
                    f"Required module not found: {str(e)}\n"
                    f"Please ensure all modules are installed."
                )
                return

            # Initialize pipeline
            self.status_update.emit("Initializing conversion pipeline...")
            pipeline = ConversionPipeline(self.config)

            # Set up progress callbacks (match thread signal signature)
            # Use lambda wrappers to emit Qt signals
            pipeline.set_progress_callback(lambda msg, pct, sub_task="", sub_pct=0: self.progress_update.emit(msg, pct, sub_task, sub_pct))
            pipeline.set_status_callback(lambda status: self.status_update.emit(status))

            # Run conversion
            self.status_update.emit("Starting conversion process...")
            success, output_path, message = pipeline.convert()

            if success:
                self.completion_signal.emit(True, output_path, message)
            else:
                self.error_signal.emit("Conversion Failed", message)

        except Exception as e:
            error_msg = f"Unexpected error in conversion thread:\n{str(e)}"
            self.error_signal.emit("Runtime Error", error_msg)
            traceback.print_exc()

    def stop(self):
        """Gracefully stop the thread"""
        self.is_running = False
        # Let run() finish; wait briefly for thread termination
        try:
            self.wait(timeout=2000)
        except Exception:
            pass


def check_dependencies():
    """Check if all required dependencies are available"""
    missing = []

    try:
        import numpy  # noqa: F401
    except ImportError:
        missing.append("numpy (pip install numpy)")

    try:
        from PIL import Image  # noqa: F401
    except ImportError:
        missing.append("Pillow (pip install Pillow)")

    # Try to import core modules
    try:
        import core.conversion_pipeline  # noqa: F401
    except ImportError:
        # This might be expected on first run
        pass

    if missing:
        msg = "Missing dependencies:\n" + "\n".join(missing)
        msg += "\n\nInstall with: pip install -r requirements.txt"
        return False, msg

    return True, "All dependencies available"


def setup_environment():
    """Set up environment variables and paths"""
    # Add current directory to Python path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)

    # Create necessary directories
    Path("logs").mkdir(exist_ok=True)
    Path("temp").mkdir(exist_ok=True)

    # Set thread name
    import threading
    threading.current_thread().name = "MainThread"


def handle_uncaught_exception(exctype, value, tb):
    """Global exception handler"""
    error_msg = f"Unhandled Exception:\n\n"
    error_msg += f"Type: {exctype.__name__}\n"
    error_msg += f"Message: {str(value)}\n\n"
    error_msg += "Traceback:\n"

    for line in traceback.format_tb(tb):
        error_msg += line

    # Log to file
    try:
        with open("logs/crash_report.log", "w") as f:
            f.write(error_msg)
    except Exception:
        pass

    # Show error dialog
    if QApplication.instance():
        QMessageBox.critical(
            None,
            "Application Crash",
            f"The application encountered an unexpected error:\n\n{str(value)}\n\n"
            f"Details have been saved to logs/crash_report.log"
        )
    else:
        print(error_msg)

    sys.exit(1)


def main():
    """Main application entry point"""

    # Set up environment
    setup_environment()

    # Set up global exception handler
    sys.excepthook = handle_uncaught_exception

    # Check dependencies
    deps_ok, deps_msg = check_dependencies()
    if not deps_ok:
        print("Dependency Check Failed!")
        print(deps_msg)

        # Try to show GUI error if possible
        app = QApplication(sys.argv)
        QMessageBox.critical(
            None,
            "Missing Dependencies",
            deps_msg
        )
        sys.exit(1)

    # Create Qt application
    app = QApplication(sys.argv)
    app.setApplicationName("GTA SA Map Converter")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("GTA Modding")
    app.setStyle("Fusion")  # Modern look

    # Apply custom style if available
    try:
        style_path = Path("resources/styles/dark.qss")
        if style_path.exists():
            with open(style_path, "r") as f:
                app.setStyleSheet(f.read())
    except Exception:
        pass  # Use default style

    # Load configuration
    config = Config()

    # Create and show main window
    window = MainWindow(config)
    window.show()

    # Center window on screen
    try:
        screen_geometry = app.primaryScreen().availableGeometry()
        window_geometry = window.frameGeometry()
        window_geometry.moveCenter(screen_geometry.center())
        window.move(window_geometry.topLeft())
    except Exception:
        pass

    # Set application icon if available
    try:
        icon_path = Path("resources/icons/app_icon.png")
        if icon_path.exists():
            from PyQt5.QtGui import QIcon
            app.setWindowIcon(QIcon(str(icon_path)))
    except Exception:
        pass

    # --- Conversion thread wiring (adds handlers so pressing START actually runs conversion) ---
    _conversion_thread: Optional[ConversionThread] = None

    def _create_and_start_conversion():
        nonlocal _conversion_thread
        # don't start another if one is already running
        if _conversion_thread is not None and _conversion_thread.isRunning():
            try:
                window.logger.info("Conversion already running; ignoring start request.")
            except Exception:
                pass
            return

        _conversion_thread = ConversionThread(config)

        def _on_progress(msg: str, percent: int, sub_task: str = "", sub_percent: int = 0):
            # Display/log progress (keep UI responsive)
            try:
                if sub_task:
                    window.logger.info(f"Progress: {msg} ({percent}%) - {sub_task} ({sub_percent}%)")
                else:
                    window.logger.info(f"Progress: {msg} ({percent}%)")
            except Exception:
                pass
            if sub_task:
                print(f"[Conversion Progress] {percent}%: {msg} - {sub_task} ({sub_percent}%)")
            else:
                print(f"[Conversion Progress] {percent}%: {msg}")

        def _on_status(status: str):
            try:
                window.logger.info(f"Status: {status}")
            except Exception:
                pass
            print(f"[Conversion Status] {status}")

        def _on_error(err_type: str, message: str):
            try:
                window.logger.error(f"Conversion error - {err_type}: {message}")
            except Exception:
                pass
            print(f"[Conversion Error] {err_type}: {message}")
            # Show critical message in GUI thread
            try:
                QMessageBox.critical(window, f"Conversion Error - {err_type}", message or "An error occurred.")
            except Exception:
                pass
            # restore UI
            try:
                window.convert_btn.setEnabled(True)
                window.cancel_btn.setEnabled(False)
            except Exception:
                pass
            # stop thread if still running
            nonlocal _conversion_thread
            if _conversion_thread:
                try:
                    _conversion_thread.stop()
                except Exception:
                    try:
                        _conversion_thread.quit()
                        _conversion_thread.wait(timeout=1000)
                    except Exception:
                        pass
                _conversion_thread = None

        def _on_completion(success: bool, output_path: str, message: str):
            try:
                window.logger.info(f"Conversion completed: {message or ''}")
            except Exception:
                pass
            print(f"[Conversion Completed] success={success} output={output_path} message={message}")
            try:
                if success:
                    QMessageBox.information(window, "Conversion Complete", message or f"Output: {output_path}")
                else:
                    QMessageBox.warning(window, "Conversion Failed", message or "Conversion failed without message.")
            except Exception:
                pass
            # restore UI and clean up
            try:
                window.convert_btn.setEnabled(True)
                window.cancel_btn.setEnabled(False)
            except Exception:
                pass
            nonlocal _conversion_thread
            if _conversion_thread:
                try:
                    _conversion_thread.quit()
                    _conversion_thread.wait(timeout=1000)
                except Exception:
                    pass
                _conversion_thread = None

        # connect signals
        _conversion_thread.progress_update.connect(_on_progress)
        _conversion_thread.status_update.connect(_on_status)
        _conversion_thread.error_signal.connect(_on_error)
        _conversion_thread.completion_signal.connect(_on_completion)

        # Start the background thread
        _conversion_thread.start()

    def _stop_conversion():
        nonlocal _conversion_thread
        if _conversion_thread:
            try:
                _conversion_thread.stop()
            except Exception:
                try:
                    _conversion_thread.quit()
                    _conversion_thread.wait(timeout=1000)
                except Exception:
                    pass
            _conversion_thread = None
        # ensure UI restored
        try:
            window.convert_btn.setEnabled(True)
            window.cancel_btn.setEnabled(False)
        except Exception:
            pass

    # Connect the MainWindow signals
    window.conversion_started.connect(_create_and_start_conversion)
    window.conversion_stopped.connect(_stop_conversion)
    # --- end wiring ---

    # Start application event loop
    return_code = app.exec_()

    # Clean up
    try:
        # Remove temp directory if empty
        temp_dir = Path("temp")
        if temp_dir.exists() and not any(temp_dir.iterdir()):
            temp_dir.rmdir()
    except Exception:
        pass

    sys.exit(return_code)


if __name__ == "__main__":
    main()
