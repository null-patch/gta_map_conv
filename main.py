import sys
import os
import traceback
from pathlib import Path

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
    progress_update = pyqtSignal(str, int)
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
            
            # Set up progress callbacks
            pipeline.set_progress_callback(self.progress_update.emit)
            pipeline.set_status_callback(self.status_update.emit)
            
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
        self.wait()


def check_dependencies():
    """Check if all required dependencies are available"""
    missing = []
    
    try:
        import numpy
    except ImportError:
        missing.append("numpy (pip install numpy)")
        
    try:
        from PIL import Image
    except ImportError:
        missing.append("Pillow (pip install Pillow)")
        
    # Try to import core modules
    try:
        import core.conversion_pipeline
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
    except:
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
    except:
        pass  # Use default style
    
    # Load configuration
    config = Config()
    
    # Create and show main window
    window = MainWindow(config)
    window.show()
    
    # Center window on screen
    screen_geometry = app.primaryScreen().availableGeometry()
    window_geometry = window.frameGeometry()
    window_geometry.moveCenter(screen_geometry.center())
    window.move(window_geometry.topLeft())
    
    # Set application icon if available
    try:
        icon_path = Path("resources/icons/app_icon.png")
        if icon_path.exists():
            from PyQt5.QtGui import QIcon
            app.setWindowIcon(QIcon(str(icon_path)))
    except:
        pass
    
    # Start application event loop
    return_code = app.exec_()
    
    # Clean up
    try:
        # Remove temp directory if empty
        temp_dir = Path("temp")
        if temp_dir.exists() and not any(temp_dir.iterdir()):
            temp_dir.rmdir()
    except:
        pass
        
    sys.exit(return_code)


if __name__ == "__main__":
    main()
