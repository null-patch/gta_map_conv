import os
import sys
import logging
import logging.handlers
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import traceback
import threading

try:
    from colorama import init, Fore, Back, Style
    COLORAMA_AVAILABLE = True
    init(autoreset=True)
except ImportError:
    COLORAMA_AVAILABLE = False
    # Create dummy color constants
    class DummyColor:
        def __getattr__(self, name):
            return ''
    Fore = Back = Style = DummyColor()


class LogLevel(Enum):
    """Logging levels"""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL
    
    @classmethod
    def from_string(cls, level_str: str) -> 'LogLevel':
        """Convert string to LogLevel"""
        level_str = level_str.upper()
        if level_str == 'DEBUG':
            return cls.DEBUG
        elif level_str == 'INFO':
            return cls.INFO
        elif level_str == 'WARNING':
            return cls.WARNING
        elif level_str == 'ERROR':
            return cls.ERROR
        elif level_str == 'CRITICAL':
            return cls.CRITICAL
        else:
            return cls.INFO


@dataclass
class LogEntry:
    """Structured log entry"""
    timestamp: float
    level: LogLevel
    message: str
    module: str = ""
    function: str = ""
    line_number: int = 0
    thread_id: int = 0
    thread_name: str = ""
    process_id: int = 0
    extra_data: Dict[str, Any] = field(default_factory=dict)
    exception_info: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'timestamp': self.timestamp,
            'timestamp_iso': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.timestamp)),
            'level': self.level.name,
            'message': self.message,
            'module': self.module,
            'function': self.function,
            'line_number': self.line_number,
            'thread_id': self.thread_id,
            'thread_name': self.thread_name,
            'process_id': self.process_id,
            'extra_data': self.extra_data,
            'exception_info': self.exception_info
        }
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), default=str)
    
    def to_string(self, include_extra: bool = False) -> str:
        """Convert to string representation"""
        time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.timestamp))
        level_str = self.level.name
        
        base_str = f"[{time_str}] [{level_str:8}] [{self.module}:{self.function}:{self.line_number}] {self.message}"
        
        if include_extra and self.extra_data:
            extra_str = json.dumps(self.extra_data, default=str)
            base_str += f" | Extra: {extra_str}"
        
        if self.exception_info:
            base_str += f"\n{self.exception_info}"
        
        return base_str


class LogFormatter(logging.Formatter):
    """Custom log formatter with colors"""
    
    # Color mappings
    COLOR_MAP = {
        'DEBUG': Fore.CYAN,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Back.WHITE + Style.BRIGHT
    }
    
    # Format strings
    SIMPLE_FORMAT = '%(asctime)s [%(levelname)s] %(message)s'
    DETAILED_FORMAT = '%(asctime)s [%(levelname)s] [%(name)s:%(funcName)s:%(lineno)d] %(message)s'
    JSON_FORMAT = '%(message)s'  # For JSON output
    
    def __init__(self, fmt: Optional[str] = None, datefmt: Optional[str] = None, 
                 style: str = '%', use_colors: bool = True, detailed: bool = False):
        if fmt is None:
            fmt = self.DETAILED_FORMAT if detailed else self.SIMPLE_FORMAT
        
        super().__init__(fmt, datefmt, style)
        self.use_colors = use_colors and COLORAMA_AVAILABLE
        self.detailed = detailed
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record"""
        # Store original levelname for colorization
        original_levelname = record.levelname
        
        # Format the message
        message = super().format(record)
        
        # Add colors if enabled
        if self.use_colors and original_levelname in self.COLOR_MAP:
            color = self.COLOR_MAP[original_levelname]
            message = f"{color}{message}{Style.RESET_ALL}"
        
        # Add exception info if present
        if record.exc_info:
            if not message.endswith('\n'):
                message += '\n'
            message += self.formatException(record.exc_info)
        
        return message
    
    def formatException(self, exc_info) -> str:
        """Format exception information"""
        formatted = super().formatException(exc_info)
        
        if self.use_colors:
            # Colorize exception traceback
            lines = formatted.split('\n')
            colored_lines = []
            for line in lines:
                if 'Traceback' in line or 'File' in line:
                    colored_lines.append(Fore.CYAN + line + Style.RESET_ALL)
                elif line.strip().startswith('^'):
                    colored_lines.append(Fore.YELLOW + line + Style.RESET_ALL)
                else:
                    colored_lines.append(Fore.RED + line + Style.RESET_ALL)
            return '\n'.join(colored_lines)
        
        return formatted


class JSONFormatter(logging.Formatter):
    """JSON log formatter for structured logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""
        log_entry = LogEntry(
            timestamp=record.created,
            level=LogLevel(record.levelno),
            message=record.getMessage(),
            module=record.module,
            function=record.funcName,
            line_number=record.lineno,
            thread_id=record.thread,
            thread_name=record.threadName,
            process_id=record.process,
            extra_data=getattr(record, 'extra_data', {})
        )
        
        # Add exception info if present
        if record.exc_info:
            log_entry.exception_info = self.formatException(record.exc_info)
        
        return log_entry.to_json()


class Logger:
    """Main logger class with multiple handlers and rotation"""
    
    def __init__(self, name: str = "GTA_SA_Converter", 
                 log_dir: Optional[str] = None,
                 level: LogLevel = LogLevel.INFO,
                 max_file_size_mb: int = 10,
                 backup_count: int = 5,
                 enable_console: bool = True,
                 enable_file: bool = True,
                 enable_json: bool = False,
                 detailed_console: bool = False):
        """
        Initialize logger
        
        Args:
            name: Logger name
            log_dir: Directory for log files
            level: Logging level
            max_file_size_mb: Maximum log file size in MB
            backup_count: Number of backup files to keep
            enable_console: Enable console output
            enable_file: Enable file output
            enable_json: Enable JSON logging
            detailed_console: Use detailed format for console
        """
        self.name = name
        self.log_dir = Path(log_dir) if log_dir else Path.home() / ".gta_map_converter" / "logs"
        self.level = level
        self.max_file_size = max_file_size_mb * 1024 * 1024
        self.backup_count = backup_count
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level.value)
        self.logger.propagate = False
        
        # Remove existing handlers
        self.logger.handlers.clear()
        
        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup handlers
        self.handlers: Dict[str, logging.Handler] = {}
        self._setup_handlers(enable_console, enable_file, enable_json, detailed_console)
        
        # Statistics
        self.message_count = 0
        self.error_count = 0
        self.warning_count = 0
        self.start_time = time.time()
        
        # Thread safety
        self._lock = threading.RLock()
    
    def _setup_handlers(self, enable_console: bool, enable_file: bool, 
                       enable_json: bool, detailed_console: bool):
        """Setup logging handlers"""
        
        # Console handler
        if enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.level.value)
            console_formatter = LogFormatter(
                use_colors=COLORAMA_AVAILABLE,
                detailed=detailed_console
            )
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
            self.handlers['console'] = console_handler
        
        # File handler (rotating)
        if enable_file:
            log_file = self.log_dir / f"{self.name}.log"
            file_handler = logging.handlers.RotatingFileHandler(
                filename=log_file,
                maxBytes=self.max_file_size,
                backupCount=self.backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(self.level.value)
            file_formatter = LogFormatter(detailed=True, use_colors=False)
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
            self.handlers['file'] = file_handler
        
        # JSON file handler
        if enable_json:
            json_file = self.log_dir / f"{self.name}_structured.json"
            json_handler = logging.handlers.RotatingFileHandler(
                filename=json_file,
                maxBytes=self.max_file_size,
                backupCount=self.backup_count,
                encoding='utf-8'
            )
            json_handler.setLevel(self.level.value)
            json_formatter = JSONFormatter()
            json_handler.setFormatter(json_formatter)
            self.logger.addHandler(json_handler)
            self.handlers['json'] = json_handler
        
        # Error file handler (only errors and critical)
        if enable_file:
            error_file = self.log_dir / f"{self.name}_errors.log"
            error_handler = logging.handlers.RotatingFileHandler(
                filename=error_file,
                maxBytes=self.max_file_size,
                backupCount=self.backup_count,
                encoding='utf-8'
            )
            error_handler.setLevel(logging.ERROR)
            error_formatter = LogFormatter(detailed=True, use_colors=False)
            error_handler.setFormatter(error_formatter)
            self.logger.addHandler(error_handler)
            self.handlers['error_file'] = error_handler
    
    def set_level(self, level: Union[LogLevel, str]):
        """Set logging level"""
        if isinstance(level, str):
            level = LogLevel.from_string(level)
        
        self.level = level
        self.logger.setLevel(level.value)
        
        for handler in self.handlers.values():
            handler.setLevel(level.value)
    
    def debug(self, message: str, extra_data: Optional[Dict[str, Any]] = None, 
              exc_info: Optional[Exception] = None):
        """Log debug message"""
        with self._lock:
            self.message_count += 1
            self._log(logging.DEBUG, message, extra_data, exc_info)
    
    def info(self, message: str, extra_data: Optional[Dict[str, Any]] = None,
             exc_info: Optional[Exception] = None):
        """Log info message"""
        with self._lock:
            self.message_count += 1
            self._log(logging.INFO, message, extra_data, exc_info)
    
    def warning(self, message: str, extra_data: Optional[Dict[str, Any]] = None,
                exc_info: Optional[Exception] = None):
        """Log warning message"""
        with self._lock:
            self.message_count += 1
            self.warning_count += 1
            self._log(logging.WARNING, message, extra_data, exc_info)
    
    def error(self, message: str, extra_data: Optional[Dict[str, Any]] = None,
              exc_info: Optional[Exception] = None):
        """Log error message"""
        with self._lock:
            self.message_count += 1
            self.error_count += 1
            self._log(logging.ERROR, message, extra_data, exc_info)
    
    def critical(self, message: str, extra_data: Optional[Dict[str, Any]] = None,
                 exc_info: Optional[Exception] = None):
        """Log critical message"""
        with self._lock:
            self.message_count += 1
            self.error_count += 1
            self._log(logging.CRITICAL, message, extra_data, exc_info)
    
    def success(self, message: str, extra_data: Optional[Dict[str, Any]] = None):
        """Log success message (custom level)"""
        with self._lock:
            self.message_count += 1
            # Log as INFO with SUCCESS prefix
            formatted_message = f"✓ {message}"
            if COLORAMA_AVAILABLE:
                formatted_message = f"{Fore.GREEN}{formatted_message}{Style.RESET_ALL}"
            self._log(logging.INFO, formatted_message, extra_data)
    
    def progress(self, current: int, total: int, message: str = "",
                 extra_data: Optional[Dict[str, Any]] = None):
        """Log progress message"""
        with self._lock:
            if total > 0:
                percentage = (current / total) * 100
                progress_msg = f"[{current}/{total}] {percentage:.1f}%"
                if message:
                    progress_msg = f"{progress_msg} - {message}"
                
                # Use extra data for structured logging
                progress_data = {
                    'current': current,
                    'total': total,
                    'percentage': percentage,
                    'message': message
                }
                if extra_data:
                    progress_data.update(extra_data)
                
                self._log(logging.INFO, progress_msg, progress_data)
    
    def _log(self, level: int, message: str, extra_data: Optional[Dict[str, Any]] = None,
             exc_info: Optional[Exception] = None):
        """Internal logging method"""
        # Get caller information
        frame = logging.currentframe()
        # Go back 3 frames: _log -> debug/info/etc -> user code
        for _ in range(3):
            if frame.f_back:
                frame = frame.f_back
        
        # Create log record with extra data
        record = self.logger.makeRecord(
            name=self.name,
            level=level,
            fn=frame.f_code.co_filename,
            lno=frame.f_lineno,
            msg=message,
            args=(),
            exc_info=exc_info,
            func=frame.f_code.co_name,
            extra=None,
            sinfo=None
        )
        
        # Add extra data to record
        if extra_data:
            record.extra_data = extra_data
        
        # Handle the record
        self.logger.handle(record)
    
    def log_exception(self, exception: Exception, context: str = "",
                      extra_data: Optional[Dict[str, Any]] = None):
        """Log exception with context"""
        with self._lock:
            self.message_count += 1
            self.error_count += 1
            
            message = f"{context}: {str(exception)}" if context else str(exception)
            self._log(logging.ERROR, message, extra_data, exc_info=exception)
    
    def start_section(self, section_name: str, extra_data: Optional[Dict[str, Any]] = None):
        """Log start of a section"""
        separator = "=" * 60
        message = f"\n{separator}\n{section_name}\n{separator}"
        self.info(message, extra_data)
    
    def end_section(self, section_name: str, success: bool = True,
                    extra_data: Optional[Dict[str, Any]] = None):
        """Log end of a section"""
        status = "COMPLETED" if success else "FAILED"
        separator = "=" * 60
        message = f"\n{section_name} {status}\n{separator}"
        
        if success:
            self.success(message, extra_data)
        else:
            self.error(message, extra_data)
    
    def add_handler(self, handler: logging.Handler):
        """Add custom handler"""
        with self._lock:
            self.logger.addHandler(handler)
            self.handlers[handler.__class__.__name__] = handler
    
    def remove_handler(self, handler_name: str):
        """Remove handler by name"""
        with self._lock:
            if handler_name in self.handlers:
                handler = self.handlers[handler_name]
                self.logger.removeHandler(handler)
                del self.handlers[handler_name]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get logging statistics"""
        with self._lock:
            runtime = time.time() - self.start_time
            
            return {
                'name': self.name,
                'level': self.level.name,
                'message_count': self.message_count,
                'error_count': self.error_count,
                'warning_count': self.warning_count,
                'start_time': self.start_time,
                'runtime_seconds': runtime,
                'messages_per_second': self.message_count / runtime if runtime > 0 else 0,
                'handlers': list(self.handlers.keys())
            }
    
    def clear_logs(self, keep_recent: bool = True, days_to_keep: int = 7):
        """Clear old log files"""
        with self._lock:
            if not self.log_dir.exists():
                return
            
            current_time = time.time()
            cutoff_time = current_time - (days_to_keep * 86400)
            
            for log_file in self.log_dir.glob("*.log"):
                if keep_recent:
                    # Check file modification time
                    try:
                        mtime = log_file.stat().st_mtime
                        if mtime > cutoff_time:
                            continue
                    except OSError:
                        continue
                
                try:
                    log_file.unlink()
                    self.info(f"Removed old log file: {log_file.name}")
                except OSError as e:
                    self.warning(f"Failed to remove log file {log_file.name}: {e}")
    
    def flush(self):
        """Flush all log handlers"""
        with self._lock:
            for handler in self.handlers.values():
                handler.flush()
    
    def close(self):
        """Close all log handlers"""
        with self._lock:
            for handler in self.handlers.values():
                handler.close()
            self.handlers.clear()
            self.logger.handlers.clear()
    
    def create_log_entry(self, level: LogLevel, message: str,
                        extra_data: Optional[Dict[str, Any]] = None) -> LogEntry:
        """Create a structured log entry without logging it"""
        frame = logging.currentframe()
        # Go back to caller
        for _ in range(3):
            if frame.f_back:
                frame = frame.f_back
        
        return LogEntry(
            timestamp=time.time(),
            level=level,
            message=message,
            module=frame.f_globals.get('__name__', ''),
            function=frame.f_code.co_name,
            line_number=frame.f_lineno,
            thread_id=threading.get_ident(),
            thread_name=threading.current_thread().name,
            process_id=os.getpid(),
            extra_data=extra_data or {}
        )


# Global logger instance
_global_logger = None

def get_logger(name: str = "GTA_SA_Converter", **kwargs) -> Logger:
    """Get global logger instance"""
    global _global_logger
    if _global_logger is None:
        _global_logger = Logger(name, **kwargs)
    return _global_logger


def setup_logging(name: str = "GTA_SA_Converter", level: str = "INFO",
                  log_dir: Optional[str] = None, **kwargs) -> Logger:
    """Setup logging with default configuration"""
    logger = get_logger(name, log_dir=log_dir, level=LogLevel.from_string(level), **kwargs)
    return logger


# Convenience functions for quick logging
def log_debug(message: str, **kwargs):
    """Quick debug log"""
    logger = get_logger()
    logger.debug(message, **kwargs)

def log_info(message: str, **kwargs):
    """Quick info log"""
    logger = get_logger()
    logger.info(message, **kwargs)

def log_warning(message: str, **kwargs):
    """Quick warning log"""
    logger = get_logger()
    logger.warning(message, **kwargs)

def log_error(message: str, **kwargs):
    """Quick error log"""
    logger = get_logger()
    logger.error(message, **kwargs)

def log_critical(message: str, **kwargs):
    """Quick critical log"""
    logger = get_logger()
    logger.critical(message, **kwargs)

def log_success(message: str, **kwargs):
    """Quick success log"""
    logger = get_logger()
    logger.success(message, **kwargs)

def log_exception(exception: Exception, context: str = "", **kwargs):
    """Quick exception log"""
    logger = get_logger()
    logger.log_exception(exception, context, **kwargs)


if __name__ == "__main__":
    # Test the logger
    print("Testing Logger")
    print("=" * 50)
    
    # Create test logger
    test_logger = Logger(
        name="TestLogger",
        level=LogLevel.DEBUG,
        enable_console=True,
        enable_file=False,
        detailed_console=True
    )
    
    # Test different log levels
    test_logger.debug("This is a debug message", {"test": "data"})
    test_logger.info("This is an info message")
    test_logger.warning("This is a warning message")
    test_logger.error("This is an error message")
    test_logger.success("This is a success message!")
    
    # Test progress logging
    for i in range(5):
        test_logger.progress(i + 1, 5, f"Processing item {i + 1}")
        time.sleep(0.1)
    
    # Test section logging
    test_logger.start_section("Test Section", {"section_id": 1})
    test_logger.info("Inside test section")
    test_logger.end_section("Test Section", success=True)
    
    # Test exception logging
    try:
        raise ValueError("Test exception")
    except ValueError as e:
        test_logger.log_exception(e, "Test context", {"extra": "info"})
    
    # Get statistics
    stats = test_logger.get_stats()
    print("\n" + "=" * 50)
    print("Logger Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Cleanup
    test_logger.close()
    print("\nLogger test completed")
