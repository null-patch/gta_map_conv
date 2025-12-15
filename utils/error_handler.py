"""
GTA SA Map Converter - Error Handler
Comprehensive error handling and recovery system
"""

import os
import sys
import traceback
import logging
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import inspect

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels"""
    INFO = 1
    WARNING = 2
    ERROR = 3
    CRITICAL = 4
    FATAL = 5


class ErrorCategory(Enum):
    """Error categories"""
    CONFIGURATION = "Configuration"
    FILE_IO = "File I/O"
    PARSING = "Parsing"
    CONVERSION = "Conversion"
    EXPORT = "Export"
    MEMORY = "Memory"
    NETWORK = "Network"
    SYSTEM = "System"
    UNKNOWN = "Unknown"


@dataclass
class ErrorInfo:
    """Detailed error information"""
    timestamp: float
    severity: ErrorSeverity
    category: ErrorCategory
    message: str
    error_code: str
    module: str = ""
    function: str = ""
    line_number: int = 0
    file_path: str = ""
    stack_trace: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    recovery_attempted: bool = False
    recovery_successful: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'timestamp': self.timestamp,
            'severity': self.severity.name,
            'category': self.category.value,
            'message': self.message,
            'error_code': self.error_code,
            'module': self.module,
            'function': self.function,
            'line_number': self.line_number,
            'file_path': self.file_path,
            'stack_trace': self.stack_trace,
            'context': self.context,
            'recovery_attempted': self.recovery_attempted,
            'recovery_successful': self.recovery_successful
        }
    
    def to_string(self, include_stack: bool = False) -> str:
        """Convert to string representation"""
        parts = [
            f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.timestamp))}]",
            f"{self.severity.name}",
            f"({self.category.value})",
            f"{self.error_code}:",
            self.message
        ]
        
        if include_stack and self.stack_trace:
            parts.append(f"\nStack Trace:\n{self.stack_trace}")
        
        return " ".join(parts)


class RecoveryStrategy(Enum):
    """Recovery strategies"""
    RETRY = "retry"
    SKIP = "skip"
    FALLBACK = "fallback"
    RESTART = "restart"
    MANUAL = "manual"
    ABORT = "abort"


@dataclass
class RecoveryAction:
    """Recovery action definition"""
    strategy: RecoveryStrategy
    description: str
    function: Optional[Callable] = None
    max_attempts: int = 3
    delay_seconds: float = 1.0
    conditions: List[Callable] = field(default_factory=list)
    
    def can_execute(self, error_info: ErrorInfo) -> bool:
        """Check if this action can be executed for the given error"""
        if not self.conditions:
            return True
        
        for condition in self.conditions:
            if not condition(error_info):
                return False
        
        return True


class ErrorHandler:
    """Main error handler with recovery capabilities"""
    
    # Error code definitions
    ERROR_CODES = {
        # Configuration errors (1000-1999)
        'CONFIG_1001': "Configuration file not found",
        'CONFIG_1002': "Invalid configuration format",
        'CONFIG_1003': "Missing required configuration",
        'CONFIG_1004': "Directory does not exist",
        'CONFIG_1005': "Insufficient permissions",
        
        # File I/O errors (2000-2999)
        'FILE_2001': "File not found",
        'FILE_2002': "Permission denied",
        'FILE_2003': "Disk full",
        'FILE_2004': "File corrupted",
        'FILE_2005': "Unsupported file format",
        'FILE_2006': "Read error",
        'FILE_2007': "Write error",
        'FILE_2008': "File too large",
        
        # Parsing errors (3000-3999)
        'PARSE_3001': "Invalid file format",
        'PARSE_3002': "Missing required section",
        'PARSE_3003': "Syntax error",
        'PARSE_3004': "Unsupported version",
        'PARSE_3005': "Data corruption detected",
        
        # Conversion errors (4000-4999)
        'CONVERT_4001': "Geometry conversion failed",
        'CONVERT_4002': "Texture conversion failed",
        'CONVERT_4003': "Material creation failed",
        'CONVERT_4004': "Coordinate system conversion failed",
        'CONVERT_4005': "Scale factor invalid",
        
        # Export errors (5000-5999)
        'EXPORT_5001': "OBJ export failed",
        'EXPORT_5002': "MTL export failed",
        'EXPORT_5003': "Texture export failed",
        'EXPORT_5004': "File size limit exceeded",
        'EXPORT_5005': "Memory allocation failed",
        
        # Memory errors (6000-6999)
        'MEMORY_6001': "Out of memory",
        'MEMORY_6002': "Memory allocation failed",
        'MEMORY_6003': "Memory corruption detected",
        
        # System errors (7000-7999)
        'SYSTEM_7001': "Operating system error",
        'SYSTEM_7002': "Library not found",
        'SYSTEM_7003': "Dependency missing",
        'SYSTEM_7004': "Timeout occurred",
        
        # Unknown errors (9000-9999)
        'UNKNOWN_9001': "Unknown error occurred"
    }
    
    def __init__(self, config=None):
        self.config = config
        self.error_log: List[ErrorInfo] = []
        self.recovery_actions: Dict[str, List[RecoveryAction]] = {}
        self.error_callbacks: List[Callable] = []
        self.recovery_callbacks: List[Callable] = []
        
        # Statistics
        self.error_count = 0
        self.warning_count = 0
        self.recovery_attempts = 0
        self.recovery_successes = 0
        
        # Setup default recovery actions
        self._setup_default_recovery_actions()
        
        # Create error log directory
        self.log_dir = Path.home() / ".gta_map_converter" / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def _setup_default_recovery_actions(self):
        """Setup default recovery actions"""
        # File not found - try alternative locations
        self.add_recovery_action(
            error_codes=['FILE_2001'],
            action=RecoveryAction(
                strategy=RecoveryStrategy.FALLBACK,
                description="Try alternative file locations",
                function=self._recover_file_not_found,
                max_attempts=2
            )
        )
        
        # Permission denied - retry with different permissions
        self.add_recovery_action(
            error_codes=['FILE_2002', 'CONFIG_1005'],
            action=RecoveryAction(
                strategy=RecoveryStrategy.RETRY,
                description="Retry with adjusted permissions",
                function=self._recover_permission_error,
                max_attempts=2,
                delay_seconds=0.5
            )
        )
        
        # Parsing errors - skip problematic file
        self.add_recovery_action(
            error_codes=['PARSE_3001', 'PARSE_3002', 'PARSE_3003'],
            action=RecoveryAction(
                strategy=RecoveryStrategy.SKIP,
                description="Skip problematic file and continue",
                function=self._recover_parsing_error,
                max_attempts=1
            )
        )
        
        # Conversion errors - use fallback converter
        self.add_recovery_action(
            error_codes=['CONVERT_4001', 'CONVERT_4002'],
            action=RecoveryAction(
                strategy=RecoveryStrategy.FALLBACK,
                description="Use alternative conversion method",
                function=self._recover_conversion_error,
                max_attempts=2
            )
        )
        
        # Memory errors - clear cache and retry
        self.add_recovery_action(
            error_codes=['MEMORY_6001', 'MEMORY_6002'],
            action=RecoveryAction(
                strategy=RecoveryStrategy.RETRY,
                description="Clear memory cache and retry",
                function=self._recover_memory_error,
                max_attempts=2,
                delay_seconds=1.0
            )
        )
    
    def add_recovery_action(self, error_codes: List[str], action: RecoveryAction):
        """Add a recovery action for specific error codes"""
        for error_code in error_codes:
            if error_code not in self.recovery_actions:
                self.recovery_actions[error_code] = []
            self.recovery_actions[error_code].append(action)
    
    def add_error_callback(self, callback: Callable):
        """Add callback for error events"""
        self.error_callbacks.append(callback)
    
    def add_recovery_callback(self, callback: Callable):
        """Add callback for recovery events"""
        self.recovery_callbacks.append(callback)
    
    def handle_error(self, error: Exception, severity: ErrorSeverity = ErrorSeverity.ERROR,
                    category: ErrorCategory = ErrorCategory.UNKNOWN,
                    error_code: str = "UNKNOWN_9001",
                    context: Optional[Dict[str, Any]] = None,
                    auto_recover: bool = True) -> Tuple[bool, Optional[ErrorInfo]]:
        """
        Handle an error with optional recovery
        
        Args:
            error: Exception object
            severity: Error severity
            category: Error category
            error_code: Error code
            context: Additional context information
            auto_recover: Whether to attempt automatic recovery
            
        Returns:
            Tuple of (recovery_successful, error_info)
        """
        # Get caller information
        frame = inspect.currentframe().f_back
        module = frame.f_globals.get('__name__', '')
        function = frame.f_code.co_name
        line_number = frame.f_lineno
        file_path = frame.f_code.co_filename
        
        # Create error info
        error_info = ErrorInfo(
            timestamp=time.time(),
            severity=severity,
            category=category,
            message=str(error),
            error_code=error_code,
            module=module,
            function=function,
            line_number=line_number,
            file_path=file_path,
            stack_trace=traceback.format_exc(),
            context=context or {}
        )
        
        # Update statistics
        if severity in [ErrorSeverity.ERROR, ErrorSeverity.CRITICAL, ErrorSeverity.FATAL]:
            self.error_count += 1
        elif severity == ErrorSeverity.WARNING:
            self.warning_count += 1
        
        # Add to log
        self.error_log.append(error_info)
        
        # Log the error
        self._log_error(error_info)
        
        # Notify callbacks
        for callback in self.error_callbacks:
            try:
                callback(error_info)
            except Exception as e:
                logger.warning(f"Error callback failed: {e}")
        
        # Attempt recovery if enabled
        recovery_successful = False
        if auto_recover and severity not in [ErrorSeverity.FATAL]:
            recovery_successful = self._attempt_recovery(error_info)
            error_info.recovery_attempted = True
            error_info.recovery_successful = recovery_successful
        
        # Save error to file if critical or fatal
        if severity in [ErrorSeverity.CRITICAL, ErrorSeverity.FATAL]:
            self._save_error_report(error_info)
        
        return recovery_successful, error_info
    
    def _attempt_recovery(self, error_info: ErrorInfo) -> bool:
        """Attempt to recover from error"""
        error_code = error_info.error_code
        
        # Check if we have recovery actions for this error
        if error_code not in self.recovery_actions:
            logger.debug(f"No recovery actions for error code: {error_code}")
            return False
        
        actions = self.recovery_actions[error_code]
        
        for action in actions:
            if not action.can_execute(error_info):
                continue
            
            logger.info(f"Attempting recovery: {action.description}")
            self.recovery_attempts += 1
            
            try:
                # Execute recovery function if provided
                if action.function:
                    success = action.function(error_info, action)
                else:
                    # Default recovery based on strategy
                    success = self._execute_recovery_strategy(action.strategy, error_info)
                
                if success:
                    self.recovery_successes += 1
                    logger.info(f"Recovery successful: {action.description}")
                    
                    # Notify recovery callbacks
                    for callback in self.recovery_callbacks:
                        try:
                            callback(error_info, action, True)
                        except Exception as e:
                            logger.warning(f"Recovery callback failed: {e}")
                    
                    return True
                else:
                    logger.warning(f"Recovery failed: {action.description}")
                    
                    # Notify recovery callbacks
                    for callback in self.recovery_callbacks:
                        try:
                            callback(error_info, action, False)
                        except Exception as e:
                            logger.warning(f"Recovery callback failed: {e}")
            
            except Exception as e:
                logger.error(f"Recovery action failed: {e}")
        
        return False
    
    def _execute_recovery_strategy(self, strategy: RecoveryStrategy, 
                                  error_info: ErrorInfo) -> bool:
        """Execute recovery strategy"""
        try:
            if strategy == RecoveryStrategy.RETRY:
                return self._recover_retry(error_info)
            elif strategy == RecoveryStrategy.SKIP:
                return self._recover_skip(error_info)
            elif strategy == RecoveryStrategy.FALLBACK:
                return self._recover_fallback(error_info)
            elif strategy == RecoveryStrategy.RESTART:
                return self._recover_restart(error_info)
            elif strategy == RecoveryStrategy.MANUAL:
                return self._recover_manual(error_info)
            else:
                return False
        except Exception as e:
            logger.error(f"Recovery strategy failed: {e}")
            return False
    
    def _recover_retry(self, error_info: ErrorInfo) -> bool:
        """Retry the failed operation"""
        # This would be implemented based on the specific error context
        # For now, just return False to indicate retry not implemented
        return False
    
    def _recover_skip(self, error_info: ErrorInfo) -> bool:
        """Skip the problematic item and continue"""
        # This is often used for file parsing errors
        # The actual skipping would be handled by the caller
        return True  # Skipping is always "successful"
    
    def _recover_fallback(self, error_info: ErrorInfo) -> bool:
        """Use fallback method"""
        # Implementation depends on the specific error
        return False
    
    def _recover_restart(self, error_info: ErrorInfo) -> bool:
        """Restart the operation"""
        # This would restart the specific component
        return False
    
    def _recover_manual(self, error_info: ErrorInfo) -> bool:
        """Require manual intervention"""
        # Log that manual intervention is needed
        logger.critical(f"Manual intervention required: {error_info.message}")
        return False
    
    # Default recovery implementations
    def _recover_file_not_found(self, error_info: ErrorInfo, action: RecoveryAction) -> bool:
        """Recover from file not found error"""
        context = error_info.context
        file_path = context.get('file_path', '')
        
        if not file_path:
            return False
        
        # Try common alternative locations
        alternatives = [
            # Check in same directory with different case
            file_path,
            file_path.lower(),
            file_path.upper(),
            
            # Check in parent directory
            os.path.join(os.path.dirname(file_path), '..', os.path.basename(file_path)),
            
            # Check in common GTA directories
            os.path.join('models', os.path.basename(file_path)),
            os.path.join('textures', os.path.basename(file_path)),
            os.path.join('maps', os.path.basename(file_path)),
        ]
        
        for alt_path in alternatives:
            if os.path.exists(alt_path):
                # Update context with found path
                error_info.context['resolved_path'] = alt_path
                logger.info(f"Found file at alternative location: {alt_path}")
                return True
        
        return False
    
    def _recover_permission_error(self, error_info: ErrorInfo, action: RecoveryAction) -> bool:
        """Recover from permission error"""
        context = error_info.context
        file_path = context.get('file_path', '')
        
        if not file_path:
            return False
        
        # Try to adjust permissions (if we have the ability)
        try:
            # This is platform-specific and may require elevated privileges
            import stat
            current_mode = os.stat(file_path).st_mode
            new_mode = current_mode | stat.S_IROTH | stat.S_IWOTH  # Add read/write for others
            os.chmod(file_path, new_mode)
            logger.info(f"Adjusted permissions for: {file_path}")
            return True
        except Exception as e:
            logger.debug(f"Could not adjust permissions: {e}")
            return False
    
    def _recover_parsing_error(self, error_info: ErrorInfo, action: RecoveryAction) -> bool:
        """Recover from parsing error"""
        # For parsing errors, we typically skip the problematic file
        # The actual skipping is handled by the caller
        return True
    
    def _recover_conversion_error(self, error_info: ErrorInfo, action: RecoveryAction) -> bool:
        """Recover from conversion error"""
        context = error_info.context
        converter_type = context.get('converter_type', '')
        file_path = context.get('file_path', '')
        
        if not converter_type or not file_path:
            return False
        
        # Try alternative conversion methods based on converter type
        if converter_type == 'dff':
            # Try different DFF conversion methods
            return self._try_alternative_dff_conversion(file_path, error_info)
        elif converter_type == 'txd':
            # Try different TXD conversion methods
            return self._try_alternative_txd_conversion(file_path, error_info)
        
        return False
    
    def _recover_memory_error(self, error_info: ErrorInfo, action: RecoveryAction) -> bool:
        """Recover from memory error"""
        # Clear caches and reduce memory usage
        import gc
        gc.collect()
        
        # Try to reduce memory footprint
        # This would be more specific in a real implementation
        logger.info("Cleared memory cache and performed garbage collection")
        return True
    
    def _try_alternative_dff_conversion(self, file_path: str, error_info: ErrorInfo) -> bool:
        """Try alternative DFF conversion methods"""
        # This would implement alternative DFF conversion strategies
        # For now, just log the attempt
        logger.info(f"Attempting alternative DFF conversion for: {file_path}")
        return False
    
    def _try_alternative_txd_conversion(self, file_path: str, error_info: ErrorInfo) -> bool:
        """Try alternative TXD conversion methods"""
        # This would implement alternative TXD conversion strategies
        logger.info(f"Attempting alternative TXD conversion for: {file_path}")
        return False
    
    def _log_error(self, error_info: ErrorInfo):
        """Log error to appropriate channels"""
        log_message = error_info.to_string()
        
        # Log based on severity
        if error_info.severity == ErrorSeverity.FATAL:
            logger.critical(log_message)
        elif error_info.severity == ErrorSeverity.CRITICAL:
            logger.critical(log_message)
        elif error_info.severity == ErrorSeverity.ERROR:
            logger.error(log_message)
        elif error_info.severity == ErrorSeverity.WARNING:
            logger.warning(log_message)
        else:
            logger.info(log_message)
        
        # Also log to file
        self._log_to_file(error_info)
    
    def _log_to_file(self, error_info: ErrorInfo):
        """Log error to file"""
        try:
            # Create daily log file
            date_str = time.strftime("%Y-%m-%d")
            log_file = self.log_dir / f"errors_{date_str}.log"
            
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(error_info.to_string(include_stack=True) + "\n\n")
        except Exception as e:
            logger.warning(f"Failed to write error to log file: {e}")
    
    def _save_error_report(self, error_info: ErrorInfo):
        """Save detailed error report"""
        try:
            # Create error report directory
            report_dir = self.log_dir / "reports"
            report_dir.mkdir(exist_ok=True)
            
            # Create report filename
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            report_file = report_dir / f"error_report_{timestamp}.json"
            
            # Save error info as JSON
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(error_info.to_dict(), f, indent=2, default=str)
            
            logger.info(f"Error report saved: {report_file}")
        except Exception as e:
            logger.warning(f"Failed to save error report: {e}")
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get error handling summary"""
        return {
            'total_errors': self.error_count,
            'total_warnings': self.warning_count,
            'recovery_attempts': self.recovery_attempts,
            'recovery_successes': self.recovery_successes,
            'recovery_rate': (self.recovery_successes / self.recovery_attempts 
                            if self.recovery_attempts > 0 else 0),
            'recent_errors': [e.to_dict() for e in self.error_log[-10:]]  # Last 10 errors
        }
    
    def clear_log(self):
        """Clear error log"""
        self.error_log.clear()
        self.error_count = 0
        self.warning_count = 0
        self.recovery_attempts = 0
        self.recovery_successes = 0
    
    def get_errors_by_severity(self, severity: ErrorSeverity) -> List[ErrorInfo]:
        """Get errors by severity"""
        return [e for e in self.error_log if e.severity == severity]
    
    def get_errors_by_category(self, category: ErrorCategory) -> List[ErrorInfo]:
        """Get errors by category"""
        return [e for e in self.error_log if e.category == category]
    
    def get_error_codes(self) -> List[str]:
        """Get list of all error codes encountered"""
        return list(set(e.error_code for e in self.error_log))
    
    def create_error_report(self, output_path: str) -> bool:
        """Create comprehensive error report"""
        try:
            report = {
                'summary': self.get_error_summary(),
                'errors': [e.to_dict() for e in self.error_log],
                'statistics': {
                    'start_time': min(e.timestamp for e in self.error_log) if self.error_log else 0,
                    'end_time': max(e.timestamp for e in self.error_log) if self.error_log else 0,
                    'error_codes': self.get_error_codes(),
                    'most_common_error': self._get_most_common_error()
                }
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Error report created: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create error report: {e}")
            return False
    
    def _get_most_common_error(self) -> Optional[Dict[str, Any]]:
        """Get most common error"""
        if not self.error_log:
            return None
        
        error_counts = {}
        for error in self.error_log:
            code = error.error_code
            error_counts[code] = error_counts.get(code, 0) + 1
        
        most_common_code = max(error_counts, key=error_counts.get)
        most_common_error = next(e for e in self.error_log if e.error_code == most_common_code)
        
        return {
            'error_code': most_common_code,
            'count': error_counts[most_common_code],
            'example': most_common_error.to_dict()
        }


# Global error handler instance
_global_error_handler = None

def get_global_error_handler() -> ErrorHandler:
    """Get global error handler instance"""
    global _global_error_handler
    if _global_error_handler is None:
        _global_error_handler = ErrorHandler()
    return _global_error_handler


# Decorator for automatic error handling
def handle_errors(error_code: str = "UNKNOWN_9001",
                  severity: ErrorSeverity = ErrorSeverity.ERROR,
                  category: ErrorCategory = ErrorCategory.UNKNOWN,
                  auto_recover: bool = True):
    """
    Decorator for automatic error handling
    
    Usage:
        @handle_errors(error_code="FILE_2001", severity=ErrorSeverity.ERROR)
        def read_file(file_path):
            # function implementation
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            error_handler = get_global_error_handler()
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Create context from function arguments
                context = {
                    'function': func.__name__,
                    'module': func.__module__,
                    'args': str(args),
                    'kwargs': str(kwargs)
                }
                
                # Handle the error
                recovery_successful, error_info = error_handler.handle_error(
                    error=e,
                    severity=severity,
                    category=category,
                    error_code=error_code,
                    context=context,
                    auto_recover=auto_recover
                )
                
                # If recovery was successful, retry the function
                if recovery_successful and auto_recover:
                    try:
                        return func(*args, **kwargs)
                    except Exception as retry_error:
                        # If retry fails, re-raise the original error
                        raise e from retry_error
                else:
                    # Re-raise the error if no recovery or recovery failed
                    raise
        
        return wrapper
    return decorator


# Convenience functions for common error types
def handle_file_error(func):
    """Decorator for file-related errors"""
    return handle_errors(
        error_code="FILE_2001",
        severity=ErrorSeverity.ERROR,
        category=ErrorCategory.FILE_IO,
        auto_recover=True
    )(func)

def handle_parse_error(func):
    """Decorator for parsing errors"""
    return handle_errors(
        error_code="PARSE_3001",
        severity=ErrorSeverity.ERROR,
        category=ErrorCategory.PARSING,
        auto_recover=True
    )(func)

def handle_conversion_error(func):
    """Decorator for conversion errors"""
    return handle_errors(
        error_code="CONVERT_4001",
        severity=ErrorSeverity.ERROR,
        category=ErrorCategory.CONVERSION,
        auto_recover=True
    )(func)


if __name__ == "__main__":
    # Test the error handler
    import sys
    
    # Setup logging
    logging.basicConfig(level=logging.DEBUG)
    
    print("Testing Error Handler")
    print("=" * 50)
    
    # Create error handler
    handler = ErrorHandler()
    
    # Test error handling
    try:
        # Simulate a file not found error
        raise FileNotFoundError("test_file.txt not found")
    except Exception as e:
        recovery_successful, error_info = handler.handle_error(
            e,
            severity=ErrorSeverity.ERROR,
            category=ErrorCategory.FILE_IO,
            error_code="FILE_2001",
            context={'file_path': 'test_file.txt'},
            auto_recover=True
        )
        
        print(f"Error handled: {error_info.message}")
        print(f"Recovery successful: {recovery_successful}")
    
    print("\n" + "=" * 50)
    print("Error Summary:")
    summary = handler.get_error_summary()
    for key, value in summary.items():
        if key != 'recent_errors':
            print(f"  {key}: {value}")
    
    print("\n" + "=" * 50)
    print("Testing error decorator:")
    
    @handle_file_error
    def test_function(file_path):
        """Test function that raises file error"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        return "File read successfully"
    
    try:
        result = test_function("non_existent_file.txt")
        print(f"Result: {result}")
    except Exception as e:
        print(f"Function raised: {type(e).__name__}: {e}")
    
    print("\nTest completed")
