"""
GTA SA Map Converter - Progress Tracker
Tracks and reports progress of conversion operations
"""

import time
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
import logging
from enum import Enum, auto

logger = logging.getLogger(__name__)

class ProgressStage(Enum):
    """Stages of the conversion process"""
    INITIALIZING = auto()
    PARSING_IDE = auto()
    PARSING_IPL = auto()
    EXTRACTING_IMG = auto()
    CONVERTING_DFF = auto()
    CONVERTING_TXD = auto()
    BUILDING_SCENE = auto()
    EXPORTING_OBJ = auto()
    FINISHING = auto()

@dataclass
class ProgressUpdate:
    """Single progress update"""
    stage: ProgressStage
    current: int
    total: int
    message: str = ""
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def progress(self) -> float:
        """Get progress as a fraction (0-1)"""
        if self.total <= 0:
            return 0.0
        return min(1.0, max(0.0, self.current / self.total))

    @property
    def percentage(self) -> float:
        """Get progress as a percentage (0-100)"""
        return self.progress * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "stage": self.stage.name,
            "current": self.current,
            "total": self.total,
            "progress": self.progress,
            "percentage": self.percentage,
            "message": self.message,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }

class ProgressCallback:
    """Callback interface for progress updates"""
    def __call__(self, update: ProgressUpdate) -> None:
        """Handle a progress update"""
        raise NotImplementedError

class LoggingProgressCallback(ProgressCallback):
    """Log progress updates to the logging system"""
    def __init__(self, level: int = logging.INFO):
        self.level = level

    def __call__(self, update: ProgressUpdate) -> None:
        """Log the progress update"""
        logger.log(
            self.level,
            f"[{update.stage.name}] {update.current}/{update.total} ({update.percentage:.1f}%) - {update.message}"
        )

class ProgressTracker:
    """Tracks progress of conversion operations"""

    def __init__(self):
        self._callbacks: List[ProgressCallback] = []
        self._current_stage: Optional[ProgressStage] = None
        self._stage_start_time: float = 0.0
        self._stage_progress: Tuple[int, int] = (0, 0)
        self._stage_metadata: Dict[str, Any] = {}
        self._overall_progress: float = 0.0
        self._stage_weights = {
            ProgressStage.INITIALIZING: 0.05,
            ProgressStage.PARSING_IDE: 0.1,
            ProgressStage.PARSING_IPL: 0.1,
            ProgressStage.EXTRACTING_IMG: 0.15,
            ProgressStage.CONVERTING_DFF: 0.25,
            ProgressStage.CONVERTING_TXD: 0.15,
            ProgressStage.BUILDING_SCENE: 0.1,
            ProgressStage.EXPORTING_OBJ: 0.05,
            ProgressStage.FINISHING: 0.05
        }

    def add_callback(self, callback: ProgressCallback) -> None:
        """Add a progress callback"""
        self._callbacks.append(callback)

    def remove_callback(self, callback: ProgressCallback) -> None:
        """Remove a progress callback"""
        self._callbacks.remove(callback)

    def clear_callbacks(self) -> None:
        """Remove all progress callbacks"""
        self._callbacks.clear()

    def begin_stage(self, stage: ProgressStage, total: int = 1, message: str = "") -> None:
        """Begin a new stage of processing"""
        self._current_stage = stage
        self._stage_start_time = time.time()
        self._stage_progress = (0, total)
        self._stage_metadata = {"message": message}
        self._update_callbacks()

    def update_stage(self, current: int, total: Optional[int] = None, message: Optional[str] = None) -> None:
        """Update progress within the current stage"""
        if self._current_stage is None:
            raise RuntimeError("No active stage to update")

        if total is not None:
            self._stage_progress = (current, total)
        else:
            self._stage_progress = (current, self._stage_progress[1])

        if message is not None:
            self._stage_metadata["message"] = message

        self._update_overall_progress()
        self._update_callbacks()

    def increment_stage(self, amount: int = 1, message: Optional[str] = None) -> None:
        """Increment progress within the current stage"""
        if self._current_stage is None:
            raise RuntimeError("No active stage to increment")

        current, total = self._stage_progress
        self.update_stage(current + amount, total, message)

    def end_stage(self, message: Optional[str] = None) -> None:
        """End the current stage"""
        if self._current_stage is None:
            raise RuntimeError("No active stage to end")

        if message is not None:
            self._stage_metadata["message"] = message

        # Force completion
        self._stage_progress = (self._stage_progress[1], self._stage_progress[1])
        self._update_overall_progress()
        self._update_callbacks()

        # Log stage completion time
        duration = time.time() - self._stage_start_time
        logger.debug(
            f"Completed stage {self._current_stage.name} in {duration:.2f} seconds"
        )

        self._current_stage = None

    def _update_overall_progress(self) -> None:
        """Update the overall progress calculation"""
        if self._current_stage is None:
            return

        # Calculate progress within current stage
        current, total = self._stage_progress
        stage_progress = current / total if total > 0 else 0.0

        # Calculate cumulative progress
        cumulative = 0.0
        for stage, weight in self._stage_weights.items():
            if stage == self._current_stage:
                cumulative += weight * stage_progress
                break
            cumulative += weight

        self._overall_progress = min(1.0, max(0.0, cumulative))

    def _update_callbacks(self) -> None:
        """Send update to all registered callbacks"""
        if self._current_stage is None:
            return

        current, total = self._stage_progress
        update = ProgressUpdate(
            stage=self._current_stage,
            current=current,
            total=total,
            message=self._stage_metadata.get("message", ""),
            metadata=self._stage_metadata
        )

        for callback in self._callbacks:
            try:
                callback(update)
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")

    @property
    def current_stage(self) -> Optional[ProgressStage]:
        """Get the current stage"""
        return self._current_stage

    @property
    def stage_progress(self) -> Tuple[int, int]:
        """Get current stage progress (current, total)"""
        return self._stage_progress

    @property
    def overall_progress(self) -> float:
        """Get overall progress (0-1)"""
        return self._overall_progress

    @property
    def overall_percentage(self) -> float:
        """Get overall progress percentage (0-100)"""
        return self._overall_progress * 100

# Default callback for logging progress
default_progress_callback = LoggingProgressCallback()

# Global progress tracker instance
_global_progress_tracker = None

def get_progress_tracker() -> ProgressTracker:
    """Get the global progress tracker instance"""
    global _global_progress_tracker
    if _global_progress_tracker is None:
        _global_progress_tracker = ProgressTracker()
        _global_progress_tracker.add_callback(default_progress_callback)
    return _global_progress_tracker
