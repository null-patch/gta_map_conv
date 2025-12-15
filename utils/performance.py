import os
import sys
import time
import threading
import multiprocessing
import psutil
import gc
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from functools import lru_cache
import logging
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import pickle

logger = logging.getLogger(__name__)

@dataclass
class SystemResources:
    cpu_count: int
    cpu_usage: float
    memory_total: int
    memory_available: int
    memory_used: int
    swap_total: int
    swap_used: int
    disk_usage: Dict[str, float]
    timestamp: float

    def get_memory_usage_percent(self) -> float:
        if self.memory_total == 0:
            return 0.0
        return (self.memory_used / self.memory_total) * 100

    def get_swap_usage_percent(self) -> float:
        if self.swap_total == 0:
            return 0.0
        return (self.swap_used / self.swap_total) * 100

    def to_dict(self) -> Dict[str, Any]:
        return {
            'cpu_count': self.cpu_count,
            'cpu_usage': self.cpu_usage,
            'memory_total': self.memory_total,
            'memory_available': self.memory_available,
            'memory_used': self.memory_used,
            'memory_usage_percent': self.get_memory_usage_percent(),
            'swap_total': self.swap_total,
            'swap_used': self.swap_used,
            'swap_usage_percent': self.get_swap_usage_percent(),
            'disk_usage': self.disk_usage,
            'timestamp': self.timestamp
        }

class MemoryManager:
    def __init__(self, max_memory_usage: float = 0.8):
        self.max_memory_usage = max_memory_usage
        self._tracked_objects: Dict[int, Any] = {}
        self._lock = threading.Lock()

    def get_system_resources(self) -> SystemResources:
        cpu_count = psutil.cpu_count()
        cpu_usage = psutil.cpu_percent()
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()

        disk_usage = {}
        for partition in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                disk_usage[partition.mountpoint] = usage.percent
            except Exception:
                continue

        return SystemResources(
            cpu_count=cpu_count,
            cpu_usage=cpu_usage,
            memory_total=mem.total,
            memory_available=mem.available,
            memory_used=mem.used,
            swap_total=swap.total,
            swap_used=swap.used,
            disk_usage=disk_usage,
            timestamp=time.time()
        )

    def check_memory_available(self, required_bytes: int) -> bool:
        mem = psutil.virtual_memory()
        return mem.available >= required_bytes

    def track_object(self, obj: Any, name: str = "") -> int:
        with self._lock:
            obj_id = id(obj)
            self._tracked_objects[obj_id] = {
                'object': obj,
                'name': name,
                'size': self.estimate_size(obj),
                'timestamp': time.time()
            }
            return obj_id

    def release_object(self, obj_id: int):
        with self._lock:
            if obj_id in self._tracked_objects:
                del self._tracked_objects[obj_id]

    def release_all_objects(self):
        with self._lock:
            self._tracked_objects.clear()

    def estimate_size(self, obj: Any) -> int:
        if isinstance(obj, (int, float, bool, str, bytes)):
            return sys.getsizeof(obj)
        elif isinstance(obj, (list, tuple, set)):
            return sum(self.estimate_size(item) for item in obj) + sys.getsizeof(obj)
        elif isinstance(obj, dict):
            return sum(self.estimate_size(k) + self.estimate_size(v) for k, v in obj.items()) + sys.getsizeof(obj)
        elif isinstance(obj, np.ndarray):
            return obj.nbytes
        else:
            return sys.getsizeof(obj)

    def optimize_memory(self, target_usage: float = 0.7):
        current_usage = self.get_system_resources().get_memory_usage_percent() / 100
        if current_usage <= target_usage:
            return

        logger.info(f"Memory usage high ({current_usage:.1%}), optimizing...")
        self.run_garbage_collection()
        self.clear_object_caches()
        self.release_old_objects()
        new_usage = self.get_system_resources().get_memory_usage_percent() / 100
        logger.info(f"Memory optimization complete. New usage: {new_usage:.1%}")

    def run_garbage_collection(self):
        logger.debug("Running garbage collection...")
        gc.collect()
        logger.debug("Garbage collection completed")

    def clear_object_caches(self):
        logger.debug("Clearing object caches...")
        for obj in gc.get_objects():
            if hasattr(obj, 'cache_clear') and callable(obj.cache_clear):
                try:
                    obj.cache_clear()
                except Exception:
                    pass
        logger.debug("Object caches cleared")

    def release_old_objects(self, min_age_seconds: int = 60):
        with self._lock:
            current_time = time.time()
            to_delete = [
                obj_id for obj_id, obj_data in self._tracked_objects.items()
                if current_time - obj_data['timestamp'] > min_age_seconds
            ]
            for obj_id in to_delete:
                del self._tracked_objects[obj_id]
            logger.debug(f"Released {len(to_delete)} old objects")

    def monitor_memory(self, interval: float = 5.0, stop_event: threading.Event = None):
        def monitor():
            while not stop_event or not stop_event.is_set():
                try:
                    resources = self.get_system_resources()
                    usage = resources.get_memory_usage_percent() / 100
                    if usage > self.max_memory_usage:
                        self.optimize_memory(target_usage=self.max_memory_usage * 0.9)
                    time.sleep(interval)
                except Exception as e:
                    logger.error(f"Memory monitoring error: {e}")
                    time.sleep(interval)

        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
        return monitor_thread

class CacheManager:
    def __init__(self, cache_dir: str = None, max_memory_items: int = 1000):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.max_memory_items = max_memory_items
        self.memory_cache: Dict[str, Tuple[Any, float]] = {}
        self._lock = threading.Lock()

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get(self, key: str, max_age_seconds: int = None) -> Optional[Any]:
        with self._lock:
            if key in self.memory_cache:
                item, timestamp = self.memory_cache[key]
                if max_age_seconds is None or (time.time() - timestamp) <= max_age_seconds:
                    return item

            if self.cache_dir:
                cache_file = self.cache_dir / f"{key}.cache"
                if cache_file.exists():
                    try:
                        mtime = cache_file.stat().st_mtime
                        if max_age_seconds is None or (time.time() - mtime) <= max_age_seconds:
                            with open(cache_file, 'rb') as f:
                                item = pickle.load(f)
                                self.memory_cache[key] = (item, time.time())
                                return item
                    except Exception as e:
                        logger.error(f"Failed to load disk cache for key '{key}': {e}")
        return None

    def set(self, key: str, value: Any, to_disk: bool = False):
        with self._lock:
            self.memory_cache[key] = (value, time.time())
            if len(self.memory_cache) > self.max_memory_items:
                oldest = sorted(self.memory_cache.items(), key=lambda x: x[1][1])[0][0]
                del self.memory_cache[oldest]

            if to_disk and self.cache_dir:
                try:
                    cache_file = self.cache_dir / f"{key}.cache"
                    with open(cache_file, 'wb') as f:
                        pickle.dump(value, f)
                except Exception as e:
                    logger.error(f"Failed to write disk cache for key '{key}': {e}")
