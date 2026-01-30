"""
Configuration file watcher with hot-reload support.

Uses watchdog library for cross-platform file watching:
- inotify on Linux
- FSEvents on macOS
- polling fallback on other platforms (but we avoid this)
"""

import logging
import threading
from pathlib import Path
from typing import Callable, Optional

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent

logger = logging.getLogger(__name__)


class ConfigFileHandler(FileSystemEventHandler):
    """
    Handles file system events for configuration files.

    Calls reload callback when watched file is modified.
    """

    def __init__(self, file_path: Path, reload_callback: Callable):
        """
        Initialize handler.

        Args:
            file_path: Path to file to watch
            reload_callback: Function to call on file change
        """
        self.file_path = file_path.resolve()
        self.reload_callback = reload_callback
        self._debounce_lock = threading.Lock()
        self._last_reload = 0

    def on_modified(self, event):
        """Handle file modification event."""
        if event.is_directory:
            return

        # Check if this is our target file
        event_path = Path(event.src_path).resolve()
        if event_path != self.file_path:
            return

        # Debounce - some editors trigger multiple events
        import time
        current_time = time.time()

        with self._debounce_lock:
            if current_time - self._last_reload < 1.0:
                # Skip rapid-fire events (less than 1 second apart)
                return
            self._last_reload = current_time

        logger.info(f"[FileWatcher] Detected change in {self.file_path.name}")

        try:
            self.reload_callback()
            logger.info(f"[FileWatcher] Reload successful")
        except Exception as e:
            logger.error(f"[FileWatcher] Reload failed: {e}", exc_info=True)


class ConfigFileWatcher:
    """
    Watches configuration file for changes and triggers reload.

    Uses watchdog library for efficient file watching:
    - inotify on Linux (no polling)
    - FSEvents on macOS (no polling)
    - ReadDirectoryChangesW on Windows
    """

    def __init__(self, file_path: str, reload_callback: Callable):
        """
        Initialize file watcher.

        Args:
            file_path: Path to configuration file to watch
            reload_callback: Function to call when file changes
        """
        self.file_path = Path(file_path)
        self.reload_callback = reload_callback

        if not self.file_path.exists():
            logger.warning(
                f"[FileWatcher] File not found: {self.file_path}. "
                "Watcher will activate when file is created."
            )

        # Create event handler
        self.event_handler = ConfigFileHandler(
            file_path=self.file_path,
            reload_callback=reload_callback,
        )

        # Create observer (uses inotify/FSEvents)
        self.observer = Observer()

        # Watch the directory containing the file
        watch_dir = self.file_path.parent
        self.observer.schedule(
            self.event_handler,
            str(watch_dir),
            recursive=False,
        )

        logger.info(
            f"[FileWatcher] Initialized watcher for {self.file_path.name} "
            f"(directory: {watch_dir})"
        )

    def start(self):
        """Start watching for file changes."""
        self.observer.start()
        logger.info(f"[FileWatcher] Started watching {self.file_path.name}")

    def stop(self):
        """Stop watching for file changes."""
        self.observer.stop()
        self.observer.join(timeout=5.0)
        logger.info(f"[FileWatcher] Stopped watching {self.file_path.name}")

    def is_alive(self) -> bool:
        """Check if watcher is running."""
        return self.observer.is_alive()


# Global watcher instance
_config_watcher: Optional[ConfigFileWatcher] = None


def start_config_watcher(file_path: str, reload_callback: Callable):
    """
    Start global configuration file watcher.

    Args:
        file_path: Path to configuration file
        reload_callback: Function to call on file change
    """
    global _config_watcher

    if _config_watcher is not None and _config_watcher.is_alive():
        logger.warning("[FileWatcher] Watcher already running")
        return

    _config_watcher = ConfigFileWatcher(file_path, reload_callback)
    _config_watcher.start()


def stop_config_watcher():
    """Stop global configuration file watcher."""
    global _config_watcher

    if _config_watcher is not None:
        _config_watcher.stop()
        _config_watcher = None
