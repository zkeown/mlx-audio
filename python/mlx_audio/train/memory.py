"""Memory monitoring and management for MLX training.

Provides utilities to monitor memory usage and take action when thresholds
are exceeded, preventing OOM kills during long training runs.
"""

from __future__ import annotations

import gc
import os
import resource
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import mlx.core as mx

if TYPE_CHECKING:
    from mlx_audio.train.trainer import Trainer


@dataclass
class MemoryStats:
    """Current memory statistics."""

    rss_mb: float  # Resident Set Size (RAM used by process)
    gpu_mb: float  # GPU/Metal memory used
    rss_limit_mb: float | None  # System memory limit if available
    gpu_limit_mb: float | None  # GPU memory limit if available

    @property
    def rss_percent(self) -> float | None:
        """RSS as percentage of limit."""
        if self.rss_limit_mb:
            return (self.rss_mb / self.rss_limit_mb) * 100
        return None

    @property
    def gpu_percent(self) -> float | None:
        """GPU memory as percentage of limit."""
        if self.gpu_limit_mb:
            return (self.gpu_mb / self.gpu_limit_mb) * 100
        return None


class MemoryMonitor:
    """Monitor and manage memory during training.

    Tracks RSS (system RAM) and GPU memory usage, triggering cleanup
    actions when thresholds are exceeded.

    Example:
        >>> monitor = MemoryMonitor(
        ...     rss_threshold_mb=8000,  # Warn at 8GB RSS
        ...     gpu_threshold_mb=6000,  # Warn at 6GB GPU
        ...     auto_cleanup=True,
        ... )
        >>> monitor.check()  # Returns True if OK, False if critical
    """

    def __init__(
        self,
        rss_threshold_mb: float | None = None,
        gpu_threshold_mb: float | None = None,
        rss_critical_mb: float | None = None,
        gpu_critical_mb: float | None = None,
        auto_cleanup: bool = True,
        cleanup_interval: int = 100,
        verbose: bool = False,
    ):
        """Initialize memory monitor.

        Args:
            rss_threshold_mb: RSS threshold for warnings/cleanup (MB).
                Defaults to 80% of system memory.
            gpu_threshold_mb: GPU threshold for warnings/cleanup (MB).
                Defaults to 80% of GPU memory.
            rss_critical_mb: RSS level that triggers aggressive cleanup (MB).
                Defaults to 90% of system memory.
            gpu_critical_mb: GPU level that triggers aggressive cleanup (MB).
                Defaults to 90% of GPU memory.
            auto_cleanup: Automatically run cleanup when thresholds exceeded.
            cleanup_interval: Run periodic cleanup every N checks.
            verbose: Print memory stats on each check.
        """
        self.auto_cleanup = auto_cleanup
        self.cleanup_interval = cleanup_interval
        self.verbose = verbose
        self._check_count = 0

        # Detect system limits
        self._rss_limit_mb = self._get_system_memory_mb()
        self._gpu_limit_mb = self._get_gpu_memory_mb()

        # Set thresholds (default to percentages of limits)
        if rss_threshold_mb is not None:
            self.rss_threshold_mb = rss_threshold_mb
        elif self._rss_limit_mb:
            self.rss_threshold_mb = self._rss_limit_mb * 0.80
        else:
            self.rss_threshold_mb = 8000  # 8GB fallback

        if gpu_threshold_mb is not None:
            self.gpu_threshold_mb = gpu_threshold_mb
        elif self._gpu_limit_mb:
            self.gpu_threshold_mb = self._gpu_limit_mb * 0.80
        else:
            self.gpu_threshold_mb = 6000  # 6GB fallback

        if rss_critical_mb is not None:
            self.rss_critical_mb = rss_critical_mb
        elif self._rss_limit_mb:
            self.rss_critical_mb = self._rss_limit_mb * 0.90
        else:
            self.rss_critical_mb = 12000  # 12GB fallback

        if gpu_critical_mb is not None:
            self.gpu_critical_mb = gpu_critical_mb
        elif self._gpu_limit_mb:
            self.gpu_critical_mb = self._gpu_limit_mb * 0.90
        else:
            self.gpu_critical_mb = 10000  # 10GB fallback

    def _get_system_memory_mb(self) -> float | None:
        """Get total system memory in MB."""
        try:
            # macOS/Unix
            import subprocess

            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return int(result.stdout.strip()) / (1024 * 1024)
        except Exception:
            pass

        try:
            # Try /proc/meminfo on Linux
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        # Format: "MemTotal:       16384000 kB"
                        kb = int(line.split()[1])
                        return kb / 1024
        except Exception:
            pass

        return None

    def _get_gpu_memory_mb(self) -> float | None:
        """Get total GPU memory in MB (Apple Silicon unified memory)."""
        # On Apple Silicon, GPU shares system memory
        # Use a reasonable default based on typical allocations
        sys_mem = self._get_system_memory_mb()
        if sys_mem:
            # GPU can typically use up to 75% of system memory on Apple Silicon
            return sys_mem * 0.75
        return None

    def get_stats(self) -> MemoryStats:
        """Get current memory statistics."""
        # RSS (Resident Set Size) - actual RAM used
        rusage = resource.getrusage(resource.RUSAGE_SELF)
        # ru_maxrss is in bytes on macOS, KB on Linux
        if os.uname().sysname == "Darwin":
            rss_mb = rusage.ru_maxrss / (1024 * 1024)
        else:
            rss_mb = rusage.ru_maxrss / 1024

        # GPU memory
        try:
            gpu_mb = mx.get_active_memory() / (1024 * 1024)
        except Exception:
            gpu_mb = 0.0

        return MemoryStats(
            rss_mb=rss_mb,
            gpu_mb=gpu_mb,
            rss_limit_mb=self._rss_limit_mb,
            gpu_limit_mb=self._gpu_limit_mb,
        )

    def cleanup(self, aggressive: bool = False) -> None:
        """Run memory cleanup.

        Args:
            aggressive: If True, run more aggressive cleanup
                (multiple GC passes, synchronous GPU cleanup).
        """
        # Clear MLX memory cache
        mx.clear_cache()

        # Run garbage collection
        if aggressive:
            # Multiple passes for thorough cleanup
            for _ in range(3):
                gc.collect()
            # Synchronize to ensure GPU operations complete
            mx.synchronize()
            mx.clear_cache()
        else:
            gc.collect()

    def check(self) -> bool:
        """Check memory status and take action if needed.

        Returns:
            True if memory is OK, False if critical threshold exceeded.
        """
        self._check_count += 1
        stats = self.get_stats()

        if self.verbose:
            print(
                f"[Memory] RSS: {stats.rss_mb:.0f}MB, "
                f"GPU: {stats.gpu_mb:.0f}MB"
            )

        # Check for critical levels
        rss_critical = stats.rss_mb >= self.rss_critical_mb
        gpu_critical = stats.gpu_mb >= self.gpu_critical_mb

        if rss_critical or gpu_critical:
            if self.auto_cleanup:
                warnings.warn(
                    f"Memory critical (RSS: {stats.rss_mb:.0f}MB, "
                    f"GPU: {stats.gpu_mb:.0f}MB). Running aggressive cleanup.",
                    ResourceWarning,
                    stacklevel=2,
                )
                self.cleanup(aggressive=True)

                # Re-check after cleanup
                stats = self.get_stats()
                if stats.rss_mb >= self.rss_critical_mb:
                    warnings.warn(
                        f"Memory still critical after cleanup: {stats.rss_mb:.0f}MB",
                        ResourceWarning,
                        stacklevel=2,
                    )
                    return False
            else:
                return False

        # Check for warning levels
        rss_warning = stats.rss_mb >= self.rss_threshold_mb
        gpu_warning = stats.gpu_mb >= self.gpu_threshold_mb

        if rss_warning or gpu_warning:
            if self.auto_cleanup:
                self.cleanup(aggressive=False)

        # Periodic cleanup even if below thresholds
        if self._check_count % self.cleanup_interval == 0:
            self.cleanup(aggressive=False)

        return True

    def format_stats(self) -> str:
        """Format current memory stats as a string."""
        stats = self.get_stats()
        parts = [f"RSS: {stats.rss_mb:.0f}MB"]

        if stats.rss_percent is not None:
            parts[0] += f" ({stats.rss_percent:.0f}%)"

        parts.append(f"GPU: {stats.gpu_mb:.0f}MB")
        if stats.gpu_percent is not None:
            parts[1] += f" ({stats.gpu_percent:.0f}%)"

        return " | ".join(parts)


# Global monitor instance for convenience
_global_monitor: MemoryMonitor | None = None


def get_memory_monitor() -> MemoryMonitor:
    """Get or create the global memory monitor."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = MemoryMonitor()
    return _global_monitor


def check_memory() -> bool:
    """Check memory using the global monitor.

    Returns:
        True if memory is OK, False if critical.
    """
    return get_memory_monitor().check()


def get_memory_stats() -> MemoryStats:
    """Get current memory stats using the global monitor."""
    return get_memory_monitor().get_stats()


def cleanup_memory(aggressive: bool = False) -> None:
    """Run memory cleanup using the global monitor."""
    get_memory_monitor().cleanup(aggressive=aggressive)
