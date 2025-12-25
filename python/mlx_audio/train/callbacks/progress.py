"""Progress bar callbacks for mlx-train."""

from __future__ import annotations

import os
import sys
import time
from typing import TYPE_CHECKING, Any

from mlx_audio.train.callbacks.base import Callback, CallbackContext, CallbackPriority

if TYPE_CHECKING:
    from mlx_audio.train.module import TrainModule
    from mlx_audio.train.trainer import Trainer


class ProgressBar(Callback):
    """Display training progress.

    Shows epoch, step, loss, metrics, and speed (samples/sec).
    Uses simple print-based output - no external dependencies.

    Example:
        >>> trainer = Trainer(callbacks=[ProgressBar()])

        Output:
        Epoch 1/10
          Step 100 | train_loss: 0.4523 | train_accuracy: 0.8750 | 1234.5 samples/sec
    """

    priority = CallbackPriority.LOW

    def __init__(
        self,
        refresh_rate: int = 10,
        show_eta: bool = True,
    ) -> None:
        """Initialize the progress bar.

        Args:
            refresh_rate: Update display every N batches
            show_eta: Whether to show estimated time remaining
        """
        self.refresh_rate = refresh_rate
        self.show_eta = show_eta

        self._epoch_start_time: float = 0
        self._fit_start_time: float = 0
        self._samples_processed: int = 0
        self._last_print_len: int = 0

    def on_fit_start(self, trainer: Trainer, module: TrainModule) -> None:
        self._fit_start_time = time.time()

    def on_train_epoch_start(
        self,
        trainer: Trainer,
        module: TrainModule,
        ctx: CallbackContext,
    ) -> None:
        self._epoch_start_time = time.time()
        self._samples_processed = 0

        epoch_str = f"Epoch {ctx.epoch + 1}"
        if ctx.max_epochs:
            epoch_str += f"/{ctx.max_epochs}"
        print(f"\n{epoch_str}")

    def on_train_batch_end(
        self,
        trainer: Trainer,
        module: TrainModule,
        batch: Any,
        outputs: dict[str, Any],
        ctx: CallbackContext,
    ) -> None:
        batch_size = self._get_batch_size(batch)
        self._samples_processed += batch_size

        if ctx.batch_idx % self.refresh_rate == 0:
            self._print_progress(ctx, outputs)

    def on_train_epoch_end(
        self,
        trainer: Trainer,
        module: TrainModule,
        ctx: CallbackContext,
    ) -> None:
        elapsed = time.time() - self._epoch_start_time
        # Clear line and print summary
        print(f"\r{' ' * self._last_print_len}", end="\r")
        print(f"  Epoch {ctx.epoch + 1} completed in {elapsed:.1f}s")

    def on_validation_start(
        self,
        trainer: Trainer,
        module: TrainModule,
        ctx: CallbackContext,
    ) -> None:
        print("  Validating...", end="\r")

    def on_validation_end(
        self,
        trainer: Trainer,
        module: TrainModule,
        ctx: CallbackContext,
        metrics: dict[str, float],
    ) -> None:
        # Format validation metrics
        metrics_str = " | ".join(f"{k}: {v:.4f}" for k, v in sorted(metrics.items()))
        print(f"  Validation: {metrics_str}")

    def on_fit_end(self, trainer: Trainer, module: TrainModule) -> None:
        total_time = time.time() - self._fit_start_time
        print(f"\nTraining completed in {self._format_time(total_time)}")

    def _print_progress(self, ctx: CallbackContext, outputs: dict[str, Any]) -> None:
        """Print progress line."""
        elapsed = time.time() - self._epoch_start_time
        samples_per_sec = self._samples_processed / elapsed if elapsed > 0 else 0

        # Build metrics string
        metrics_parts = []
        for k, v in sorted(ctx.metrics.items()):
            if isinstance(v, float):
                metrics_parts.append(f"{k}: {v:.4f}")

        metrics_str = " | ".join(metrics_parts)

        # Build progress string
        progress = f"  Step {ctx.global_step}"
        if metrics_str:
            progress += f" | {metrics_str}"
        progress += f" | {samples_per_sec:.1f} samples/sec"

        # Add ETA if enabled
        if self.show_eta and ctx.max_steps:
            remaining_steps = ctx.max_steps - ctx.global_step
            steps_per_sec = ctx.global_step / elapsed if elapsed > 0 else 0
            if steps_per_sec > 0:
                eta_sec = remaining_steps / steps_per_sec
                progress += f" | ETA: {self._format_time(eta_sec)}"

        # Print with carriage return for in-place update
        self._last_print_len = len(progress)
        print(f"\r{progress}", end="", flush=True)

    def _get_batch_size(self, batch: Any) -> int:
        """Extract batch size from batch data."""
        if isinstance(batch, tuple | list) and len(batch) > 0:
            first = batch[0]
            if hasattr(first, "shape"):
                return first.shape[0]
        elif hasattr(batch, "shape"):
            return batch.shape[0]
        return 1

    def _format_time(self, seconds: float) -> str:
        """Format seconds as human-readable time."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins}m {secs}s"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            return f"{hours}h {mins}m"


class RichProgressBar(Callback):
    """Enhanced progress display with rich formatting.

    Features:
    - Boxed training configuration summary
    - Visual progress bars for epochs
    - Color-coded metrics (where terminal supports it)
    - Throughput history and trends
    - Live updating display

    Example:
        >>> trainer = Trainer(callbacks=[RichProgressBar()])

        Output:
        ╭──────────────────────────────────────────────────────────────╮
        │  DrumUX Training                                             │
        │  Model: DrumuxTrainModule (2.1M params)                      │
        │  Config: batch=2, accum=4, lr=1e-4, epochs=30               │
        ├──────────────────────────────────────────────────────────────┤
        │  Epoch 3/30  ████████░░░░░░░░░░░░░░░░░░░░░░  10%             │
        │  Step 1,234/12,340                                           │
        │                                                              │
        │  train_loss    0.4523  ▼                                     │
        │  train_f1      0.8234  ▲                                     │
        │                                                              │
        │  ⚡ 16.5 samples/sec   ETA: 2h 34m                           │
        ╰──────────────────────────────────────────────────────────────╯
    """

    priority = CallbackPriority.LOW

    # Box drawing characters
    BOX_TL = "╭"
    BOX_TR = "╮"
    BOX_BL = "╰"
    BOX_BR = "╯"
    BOX_H = "─"
    BOX_V = "│"
    BOX_ML = "├"
    BOX_MR = "┤"

    # Progress bar characters
    BAR_FULL = "█"
    BAR_EMPTY = "░"
    BAR_PARTIAL = ["▏", "▎", "▍", "▌", "▋", "▊", "▉"]

    # Trend indicators
    TREND_UP = "▲"
    TREND_DOWN = "▼"
    TREND_FLAT = "─"

    def __init__(
        self,
        refresh_rate: int = 10,
        width: int = 70,
        show_throughput_history: bool = True,
        use_colors: bool = True,
        show_sparklines: bool = True,
    ) -> None:
        """Initialize the rich progress bar.

        Args:
            refresh_rate: Update display every N batches
            width: Width of the display box (min 50)
            show_throughput_history: Show throughput trend sparkline
            use_colors: Use ANSI colors (auto-disabled if not supported)
            show_sparklines: Show mini graphs for metrics
        """
        self.refresh_rate = refresh_rate
        self.width = max(50, width)
        self.show_throughput_history = show_throughput_history
        self.show_sparklines = show_sparklines

        # Auto-detect color support
        self.use_colors = use_colors and self._supports_colors()

        # State
        self._fit_start_time: float = 0
        self._epoch_start_time: float = 0
        self._samples_processed: int = 0
        self._total_samples: int = 0
        self._last_display_lines: int = 0

        # History for trends
        self._loss_history: list[float] = []
        self._throughput_history: list[float] = []
        self._metric_history: dict[str, list[float]] = {}

        # Training config (captured at start)
        self._model_name: str = ""
        self._num_params: int = 0
        self._batch_size: int = 0
        self._accumulate_grad_batches: int = 1
        self._learning_rate: float = 0.0

    def _supports_colors(self) -> bool:
        """Check if terminal supports colors."""
        # Check NO_COLOR environment variable
        if os.environ.get("NO_COLOR"):
            return False

        # Check if stdout is a TTY
        if not hasattr(sys.stdout, "isatty") or not sys.stdout.isatty():
            return False

        # Check TERM environment variable
        term = os.environ.get("TERM", "")
        return term not in ("dumb", "")

    def _color(self, text: str, color: str) -> str:
        """Apply ANSI color to text if colors enabled."""
        if not self.use_colors:
            return text

        colors = {
            "reset": "\033[0m",
            "bold": "\033[1m",
            "dim": "\033[2m",
            "red": "\033[31m",
            "green": "\033[32m",
            "yellow": "\033[33m",
            "blue": "\033[34m",
            "magenta": "\033[35m",
            "cyan": "\033[36m",
            "white": "\033[37m",
            "bright_green": "\033[92m",
            "bright_yellow": "\033[93m",
            "bright_blue": "\033[94m",
            "bright_cyan": "\033[96m",
        }
        return f"{colors.get(color, '')}{text}{colors['reset']}"

    def _box_line(self, content: str, align: str = "left") -> str:
        """Create a boxed line with proper padding."""
        # Strip ANSI codes for length calculation
        visible_len = len(self._strip_ansi(content))
        inner_width = self.width - 4  # Account for │ and spaces on both sides

        if align == "center":
            pad_total = inner_width - visible_len
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            padded = " " * pad_left + content + " " * pad_right
        elif align == "right":
            padded = " " * (inner_width - visible_len) + content
        else:
            padded = content + " " * (inner_width - visible_len)

        return f"{self.BOX_V} {padded} {self.BOX_V}"

    def _strip_ansi(self, text: str) -> str:
        """Remove ANSI escape codes from text."""
        import re
        return re.sub(r"\033\[[0-9;]*m", "", text)

    def _box_top(self) -> str:
        """Create top border of box."""
        return f"{self.BOX_TL}{self.BOX_H * (self.width - 2)}{self.BOX_TR}"

    def _box_bottom(self) -> str:
        """Create bottom border of box."""
        return f"{self.BOX_BL}{self.BOX_H * (self.width - 2)}{self.BOX_BR}"

    def _box_divider(self) -> str:
        """Create horizontal divider within box."""
        return f"{self.BOX_ML}{self.BOX_H * (self.width - 2)}{self.BOX_MR}"

    def _progress_bar(self, progress: float, width: int = 30) -> str:
        """Create a progress bar.

        Args:
            progress: Progress from 0.0 to 1.0
            width: Width in characters

        Returns:
            Progress bar string like "████████░░░░░░░░░░░░"
        """
        progress = max(0.0, min(1.0, progress))
        filled = int(progress * width)
        partial_idx = int((progress * width - filled) * len(self.BAR_PARTIAL))

        bar = self.BAR_FULL * filled
        if filled < width and partial_idx > 0:
            bar += self.BAR_PARTIAL[partial_idx - 1]
            filled += 1
        bar += self.BAR_EMPTY * (width - filled)

        if self.use_colors:
            return self._color(bar, "bright_cyan")
        return bar

    def _sparkline(self, values: list[float], width: int = 10) -> str:
        """Create a sparkline from values.

        Args:
            values: List of values to plot
            width: Width in characters

        Returns:
            Sparkline string like "▁▂▃▅▆▇█▆▅▃"
        """
        if not values:
            return " " * width

        # Take last `width` values
        values = values[-width:]
        if len(values) < 2:
            return " " * width

        chars = "▁▂▃▄▅▆▇█"
        min_val = min(values)
        max_val = max(values)

        if max_val == min_val:
            return chars[3] * len(values) + " " * (width - len(values))

        result = ""
        for v in values:
            idx = int((v - min_val) / (max_val - min_val) * (len(chars) - 1))
            result += chars[idx]

        return result.ljust(width)

    def _trend_indicator(self, values: list[float], lower_is_better: bool = True) -> str:
        """Get trend indicator based on recent values."""
        if len(values) < 2:
            return self._color(self.TREND_FLAT, "dim")

        recent = values[-5:] if len(values) >= 5 else values
        if recent[-1] < recent[0] * 0.99:  # Decreased by > 1%
            color = "green" if lower_is_better else "red"
            return self._color(self.TREND_DOWN, color)
        elif recent[-1] > recent[0] * 1.01:  # Increased by > 1%
            color = "red" if lower_is_better else "green"
            return self._color(self.TREND_UP, color)
        return self._color(self.TREND_FLAT, "dim")

    def _format_number(self, n: int) -> str:
        """Format number with thousand separators."""
        return f"{n:,}"

    def _format_time(self, seconds: float) -> str:
        """Format seconds as human-readable time."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins}m {secs}s"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            return f"{hours}h {mins}m"

    def _format_params(self, n: int) -> str:
        """Format parameter count (e.g., 2.1M, 345K)."""
        if n >= 1_000_000_000:
            return f"{n / 1_000_000_000:.1f}B"
        elif n >= 1_000_000:
            return f"{n / 1_000_000:.1f}M"
        elif n >= 1_000:
            return f"{n / 1_000:.1f}K"
        return str(n)

    def _clear_display(self) -> None:
        """Clear previous display lines."""
        if self._last_display_lines > 0:
            # Move cursor up and clear lines
            for _ in range(self._last_display_lines):
                sys.stdout.write("\033[F\033[K")
            sys.stdout.flush()

    def _get_batch_size(self, batch: Any) -> int:
        """Extract batch size from batch data."""
        if isinstance(batch, tuple | list) and len(batch) > 0:
            first = batch[0]
            if hasattr(first, "shape"):
                return first.shape[0]
        elif hasattr(batch, "shape"):
            return batch.shape[0]
        return 1

    # === Callback Hooks ===

    def on_fit_start(self, trainer: Trainer, module: TrainModule) -> None:
        self._fit_start_time = time.time()

        # Capture training configuration
        self._model_name = module.__class__.__name__
        self._num_params = getattr(module, "num_parameters", 0)
        self._batch_size = getattr(trainer, "_current_batch_size", 0)
        self._accumulate_grad_batches = getattr(trainer, "accumulate_grad_batches", 1)
        self._learning_rate = getattr(module, "learning_rate", 0.0)

        # Try to get from optimizer config
        if self._learning_rate == 0 and hasattr(module, "configure_optimizers"):
            try:
                opt_config = module.configure_optimizers()
                if isinstance(opt_config, dict):
                    self._learning_rate = opt_config.get("learning_rate", 0)
            except Exception:
                pass

        # Reset history
        self._loss_history = []
        self._throughput_history = []
        self._metric_history = {}

    def on_train_epoch_start(
        self,
        trainer: Trainer,
        module: TrainModule,
        ctx: CallbackContext,
    ) -> None:
        self._epoch_start_time = time.time()
        self._samples_processed = 0

        # Estimate total samples in epoch
        if hasattr(trainer, "_train_dataloader"):
            try:
                dl = trainer._train_dataloader
                if hasattr(dl, "__len__"):
                    self._total_samples = len(dl) * self._batch_size
            except Exception:
                pass

    def on_train_batch_end(
        self,
        trainer: Trainer,
        module: TrainModule,
        batch: Any,
        outputs: dict[str, Any],
        ctx: CallbackContext,
    ) -> None:
        batch_size = self._get_batch_size(batch)
        self._samples_processed += batch_size

        if self._batch_size == 0:
            self._batch_size = batch_size

        # Record loss history
        if "loss" in outputs:
            loss_val = float(outputs["loss"])
            self._loss_history.append(loss_val)
            if len(self._loss_history) > 100:
                self._loss_history = self._loss_history[-100:]

        # Record metric history
        for key, value in ctx.metrics.items():
            if isinstance(value, float):
                if key not in self._metric_history:
                    self._metric_history[key] = []
                self._metric_history[key].append(value)
                if len(self._metric_history[key]) > 100:
                    self._metric_history[key] = self._metric_history[key][-100:]

        # Record throughput
        elapsed = time.time() - self._epoch_start_time
        if elapsed > 0:
            throughput = self._samples_processed / elapsed
            self._throughput_history.append(throughput)
            if len(self._throughput_history) > 50:
                self._throughput_history = self._throughput_history[-50:]

        if ctx.batch_idx % self.refresh_rate == 0:
            self._render_display(trainer, module, ctx)

    def on_train_epoch_end(
        self,
        trainer: Trainer,
        module: TrainModule,
        ctx: CallbackContext,
    ) -> None:
        elapsed = time.time() - self._epoch_start_time
        self._clear_display()

        # Print epoch summary
        lines = []
        lines.append(self._box_top())
        lines.append(self._box_line(
            self._color(f"Epoch {ctx.epoch + 1} Complete", "bold"),
            align="center"
        ))
        lines.append(self._box_divider())

        elapsed_str = self._format_time(elapsed)
        avg_throughput = self._samples_processed / elapsed if elapsed > 0 else 0
        lines.append(self._box_line(
            f"Duration: {elapsed_str}  |  "
            f"Samples: {self._format_number(self._samples_processed)}  |  "
            f"Avg: {avg_throughput:.1f} samples/sec"
        ))

        # Show final metrics for epoch
        if ctx.metrics:
            lines.append(self._box_line(""))
            for key, value in sorted(ctx.metrics.items()):
                if isinstance(value, float):
                    lower_is_better = "loss" in key.lower()
                    trend = self._trend_indicator(
                        self._metric_history.get(key, []), lower_is_better
                    )
                    lines.append(self._box_line(f"  {key}: {value:.4f} {trend}"))

        lines.append(self._box_bottom())

        print("\n".join(lines))
        self._last_display_lines = 0

    def on_validation_start(
        self,
        trainer: Trainer,
        module: TrainModule,
        ctx: CallbackContext,
    ) -> None:
        self._clear_display()
        print(self._color("  Validating...", "dim"))
        self._last_display_lines = 1

    def on_validation_end(
        self,
        trainer: Trainer,
        module: TrainModule,
        ctx: CallbackContext,
        metrics: dict[str, float],
    ) -> None:
        self._clear_display()

        # Print validation results in a nice box
        lines = []
        lines.append(self._box_divider())
        lines.append(self._box_line(self._color("Validation Results", "bright_cyan")))

        for key, value in sorted(metrics.items()):
            # Color code based on metric name
            if "loss" in key.lower():
                color = "yellow"
            elif any(m in key.lower() for m in ["acc", "f1", "precision", "recall"]):
                color = "green" if value > 0.5 else "yellow"
            else:
                color = "white"

            lines.append(self._box_line(
                f"  {key}: {self._color(f'{value:.4f}', color)}"
            ))

        print("\n".join(lines))
        self._last_display_lines = len(lines)

    def on_fit_end(self, trainer: Trainer, module: TrainModule) -> None:
        total_time = time.time() - self._fit_start_time
        self._clear_display()

        # Print final summary
        lines = []
        lines.append("")
        lines.append(self._box_top())
        lines.append(self._box_line(
            self._color("Training Complete!", "bright_green"),
            align="center"
        ))
        lines.append(self._box_divider())
        lines.append(self._box_line(f"Total time: {self._format_time(total_time)}"))

        # Final throughput stats
        if self._throughput_history:
            avg_throughput = sum(self._throughput_history) / len(self._throughput_history)
            max_throughput = max(self._throughput_history)
            lines.append(self._box_line(
                f"Throughput: avg {avg_throughput:.1f}, peak {max_throughput:.1f} samples/sec"
            ))

        lines.append(self._box_bottom())

        print("\n".join(lines))

    def _render_display(
        self,
        trainer: Trainer,
        module: TrainModule,
        ctx: CallbackContext,
    ) -> None:
        """Render the full progress display."""
        self._clear_display()

        lines = []

        # Header with model info
        lines.append(self._box_top())
        lines.append(self._box_line(
            self._color(self._model_name, "bold"),
            align="center"
        ))

        # Config line
        config_parts = []
        if self._num_params > 0:
            config_parts.append(f"{self._format_params(self._num_params)} params")
        if self._batch_size > 0:
            eff_batch = self._batch_size * self._accumulate_grad_batches
            if self._accumulate_grad_batches > 1:
                config_parts.append(f"batch={self._batch_size}x{self._accumulate_grad_batches}={eff_batch}")
            else:
                config_parts.append(f"batch={self._batch_size}")
        if self._learning_rate > 0:
            config_parts.append(f"lr={self._learning_rate:.0e}")

        if config_parts:
            lines.append(self._box_line(
                self._color(" | ".join(config_parts), "dim"),
                align="center"
            ))

        lines.append(self._box_divider())

        # Epoch progress
        epoch_progress = (ctx.epoch) / ctx.max_epochs if ctx.max_epochs else 0
        step_in_epoch = ctx.batch_idx + 1
        steps_per_epoch = ctx.max_steps // ctx.max_epochs if ctx.max_epochs and ctx.max_steps else 0

        epoch_str = f"Epoch {ctx.epoch + 1}"
        if ctx.max_epochs:
            epoch_str += f"/{ctx.max_epochs}"

        # Calculate epoch-level progress including current position
        if steps_per_epoch > 0:
            epoch_progress = (ctx.epoch + step_in_epoch / steps_per_epoch) / ctx.max_epochs
        progress_bar = self._progress_bar(epoch_progress, width=35)
        pct = f"{epoch_progress * 100:.0f}%"

        lines.append(self._box_line(f"{epoch_str}  {progress_bar}  {pct}"))

        # Step info
        step_str = f"Step {self._format_number(ctx.global_step)}"
        if ctx.max_steps:
            step_str += f"/{self._format_number(ctx.max_steps)}"
        lines.append(self._box_line(self._color(step_str, "dim")))

        lines.append(self._box_line(""))

        # Metrics with trends
        displayed_metrics = 0
        for key, value in sorted(ctx.metrics.items()):
            if isinstance(value, float) and displayed_metrics < 6:  # Limit to 6 metrics
                history = self._metric_history.get(key, [])
                lower_is_better = "loss" in key.lower()
                trend = self._trend_indicator(history, lower_is_better)

                # Sparkline if enabled
                sparkline = ""
                if self.show_sparklines and len(history) > 2:
                    sparkline = self._color(self._sparkline(history, 12), "dim")

                # Color code the value
                if "loss" in key.lower():
                    value_str = self._color(f"{value:.4f}", "yellow")
                elif any(m in key.lower() for m in ["acc", "f1", "precision", "recall", "auc"]):
                    value_str = self._color(f"{value:.4f}", "green" if value > 0.5 else "yellow")
                else:
                    value_str = f"{value:.4f}"

                metric_line = f"  {key:20s} {value_str} {trend}"
                if sparkline:
                    metric_line += f"  {sparkline}"
                lines.append(self._box_line(metric_line))
                displayed_metrics += 1

        lines.append(self._box_line(""))

        # Throughput and ETA
        elapsed = time.time() - self._epoch_start_time
        samples_per_sec = self._samples_processed / elapsed if elapsed > 0 else 0

        throughput_str = f"{self._color('⚡', 'bright_yellow')} {samples_per_sec:.1f} samples/sec"

        # ETA calculation
        if ctx.max_steps and ctx.global_step > 0:
            total_elapsed = time.time() - self._fit_start_time
            steps_per_sec = ctx.global_step / total_elapsed
            if steps_per_sec > 0:
                remaining_steps = ctx.max_steps - ctx.global_step
                eta_sec = remaining_steps / steps_per_sec
                throughput_str += f"   ETA: {self._format_time(eta_sec)}"

        lines.append(self._box_line(throughput_str))

        lines.append(self._box_bottom())

        # Print all lines
        output = "\n".join(lines)
        print(output)
        self._last_display_lines = len(lines)
