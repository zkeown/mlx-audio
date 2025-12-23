"""TensorBoard logger for mlx-train."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mlx_audio.train.loggers.base import Logger


class TensorBoardLogger(Logger):
    """TensorBoard logger for experiment tracking.

    Logs metrics, hyperparameters, and optional histograms/audio to TensorBoard.

    Example:
        >>> logger = TensorBoardLogger(
        ...     log_dir="./runs/experiment_1",
        ...     name="my_experiment",
        ...     comment="_lr_0.001"
        ... )
        >>> trainer = Trainer(logger=logger)

    Note:
        Requires tensorboardX to be installed: pip install tensorboardX
    """

    def __init__(
        self,
        log_dir: str | Path = "./runs",
        name: str | None = None,
        comment: str = "",
        purge_step: int | None = None,
        flush_secs: int = 120,
        **kwargs: Any,
    ) -> None:
        """Initialize the TensorBoard logger.

        Args:
            log_dir: Base directory for TensorBoard logs
            name: Experiment name (subdirectory within log_dir)
            comment: Comment appended to the run folder name
            purge_step: Step from which to purge old data (for resuming)
            flush_secs: How often to flush to disk (seconds)
            **kwargs: Additional arguments passed to SummaryWriter
        """
        try:
            from tensorboardX import SummaryWriter

            self._tb_module = "tensorboardX"
        except ImportError:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self._tb_module = "torch"
            except ImportError as e:
                raise ImportError(
                    "TensorBoard support requires tensorboardX or torch. "
                    "Install with: pip install tensorboardX"
                ) from e

        # Construct log directory
        log_dir = Path(log_dir)
        if name:
            log_dir = log_dir / name

        self._log_dir = log_dir
        self._writer = SummaryWriter(
            logdir=str(log_dir),
            comment=comment,
            purge_step=purge_step,
            flush_secs=flush_secs,
            **kwargs,
        )

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        """Log scalar metrics to TensorBoard.

        Args:
            metrics: Dictionary of metric name to value
            step: Current training step
        """
        for name, value in metrics.items():
            self._writer.add_scalar(name, value, global_step=step)

    def log_hyperparams(self, params: dict[str, Any]) -> None:
        """Log hyperparameters to TensorBoard.

        Args:
            params: Dictionary of hyperparameter name to value
        """
        # TensorBoard stores hparams with associated metrics
        # We log with empty metrics initially; they get updated during training
        self._writer.add_hparams(params, {})

    def finalize(self) -> None:
        """Close the TensorBoard writer."""
        self._writer.close()

    @property
    def experiment(self) -> Any:
        """Access the underlying SummaryWriter for advanced usage.

        Returns:
            The SummaryWriter object
        """
        return self._writer

    @property
    def log_dir(self) -> Path:
        """Get the log directory path.

        Returns:
            Path to the TensorBoard log directory
        """
        return self._log_dir

    # === Extension methods for audio ML ===

    def log_histogram(
        self,
        tag: str,
        values: Any,
        step: int,
        bins: str = "auto",
    ) -> None:
        """Log a histogram (e.g., for weights/gradients).

        Args:
            tag: Name for the histogram
            values: Values to histogram (numpy array or similar)
            step: Current training step
            bins: Histogram binning strategy
        """
        self._writer.add_histogram(tag, values, global_step=step, bins=bins)

    def log_image(
        self,
        tag: str,
        img_tensor: Any,
        step: int,
        dataformats: str = "CHW",
    ) -> None:
        """Log an image (e.g., spectrogram visualization).

        Args:
            tag: Name for the image
            img_tensor: Image tensor
            step: Current training step
            dataformats: Format of the image tensor (CHW, HWC, etc.)
        """
        self._writer.add_image(tag, img_tensor, global_step=step, dataformats=dataformats)

    def log_audio(
        self,
        tag: str,
        snd_tensor: Any,
        step: int,
        sample_rate: int = 44100,
    ) -> None:
        """Log audio (especially useful for mlx-audio).

        Args:
            tag: Name for the audio
            snd_tensor: Audio tensor (1D or 2D)
            step: Current training step
            sample_rate: Audio sample rate in Hz
        """
        self._writer.add_audio(tag, snd_tensor, global_step=step, sample_rate=sample_rate)

    def log_text(self, tag: str, text: str, step: int) -> None:
        """Log text.

        Args:
            tag: Name for the text
            text: Text content
            step: Current training step
        """
        self._writer.add_text(tag, text, global_step=step)

    def log_figure(self, tag: str, figure: Any, step: int) -> None:
        """Log a matplotlib figure.

        Args:
            tag: Name for the figure
            figure: Matplotlib figure object
            step: Current training step
        """
        self._writer.add_figure(tag, figure, global_step=step)
