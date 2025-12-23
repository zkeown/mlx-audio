"""Weights & Biases logger for mlx-train."""

from __future__ import annotations

from typing import Any

from mlx_audio.train.loggers.base import Logger


class WandbLogger(Logger):
    """Weights & Biases logger.

    Logs metrics, hyperparameters, and artifacts to W&B.

    Example:
        >>> logger = WandbLogger(
        ...     project="my-mlx-project",
        ...     name="experiment-1",
        ...     config={"learning_rate": 1e-4, "batch_size": 32}
        ... )
        >>> trainer = Trainer(logger=logger)

    Note:
        Requires wandb to be installed: pip install wandb
    """

    def __init__(
        self,
        project: str,
        name: str | None = None,
        config: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        notes: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the W&B logger.

        Args:
            project: W&B project name
            name: Run name (optional, W&B will generate one if not provided)
            config: Hyperparameters to log
            tags: Tags for the run
            notes: Notes/description for the run
            **kwargs: Additional arguments passed to wandb.init()
        """
        try:
            import wandb
        except ImportError as e:
            raise ImportError(
                "wandb is required for WandbLogger. Install with: pip install wandb"
            ) from e

        self._wandb = wandb

        self._run = wandb.init(
            project=project,
            name=name,
            config=config,
            tags=tags,
            notes=notes,
            **kwargs,
        )

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        """Log metrics to W&B.

        Args:
            metrics: Dictionary of metric name to value
            step: Current training step
        """
        self._wandb.log(metrics, step=step)

    def log_hyperparams(self, params: dict[str, Any]) -> None:
        """Log/update hyperparameters in W&B config.

        Args:
            params: Dictionary of hyperparameter name to value
        """
        self._wandb.config.update(params)

    def finalize(self) -> None:
        """Finish the W&B run."""
        self._wandb.finish()

    @property
    def experiment(self) -> Any:
        """Access the underlying W&B run for advanced usage.

        Returns:
            The wandb.Run object
        """
        return self._run

    def log_artifact(
        self,
        artifact_path: str,
        name: str,
        artifact_type: str = "model",
    ) -> None:
        """Log an artifact (file/directory) to W&B.

        Args:
            artifact_path: Path to the artifact
            name: Name for the artifact
            artifact_type: Type of artifact (e.g., "model", "dataset")
        """
        artifact = self._wandb.Artifact(name=name, type=artifact_type)
        artifact.add_file(artifact_path)
        self._run.log_artifact(artifact)
