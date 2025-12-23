"""MLflow logger for mlx-train."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mlx_audio.train.loggers.base import Logger


class MLflowLogger(Logger):
    """MLflow logger for experiment tracking.

    Logs metrics, hyperparameters, and artifacts to MLflow.
    Supports both local and remote MLflow tracking servers.

    Example:
        >>> logger = MLflowLogger(
        ...     experiment_name="my-mlx-project",
        ...     run_name="experiment-1",
        ...     tracking_uri="http://localhost:5000"
        ... )
        >>> trainer = Trainer(logger=logger)

    Note:
        Requires mlflow to be installed: pip install mlflow
    """

    def __init__(
        self,
        experiment_name: str,
        run_name: str | None = None,
        tracking_uri: str | None = None,
        tags: dict[str, str] | None = None,
        artifact_location: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the MLflow logger.

        Args:
            experiment_name: Name of the MLflow experiment
            run_name: Name for this specific run (optional)
            tracking_uri: URI of the MLflow tracking server (optional, defaults to local)
            tags: Tags to apply to the run
            artifact_location: Default location for artifacts
            **kwargs: Additional arguments passed to mlflow.start_run()
        """
        try:
            import mlflow
        except ImportError as e:
            raise ImportError(
                "mlflow is required for MLflowLogger. Install with: pip install mlflow"
            ) from e

        self._mlflow = mlflow

        # Set tracking URI if provided
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        # Set or create experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(
                experiment_name,
                artifact_location=artifact_location,
            )
        else:
            experiment_id = experiment.experiment_id

        # Start the run
        self._run = mlflow.start_run(
            experiment_id=experiment_id,
            run_name=run_name,
            tags=tags,
            **kwargs,
        )
        self._run_id = self._run.info.run_id

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        """Log metrics to MLflow.

        Args:
            metrics: Dictionary of metric name to value
            step: Current training step
        """
        self._mlflow.log_metrics(metrics, step=step)

    def log_hyperparams(self, params: dict[str, Any]) -> None:
        """Log hyperparameters to MLflow.

        Args:
            params: Dictionary of hyperparameter name to value
        """
        # Flatten nested dicts and convert to strings for MLflow compatibility
        flat_params = {}
        for key, value in params.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    flat_params[f"{key}.{sub_key}"] = str(sub_value)
            else:
                flat_params[key] = str(value) if not isinstance(value, (int, float)) else value
        self._mlflow.log_params(flat_params)

    def finalize(self) -> None:
        """End the MLflow run."""
        self._mlflow.end_run()

    @property
    def experiment(self) -> Any:
        """Access the underlying MLflow run for advanced usage.

        Returns:
            The mlflow.ActiveRun object
        """
        return self._run

    @property
    def run_id(self) -> str:
        """Get the MLflow run ID.

        Returns:
            The run ID string
        """
        return self._run_id

    # === Extension methods ===

    def log_artifact(self, local_path: str | Path, artifact_path: str | None = None) -> None:
        """Log a local file or directory as an artifact.

        Args:
            local_path: Path to the local file or directory
            artifact_path: Path within the artifact store (optional)
        """
        self._mlflow.log_artifact(str(local_path), artifact_path=artifact_path)

    def log_artifacts(self, local_dir: str | Path, artifact_path: str | None = None) -> None:
        """Log all files in a local directory as artifacts.

        Args:
            local_dir: Path to the local directory
            artifact_path: Path within the artifact store (optional)
        """
        self._mlflow.log_artifacts(str(local_dir), artifact_path=artifact_path)

    def log_model(
        self,
        model: Any,
        artifact_path: str = "model",
    ) -> None:
        """Log an MLX model weights to MLflow.

        Note: MLX models are saved as safetensors since MLflow doesn't have
        native MLX support yet.

        Args:
            model: The MLX model to log (must have .parameters() method)
            artifact_path: Path within the artifact store
        """
        import tempfile

        import mlx.nn as nn

        # Save model weights to temp directory and log
        with tempfile.TemporaryDirectory() as tmpdir:
            weights_path = Path(tmpdir) / "weights.safetensors"
            nn.save_weights(str(weights_path), dict(model.parameters()))
            self._mlflow.log_artifact(str(weights_path), artifact_path=artifact_path)

    def set_tag(self, key: str, value: str) -> None:
        """Set a tag on the current run.

        Args:
            key: Tag key
            value: Tag value
        """
        self._mlflow.set_tag(key, value)

    def log_figure(self, figure: Any, artifact_file: str) -> None:
        """Log a matplotlib figure.

        Args:
            figure: Matplotlib figure object
            artifact_file: Filename for the artifact
        """
        self._mlflow.log_figure(figure, artifact_file)

    def log_dict(self, dictionary: dict, artifact_file: str) -> None:
        """Log a dictionary as a JSON or YAML artifact.

        Args:
            dictionary: Dictionary to log
            artifact_file: Filename (should end in .json or .yaml)
        """
        self._mlflow.log_dict(dictionary, artifact_file)
