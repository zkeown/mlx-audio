"""Tests for mlx-train loggers."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mlx_audio.train.loggers.base import Logger


class TestTensorBoardLogger:
    """Tests for TensorBoardLogger."""

    def test_import_error_message(self):
        """Test informative error when tensorboardX not installed."""
        with patch.dict("sys.modules", {"tensorboardX": None, "torch": None}):
            # Need to reload the module to trigger the import error
            import importlib
            import sys

            # Remove cached module if present
            if "mlx_audio.train.loggers.tensorboard" in sys.modules:
                del sys.modules["mlx_audio.train.loggers.tensorboard"]

            with pytest.raises(ImportError) as exc_info:
                from mlx_audio.train.loggers.tensorboard import TensorBoardLogger

                TensorBoardLogger()

            assert "tensorboardX" in str(exc_info.value)

    def test_log_metrics(self):
        """Test logging scalar metrics."""
        pytest.importorskip("tensorboardX")
        from mlx_audio.train.loggers.tensorboard import TensorBoardLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TensorBoardLogger(log_dir=tmpdir)

            # Mock the writer after initialization
            mock_writer = MagicMock()
            logger._writer = mock_writer

            logger.log_metrics({"loss": 0.5, "accuracy": 0.9}, step=100)

            assert mock_writer.add_scalar.call_count == 2
            mock_writer.add_scalar.assert_any_call("loss", 0.5, global_step=100)
            mock_writer.add_scalar.assert_any_call("accuracy", 0.9, global_step=100)

    def test_log_hyperparams(self):
        """Test logging hyperparameters."""
        pytest.importorskip("tensorboardX")
        from mlx_audio.train.loggers.tensorboard import TensorBoardLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TensorBoardLogger(log_dir=tmpdir)

            mock_writer = MagicMock()
            logger._writer = mock_writer

            logger.log_hyperparams({"lr": 0.001, "batch_size": 32})

            mock_writer.add_hparams.assert_called_once_with({"lr": 0.001, "batch_size": 32}, {})

    def test_finalize(self):
        """Test cleanup on finalize."""
        pytest.importorskip("tensorboardX")
        from mlx_audio.train.loggers.tensorboard import TensorBoardLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TensorBoardLogger(log_dir=tmpdir)

            mock_writer = MagicMock()
            logger._writer = mock_writer

            logger.finalize()

            mock_writer.close.assert_called_once()

    def test_experiment_property(self):
        """Test access to underlying writer."""
        pytest.importorskip("tensorboardX")
        from mlx_audio.train.loggers.tensorboard import TensorBoardLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TensorBoardLogger(log_dir=tmpdir)
            assert logger.experiment is logger._writer
            logger.finalize()

    def test_log_dir_property(self):
        """Test log_dir property."""
        pytest.importorskip("tensorboardX")
        from mlx_audio.train.loggers.tensorboard import TensorBoardLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TensorBoardLogger(log_dir=tmpdir, name="test_run")
            assert logger.log_dir == Path(tmpdir) / "test_run"
            logger.finalize()

    def test_extension_methods(self):
        """Test extension methods exist and are callable."""
        pytest.importorskip("tensorboardX")
        from mlx_audio.train.loggers.tensorboard import TensorBoardLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TensorBoardLogger(log_dir=tmpdir)

            mock_writer = MagicMock()
            logger._writer = mock_writer

            # Test all extension methods exist
            import numpy as np

            logger.log_histogram("weights", np.array([1, 2, 3]), step=1)
            logger.log_text("description", "hello world", step=1)

            mock_writer.add_histogram.assert_called_once()
            mock_writer.add_text.assert_called_once()


class TestMLflowLogger:
    """Tests for MLflowLogger."""

    def test_import_error_message(self):
        """Test informative error when mlflow not installed."""
        with patch.dict("sys.modules", {"mlflow": None}):
            import importlib
            import sys

            if "mlx_audio.train.loggers.mlflow" in sys.modules:
                del sys.modules["mlx_audio.train.loggers.mlflow"]

            with pytest.raises(ImportError) as exc_info:
                from mlx_audio.train.loggers.mlflow import MLflowLogger

                MLflowLogger(experiment_name="test")

            assert "mlflow" in str(exc_info.value)

    def test_log_metrics(self):
        """Test logging metrics."""
        mlflow = pytest.importorskip("mlflow")

        with patch.object(mlflow, "log_metrics") as mock_log:
            with patch.object(mlflow, "get_experiment_by_name") as mock_get_exp:
                with patch.object(mlflow, "start_run") as mock_start:
                    with patch.object(mlflow, "end_run"):
                        mock_get_exp.return_value = MagicMock(experiment_id="123")
                        mock_start.return_value = MagicMock(info=MagicMock(run_id="abc"))

                        from mlx_audio.train.loggers.mlflow import MLflowLogger

                        logger = MLflowLogger(experiment_name="test")
                        logger.log_metrics({"loss": 0.5}, step=100)

                        mock_log.assert_called_once_with({"loss": 0.5}, step=100)
                        logger.finalize()

    def test_log_hyperparams(self):
        """Test logging hyperparameters."""
        mlflow = pytest.importorskip("mlflow")

        with patch.object(mlflow, "log_params") as mock_log:
            with patch.object(mlflow, "get_experiment_by_name") as mock_get_exp:
                with patch.object(mlflow, "start_run") as mock_start:
                    with patch.object(mlflow, "end_run"):
                        mock_get_exp.return_value = MagicMock(experiment_id="123")
                        mock_start.return_value = MagicMock(info=MagicMock(run_id="abc"))

                        from mlx_audio.train.loggers.mlflow import MLflowLogger

                        logger = MLflowLogger(experiment_name="test")
                        logger.log_hyperparams({"lr": 0.001})

                        mock_log.assert_called_once()
                        logger.finalize()

    def test_finalize(self):
        """Test cleanup on finalize."""
        mlflow = pytest.importorskip("mlflow")

        with patch.object(mlflow, "end_run") as mock_end:
            with patch.object(mlflow, "get_experiment_by_name") as mock_get_exp:
                with patch.object(mlflow, "start_run") as mock_start:
                    mock_get_exp.return_value = MagicMock(experiment_id="123")
                    mock_start.return_value = MagicMock(info=MagicMock(run_id="abc"))

                    from mlx_audio.train.loggers.mlflow import MLflowLogger

                    logger = MLflowLogger(experiment_name="test")
                    logger.finalize()

                    mock_end.assert_called_once()

    def test_run_id_property(self):
        """Test run_id property."""
        mlflow = pytest.importorskip("mlflow")

        with patch.object(mlflow, "get_experiment_by_name") as mock_get_exp:
            with patch.object(mlflow, "start_run") as mock_start:
                with patch.object(mlflow, "end_run"):
                    mock_get_exp.return_value = MagicMock(experiment_id="123")
                    mock_start.return_value = MagicMock(info=MagicMock(run_id="test-run-id"))

                    from mlx_audio.train.loggers.mlflow import MLflowLogger

                    logger = MLflowLogger(experiment_name="test")
                    assert logger.run_id == "test-run-id"
                    logger.finalize()


class TestLoggerInterface:
    """Tests for Logger interface compliance."""

    def test_tensorboard_implements_interface(self):
        """Test TensorBoardLogger implements Logger interface."""
        pytest.importorskip("tensorboardX")
        from mlx_audio.train.loggers.tensorboard import TensorBoardLogger

        assert issubclass(TensorBoardLogger, Logger)

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TensorBoardLogger(log_dir=tmpdir)

            # Verify abstract methods exist
            assert callable(getattr(logger, "log_metrics", None))
            assert callable(getattr(logger, "log_hyperparams", None))
            assert callable(getattr(logger, "finalize", None))

            logger.finalize()

    def test_mlflow_implements_interface(self):
        """Test MLflowLogger implements Logger interface."""
        mlflow = pytest.importorskip("mlflow")

        with patch.object(mlflow, "get_experiment_by_name") as mock_get_exp:
            with patch.object(mlflow, "start_run") as mock_start:
                with patch.object(mlflow, "end_run"):
                    mock_get_exp.return_value = MagicMock(experiment_id="123")
                    mock_start.return_value = MagicMock(info=MagicMock(run_id="abc"))

                    from mlx_audio.train.loggers.mlflow import MLflowLogger

                    assert issubclass(MLflowLogger, Logger)

                    logger = MLflowLogger(experiment_name="test")

                    assert callable(getattr(logger, "log_metrics", None))
                    assert callable(getattr(logger, "log_hyperparams", None))
                    assert callable(getattr(logger, "finalize", None))

                    logger.finalize()


@pytest.mark.integration
class TestTensorBoardLoggerIntegration:
    """Integration tests requiring tensorboardX."""

    def test_creates_log_files(self):
        """Test that TensorBoard actually creates event files."""
        pytest.importorskip("tensorboardX")
        from mlx_audio.train.loggers.tensorboard import TensorBoardLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TensorBoardLogger(log_dir=tmpdir, name="test_run")
            logger.log_metrics({"loss": 0.5}, step=1)
            logger.log_metrics({"loss": 0.3}, step=2)
            logger.finalize()

            # Check event file was created
            log_dir = Path(tmpdir) / "test_run"
            event_files = list(log_dir.glob("events.out.tfevents.*"))
            assert len(event_files) >= 1, f"Expected event files in {log_dir}"


@pytest.mark.integration
class TestMLflowLoggerIntegration:
    """Integration tests requiring mlflow."""

    def test_creates_run(self):
        """Test that MLflow actually creates a run."""
        mlflow = pytest.importorskip("mlflow")

        with tempfile.TemporaryDirectory() as tmpdir:
            mlflow.set_tracking_uri(f"file://{tmpdir}")

            from mlx_audio.train.loggers.mlflow import MLflowLogger

            logger = MLflowLogger(
                experiment_name="test_experiment",
                run_name="test_run",
            )
            logger.log_metrics({"loss": 0.5}, step=1)
            run_id = logger.run_id
            logger.finalize()

            # Verify run exists
            run = mlflow.get_run(run_id)
            assert run is not None
            assert run.data.metrics.get("loss") == 0.5
