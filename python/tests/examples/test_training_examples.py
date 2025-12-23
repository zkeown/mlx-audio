"""Tests for training examples.

These tests verify that the example scripts work correctly with synthetic data.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import mlx.core as mx
import numpy as np
import pytest


@pytest.mark.examples
class TestMNISTExample:
    """Tests for MNIST classifier example."""

    def test_synthetic_mnist_dataset(self):
        """Test synthetic MNIST dataset creation."""
        from examples.train_mnist import SyntheticMNIST

        dataset = SyntheticMNIST(size=100, train=True)

        assert len(dataset) == 100

        image, label = dataset[0]
        assert image.shape == (28, 28)
        assert image.dtype == np.float32
        assert 0 <= label <= 9

    def test_mnist_classifier_forward(self):
        """Test MNISTClassifier forward pass."""
        from examples.train_mnist import MNISTClassifier

        model = MNISTClassifier()

        # Test forward pass
        x = mx.random.normal((4, 28, 28))
        logits = model(x)

        assert logits.shape == (4, 10)

    def test_mnist_classifier_compute_loss(self):
        """Test MNISTClassifier compute_loss."""
        from examples.train_mnist import MNISTClassifier

        model = MNISTClassifier()

        # Create synthetic batch
        images = mx.random.normal((4, 28, 28))
        labels = mx.array([0, 1, 2, 3])

        # Mock trainer for configure_optimizers
        model._trainer = MagicMock()
        model._trainer.max_steps = 100

        loss, metrics = model.compute_loss((images, labels))

        # Verify loss is scalar
        assert loss.shape == ()
        assert "accuracy" in metrics
        assert metrics["accuracy"].shape == ()

    def test_to_mlx_batch(self):
        """Test batch conversion function."""
        from examples.train_mnist import to_mlx_batch

        batch = [
            (np.random.randn(28, 28).astype(np.float32), 0),
            (np.random.randn(28, 28).astype(np.float32), 1),
            (np.random.randn(28, 28).astype(np.float32), 2),
        ]

        images, labels = to_mlx_batch(batch)

        assert isinstance(images, mx.array)
        assert isinstance(labels, mx.array)
        assert images.shape == (3, 28, 28)
        assert labels.shape == (3,)

    @pytest.mark.slow
    def test_mnist_training_smoke(self):
        """Smoke test: verify training loop runs without errors."""
        from examples.train_mnist import MNISTClassifier, SyntheticMNIST, to_mlx_batch

        from mlx_audio.data import DataLoader
        from mlx_audio.train import Trainer

        # Create tiny dataset
        train_dataset = SyntheticMNIST(size=16, train=True)
        train_loader = DataLoader(
            train_dataset,
            batch_size=4,
            shuffle=True,
            collate_fn=lambda x: x,
            mlx_transforms=to_mlx_batch,
        )

        model = MNISTClassifier()

        # Train for just 2 steps (disable compile for simpler test)
        trainer = Trainer(max_steps=2, callbacks=[], compile=False)
        trainer.fit(model, train_loader)

        assert trainer.global_step == 2


@pytest.mark.examples
class TestCLAPExample:
    """Tests for CLAP fine-tuning example."""

    def test_synthetic_audio_text_dataset(self):
        """Test synthetic audio-text dataset creation."""
        from examples.train_clap_finetune import SyntheticAudioTextDataset

        dataset = SyntheticAudioTextDataset(size=50, train=True)

        assert len(dataset) == 50

        sample = dataset[0]
        assert "mel" in sample
        assert "input_ids" in sample
        assert "attention_mask" in sample
        assert sample["mel"].shape == (1, 64, 256)
        assert sample["input_ids"].shape == (64,)

    def test_collate_fn(self):
        """Test collate function for CLAP."""
        from examples.train_clap_finetune import SyntheticAudioTextDataset, collate_fn

        dataset = SyntheticAudioTextDataset(size=10, train=True)
        batch = [dataset[i] for i in range(4)]

        mels, input_ids, attention_masks = collate_fn(batch)

        assert mels.shape == (4, 1, 64, 256)
        assert input_ids.shape == (4, 64)
        assert attention_masks.shape == (4, 64)

    def test_to_mlx_batch(self):
        """Test MLX conversion for CLAP batches."""
        from examples.train_clap_finetune import to_mlx_batch

        batch = (
            np.random.randn(4, 1, 64, 256).astype(np.float32),
            np.random.randint(0, 1000, (4, 64)).astype(np.int32),
            np.ones((4, 64), dtype=np.int32),
        )

        mels, input_ids, attention_masks = to_mlx_batch(batch)

        assert isinstance(mels, mx.array)
        assert isinstance(input_ids, mx.array)
        assert isinstance(attention_masks, mx.array)

    def test_clap_train_module_creation(self):
        """Test CLAPTrainModule can be created."""
        # This test requires CLAP model which may not be fully available
        # in all test environments, so we skip if import fails
        try:
            from examples.train_clap_finetune import CLAPTrainModule

            model = CLAPTrainModule(freeze_audio=False, freeze_text=True)
            assert model is not None
        except ImportError as e:
            pytest.skip(f"CLAP dependencies not available: {e}")

    def test_clap_compute_loss(self):
        """Test CLAPTrainModule compute_loss."""
        try:
            from examples.train_clap_finetune import CLAPTrainModule

            model = CLAPTrainModule(freeze_text=True)

            # Mock trainer
            model._trainer = MagicMock()
            model._trainer.max_steps = 100

            # Create synthetic batch
            batch = (
                mx.random.normal((2, 1, 64, 256)),  # mel
                mx.zeros((2, 64), dtype=mx.int32),  # input_ids
                mx.ones((2, 64), dtype=mx.int32),   # attention_mask
            )

            loss, metrics = model.compute_loss(batch)

            assert loss.shape == ()
            assert "audio_acc" in metrics
            assert "text_acc" in metrics
            assert "logit_scale" in metrics
        except ImportError as e:
            pytest.skip(f"CLAP dependencies not available: {e}")


@pytest.mark.examples
class TestLoggerIntegration:
    """Test logger integration in examples."""

    def test_create_logger_none(self):
        """Test create_logger returns None when no flags set."""
        from examples.train_mnist import create_logger

        args = MagicMock()
        args.wandb = False
        args.tensorboard = False
        args.mlflow = False
        args.log_dir = "./runs"

        logger = create_logger(args)
        assert logger is None

    def test_create_logger_tensorboard(self):
        """Test create_logger creates TensorBoardLogger."""
        pytest.importorskip("tensorboardX")
        import tempfile

        from examples.train_mnist import create_logger

        args = MagicMock()
        args.wandb = False
        args.tensorboard = True
        args.mlflow = False

        with tempfile.TemporaryDirectory() as tmpdir:
            args.log_dir = tmpdir
            logger = create_logger(args)

            assert logger is not None
            from mlx_audio.train.loggers import TensorBoardLogger

            assert isinstance(logger, TensorBoardLogger)
            logger.finalize()


@pytest.mark.slow
@pytest.mark.integration
class TestEndToEndExamples:
    """End-to-end integration tests for examples."""

    def test_mnist_full_training_loop(self):
        """Test complete MNIST training loop with validation."""
        from examples.train_mnist import MNISTClassifier, SyntheticMNIST, to_mlx_batch

        from mlx_audio.data import DataLoader
        from mlx_audio.train import Trainer
        from mlx_audio.train.callbacks import ProgressBar

        # Create datasets
        train_dataset = SyntheticMNIST(size=32, train=True)
        val_dataset = SyntheticMNIST(size=16, train=False)

        train_loader = DataLoader(
            train_dataset,
            batch_size=8,
            shuffle=True,
            collate_fn=lambda x: x,
            mlx_transforms=to_mlx_batch,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=8,
            shuffle=False,
            collate_fn=lambda x: x,
            mlx_transforms=to_mlx_batch,
        )

        model = MNISTClassifier()

        # Train for 1 epoch (disable compile for simpler test)
        trainer = Trainer(
            max_epochs=1,
            callbacks=[ProgressBar(refresh_rate=1)],
            compile=False,
        )
        trainer.fit(model, train_loader, val_loader)

        # Verify training completed
        assert trainer.current_epoch == 1
        assert trainer.global_step > 0

    def test_mnist_with_tensorboard_logger(self):
        """Test MNIST training with TensorBoard logging."""
        pytest.importorskip("tensorboardX")
        import tempfile
        from pathlib import Path

        from examples.train_mnist import MNISTClassifier, SyntheticMNIST, to_mlx_batch

        from mlx_audio.data import DataLoader
        from mlx_audio.train import Trainer
        from mlx_audio.train.loggers import TensorBoardLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dataset and loader
            train_dataset = SyntheticMNIST(size=16, train=True)
            train_loader = DataLoader(
                train_dataset,
                batch_size=4,
                shuffle=True,
                collate_fn=lambda x: x,
                mlx_transforms=to_mlx_batch,
            )

            model = MNISTClassifier()

            # Create logger
            logger = TensorBoardLogger(log_dir=tmpdir, name="test")

            # Train (disable compile for simpler test)
            trainer = Trainer(max_steps=5, logger=logger, compile=False)
            trainer.fit(model, train_loader)

            # Verify event files were created
            log_dir = Path(tmpdir) / "test"
            event_files = list(log_dir.glob("events.out.tfevents.*"))
            assert len(event_files) >= 1
