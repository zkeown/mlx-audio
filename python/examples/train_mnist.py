#!/usr/bin/env python3
"""MNIST classifier training example.

This example demonstrates the mlx-audio training framework with a simple CNN classifier.
It shows how to:
- Subclass TrainModule with compute_loss and configure_optimizers
- Use DataLoader with transforms
- Configure callbacks (ProgressBar, ModelCheckpoint, EarlyStopping, LearningRateMonitor)
- Optionally use loggers (WandB, TensorBoard, MLflow)

Usage:
    # Basic training
    python train_mnist.py --epochs 5 --batch-size 64

    # With TensorBoard logging
    python train_mnist.py --epochs 5 --tensorboard

    # With W&B logging
    python train_mnist.py --epochs 5 --wandb

    # Resume from checkpoint
    python train_mnist.py --epochs 10 --resume checkpoints/last
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from mlx_audio.data import DataLoader
from mlx_audio.data.data.dataset import Dataset
from mlx_audio.train import OptimizerConfig, Trainer, TrainModule
from mlx_audio.train.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    ProgressBar,
)
from mlx_audio.train.schedulers import WarmupCosineScheduler


class SyntheticMNIST(Dataset):
    """Synthetic MNIST-like dataset for demonstration.

    In a real scenario, you would use mlx_audio.data.datasets.common.mnist.MNIST
    and download actual MNIST data. This synthetic version allows the example
    to run without any data downloads.
    """

    def __init__(self, size: int = 1000, train: bool = True) -> None:
        self.size = size
        self.train = train
        # Generate random "digit-like" patterns
        np.random.seed(42 if train else 123)
        self._images = np.random.randn(size, 28, 28).astype(np.float32) * 0.3
        self._labels = np.random.randint(0, 10, size=size)

        # Make patterns somewhat digit-dependent for learnability
        for i in range(size):
            digit = self._labels[i]
            # Add digit-specific pattern (makes task learnable)
            self._images[i, digit * 2 : digit * 2 + 5, digit * 2 : digit * 2 + 5] += 1.0

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> tuple[np.ndarray, int]:
        return self._images[idx], int(self._labels[idx])


class MNISTClassifier(TrainModule):
    """Simple CNN for MNIST classification.

    Architecture:
        Conv2d(1, 32) -> ReLU -> MaxPool
        Conv2d(32, 64) -> ReLU -> MaxPool
        Flatten -> Linear(64*7*7, 128) -> ReLU -> Linear(128, 10)
    """

    def __init__(self) -> None:
        super().__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # Pooling layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: Input images of shape [B, 28, 28]

        Returns:
            Logits of shape [B, 10]
        """
        # MLX uses NHWC format: [B, H, W, C]
        # Add channel dimension: [B, 28, 28] -> [B, 28, 28, 1]
        if x.ndim == 3:
            x = x[:, :, :, None]

        # Conv block 1
        x = nn.relu(self.conv1(x))
        x = self.pool(x)  # [B, 14, 14, 32]

        # Conv block 2
        x = nn.relu(self.conv2(x))
        x = self.pool(x)  # [B, 7, 7, 64]

        # Flatten and FC layers
        x = x.reshape(x.shape[0], -1)  # [B, 7*7*64]
        x = nn.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def compute_loss(self, batch: tuple) -> tuple[mx.array, dict[str, mx.array]]:
        """Compute cross-entropy loss and accuracy.

        Args:
            batch: Tuple of (images, labels) from dataloader

        Returns:
            Tuple of (loss, metrics_dict)
        """
        images, labels = batch

        # Forward pass
        logits = self(images)

        # Cross-entropy loss
        loss = mx.mean(nn.losses.cross_entropy(logits, labels))

        # Compute accuracy
        predictions = mx.argmax(logits, axis=-1)
        accuracy = mx.mean(predictions == labels)

        return loss, {"accuracy": accuracy}

    def configure_optimizers(self) -> OptimizerConfig:
        """Configure AdamW optimizer with warmup + cosine decay schedule."""
        # Calculate total steps (estimate if max_steps not set)
        total_steps = self.trainer.max_steps or 1000
        warmup_steps = min(100, total_steps // 10)

        schedule = WarmupCosineScheduler(
            peak_lr=1e-3,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr=1e-5,
        )

        return OptimizerConfig(
            optimizer=optim.AdamW(learning_rate=schedule, weight_decay=0.01),
            lr_schedule_name="warmup_cosine",
        )


def to_mlx_batch(batch: list) -> tuple[mx.array, mx.array]:
    """Convert a batch of numpy arrays to MLX arrays.

    This is the mlx_transforms function that runs in the main thread
    after collation.
    """
    images = np.stack([item[0] for item in batch])
    labels = np.array([item[1] for item in batch])
    return mx.array(images), mx.array(labels)


def create_logger(args: argparse.Namespace):
    """Create logger based on command line arguments."""
    if args.wandb:
        try:
            from mlx_audio.train.loggers import WandbLogger

            return WandbLogger(
                project="mlx-audio-examples",
                name="mnist-classifier",
                config={
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "learning_rate": 1e-3,
                },
            )
        except ImportError:
            print("Warning: wandb not installed. Skipping W&B logging.")
            return None

    if args.tensorboard:
        try:
            from mlx_audio.train.loggers import TensorBoardLogger

            return TensorBoardLogger(
                log_dir=args.log_dir,
                name="mnist-classifier",
            )
        except ImportError:
            print("Warning: tensorboardX not installed. Skipping TensorBoard logging.")
            return None

    if args.mlflow:
        try:
            from mlx_audio.train.loggers import MLflowLogger

            return MLflowLogger(
                experiment_name="mlx-audio-examples",
                run_name="mnist-classifier",
            )
        except ImportError:
            print("Warning: mlflow not installed. Skipping MLflow logging.")
            return None

    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MNIST classifier with mlx-audio")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--train-size", type=int, default=5000, help="Training set size")
    parser.add_argument("--val-size", type=int, default=1000, help="Validation set size")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/mnist")
    parser.add_argument("--log-dir", type=str, default="./runs")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--tensorboard", action="store_true", help="Enable TensorBoard logging")
    parser.add_argument("--mlflow", action="store_true", help="Enable MLflow logging")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Create datasets
    print(f"Creating synthetic MNIST dataset (train={args.train_size}, val={args.val_size})...")
    train_dataset = SyntheticMNIST(size=args.train_size, train=True)
    val_dataset = SyntheticMNIST(size=args.val_size, train=False)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda x: x,  # Return list, handle in mlx_transforms
        mlx_transforms=to_mlx_batch,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda x: x,
        mlx_transforms=to_mlx_batch,
    )

    # Create model
    model = MNISTClassifier()
    print(f"Model parameters: {sum(p.size for p in model.parameters().values()):,}")

    # Create callbacks
    callbacks = [
        ProgressBar(refresh_rate=10),
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(
            dirpath=args.checkpoint_dir,
            monitor="val_accuracy",
            mode="max",
            save_top_k=2,
        ),
        EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=5,
            min_delta=0.001,
        ),
    ]

    # Create logger
    logger = create_logger(args)

    # Create trainer
    trainer = Trainer(
        max_epochs=args.epochs,
        gradient_clip_val=1.0,
        callbacks=callbacks,
        logger=logger,
        seed=args.seed,
    )

    # Train
    print(f"\nStarting training for {args.epochs} epochs...")
    trainer.fit(model, train_loader, val_loader, ckpt_path=args.resume)

    print("\nTraining complete!")
    if Path(args.checkpoint_dir).exists():
        print(f"Checkpoints saved to: {args.checkpoint_dir}")


if __name__ == "__main__":
    main()
