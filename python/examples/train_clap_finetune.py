#!/usr/bin/env python3
"""CLAP fine-tuning example.

This example demonstrates fine-tuning a CLAP (Contrastive Language-Audio Pretraining)
model using the mlx-audio training framework. It shows:
- Loading pretrained CLAP model and wrapping in TrainModule
- Freezing encoder layers for efficient fine-tuning
- Contrastive loss (InfoNCE) implementation
- Custom audio-text dataset with collate function
- Full callback suite including gradient clipping
- Resume from checkpoint

Usage:
    # Fine-tune with synthetic data (demo mode)
    python train_clap_finetune.py --epochs 3 --batch-size 8

    # Fine-tune on your audio-text data
    python train_clap_finetune.py \
        --data-dir /path/to/audio_text_pairs \
        --epochs 10 \
        --wandb

    # Resume from checkpoint
    python train_clap_finetune.py --resume checkpoints/clap/last
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
    GradientClipper,
    LearningRateMonitor,
    ModelCheckpoint,
    ProgressBar,
)
from mlx_audio.train.schedulers import WarmupLinearScheduler


class SyntheticAudioTextDataset(Dataset):
    """Synthetic audio-text dataset for demonstration.

    In a real scenario, you would load actual audio files and their captions.
    This synthetic version generates random mel spectrograms and fake token IDs.
    """

    def __init__(
        self,
        size: int = 500,
        n_mels: int = 64,
        max_length: int = 512,
        vocab_size: int = 50265,
        train: bool = True,
    ) -> None:
        self.size = size
        self.n_mels = n_mels
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.train = train

        # Use different seeds for train/val
        np.random.seed(42 if train else 123)

        # Pre-generate synthetic data
        # In practice, you'd load audio files and tokenize captions
        self._mels = [
            np.random.randn(1, n_mels, 256).astype(np.float32) * 0.5
            for _ in range(size)
        ]
        self._input_ids = [
            np.random.randint(0, vocab_size, size=64).astype(np.int32)
            for _ in range(size)
        ]
        self._attention_masks = [
            np.ones(64, dtype=np.int32)
            for _ in range(size)
        ]

        # Create learnable patterns (matching indices have similar patterns)
        for i in range(size):
            # Add pattern based on index to make task learnable
            pattern_idx = i % 10
            self._mels[i][0, pattern_idx * 5 : pattern_idx * 5 + 10, :50] += 1.0
            # Add special token at matching position in text
            self._input_ids[i][pattern_idx * 5] = pattern_idx + 1000

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> dict:
        return {
            "mel": self._mels[idx],
            "input_ids": self._input_ids[idx],
            "attention_mask": self._attention_masks[idx],
        }


class CLAPTrainModule(TrainModule):
    """TrainModule wrapper for fine-tuning CLAP.

    Implements contrastive loss (InfoNCE) for audio-text alignment.
    Supports freezing audio or text encoders for efficient fine-tuning.
    """

    def __init__(
        self,
        freeze_audio: bool = False,
        freeze_text: bool = True,
    ) -> None:
        """Initialize CLAP training module.

        Args:
            freeze_audio: Freeze audio encoder weights
            freeze_text: Freeze text encoder weights (common for fine-tuning)
        """
        super().__init__()

        # Create CLAP model (use default config for demo)
        from mlx_audio.models.clap import CLAP

        self.clap = CLAP()

        # Freeze encoders if requested
        # In MLX, call freeze() on the module to mark all its params non-trainable
        if freeze_audio:
            self.clap.audio_encoder.freeze()
            self.clap.audio_projection.freeze()
        if freeze_text:
            self.clap.text_encoder.freeze()

    def __call__(
        self,
        audio: mx.array | None = None,
        input_ids: mx.array | None = None,
        attention_mask: mx.array | None = None,
    ) -> dict[str, mx.array]:
        """Forward pass."""
        return self.clap(audio, input_ids, attention_mask)

    def compute_loss(self, batch: tuple) -> tuple[mx.array, dict[str, mx.array]]:
        """Compute contrastive loss (InfoNCE).

        The loss encourages audio and text embeddings to be close for matching
        pairs and far for non-matching pairs within the batch.

        Args:
            batch: Tuple of (mel, input_ids, attention_mask)

        Returns:
            Tuple of (loss, metrics_dict)
        """
        mel, input_ids, attention_mask = batch

        # Forward pass
        outputs = self(audio=mel, input_ids=input_ids, attention_mask=attention_mask)

        # Get logits (scaled similarity matrices)
        logits_audio = outputs["logits_per_audio"]  # [B, B]
        logits_text = outputs["logits_per_text"]    # [B, B]

        batch_size = mel.shape[0]

        # Labels: diagonal is the correct match
        labels = mx.arange(batch_size)

        # Symmetric cross-entropy loss
        loss_audio = mx.mean(nn.losses.cross_entropy(logits_audio, labels))
        loss_text = mx.mean(nn.losses.cross_entropy(logits_text, labels))
        loss = (loss_audio + loss_text) / 2

        # Compute accuracy: how often is the diagonal the max?
        audio_preds = mx.argmax(logits_audio, axis=-1)
        text_preds = mx.argmax(logits_text, axis=-1)
        audio_accuracy = mx.mean(audio_preds == labels)
        text_accuracy = mx.mean(text_preds == labels)

        metrics = {
            "audio_acc": audio_accuracy,
            "text_acc": text_accuracy,
            "logit_scale": mx.exp(self.clap.logit_scale[0]),
        }

        return loss, metrics

    def configure_optimizers(self) -> OptimizerConfig:
        """Configure AdamW with warmup + linear decay for fine-tuning."""
        total_steps = self.trainer.max_steps or 1000
        warmup_steps = min(200, total_steps // 5)

        # Lower learning rate for fine-tuning
        schedule = WarmupLinearScheduler(
            peak_lr=5e-5,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
        )

        return OptimizerConfig(
            optimizer=optim.AdamW(learning_rate=schedule, weight_decay=0.01),
            lr_schedule_name="warmup_linear",
        )


def collate_fn(batch: list) -> tuple:
    """Collate audio-text pairs into batches.

    Args:
        batch: List of dicts with 'mel', 'input_ids', 'attention_mask'

    Returns:
        Tuple of (mel, input_ids, attention_mask) as numpy arrays
    """
    mels = np.stack([item["mel"] for item in batch])
    input_ids = np.stack([item["input_ids"] for item in batch])
    attention_masks = np.stack([item["attention_mask"] for item in batch])
    return mels, input_ids, attention_masks


def to_mlx_batch(batch: tuple) -> tuple:
    """Convert numpy batch to MLX arrays."""
    mels, input_ids, attention_masks = batch
    return (
        mx.array(mels),
        mx.array(input_ids),
        mx.array(attention_masks),
    )


def create_logger(args: argparse.Namespace):
    """Create logger based on command line arguments."""
    if args.wandb:
        try:
            from mlx_audio.train.loggers import WandbLogger

            return WandbLogger(
                project="mlx-audio-examples",
                name="clap-finetune",
                config={
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "freeze_audio": args.freeze_audio,
                    "freeze_text": args.freeze_text,
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
                name="clap-finetune",
            )
        except ImportError:
            print("Warning: tensorboardX not installed. Skipping TensorBoard logging.")
            return None

    if args.mlflow:
        try:
            from mlx_audio.train.loggers import MLflowLogger

            return MLflowLogger(
                experiment_name="mlx-audio-examples",
                run_name="clap-finetune",
            )
        except ImportError:
            print("Warning: mlflow not installed. Skipping MLflow logging.")
            return None

    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune CLAP model with mlx-audio")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--train-size", type=int, default=500, help="Training set size (synthetic)")
    parser.add_argument("--val-size", type=int, default=100, help="Validation set size (synthetic)")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to real audio-text data")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/clap")
    parser.add_argument("--log-dir", type=str, default="./runs")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume")
    parser.add_argument("--freeze-audio", action="store_true", help="Freeze audio encoder")
    parser.add_argument("--freeze-text", action="store_true", default=True, help="Freeze text encoder")
    parser.add_argument("--no-freeze-text", dest="freeze_text", action="store_false")
    parser.add_argument("--gradient-clip", type=float, default=1.0, help="Gradient clipping value")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--tensorboard", action="store_true", help="Enable TensorBoard logging")
    parser.add_argument("--mlflow", action="store_true", help="Enable MLflow logging")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Create datasets
    if args.data_dir:
        print(f"Loading data from: {args.data_dir}")
        # In a real implementation, you would load actual audio-text pairs here
        raise NotImplementedError(
            "Real data loading not yet implemented. "
            "Use synthetic mode (omit --data-dir) for demonstration."
        )
    else:
        print(f"Creating synthetic dataset (train={args.train_size}, val={args.val_size})...")
        train_dataset = SyntheticAudioTextDataset(size=args.train_size, train=True)
        val_dataset = SyntheticAudioTextDataset(size=args.val_size, train=False)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        mlx_transforms=to_mlx_batch,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        mlx_transforms=to_mlx_batch,
    )

    # Create model
    print(f"Creating CLAP model (freeze_audio={args.freeze_audio}, freeze_text={args.freeze_text})")
    model = CLAPTrainModule(
        freeze_audio=args.freeze_audio,
        freeze_text=args.freeze_text,
    )

    # Count trainable parameters
    total_params = sum(p.size for p in model.parameters().values())
    trainable_params = sum(
        p.size for p in model.trainable_parameters().values()
    )
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Create callbacks
    callbacks = [
        ProgressBar(refresh_rate=10),
        LearningRateMonitor(logging_interval="step"),
        GradientClipper(max_norm=args.gradient_clip),
        ModelCheckpoint(
            dirpath=args.checkpoint_dir,
            monitor="val_loss",
            mode="min",
            save_top_k=2,
        ),
        EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=5,
            min_delta=0.001,
        ),
    ]

    # Create logger
    logger = create_logger(args)

    # Create trainer
    trainer = Trainer(
        max_epochs=args.epochs,
        callbacks=callbacks,
        logger=logger,
        seed=args.seed,
    )

    # Train
    print(f"\nStarting fine-tuning for {args.epochs} epochs...")
    trainer.fit(model, train_loader, val_loader, ckpt_path=args.resume)

    print("\nFine-tuning complete!")
    if Path(args.checkpoint_dir).exists():
        print(f"Checkpoints saved to: {args.checkpoint_dir}")


if __name__ == "__main__":
    main()
