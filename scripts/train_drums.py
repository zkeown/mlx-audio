#!/usr/bin/env python3
"""Train drum transcription model with MLX.

Usage:
    python scripts/train_drums.py --epochs 30 --batch-size 32

    # Quick test run
    python scripts/train_drums.py --epochs 2 --batch-size 4 --seq-length 128
"""

import argparse
from pathlib import Path

from mlx_audio.models.drums.config import DrumTranscriberConfig
from mlx_audio.models.drums.data import EGMDDataset, collate_fn, to_mlx
from mlx_audio.models.drums.train_module import DrumTranscriberModule
from mlx_audio.train import Trainer
from mlx_audio.train.callbacks import ModelCheckpoint, ProgressBar


def main():
    parser = argparse.ArgumentParser(
        description="Train drum transcription model"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("/Users/zakkeown/ml/drums/data/e-gmd"),
        help="Path to E-GMD dataset",
    )
    parser.add_argument(
        "--epochs", type=int, default=30, help="Number of epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size"
    )
    parser.add_argument(
        "--seq-length", type=int, default=512, help="Sequence length"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument(
        "--model",
        choices=["lightweight", "standard", "fast"],
        default="lightweight",
        help="Model preset",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/drums-mlx"),
        help="Checkpoint directory",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    print("=" * 60)
    print("Drum Transcription Training (MLX)")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Sequence length: {args.seq_length}")
    print(f"Learning rate: {args.lr}")
    print()

    # Create datasets
    print("Loading datasets...")
    train_dataset = EGMDDataset(
        args.dataset, split="train", seq_length=args.seq_length
    )
    val_dataset = EGMDDataset(
        args.dataset, split="validation", seq_length=args.seq_length
    )
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Create dataloaders
    from mlx_audio.data import DataLoader

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        mlx_transforms=to_mlx,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        mlx_transforms=to_mlx,
    )

    # Estimate total steps for LR schedule
    steps_per_epoch = len(train_dataset) // args.batch_size
    total_steps = steps_per_epoch * args.epochs
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Total steps: {total_steps}")

    # Create model config
    presets = {
        "lightweight": DrumTranscriberConfig(
            encoder_type="lightweight",
            embed_dim=256,
            num_layers=3,
            num_heads=4,
        ),
        "standard": DrumTranscriberConfig(
            encoder_type="standard",
            embed_dim=512,
            num_layers=4,
            num_heads=8,
        ),
        "fast": DrumTranscriberConfig(
            encoder_type="lightweight",
            embed_dim=256,
            num_layers=2,
            num_heads=4,
            use_local_attention=True,
            window_size=32,
        ),
    }
    config = presets[args.model]

    # Create training module
    module = DrumTranscriberModule(
        config=config,
        onset_weight=1.0,
        velocity_weight=0.5,
        pos_weight=150.0,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_steps=500,
        total_steps=total_steps,
    )
    print(f"Model parameters: {module.model.num_parameters:,}")

    # Create callbacks
    callbacks = [
        ProgressBar(refresh_rate=10),
        ModelCheckpoint(
            dirpath=str(args.checkpoint_dir),
            monitor="val_loss",
            mode="min",
            save_top_k=2,
            save_last=True,
        ),
    ]

    # Create trainer
    trainer = Trainer(
        max_epochs=args.epochs,
        gradient_clip_val=1.0,
        callbacks=callbacks,
        seed=args.seed,
        compile=True,
    )

    # Train
    print()
    print("Starting training...")
    trainer.fit(module, train_loader, val_loader)

    print()
    print("Training complete!")
    print(f"Checkpoints saved to: {args.checkpoint_dir}")


if __name__ == "__main__":
    main()
