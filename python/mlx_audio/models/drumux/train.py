#!/usr/bin/env python3
"""Train Drumux drum transcription model with MLX.

Usage:
    python -m mlx_audio.models.drumux.train --data /path/to/e-gmd --epochs 30
"""

import argparse
from pathlib import Path

import mlx.core as mx

from mlx_audio.train import Trainer
from mlx_audio.train.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    RichProgressBar,
)

from .model import DrumTranscriberConfig
from .train_module import DrumuxTrainModule
from .data import EGMDDataset, SpectrogramConfig, create_dataloader, compute_class_weights


def main():
    parser = argparse.ArgumentParser(description="Train Drumux drum transcription model")
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to E-GMD dataset directory",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="standard",
        choices=["standard", "lightweight", "fast"],
        help="Model preset",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/drumux-mlx"),
        help="Checkpoint directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--accumulate-grad-batches",
        type=int,
        default=1,
        help="Accumulate gradients over N batches (effective batch = batch-size * N)",
    )

    args = parser.parse_args()

    # Set seed
    mx.random.seed(args.seed)

    print("=" * 60)
    print("Drumux Training (MLX)")
    print("=" * 60)

    # Load datasets
    print(f"\nLoading data from {args.data}...")
    spec_config = SpectrogramConfig()

    train_dataset = EGMDDataset(
        args.data,
        split="train",
        config=spec_config,
        seq_length=512,
        stride=256,
        cache_spectrograms=False,  # Disable to save memory
        cache_midi=True,  # MIDI is small, keep cached
    )
    val_dataset = EGMDDataset(
        args.data,
        split="validation",
        config=spec_config,
        seq_length=512,
        stride=256,
        cache_spectrograms=False,
        cache_midi=True,
    )

    print(f"Train samples: {len(train_dataset):,}")
    print(f"Val samples: {len(val_dataset):,}")

    # Compute class weights
    print("\nComputing class weights...")
    pos_weight = compute_class_weights(train_dataset)
    print(f"Class weights: min={float(pos_weight.min()):.1f}, max={float(pos_weight.max()):.1f}")

    # Create dataloaders
    train_loader = create_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = create_dataloader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(1, args.num_workers // 2),
    )

    # Calculate steps
    steps_per_epoch = len(train_dataset) // args.batch_size
    total_steps = steps_per_epoch * args.epochs

    effective_batch = args.batch_size * args.accumulate_grad_batches
    print(f"\nSteps per epoch: {steps_per_epoch:,}")
    print(f"Total steps: {total_steps:,}")
    if args.accumulate_grad_batches > 1:
        print(f"Effective batch size: {effective_batch} (batch={args.batch_size} × accum={args.accumulate_grad_batches})")

    # Create model
    print(f"\nCreating {args.preset} model...")
    module = DrumuxTrainModule(
        preset=args.preset,
        pos_weight=pos_weight,
        learning_rate=args.lr,
        warmup_steps=500,
        total_steps=total_steps,
    )
    print(f"Parameters: {module.num_parameters:,}")

    # Create trainer
    trainer = Trainer(
        max_epochs=args.epochs,
        val_check_interval=1.0,  # Validate every epoch
        gradient_clip_val=1.0,
        accumulate_grad_batches=args.accumulate_grad_batches,
        default_root_dir=str(args.checkpoint_dir),
        enable_checkpointing=True,
        compile=True,
        seed=args.seed,
        callbacks=[
            RichProgressBar(),
            ModelCheckpoint(
                dirpath=str(args.checkpoint_dir),
                filename="best_model",
                monitor="val_f1",
                mode="max",
                save_top_k=1,
            ),
            EarlyStopping(
                monitor="val_f1",
                patience=10,
                mode="max",
            ),
        ],
    )

    # Train
    print("\nStarting training...")
    trainer.fit(module, train_loader, val_loader)

    print("\n✓ Training complete!")
    print(f"Best checkpoint saved to: {args.checkpoint_dir}")


if __name__ == "__main__":
    main()
