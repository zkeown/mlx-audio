#!/usr/bin/env python3
"""Benchmark comparing PyTorch vs MLX training for drum transcription.

This benchmark measures training throughput (samples/sec) for both frameworks
using the E-GMD dataset.

Usage:
    python benchmarks/drums_training_benchmark.py --dataset /path/to/e-gmd

Requirements:
    - E-GMD dataset with precomputed spectrograms (.spec.pt files)
    - PyTorch drums model (from /Users/zakkeown/ml/drums)
    - MLX drums model (from mlx_audio.models.drums)
"""

import argparse
import sys
import time
from pathlib import Path


def benchmark_pytorch_training(
    dataset_path: Path,
    num_batches: int = 100,
    batch_size: int = 32,
    seq_length: int = 512,
) -> dict:
    """Benchmark PyTorch training throughput.

    Args:
        dataset_path: Path to E-GMD dataset
        num_batches: Number of batches to run
        batch_size: Batch size
        seq_length: Sequence length (frames)

    Returns:
        Dict with timing results
    """
    import torch
    from torch.optim import AdamW

    # Add drums project to path
    drums_path = Path("/Users/zakkeown/ml/drums")
    sys.path.insert(0, str(drums_path / "src"))

    from drums.data.egmd import EGMDDataset
    from drums.model.loss import DrumTranscriptionLoss
    from drums.model.model import create_model

    print("\n=== PyTorch Benchmark ===")

    # Select device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Create model
    model = create_model("lightweight")  # Use lightweight for faster benchmark
    model = model.to(device)
    print(f"Model parameters: {model.num_parameters:,}")

    # Create dataset and dataloader
    dataset = EGMDDataset(dataset_path, split="train", seq_length=seq_length)
    print(f"Dataset samples: {len(dataset)}")

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Single worker for fair comparison
        pin_memory=False,
    )

    # Create optimizer and loss
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    criterion = DrumTranscriptionLoss(pos_weight=150.0)

    # Warmup
    print("Warming up...")
    model.train()
    for i, batch in enumerate(dataloader):
        if i >= 2:
            break
        spec = batch["spectrogram"].to(device)
        onset_target = batch["onset_target"].to(device)
        velocity_target = batch["velocity_target"].to(device)

        optimizer.zero_grad()
        onset_logits, velocity = model(spec)
        losses = criterion(onset_logits, velocity, onset_target, velocity_target)
        losses["loss"].backward()
        optimizer.step()

    # Benchmark
    print(f"Running {num_batches} batches...")
    times = []
    total_samples = 0

    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break

        spec = batch["spectrogram"].to(device)
        onset_target = batch["onset_target"].to(device)
        velocity_target = batch["velocity_target"].to(device)

        start = time.perf_counter()

        optimizer.zero_grad()
        onset_logits, velocity = model(spec)
        losses = criterion(onset_logits, velocity, onset_target, velocity_target)
        losses["loss"].backward()
        optimizer.step()

        # Sync for accurate timing
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()

        elapsed = time.perf_counter() - start
        times.append(elapsed)
        total_samples += spec.shape[0]

    avg_time = sum(times) / len(times)
    samples_per_sec = total_samples / sum(times)

    print(f"Average time per batch: {avg_time*1000:.2f} ms")
    print(f"Throughput: {samples_per_sec:.1f} samples/sec")

    return {
        "framework": "PyTorch",
        "device": str(device),
        "num_batches": len(times),
        "total_samples": total_samples,
        "total_time": sum(times),
        "avg_time_per_batch": avg_time,
        "samples_per_sec": samples_per_sec,
    }


def benchmark_mlx_training(
    dataset_path: Path,
    num_batches: int = 100,
    batch_size: int = 32,
    seq_length: int = 512,
) -> dict:
    """Benchmark MLX training throughput.

    Args:
        dataset_path: Path to E-GMD dataset
        num_batches: Number of batches to run
        batch_size: Batch size
        seq_length: Sequence length (frames)

    Returns:
        Dict with timing results
    """
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

    from mlx_audio.models.drums.config import DrumTranscriberConfig
    from mlx_audio.models.drums.data import EGMDDataset, collate_fn, to_mlx
    from mlx_audio.models.drums.loss import DrumTranscriptionLoss
    from mlx_audio.models.drums.model import DrumTranscriber

    print("\n=== MLX Benchmark ===")
    print("Device: Apple Silicon (unified memory)")

    # Create model (lightweight for faster benchmark)
    config = DrumTranscriberConfig(
        encoder_type="lightweight",
        embed_dim=256,
        num_layers=3,
        num_heads=4,
    )
    model = DrumTranscriber(config)
    mx.eval(model.parameters())
    print(f"Model parameters: {model.num_parameters:,}")

    # Create dataset
    dataset = EGMDDataset(dataset_path, split="train", seq_length=seq_length)
    print(f"Dataset samples: {len(dataset)}")

    # Create simple iterator (no DataLoader for direct comparison)
    def iter_batches():
        indices = list(range(len(dataset)))
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i : i + batch_size]
            if len(batch_indices) < batch_size:
                continue
            samples = [dataset[j] for j in batch_indices]
            batch = collate_fn(samples)
            yield to_mlx(batch)

    # Create optimizer and loss
    optimizer = optim.AdamW(learning_rate=1e-4, weight_decay=0.01)
    criterion = DrumTranscriptionLoss(pos_weight=150.0)

    # Create training step function
    def loss_fn(model, batch):
        onset_logits, velocity = model(batch["spectrogram"])
        losses = criterion(
            onset_logits,
            velocity,
            batch["onset_target"],
            batch["velocity_target"],
        )
        return losses["loss"]

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    def train_step(batch):
        loss, grads = loss_and_grad_fn(model, batch)
        optimizer.update(model, grads)
        return loss

    # Warmup
    print("Warming up...")
    batch_iter = iter_batches()
    for i in range(2):
        try:
            batch = next(batch_iter)
            loss = train_step(batch)
            mx.eval(loss, model.parameters(), optimizer.state)
        except StopIteration:
            batch_iter = iter_batches()

    # Benchmark
    print(f"Running {num_batches} batches...")
    times = []
    total_samples = 0
    batch_iter = iter_batches()

    for i in range(num_batches):
        try:
            batch = next(batch_iter)
        except StopIteration:
            batch_iter = iter_batches()
            batch = next(batch_iter)

        start = time.perf_counter()

        loss = train_step(batch)
        # Force evaluation to get accurate timing
        mx.eval(loss, model.parameters(), optimizer.state)

        elapsed = time.perf_counter() - start
        times.append(elapsed)
        total_samples += batch["spectrogram"].shape[0]

    avg_time = sum(times) / len(times)
    samples_per_sec = total_samples / sum(times)

    print(f"Average time per batch: {avg_time*1000:.2f} ms")
    print(f"Throughput: {samples_per_sec:.1f} samples/sec")

    return {
        "framework": "MLX",
        "device": "Apple Silicon",
        "num_batches": len(times),
        "total_samples": total_samples,
        "total_time": sum(times),
        "avg_time_per_batch": avg_time,
        "samples_per_sec": samples_per_sec,
    }


def run_benchmark(
    dataset_path: Path,
    num_batches: int = 100,
    batch_size: int = 32,
    seq_length: int = 512,
) -> None:
    """Run complete benchmark comparing PyTorch and MLX.

    Args:
        dataset_path: Path to E-GMD dataset
        num_batches: Number of batches per framework
        batch_size: Batch size
        seq_length: Sequence length (frames)
    """
    print("=" * 60)
    print("Drum Transcription Training Benchmark")
    print("=" * 60)
    print(f"Dataset: {dataset_path}")
    print(f"Batch size: {batch_size}")
    print(f"Sequence length: {seq_length} frames")
    print(f"Batches per framework: {num_batches}")

    # Run benchmarks
    pt_results = benchmark_pytorch_training(
        dataset_path, num_batches, batch_size, seq_length
    )
    mlx_results = benchmark_mlx_training(
        dataset_path, num_batches, batch_size, seq_length
    )

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Framework':<15} {'Device':<20} {'Samples/sec':>15}")
    print("-" * 60)
    print(
        f"{'PyTorch':<15} {pt_results['device']:<20} {pt_results['samples_per_sec']:>15.1f}"
    )
    print(
        f"{'MLX':<15} {mlx_results['device']:<20} {mlx_results['samples_per_sec']:>15.1f}"
    )
    print("-" * 60)

    ratio = mlx_results["samples_per_sec"] / pt_results["samples_per_sec"]
    if ratio >= 1:
        print(f"MLX is {ratio:.2f}x faster than PyTorch")
    else:
        print(f"PyTorch is {1/ratio:.2f}x faster than MLX")

    print("\nNote: Performance depends on model size, batch size, and hardware.")
    print("MLX performance typically improves with larger batch sizes.")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark PyTorch vs MLX drum transcription training"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("/Users/zakkeown/ml/drums/data/e-gmd"),
        help="Path to E-GMD dataset with precomputed spectrograms",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=10,  # Start very small
        help="Number of batches to benchmark",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,  # Start very small
        help="Batch size",
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=128,  # Start very small (~1.3 seconds)
        help="Sequence length in frames",
    )
    parser.add_argument(
        "--pytorch-only",
        action="store_true",
        help="Only run PyTorch benchmark",
    )
    parser.add_argument(
        "--mlx-only",
        action="store_true",
        help="Only run MLX benchmark",
    )

    args = parser.parse_args()

    if args.pytorch_only:
        benchmark_pytorch_training(
            args.dataset, args.num_batches, args.batch_size, args.seq_length
        )
    elif args.mlx_only:
        benchmark_mlx_training(
            args.dataset, args.num_batches, args.batch_size, args.seq_length
        )
    else:
        run_benchmark(
            args.dataset, args.num_batches, args.batch_size, args.seq_length
        )


if __name__ == "__main__":
    main()
