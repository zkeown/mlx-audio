#!/usr/bin/env python3
"""Generate Banquet parity fixtures for Swift tests.

This script generates intermediate outputs from the Python Banquet implementation
to verify the Swift implementation produces identical results.

Usage:
    python tests/generate_banquet_fixtures.py --output-dir swift/Tests/Fixtures/Banquet

Run from the python/ directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

# Set seed for reproducibility
mx.random.seed(42)
np.random.seed(42)


def save_arrays(path: Path, arrays: dict[str, mx.array]) -> None:
    """Save arrays to safetensors file."""
    mx.save_safetensors(str(path), arrays)
    print(f"  Saved: {path.name}")


def save_config(path: Path, config: dict) -> None:
    """Save config as JSON file."""
    with open(path, "w") as f:
        json.dump(config, f, indent=2)


def flatten_params(params: dict, prefix: str = "") -> dict[str, mx.array]:
    """Flatten nested parameter dict with dot-separated keys."""
    flat = {}
    for k, v in params.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            flat.update(flatten_params(v, key))
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    flat.update(flatten_params(item, f"{key}.{i}"))
                elif isinstance(item, mx.array):
                    flat[f"{key}.{i}"] = item
        elif isinstance(v, mx.array):
            flat[key] = v
    return flat


def generate_film_fixtures(output_dir: Path) -> None:
    """Generate FiLM conditioning fixtures."""
    print("Generating FiLM fixtures...")

    from mlx_audio.models.banquet import FiLM

    # Small config for testing
    cond_dim = 768
    channels = 128

    film = FiLM(
        cond_embedding_dim=cond_dim,
        channels=channels,
        additive=True,
        multiplicative=True,
        depth=2,
        channels_per_group=16,
    )

    # Generate deterministic weights
    def init_weights(shape):
        # Use small random values
        return mx.random.normal(shape) * 0.01

    # Initialize with reproducible weights
    film.gn.weight = init_weights([channels])
    film.gn.bias = mx.zeros([channels])

    # Gamma network (2 layers)
    film.gamma.layers[0].weight = init_weights([channels, cond_dim])
    film.gamma.layers[0].bias = mx.zeros([channels])
    film.gamma.layers[2].weight = init_weights([channels, channels])
    film.gamma.layers[2].bias = mx.zeros([channels])

    # Beta network (2 layers)
    film.beta.layers[0].weight = init_weights([channels, cond_dim])
    film.beta.layers[0].bias = mx.zeros([channels])
    film.beta.layers[2].weight = init_weights([channels, channels])
    film.beta.layers[2].bias = mx.zeros([channels])

    mx.eval(film.parameters())

    # Generate test inputs
    batch = 2
    n_bands = 64
    n_time = 50

    # Input: [batch, channels, n_bands, n_time]
    x = mx.random.normal([batch, channels, n_bands, n_time]) * 0.1
    w = mx.random.normal([batch, cond_dim]) * 0.1
    mx.eval(x, w)

    # Forward pass
    output = film(x, w)
    mx.eval(output)

    # Save fixtures
    save_arrays(output_dir / "film.safetensors", {
        "input": x,
        "conditioning": w,
        "output": output,
    })

    # Save weights (flattened for loading)
    weights = flatten_params(film.parameters())
    save_arrays(output_dir / "film_weights.safetensors", weights)

    # Save config
    save_config(output_dir / "film_config.json", {
        "cond_embedding_dim": cond_dim,
        "channels": channels,
        "additive": True,
        "multiplicative": True,
        "depth": 2,
        "channels_per_group": 16,
    })

    print(f"  Saved FiLM fixtures: input {x.shape}, output {output.shape}")


def generate_bandsplit_fixtures(output_dir: Path) -> None:
    """Generate BandSplit fixtures."""
    print("Generating BandSplit fixtures...")

    from mlx_audio.models.banquet import BandSplitModule, BanquetConfig
    from mlx_audio.models.banquet.utils import MusicalBandsplitSpecification

    config = BanquetConfig(n_bands=16)  # Smaller for testing
    spec = MusicalBandsplitSpecification(
        nfft=config.n_fft,
        fs=config.sample_rate,
        n_bands=config.n_bands,
    )

    band_split = BandSplitModule(
        band_specs=spec.get_band_specs(),
        in_channel=config.in_channel,
        emb_dim=config.emb_dim,
    )

    # Initialize with reproducible weights
    for module in band_split.norm_fc_modules:
        module.norm.weight = mx.ones(module.norm.weight.shape)
        module.norm.bias = mx.zeros(module.norm.bias.shape)
        module.fc.weight = mx.random.normal(module.fc.weight.shape) * 0.01
        module.fc.bias = mx.zeros(module.fc.bias.shape)

    mx.eval(band_split.parameters())

    # Generate complex input
    batch = 2
    freq = config.freq_bins
    time = 50

    real = mx.random.normal([batch, config.in_channel, freq, time]) * 0.1
    imag = mx.random.normal([batch, config.in_channel, freq, time]) * 0.1
    x = real + 1j * imag
    mx.eval(x)

    # Forward pass
    output = band_split(x)
    mx.eval(output)

    # Save as real/imag for safetensors compatibility
    save_arrays(output_dir / "bandsplit.safetensors", {
        "input_real": mx.real(x),
        "input_imag": mx.imag(x),
        "output": output,
    })

    # Save weights
    weights = {}
    for i, module in enumerate(band_split.norm_fc_modules):
        weights[f"norm_fc_modules.{i}.norm.weight"] = module.norm.weight
        weights[f"norm_fc_modules.{i}.norm.bias"] = module.norm.bias
        weights[f"norm_fc_modules.{i}.fc.weight"] = module.fc.weight
        weights[f"norm_fc_modules.{i}.fc.bias"] = module.fc.bias
    save_arrays(output_dir / "bandsplit_weights.safetensors", weights)

    # Save config
    save_config(output_dir / "bandsplit_config.json", {
        "n_fft": config.n_fft,
        "sample_rate": config.sample_rate,
        "n_bands": config.n_bands,
        "in_channel": config.in_channel,
        "emb_dim": config.emb_dim,
    })

    print(f"  Saved BandSplit fixtures: input {x.shape}, output {output.shape}")


def generate_seqband_fixtures(output_dir: Path) -> None:
    """Generate SeqBandModelling fixtures."""
    print("Generating SeqBandModelling fixtures...")

    from mlx_audio.models.banquet import SeqBandModellingModule

    # Small config for testing
    n_modules = 2  # Just 2 modules for faster testing
    emb_dim = 64
    rnn_dim = 128

    seq_band = SeqBandModellingModule(
        n_modules=n_modules,
        emb_dim=emb_dim,
        rnn_dim=rnn_dim,
        bidirectional=True,
        rnn_type="LSTM",
    )

    # Initialize with reproducible weights
    for module in seq_band.seqband:
        module.norm.weight = mx.ones(module.norm.weight.shape)
        module.norm.bias = mx.zeros(module.norm.bias.shape)

        # LSTM weights (simplified initialization)
        for param in module.rnn.parameters().values():
            if isinstance(param, mx.array):
                param_shape = param.shape
                mx.random.seed(42)  # Reset for reproducibility
                new_val = mx.random.normal(param_shape) * 0.01
                # Note: Can't directly assign, need to use update

        module.fc.weight = mx.random.normal(module.fc.weight.shape) * 0.01
        module.fc.bias = mx.zeros(module.fc.bias.shape)

    mx.eval(seq_band.parameters())

    # Generate input
    batch = 2
    n_bands = 16
    n_time = 20

    x = mx.random.normal([batch, n_bands, n_time, emb_dim]) * 0.1
    mx.eval(x)

    # Forward pass
    output = seq_band(x)
    mx.eval(output)

    # Save fixtures
    save_arrays(output_dir / "seqband.safetensors", {
        "input": x,
        "output": output,
    })

    # Save config
    save_config(output_dir / "seqband_config.json", {
        "n_modules": n_modules,
        "emb_dim": emb_dim,
        "rnn_dim": rnn_dim,
        "bidirectional": True,
        "rnn_type": "LSTM",
    })

    print(f"  Saved SeqBandModelling fixtures: input {x.shape}, output {output.shape}")


def generate_passt_fixtures(output_dir: Path) -> None:
    """Generate PaSST encoder fixtures."""
    print("Generating PaSST fixtures...")

    from mlx_audio.models.banquet import PaSST, PaSSTConfig

    # Smaller config for testing
    config = PaSSTConfig(
        embed_dim=256,
        num_heads=4,
        num_layers=2,
        mlp_ratio=2.0,
    )

    passt = PaSST(config)

    # Initialize with reproducible weights
    mx.random.seed(42)

    # Patch embed
    passt.patch_embed.proj.weight = mx.random.normal(passt.patch_embed.proj.weight.shape) * 0.02

    # Position embeddings
    passt.cls_token = mx.random.normal(passt.cls_token.shape) * 0.02
    passt.dist_token = mx.random.normal(passt.dist_token.shape) * 0.02
    passt.time_pos_embed = mx.random.normal(passt.time_pos_embed.shape) * 0.02
    passt.freq_pos_embed = mx.random.normal(passt.freq_pos_embed.shape) * 0.02
    passt.pos_embed = mx.random.normal(passt.pos_embed.shape) * 0.02

    # Transformer blocks
    for block in passt.blocks:
        block.norm1.weight = mx.ones(block.norm1.weight.shape)
        block.norm1.bias = mx.zeros(block.norm1.bias.shape)
        block.attn.qkv.weight = mx.random.normal(block.attn.qkv.weight.shape) * 0.02
        block.attn.qkv.bias = mx.zeros(block.attn.qkv.bias.shape)
        block.attn.proj.weight = mx.random.normal(block.attn.proj.weight.shape) * 0.02
        block.attn.proj.bias = mx.zeros(block.attn.proj.bias.shape)
        block.norm2.weight = mx.ones(block.norm2.weight.shape)
        block.norm2.bias = mx.zeros(block.norm2.bias.shape)
        block.mlp.fc1.weight = mx.random.normal(block.mlp.fc1.weight.shape) * 0.02
        block.mlp.fc1.bias = mx.zeros(block.mlp.fc1.bias.shape)
        block.mlp.fc2.weight = mx.random.normal(block.mlp.fc2.weight.shape) * 0.02
        block.mlp.fc2.bias = mx.zeros(block.mlp.fc2.bias.shape)

    passt.norm.weight = mx.ones(passt.norm.weight.shape)
    passt.norm.bias = mx.zeros(passt.norm.bias.shape)

    mx.eval(passt.parameters())

    # Generate input: mel spectrogram [batch, 1, 128, 998]
    batch = 2
    x = mx.random.normal([batch, 1, 128, 998]) * 0.1
    mx.eval(x)

    # Forward pass
    output = passt(x)
    mx.eval(output)

    # Save fixtures
    save_arrays(output_dir / "passt.safetensors", {
        "input": x,
        "output": output,
    })

    # Save config
    save_config(output_dir / "passt_config.json", {
        "embed_dim": config.embed_dim,
        "num_heads": config.num_heads,
        "num_layers": config.num_layers,
        "mlp_ratio": config.mlp_ratio,
        "patch_size": list(config.patch_size),
    })

    print(f"  Saved PaSST fixtures: input {x.shape}, output {output.shape}")


def generate_maskestim_fixtures(output_dir: Path) -> None:
    """Generate MaskEstimation fixtures."""
    print("Generating MaskEstimation fixtures...")

    from mlx_audio.models.banquet import OverlappingMaskEstimationModule, BanquetConfig
    from mlx_audio.models.banquet.utils import MusicalBandsplitSpecification

    config = BanquetConfig(n_bands=16, emb_dim=64, mlp_dim=128)
    spec = MusicalBandsplitSpecification(
        nfft=config.n_fft,
        fs=config.sample_rate,
        n_bands=config.n_bands,
    )

    mask_estim = OverlappingMaskEstimationModule(
        in_channel=config.in_channel,
        band_specs=spec.get_band_specs(),
        freq_weights=spec.get_freq_weights(),
        n_freq=config.freq_bins,
        emb_dim=config.emb_dim,
        mlp_dim=config.mlp_dim,
        complex_mask=True,
    )

    # Initialize with reproducible weights
    mx.random.seed(42)
    for mlp in mask_estim.norm_mlp:
        mlp.norm.weight = mx.ones(mlp.norm.weight.shape)
        mlp.norm.bias = mx.zeros(mlp.norm.bias.shape)
        mlp.hidden_linear.weight = mx.random.normal(mlp.hidden_linear.weight.shape) * 0.01
        mlp.hidden_linear.bias = mx.zeros(mlp.hidden_linear.bias.shape)
        mlp.output_linear.weight = mx.random.normal(mlp.output_linear.weight.shape) * 0.01
        mlp.output_linear.bias = mx.zeros(mlp.output_linear.bias.shape)

    mx.eval(mask_estim.parameters())

    # Generate input
    batch = 2
    n_time = 20

    x = mx.random.normal([batch, config.n_bands, n_time, config.emb_dim]) * 0.1
    mx.eval(x)

    # Forward pass
    output = mask_estim(x)
    mx.eval(output)

    # Save fixtures
    save_arrays(output_dir / "maskestim.safetensors", {
        "input": x,
        "output": output,
    })

    # Save config
    save_config(output_dir / "maskestim_config.json", {
        "in_channel": config.in_channel,
        "n_freq": config.freq_bins,
        "n_bands": config.n_bands,
        "emb_dim": config.emb_dim,
        "mlp_dim": config.mlp_dim,
        "complex_mask": True,
    })

    print(f"  Saved MaskEstimation fixtures: input {x.shape}, output {output.shape}")


def generate_small_model_fixtures(output_dir: Path) -> None:
    """Generate small Banquet model fixtures for integration test."""
    print("Generating small Banquet model fixtures...")

    from mlx_audio.models.banquet import Banquet, BanquetConfig, PaSSTConfig

    # Very small config for fast testing
    # Note: cond_emb_dim must match passt embed_dim
    passt_embed_dim = 128
    config = BanquetConfig(
        n_bands=8,
        emb_dim=32,
        rnn_dim=64,
        n_sqm_modules=1,
        mlp_dim=64,
        cond_emb_dim=passt_embed_dim,  # Must match PaSST output
    )
    passt_config = PaSSTConfig(
        embed_dim=passt_embed_dim,
        num_heads=2,
        num_layers=1,
        mlp_ratio=2.0,
    )

    model = Banquet(config=config, passt_config=passt_config)
    mx.eval(model.parameters())

    # Generate inputs
    batch = 1
    samples = 22050  # 0.5 seconds at 44.1kHz

    mixture = mx.random.normal([batch, 2, samples]) * 0.1
    query_embedding = mx.random.normal([batch, passt_config.embed_dim]) * 0.1
    mx.eval(mixture, query_embedding)

    # Forward pass
    output = model(mixture, query_embedding)
    mx.eval(output.audio, output.mask)

    # Save fixtures
    save_arrays(output_dir / "small_model.safetensors", {
        "mixture": mixture,
        "query_embedding": query_embedding,
        "output_audio": output.audio,
        "output_mask": output.mask,
    })

    # Save configs
    save_config(output_dir / "small_model_config.json", {
        "banquet": {
            "n_bands": config.n_bands,
            "emb_dim": config.emb_dim,
            "rnn_dim": config.rnn_dim,
            "n_sqm_modules": config.n_sqm_modules,
            "mlp_dim": config.mlp_dim,
            "cond_emb_dim": config.cond_emb_dim,
            "sample_rate": config.sample_rate,
            "n_fft": config.n_fft,
            "hop_length": config.hop_length,
            "in_channel": config.in_channel,
        },
        "passt": {
            "embed_dim": passt_config.embed_dim,
            "num_heads": passt_config.num_heads,
            "num_layers": passt_config.num_layers,
            "mlp_ratio": passt_config.mlp_ratio,
        },
    })

    print(f"  Saved small model fixtures: mixture {mixture.shape}, output {output.audio.shape}")


def main():
    parser = argparse.ArgumentParser(description="Generate Banquet parity fixtures")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("../swift/Tests/Fixtures/Banquet"),
        help="Output directory for fixtures",
    )
    parser.add_argument(
        "--components",
        nargs="+",
        default=["all"],
        choices=["all", "film", "bandsplit", "seqband", "passt", "maskestim", "model"],
        help="Which components to generate fixtures for",
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    components = args.components
    if "all" in components:
        components = ["film", "bandsplit", "seqband", "passt", "maskestim", "model"]

    print(f"Generating Banquet fixtures in {args.output_dir}")
    print("=" * 60)

    if "film" in components:
        generate_film_fixtures(args.output_dir)

    if "bandsplit" in components:
        generate_bandsplit_fixtures(args.output_dir)

    if "seqband" in components:
        generate_seqband_fixtures(args.output_dir)

    if "passt" in components:
        generate_passt_fixtures(args.output_dir)

    if "maskestim" in components:
        generate_maskestim_fixtures(args.output_dir)

    if "model" in components:
        generate_small_model_fixtures(args.output_dir)

    print("=" * 60)
    print("Done! Fixtures saved to:", args.output_dir)


if __name__ == "__main__":
    main()
