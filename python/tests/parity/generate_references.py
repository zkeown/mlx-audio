#!/usr/bin/env python3
"""Unified parity reference generator for all mlx-audio models.

This script generates test fixtures for all models in a single command,
making it easy to regenerate references for Swift parity tests.

Usage:
    # Generate fixtures for all models
    python tests/parity/generate_references.py --model all

    # Generate fixtures for specific models
    python tests/parity/generate_references.py --model htdemucs
    python tests/parity/generate_references.py --model encodec clap

    # Specify output directory
    python tests/parity/generate_references.py --model all --output-dir swift/Tests/Fixtures

Models:
    - htdemucs: Source separation model
    - encodec: Neural audio codec
    - whisper: Speech recognition
    - clap: Audio-text embeddings
    - musicgen: Text-to-music generation
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Valid models
VALID_MODELS = {"htdemucs", "encodec", "whisper", "clap", "musicgen"}

# Script locations relative to tests/ directory
GENERATOR_SCRIPTS = {
    "htdemucs": "generate_swift_fixtures.py",
    "encodec": "generate_encodec_fixtures.py",
    "whisper": "models/whisper/generate_swift_parity_data.py",
    "clap": "generate_clap_fixtures.py",
    "musicgen": "generate_musicgen_fixtures.py",
}


def get_tests_dir() -> Path:
    """Get the tests directory."""
    return Path(__file__).parent.parent


def generate_model_fixtures(
    model: str,
    output_base_dir: Path,
    seed: int = 42,
    layers_only: bool = False,
    verbose: bool = False,
) -> bool:
    """Generate fixtures for a single model.

    Args:
        model: Model name
        output_base_dir: Base output directory (model subdir will be created)
        seed: Random seed for reproducibility
        layers_only: Only generate layer fixtures (faster)
        verbose: Print verbose output

    Returns:
        True if successful, False otherwise
    """
    if model not in VALID_MODELS:
        print(f"Error: Unknown model '{model}'")
        print(f"Valid models: {', '.join(sorted(VALID_MODELS))}")
        return False

    script_name = GENERATOR_SCRIPTS.get(model)
    if not script_name:
        print(f"Warning: No generator script found for '{model}'")
        return False

    tests_dir = get_tests_dir()
    script_path = tests_dir / script_name

    if not script_path.exists():
        print(f"Warning: Script not found: {script_path}")
        return False

    # Determine output directory
    output_dir = output_base_dir / model.title().replace("_", "")
    if model == "htdemucs":
        output_dir = output_base_dir / "HTDemucs"
    elif model == "encodec":
        output_dir = output_base_dir / "EnCodec"
    elif model == "whisper":
        output_dir = output_base_dir / "Whisper"
    elif model == "clap":
        output_dir = output_base_dir / "CLAP"
    elif model == "musicgen":
        output_dir = output_base_dir / "MusicGen"

    print(f"\n{'='*60}")
    print(f"Generating {model.upper()} fixtures")
    print(f"Script: {script_path}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")

    # Build command
    cmd = [sys.executable, str(script_path)]

    # Add output directory argument
    if model == "whisper":
        # Whisper uses a different output directory structure
        cmd.extend(["--output-dir", str(output_dir)])
    else:
        cmd.extend(["--output-dir", str(output_dir)])

    # Add seed
    if "--seed" in get_script_args(script_path):
        cmd.extend(["--seed", str(seed)])

    # Add layers-only flag
    if layers_only and "--layers-only" in get_script_args(script_path):
        cmd.append("--layers-only")

    # Run generator
    try:
        result = subprocess.run(
            cmd,
            capture_output=not verbose,
            text=True,
            cwd=str(tests_dir.parent),  # Run from python/ directory
        )

        if result.returncode != 0:
            print(f"Error generating {model} fixtures:")
            if result.stderr:
                print(result.stderr)
            return False

        if verbose and result.stdout:
            print(result.stdout)

        print(f"✓ {model.upper()} fixtures generated successfully")
        return True

    except Exception as e:
        print(f"Error running generator for {model}: {e}")
        return False


def get_script_args(script_path: Path) -> set[str]:
    """Get available command-line arguments from a script."""
    args = set()
    try:
        content = script_path.read_text()
        if "--seed" in content:
            args.add("--seed")
        if "--layers-only" in content:
            args.add("--layers-only")
        if "--output-dir" in content:
            args.add("--output-dir")
    except Exception:
        pass
    return args


def main():
    parser = argparse.ArgumentParser(
        description="Generate parity test fixtures for mlx-audio models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model",
        nargs="+",
        default=["all"],
        help="Model(s) to generate fixtures for. Use 'all' for all models.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("swift/Tests/Fixtures"),
        help="Base output directory for fixtures",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--layers-only",
        action="store_true",
        help="Only generate layer fixtures (faster)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print verbose output from generators",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models and exit",
    )
    args = parser.parse_args()

    if args.list:
        print("Available models:")
        for model in sorted(VALID_MODELS):
            script = GENERATOR_SCRIPTS.get(model, "N/A")
            print(f"  - {model}: {script}")
        return 0

    # Determine which models to generate
    models = set()
    for m in args.model:
        if m.lower() == "all":
            models = VALID_MODELS.copy()
            break
        elif m.lower() in VALID_MODELS:
            models.add(m.lower())
        else:
            print(f"Warning: Unknown model '{m}'")

    if not models:
        print("No valid models specified.")
        print(f"Available: {', '.join(sorted(VALID_MODELS))}")
        return 1

    print(f"Generating fixtures for: {', '.join(sorted(models))}")
    print(f"Output directory: {args.output_dir}")
    print(f"Seed: {args.seed}")
    if args.layers_only:
        print("Mode: layers only")

    # Generate fixtures
    results = {}
    for model in sorted(models):
        success = generate_model_fixtures(
            model=model,
            output_base_dir=args.output_dir,
            seed=args.seed,
            layers_only=args.layers_only,
            verbose=args.verbose,
        )
        results[model] = success

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    succeeded = sum(1 for v in results.values() if v)
    failed = sum(1 for v in results.values() if not v)

    for model, success in sorted(results.items()):
        status = "✓" if success else "✗"
        print(f"  {status} {model}")

    print()
    print(f"Succeeded: {succeeded}/{len(results)}")
    if failed > 0:
        print(f"Failed: {failed}/{len(results)}")
        return 1

    print("\nAll fixtures generated successfully!")
    print(f"Output: {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
