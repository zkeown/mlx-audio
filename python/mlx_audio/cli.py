"""Command-line interface for mlx-audio."""

from __future__ import annotations


def main() -> None:
    """Main entry point for mlx-audio CLI."""
    print("mlx-audio CLI")
    print("=============")
    print()
    print("Available commands:")
    print("  mlx-audio separate <input> - Separate audio into stems")
    print("  mlx-audio transcribe <input> - Transcribe speech to text")
    print("  mlx-audio generate <prompt> - Generate audio from text")
    print()
    print("For more information, see: https://github.com/zkeown/mlx-audio")


if __name__ == "__main__":
    main()
