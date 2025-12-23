"""Quality benchmark tests for mlx-audio models.

These tests verify that mlx-audio models produce outputs with quality
comparable to reference implementations.

Benchmarks include:
- HTDemucs: Signal-to-Distortion Ratio (SDR) for source separation
- Whisper: Word Error Rate (WER) for speech recognition
- CLAP: Retrieval accuracy for audio-text embeddings
- EnCodec: Reconstruction SDR for audio compression
"""
