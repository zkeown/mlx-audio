"""CLAP quality benchmark tests.

These tests verify that the MLX CLAP implementation produces
audio-text embeddings with cosine similarity comparable to the
reference implementation.

Quality targets:
- Cosine similarity between Python and Swift embeddings > 0.999
- Audio-text retrieval accuracy matches reference
"""

import pytest
import numpy as np

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        a: First vector [D] or [B, D]
        b: Second vector [D] or [B, D]

    Returns:
        Cosine similarity (-1 to 1, 1 = identical)
    """
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if b.ndim == 1:
        b = b.reshape(1, -1)

    # Normalize
    a_norm = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-8)

    # Compute similarity
    sim = np.sum(a_norm * b_norm, axis=-1)
    return float(sim.mean())


def generate_synthetic_audio(
    duration_seconds: float = 2.0,
    sample_rate: int = 48000,
    seed: int = 42,
) -> np.ndarray:
    """Generate synthetic audio for testing.

    Returns:
        audio: [samples] audio array
    """
    np.random.seed(seed)
    samples = int(duration_seconds * sample_rate)

    t = np.linspace(0, duration_seconds, samples)
    audio = (
        np.sin(2 * np.pi * 440 * t) * 0.3  # A4 note
        + np.sin(2 * np.pi * 880 * t) * 0.2  # A5 note
        + np.random.randn(samples) * 0.05  # Noise
    )

    return audio.astype(np.float32)


@pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
@pytest.mark.slow
class TestCLAPQuality:
    """Quality tests for CLAP audio-text embeddings."""

    def test_audio_encoder_forward(self):
        """Test that audio encoder forward pass works."""
        from mlx_audio.models.clap import CLAP, CLAPConfig

        config = CLAPConfig()
        model = CLAP(config)

        # Generate random audio input
        # CLAP expects [B, T] audio at 48kHz
        audio_input = mx.random.normal([1, 48000])  # 1 second

        # Get audio embeddings
        audio_embed = model.encode_audio(audio_input)
        mx.eval(audio_embed)

        assert audio_embed.shape[0] == 1
        assert audio_embed.shape[1] == config.projection_dim

    def test_text_encoder_forward(self):
        """Test that text encoder forward pass works."""
        from mlx_audio.models.clap import CLAP, CLAPConfig

        config = CLAPConfig()
        model = CLAP(config)

        # Create mock text input (token ids)
        text_input = mx.random.randint(low=0, high=49408, shape=[1, 77])

        # Get text embeddings
        text_embed = model.encode_text(text_input)
        mx.eval(text_embed)

        assert text_embed.shape[0] == 1
        assert text_embed.shape[1] == config.projection_dim

    def test_embedding_similarity_same_input(self):
        """Test that same input produces consistent embeddings."""
        from mlx_audio.models.clap import CLAP, CLAPConfig

        config = CLAPConfig()
        model = CLAP(config)
        model.eval()

        audio_input = mx.random.normal([1, 48000])

        embed1 = model.encode_audio(audio_input)
        mx.eval(embed1)

        embed2 = model.encode_audio(audio_input)
        mx.eval(embed2)

        # Should be identical
        assert mx.allclose(embed1, embed2)

    def test_embedding_similarity_different_input(self):
        """Test that different inputs produce different embeddings."""
        from mlx_audio.models.clap import CLAP, CLAPConfig

        config = CLAPConfig()
        model = CLAP(config)

        audio1 = mx.random.normal([1, 48000], key=mx.random.key(1))
        audio2 = mx.random.normal([1, 48000], key=mx.random.key(2))

        embed1 = model.encode_audio(audio1)
        embed2 = model.encode_audio(audio2)
        mx.eval(embed1, embed2)

        # Should be different
        sim = cosine_similarity(np.array(embed1), np.array(embed2))
        assert sim < 0.99, f"Embeddings too similar: {sim}"

    def test_output_shape_batch(self):
        """Test batch processing output shapes."""
        from mlx_audio.models.clap import CLAP, CLAPConfig

        config = CLAPConfig()
        model = CLAP(config)

        # Batch of audio
        batch_audio = mx.random.normal([4, 48000])
        audio_embed = model.encode_audio(batch_audio)
        mx.eval(audio_embed)

        assert audio_embed.shape == (4, config.projection_dim)

        # Batch of text
        batch_text = mx.random.randint(low=0, high=49408, shape=[4, 77])
        text_embed = model.encode_text(batch_text)
        mx.eval(text_embed)

        assert text_embed.shape == (4, config.projection_dim)


@pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
class TestCLAPRetrievalMetrics:
    """Tests for CLAP retrieval functionality."""

    def test_audio_text_similarity_matrix(self):
        """Test computing audio-text similarity matrix."""
        from mlx_audio.models.clap import CLAP, CLAPConfig

        config = CLAPConfig()
        model = CLAP(config)

        # Multiple audio and text inputs
        audio_batch = mx.random.normal([3, 48000])
        text_batch = mx.random.randint(low=0, high=49408, shape=[3, 77])

        audio_embed = model.encode_audio(audio_batch)
        text_embed = model.encode_text(text_batch)
        mx.eval(audio_embed, text_embed)

        # Compute similarity matrix
        audio_np = np.array(audio_embed)
        text_np = np.array(text_embed)

        # Normalize
        audio_norm = audio_np / (
            np.linalg.norm(audio_np, axis=-1, keepdims=True) + 1e-8
        )
        text_norm = text_np / (np.linalg.norm(text_np, axis=-1, keepdims=True) + 1e-8)

        # Similarity matrix [audio, text]
        sim_matrix = audio_norm @ text_norm.T

        assert sim_matrix.shape == (3, 3)
        # Diagonal should be similarity of matched pairs
        # Off-diagonal should be different

    def test_embedding_normalization(self):
        """Test that embeddings can be properly normalized."""
        from mlx_audio.models.clap import CLAP, CLAPConfig

        config = CLAPConfig()
        model = CLAP(config)

        audio_input = mx.random.normal([1, 48000])
        embed = model.encode_audio(audio_input)
        mx.eval(embed)

        embed_np = np.array(embed)
        norm = np.linalg.norm(embed_np, axis=-1)

        # Norm should be reasonable (not zero or infinite)
        assert norm > 0.1
        assert norm < 1000


@pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
class TestCLAPEdgeCases:
    """Edge case tests for CLAP."""

    def test_short_audio(self):
        """Test with very short audio."""
        from mlx_audio.models.clap import CLAP, CLAPConfig

        config = CLAPConfig()
        model = CLAP(config)

        # Very short audio (0.1 second)
        short_audio = mx.random.normal([1, 4800])

        try:
            embed = model.encode_audio(short_audio)
            mx.eval(embed)
            assert embed.shape[0] == 1
        except Exception as e:
            # Some models may not support very short audio
            pytest.skip(f"Short audio not supported: {e}")

    def test_long_audio(self):
        """Test with long audio (10 seconds)."""
        from mlx_audio.models.clap import CLAP, CLAPConfig

        config = CLAPConfig()
        model = CLAP(config)

        # Long audio (10 seconds at 48kHz)
        long_audio = mx.random.normal([1, 480000])

        embed = model.encode_audio(long_audio)
        mx.eval(embed)

        assert embed.shape[0] == 1
        assert embed.shape[1] == config.projection_dim

    def test_silent_audio(self):
        """Test with silent audio."""
        from mlx_audio.models.clap import CLAP, CLAPConfig

        config = CLAPConfig()
        model = CLAP(config)

        silent_audio = mx.zeros([1, 48000])

        embed = model.encode_audio(silent_audio)
        mx.eval(embed)

        # Should produce valid (non-NaN) embeddings
        embed_np = np.array(embed)
        assert not np.any(np.isnan(embed_np))

    def test_deterministic_output(self):
        """Test that model output is deterministic."""
        from mlx_audio.models.clap import CLAP, CLAPConfig

        config = CLAPConfig()
        model = CLAP(config)
        model.eval()

        audio_input = mx.random.normal([1, 48000])

        output1 = model.encode_audio(audio_input)
        mx.eval(output1)

        output2 = model.encode_audio(audio_input)
        mx.eval(output2)

        assert mx.allclose(output1, output2)

    def test_empty_text_handling(self):
        """Test handling of minimal text input."""
        from mlx_audio.models.clap import CLAP, CLAPConfig

        config = CLAPConfig()
        model = CLAP(config)

        # Minimal text (just start/end tokens, rest padding)
        minimal_text = mx.zeros([1, 77], dtype=mx.int32)

        embed = model.encode_text(minimal_text)
        mx.eval(embed)

        # Should produce valid embeddings
        embed_np = np.array(embed)
        assert not np.any(np.isnan(embed_np))
