"""Tests for EnCodec model."""

import pytest
import mlx.core as mx

from mlx_audio.models.encodec.config import EnCodecConfig
from mlx_audio.models.encodec.model import EnCodec
from mlx_audio.models.encodec.layers.quantizer import (
    VectorQuantizer,
    ResidualVectorQuantizer,
)


class TestVectorQuantizer:
    """Tests for single codebook vector quantizer."""

    def test_encode_output_shape(self):
        """Test encode produces correct shape."""
        vq = VectorQuantizer(codebook_size=1024, codebook_dim=128)

        x = mx.random.normal((2, 50, 128))  # [B, T, D]
        codes = vq.encode(x)

        assert codes.shape == (2, 50)  # [B, T]

    def test_decode_output_shape(self):
        """Test decode produces correct shape."""
        vq = VectorQuantizer(codebook_size=1024, codebook_dim=128)

        codes = mx.array([[0, 1, 2], [3, 4, 5]])  # [B, T]
        embeddings = vq.decode(codes)

        assert embeddings.shape == (2, 3, 128)  # [B, T, D]

    def test_encode_decode_consistency(self):
        """Test that encode->decode->encode is idempotent."""
        vq = VectorQuantizer(codebook_size=1024, codebook_dim=128)

        # Encode some input
        x = mx.random.normal((1, 10, 128))
        codes1 = vq.encode(x)

        # Decode and re-encode
        embeddings = vq.decode(codes1)
        codes2 = vq.encode(embeddings)

        # Should get same codes
        assert mx.array_equal(codes1, codes2)


class TestResidualVectorQuantizer:
    """Tests for residual vector quantizer."""

    def test_encode_output_shape(self):
        """Test encode produces correct shape."""
        rvq = ResidualVectorQuantizer(
            num_codebooks=4,
            codebook_size=1024,
            codebook_dim=128,
        )

        x = mx.random.normal((2, 50, 128))  # [B, T, D]
        codes = rvq.encode(x)

        assert codes.shape == (2, 4, 50)  # [B, K, T]

    def test_decode_output_shape(self):
        """Test decode produces correct shape."""
        rvq = ResidualVectorQuantizer(
            num_codebooks=4,
            codebook_size=1024,
            codebook_dim=128,
        )

        codes = mx.zeros((2, 4, 50), dtype=mx.int32)  # [B, K, T]
        embeddings = rvq.decode(codes)

        assert embeddings.shape == (2, 50, 128)  # [B, T, D]

    def test_encode_decode_roundtrip(self):
        """Test encode->decode produces valid reconstruction."""
        rvq = ResidualVectorQuantizer(
            num_codebooks=4,
            codebook_size=1024,
            codebook_dim=128,
        )

        x = mx.random.normal((1, 10, 128))
        quantized, codes = rvq(x)

        assert quantized.shape == x.shape
        assert codes.shape == (1, 4, 10)

    def test_get_codebook(self):
        """Test codebook retrieval."""
        rvq = ResidualVectorQuantizer(
            num_codebooks=4,
            codebook_size=1024,
            codebook_dim=128,
        )

        for i in range(4):
            codebook = rvq.get_codebook(i)
            assert codebook.shape == (1024, 128)


class TestEnCodecModel:
    """Tests for full EnCodec model."""

    @pytest.fixture
    def small_config(self):
        """Create a small config for fast testing."""
        return EnCodecConfig(
            sample_rate=16000,
            channels=1,
            num_codebooks=2,
            codebook_size=512,
            codebook_dim=64,
            num_filters=16,
            ratios=(4, 4, 4, 4),
            lstm_layers=0,  # Disable LSTM for faster tests
        )

    @pytest.fixture
    def small_model(self, small_config):
        """Create a small model for testing."""
        return EnCodec(small_config)

    def test_encode_output_shape(self, small_model, small_config):
        """Test encode produces correct shape."""
        # 1 second of audio
        audio = mx.random.normal((1, 1, small_config.sample_rate))
        codes = small_model.encode(audio)

        # Expected frames = samples / hop_length (allow +/- 1 due to padding)
        expected_frames = small_config.sample_rate // small_config.hop_length
        assert codes.shape[0] == 1
        assert codes.shape[1] == small_config.num_codebooks
        assert abs(codes.shape[2] - expected_frames) <= 1

    def test_decode_output_shape(self, small_model, small_config):
        """Test decode produces correct shape."""
        num_frames = 50
        codes = mx.zeros(
            (1, small_config.num_codebooks, num_frames),
            dtype=mx.int32
        )
        audio = small_model.decode(codes)

        expected_samples = num_frames * small_config.hop_length
        assert audio.shape == (1, small_config.channels, expected_samples)

    def test_full_forward_pass(self, small_model, small_config):
        """Test full encode-decode forward pass."""
        audio = mx.random.normal((1, 1, small_config.sample_rate))
        reconstructed, codes = small_model(audio)

        assert reconstructed.shape[0] == 1
        assert reconstructed.shape[1] == small_config.channels
        assert codes.shape[1] == small_config.num_codebooks

    def test_mono_input_handling(self, small_model, small_config):
        """Test that mono input without channel dim is handled."""
        # [B, T] format
        audio = mx.random.normal((1, small_config.sample_rate))
        codes = small_model.encode(audio)

        assert codes.ndim == 3  # [B, K, T']

    def test_properties(self, small_model, small_config):
        """Test model properties."""
        assert small_model.sample_rate == small_config.sample_rate
        assert small_model.channels == small_config.channels
        assert small_model.num_codebooks == small_config.num_codebooks
        assert small_model.codebook_size == small_config.codebook_size
        assert small_model.hop_length == small_config.hop_length
