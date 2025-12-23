"""Tests for EnCodec configuration."""

import pytest
from mlx_audio.models.encodec.config import EnCodecConfig


class TestEnCodecConfig:
    """Tests for EnCodecConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = EnCodecConfig()

        assert config.sample_rate == 32000
        assert config.channels == 1
        assert config.num_codebooks == 4
        assert config.codebook_size == 2048
        assert config.codebook_dim == 128

    def test_frame_rate_calculation(self):
        """Test frame rate property."""
        config = EnCodecConfig(sample_rate=32000, ratios=(8, 5, 4, 4))
        # hop_length = 8 * 5 * 4 * 4 = 640
        # frame_rate = 32000 / 640 = 50
        assert config.frame_rate == 50.0

    def test_hop_length_calculation(self):
        """Test hop length property."""
        config = EnCodecConfig(ratios=(8, 5, 4, 4))
        assert config.hop_length == 8 * 5 * 4 * 4  # 640

    def test_encodec_24khz_preset(self):
        """Test 24kHz preset configuration."""
        config = EnCodecConfig.encodec_24khz()

        assert config.sample_rate == 24000
        assert config.channels == 1
        assert config.num_codebooks == 8
        assert config.codebook_size == 1024

    def test_encodec_32khz_preset(self):
        """Test 32kHz preset configuration."""
        config = EnCodecConfig.encodec_32khz()

        assert config.sample_rate == 32000
        assert config.channels == 1
        assert config.num_codebooks == 4
        assert config.codebook_size == 2048

    def test_encodec_48khz_stereo_preset(self):
        """Test 48kHz stereo preset configuration."""
        config = EnCodecConfig.encodec_48khz_stereo()

        assert config.sample_rate == 48000
        assert config.channels == 2
        assert config.num_codebooks == 8

    def test_from_name(self):
        """Test creating config from model name."""
        config = EnCodecConfig.from_name("encodec_24khz")
        assert config.sample_rate == 24000

        config = EnCodecConfig.from_name("32khz")
        assert config.sample_rate == 32000

    def test_from_name_invalid(self):
        """Test invalid model name raises error."""
        with pytest.raises(ValueError, match="Unknown EnCodec model"):
            EnCodecConfig.from_name("invalid_model")

    def test_from_dict(self):
        """Test creating config from dictionary."""
        d = {
            "sample_rate": 16000,
            "channels": 2,
            "num_codebooks": 8,
            "ratios": [4, 4, 4, 4],
        }
        config = EnCodecConfig.from_dict(d)

        assert config.sample_rate == 16000
        assert config.channels == 2
        assert config.num_codebooks == 8
        assert config.ratios == (4, 4, 4, 4)

    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = EnCodecConfig(sample_rate=24000, channels=1)
        d = config.to_dict()

        assert d["sample_rate"] == 24000
        assert d["channels"] == 1
        assert isinstance(d["ratios"], list)  # Should be list for JSON
