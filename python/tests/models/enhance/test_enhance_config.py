"""Tests for DeepFilterNet configuration."""

import pytest

from mlx_audio.models.enhance import DeepFilterNetConfig


class TestDeepFilterNetConfig:
    """Tests for DeepFilterNet configuration dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = DeepFilterNetConfig()

        assert config.sample_rate == 48000
        assert config.fft_size == 1920
        assert config.hop_size == 480
        assert config.erb_bands == 32
        assert config.df_bins == 96
        assert config.hidden_size == 256
        assert config.num_groups == 1

    def test_deepfilternet2(self):
        """Test DeepFilterNet2 48kHz preset."""
        config = DeepFilterNetConfig.deepfilternet2()

        assert config.sample_rate == 48000
        assert config.fft_size == 1920
        assert config.hop_size == 480

    def test_deepfilternet2_16k(self):
        """Test DeepFilterNet2 16kHz preset."""
        config = DeepFilterNetConfig.deepfilternet2_16k()

        assert config.sample_rate == 16000
        assert config.fft_size == 640
        assert config.hop_size == 160

    def test_custom_config(self):
        """Test custom configuration."""
        config = DeepFilterNetConfig(
            sample_rate=24000,
            hidden_size=128,
            erb_bands=64,
        )

        assert config.sample_rate == 24000
        assert config.hidden_size == 128
        assert config.erb_bands == 64
        # Defaults should still apply
        assert config.num_groups == 1

    def test_config_immutability(self):
        """Test that config fields can be set."""
        config = DeepFilterNetConfig()

        # Dataclass is mutable by default
        config.hidden_size = 512
        assert config.hidden_size == 512
