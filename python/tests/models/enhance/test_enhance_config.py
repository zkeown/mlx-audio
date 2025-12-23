"""Tests for DeepFilterNet configuration."""

import pytest

from mlx_audio.models.enhance import DeepFilterNetConfig


class TestDeepFilterNetConfig:
    """Tests for DeepFilterNet configuration dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = DeepFilterNetConfig()

        assert config.sample_rate == 48000
        assert config.fft_size == 960
        assert config.hop_size == 480
        assert config.nb_erb == 32
        assert config.nb_df == 96
        assert config.hidden_dim == 256
        assert config.gru_groups == 1
        assert config.linear_groups == 1
        assert config.lookahead == 2

    def test_preset_48k(self):
        """Test 48kHz preset."""
        config = DeepFilterNetConfig.preset_48k()

        assert config.sample_rate == 48000
        assert config.fft_size == 960
        assert config.hop_size == 480

    def test_preset_16k(self):
        """Test 16kHz preset."""
        config = DeepFilterNetConfig.preset_16k()

        assert config.sample_rate == 16000
        assert config.fft_size == 320
        assert config.hop_size == 160

    def test_custom_config(self):
        """Test custom configuration."""
        config = DeepFilterNetConfig(
            sample_rate=24000,
            hidden_dim=128,
            nb_erb=64,
        )

        assert config.sample_rate == 24000
        assert config.hidden_dim == 128
        assert config.nb_erb == 64
        # Defaults should still apply
        assert config.gru_groups == 1

    def test_config_immutability(self):
        """Test that config fields can be set."""
        config = DeepFilterNetConfig()

        # Dataclass is mutable by default
        config.hidden_dim = 512
        assert config.hidden_dim == 512
