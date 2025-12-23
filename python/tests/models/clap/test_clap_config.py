"""Tests for CLAP configuration."""

import pytest


class TestCLAPAudioConfig:
    """Tests for CLAPAudioConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        from mlx_audio.models.clap.config import CLAPAudioConfig

        config = CLAPAudioConfig()

        assert config.sample_rate == 48000
        assert config.n_mels == 64
        assert config.n_fft == 1024
        assert config.hop_length == 480
        assert config.patch_size == 4
        assert config.embed_dim == 96
        assert config.depths == (2, 2, 6, 2)
        assert config.num_heads == (4, 8, 16, 32)
        assert config.window_size == 8
        assert config.hidden_size == 768

    def test_custom_values(self):
        """Test custom configuration values."""
        from mlx_audio.models.clap.config import CLAPAudioConfig

        config = CLAPAudioConfig(
            sample_rate=44100,
            n_mels=128,
            hidden_size=1024,
        )

        assert config.sample_rate == 44100
        assert config.n_mels == 128
        assert config.hidden_size == 1024


class TestCLAPTextConfig:
    """Tests for CLAPTextConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        from mlx_audio.models.clap.config import CLAPTextConfig

        config = CLAPTextConfig()

        assert config.vocab_size == 50265
        assert config.hidden_size == 768
        assert config.num_hidden_layers == 12
        assert config.num_attention_heads == 12
        assert config.intermediate_size == 3072
        assert config.max_position_embeddings == 514


class TestCLAPConfig:
    """Tests for CLAPConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        from mlx_audio.models.clap.config import CLAPConfig

        config = CLAPConfig()

        assert config.projection_dim == 512
        assert isinstance(config.audio.sample_rate, int)
        assert isinstance(config.text.vocab_size, int)

    def test_htsat_tiny_factory(self):
        """Test HTSAT-tiny factory method."""
        from mlx_audio.models.clap.config import CLAPConfig

        config = CLAPConfig.htsat_tiny()

        assert config.audio.embed_dim == 96
        assert config.audio.hidden_size == 768

    def test_htsat_base_factory(self):
        """Test HTSAT-base factory method."""
        from mlx_audio.models.clap.config import CLAPConfig

        config = CLAPConfig.htsat_base()

        assert config.audio.embed_dim == 128
        assert config.audio.hidden_size == 1024

    def test_from_dict(self):
        """Test creating config from dictionary."""
        from mlx_audio.models.clap.config import CLAPConfig

        config_dict = {
            "audio": {
                "sample_rate": 44100,
                "n_mels": 128,
            },
            "text": {
                "hidden_size": 512,
            },
            "projection_dim": 256,
        }

        config = CLAPConfig.from_dict(config_dict)

        assert config.audio.sample_rate == 44100
        assert config.audio.n_mels == 128
        assert config.text.hidden_size == 512
        assert config.projection_dim == 256

    def test_to_dict(self):
        """Test converting config to dictionary."""
        from mlx_audio.models.clap.config import CLAPConfig

        config = CLAPConfig()
        config_dict = config.to_dict()

        assert "audio" in config_dict
        assert "text" in config_dict
        assert "projection_dim" in config_dict
        assert config_dict["projection_dim"] == 512
