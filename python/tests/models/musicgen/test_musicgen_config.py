"""Tests for MusicGen configuration."""

import pytest
from mlx_audio.models.musicgen.config import MusicGenConfig


class TestMusicGenConfig:
    """Tests for MusicGenConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = MusicGenConfig()

        assert config.num_codebooks == 4
        assert config.codebook_size == 2048
        assert config.hidden_size == 1024
        assert config.num_hidden_layers == 24
        assert config.sample_rate == 32000
        assert config.frame_rate == 50

    def test_max_new_tokens_property(self):
        """Test max_new_tokens calculation."""
        config = MusicGenConfig(max_duration=10.0, frame_rate=50)
        assert config.max_new_tokens == 500

    def test_head_dim_property(self):
        """Test head_dim calculation."""
        config = MusicGenConfig(hidden_size=1024, num_attention_heads=16)
        assert config.head_dim == 64

    def test_vocab_size_property(self):
        """Test vocab_size includes special tokens."""
        config = MusicGenConfig(codebook_size=2048)
        assert config.vocab_size == 2049  # +1 for pad/bos/eos

    def test_small_preset(self):
        """Test small preset configuration."""
        config = MusicGenConfig.small()

        assert config.hidden_size == 1024
        assert config.num_hidden_layers == 24
        assert config.num_attention_heads == 16

    def test_medium_preset(self):
        """Test medium preset configuration."""
        config = MusicGenConfig.medium()

        assert config.hidden_size == 1536
        assert config.num_hidden_layers == 48
        assert config.num_attention_heads == 24

    def test_large_preset(self):
        """Test large preset configuration."""
        config = MusicGenConfig.large()

        assert config.hidden_size == 2048
        assert config.num_hidden_layers == 48
        assert config.num_attention_heads == 32

    def test_melody_preset(self):
        """Test melody preset configuration."""
        config = MusicGenConfig.melody()

        # Melody uses medium config as base
        assert config.hidden_size == 1536
        assert config.num_hidden_layers == 48

    def test_from_name(self):
        """Test creating config from model name."""
        config = MusicGenConfig.from_name("small")
        assert config.hidden_size == 1024

        config = MusicGenConfig.from_name("medium")
        assert config.hidden_size == 1536

        config = MusicGenConfig.from_name("musicgen_large")
        assert config.hidden_size == 2048

    def test_from_name_invalid(self):
        """Test invalid model name raises error."""
        with pytest.raises(ValueError, match="Unknown MusicGen model"):
            MusicGenConfig.from_name("invalid_model")

    def test_from_dict(self):
        """Test creating config from dictionary."""
        d = {
            "hidden_size": 512,
            "num_hidden_layers": 12,
            "num_attention_heads": 8,
            "num_codebooks": 2,
        }
        config = MusicGenConfig.from_dict(d)

        assert config.hidden_size == 512
        assert config.num_hidden_layers == 12
        assert config.num_attention_heads == 8
        assert config.num_codebooks == 2

    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = MusicGenConfig(hidden_size=512)
        d = config.to_dict()

        assert d["hidden_size"] == 512
        assert "num_hidden_layers" in d
        assert "sample_rate" in d
