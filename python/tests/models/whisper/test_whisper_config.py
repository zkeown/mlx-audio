"""Tests for Whisper configuration."""

import pytest

from mlx_audio.models.whisper.config import WhisperConfig


class TestWhisperConfig:
    """Tests for WhisperConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = WhisperConfig()
        assert config.n_mels == 80
        assert config.sample_rate == 16000
        assert config.n_fft == 400
        assert config.hop_length == 160
        assert config.chunk_length == 30

    def test_tiny_config(self):
        """Test tiny model configuration."""
        config = WhisperConfig.tiny()
        assert config.n_audio_state == 384
        assert config.n_audio_head == 6
        assert config.n_audio_layer == 4
        assert config.n_text_state == 384
        assert config.n_text_head == 6
        assert config.n_text_layer == 4

    def test_base_config(self):
        """Test base model configuration."""
        config = WhisperConfig.base()
        assert config.n_audio_state == 512
        assert config.n_audio_head == 8
        assert config.n_audio_layer == 6
        assert config.n_text_state == 512
        assert config.n_text_head == 8
        assert config.n_text_layer == 6

    def test_small_config(self):
        """Test small model configuration."""
        config = WhisperConfig.small()
        assert config.n_audio_state == 768
        assert config.n_audio_head == 12
        assert config.n_audio_layer == 12

    def test_medium_config(self):
        """Test medium model configuration."""
        config = WhisperConfig.medium()
        assert config.n_audio_state == 1024
        assert config.n_audio_head == 16
        assert config.n_audio_layer == 24

    def test_large_config(self):
        """Test large model configuration."""
        config = WhisperConfig.large()
        assert config.n_audio_state == 1280
        assert config.n_audio_head == 20
        assert config.n_audio_layer == 32
        assert config.n_mels == 80  # v1/v2 use 80 mel bins

    def test_large_v3_config(self):
        """Test large-v3 model configuration."""
        config = WhisperConfig.large_v3()
        assert config.n_audio_state == 1280
        assert config.n_mels == 128  # v3 uses 128 mel bins
        assert config.is_v3 is True

    def test_large_v3_turbo_config(self):
        """Test large-v3-turbo model configuration."""
        config = WhisperConfig.large_v3_turbo()
        assert config.n_audio_state == 1280
        assert config.n_audio_layer == 32  # Full encoder
        assert config.n_text_layer == 4  # Reduced decoder
        assert config.n_mels == 128
        assert config.is_v3 is True

    def test_turbo_alias(self):
        """Test turbo is alias for large_v3_turbo."""
        turbo = WhisperConfig.turbo()
        v3_turbo = WhisperConfig.large_v3_turbo()
        assert turbo.n_text_layer == v3_turbo.n_text_layer

    def test_from_name(self):
        """Test creating config from model name."""
        config = WhisperConfig.from_name("tiny")
        assert config.n_audio_state == 384

        config = WhisperConfig.from_name("large-v3-turbo")
        assert config.n_text_layer == 4

        config = WhisperConfig.from_name("whisper-base")
        assert config.n_audio_state == 512

    def test_from_name_invalid(self):
        """Test from_name with invalid model name."""
        with pytest.raises(ValueError, match="Unknown Whisper model"):
            WhisperConfig.from_name("invalid-model")

    def test_n_samples_property(self):
        """Test n_samples computed property."""
        config = WhisperConfig()
        # 30 seconds * 16000 Hz = 480000 samples
        assert config.n_samples == 480000

    def test_n_frames_property(self):
        """Test n_frames computed property."""
        config = WhisperConfig()
        # 480000 samples / 160 hop = 3000 frames
        assert config.n_frames == 3000

    def test_is_multilingual(self):
        """Test is_multilingual property."""
        config = WhisperConfig.tiny()
        assert config.is_multilingual is True

        config = WhisperConfig.tiny_en()
        assert config.is_multilingual is False

    def test_to_dict(self):
        """Test config serialization to dict."""
        config = WhisperConfig.tiny()
        d = config.to_dict()
        assert d["n_audio_state"] == 384
        assert d["n_mels"] == 80
        assert isinstance(d, dict)

    def test_from_dict(self):
        """Test config creation from dict."""
        d = {
            "n_mels": 128,
            "n_audio_state": 1280,
            "n_audio_layer": 32,
            "n_text_layer": 4,
        }
        config = WhisperConfig.from_dict(d)
        assert config.n_mels == 128
        assert config.n_audio_state == 1280
        assert config.n_text_layer == 4

    def test_from_dict_ignores_unknown(self):
        """Test from_dict ignores unknown fields."""
        d = {
            "n_mels": 80,
            "unknown_field": "value",
        }
        config = WhisperConfig.from_dict(d)
        assert config.n_mels == 80
        assert not hasattr(config, "unknown_field")
