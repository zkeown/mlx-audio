"""Tests for VAD configuration."""

import pytest

from mlx_audio.models.vad import VADConfig


class TestVADConfig:
    """Tests for VADConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = VADConfig()

        assert config.sample_rate == 16000
        assert config.window_size_samples == 512
        assert config.context_size_samples == 64
        assert config.hidden_size == 128
        assert config.num_layers == 2
        assert config.threshold == 0.5

    def test_silero_vad_16k(self):
        """Test 16kHz preset configuration."""
        config = VADConfig.silero_vad_16k()

        assert config.sample_rate == 16000
        assert config.window_size_samples == 512
        assert config.context_size_samples == 64
        assert config.hidden_size == 128
        assert config.num_layers == 2

    def test_silero_vad_8k(self):
        """Test 8kHz preset configuration."""
        config = VADConfig.silero_vad_8k()

        assert config.sample_rate == 8000
        assert config.window_size_samples == 256
        assert config.context_size_samples == 32
        assert config.hidden_size == 128
        assert config.num_layers == 2

    def test_window_duration_ms(self):
        """Test window duration calculation."""
        config = VADConfig.silero_vad_16k()
        # 512 samples at 16kHz = 32ms
        assert config.window_duration_ms == 32.0

        config = VADConfig.silero_vad_8k()
        # 256 samples at 8kHz = 32ms
        assert config.window_duration_ms == 32.0

    def test_context_duration_ms(self):
        """Test context duration calculation."""
        config = VADConfig.silero_vad_16k()
        # 64 samples at 16kHz = 4ms
        assert config.context_duration_ms == 4.0

        config = VADConfig.silero_vad_8k()
        # 32 samples at 8kHz = 4ms
        assert config.context_duration_ms == 4.0

    def test_min_speech_duration_samples(self):
        """Test minimum speech duration calculation."""
        config = VADConfig.silero_vad_16k()
        # 250ms at 16kHz = 4000 samples
        assert config.min_speech_duration_samples == 4000

        config = VADConfig.silero_vad_8k()
        # 250ms at 8kHz = 2000 samples
        assert config.min_speech_duration_samples == 2000

    def test_min_silence_duration_samples(self):
        """Test minimum silence duration calculation."""
        config = VADConfig.silero_vad_16k()
        # 100ms at 16kHz = 1600 samples
        assert config.min_silence_duration_samples == 1600

        config = VADConfig.silero_vad_8k()
        # 100ms at 8kHz = 800 samples
        assert config.min_silence_duration_samples == 800

    def test_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            "sample_rate": 16000,
            "window_size_samples": 512,
            "hidden_size": 256,
            "num_layers": 4,
        }
        config = VADConfig.from_dict(config_dict)

        assert config.sample_rate == 16000
        assert config.window_size_samples == 512
        assert config.hidden_size == 256
        assert config.num_layers == 4

    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = VADConfig.silero_vad_16k()
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict["sample_rate"] == 16000
        assert config_dict["window_size_samples"] == 512
        assert config_dict["hidden_size"] == 128

    def test_from_name(self):
        """Test creating config from preset name."""
        config = VADConfig.from_name("silero_vad_16k")
        assert config.sample_rate == 16000

        config = VADConfig.from_name("silero_vad_8k")
        assert config.sample_rate == 8000

    def test_from_name_default(self):
        """Test default preset name."""
        config = VADConfig.from_name("silero_vad")
        assert config.sample_rate == 16000  # Default is 16k

    def test_from_name_invalid(self):
        """Test invalid preset name raises error."""
        with pytest.raises(ValueError):
            VADConfig.from_name("invalid_model")

    def test_custom_config(self):
        """Test creating config with custom values."""
        config = VADConfig(
            sample_rate=22050,
            window_size_samples=1024,
            context_size_samples=128,
            hidden_size=256,
            num_layers=4,
            threshold=0.7,
        )

        assert config.sample_rate == 22050
        assert config.window_size_samples == 1024
        assert config.context_size_samples == 128
        assert config.hidden_size == 256
        assert config.num_layers == 4
        assert config.threshold == 0.7
