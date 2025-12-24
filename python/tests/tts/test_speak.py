"""Tests for the speak() functional API."""

import pytest


class TestSpeakAPI:
    """Tests for mlx_audio.speak() function."""

    def test_speak_import(self):
        """Test that speak can be imported from mlx_audio."""
        from mlx_audio import speak

        assert callable(speak)

    def test_speech_result_import(self):
        """Test that SpeechResult can be imported."""
        from mlx_audio.types.results import SpeechResult

        assert SpeechResult is not None

    def test_parler_tts_import(self):
        """Test that ParlerTTS model can be imported."""
        from mlx_audio.models.tts import ParlerTTS, ParlerTTSConfig

        assert ParlerTTS is not None
        assert ParlerTTSConfig is not None

    def test_config_mini(self):
        """Test ParlerTTSConfig.mini() preset."""
        from mlx_audio.models.tts import ParlerTTSConfig

        config = ParlerTTSConfig.mini()
        assert config.hidden_size == 1024
        assert config.num_hidden_layers == 24
        assert config.num_codebooks == 9
        assert config.sample_rate == 24000

    def test_config_large(self):
        """Test ParlerTTSConfig.large() preset."""
        from mlx_audio.models.tts import ParlerTTSConfig

        config = ParlerTTSConfig.large()
        assert config.hidden_size == 1536
        assert config.num_hidden_layers == 36
        assert config.num_codebooks == 9

    def test_config_from_name(self):
        """Test ParlerTTSConfig.from_name()."""
        from mlx_audio.models.tts import ParlerTTSConfig

        mini = ParlerTTSConfig.from_name("mini")
        assert mini.hidden_size == 1024

        large = ParlerTTSConfig.from_name("large")
        assert large.hidden_size == 1536

    def test_registry_has_speech_task(self):
        """Test that SPEECH task type exists in registry."""
        from mlx_audio.hub.registry import TaskType

        assert hasattr(TaskType, "SPEECH")
        assert TaskType.SPEECH.value == "speech"

    def test_registry_has_parler_tts_models(self):
        """Test that Parler-TTS models are registered."""
        from mlx_audio.hub.registry import ModelRegistry, TaskType

        registry = ModelRegistry.get()

        # Check parler-tts-mini is registered
        mini_spec = registry.get_spec("parler-tts-mini")
        assert mini_spec is not None
        assert mini_spec.task == TaskType.SPEECH
        assert "parler-tts-mini" in mini_spec.name

        # Check parler-tts-large is registered
        large_spec = registry.get_spec("parler-tts-large")
        assert large_spec is not None
        assert large_spec.task == TaskType.SPEECH

    def test_registry_default_speech_model(self):
        """Test that parler-tts-mini is the default for SPEECH task."""
        from mlx_audio.hub.registry import ModelRegistry, TaskType

        registry = ModelRegistry.get()
        default_spec = registry.get_default_for_task(TaskType.SPEECH)

        assert default_spec is not None
        assert default_spec.name == "parler-tts-mini"


@pytest.mark.integration
@pytest.mark.skip(reason="Model mlx-community/parler-tts-mini not yet available on HuggingFace")
class TestSpeakIntegration:
    """Integration tests for speak() - require model download."""

    def test_speak_basic(self):
        """Test basic speak functionality."""
        import mlx_audio

        result = mlx_audio.speak("Hello world", model="parler-tts-mini")
        assert result.sample_rate == 24000
        assert result.duration > 0.5
        assert result.text == "Hello world"

    def test_speak_with_description(self):
        """Test speak with voice description."""
        import mlx_audio

        result = mlx_audio.speak(
            "Hello",
            description="A calm female voice",
            model="parler-tts-mini",
        )
        assert result.description == "A calm female voice"
        assert result.text == "Hello"

    def test_speak_with_seed(self):
        """Test speak with seed for reproducibility."""
        import mlx_audio

        result1 = mlx_audio.speak("Test", model="parler-tts-mini", seed=42)
        result2 = mlx_audio.speak("Test", model="parler-tts-mini", seed=42)

        # Same seed should produce same output
        assert result1.array.shape == result2.array.shape

    def test_speech_result_save(self, tmp_path):
        """Test saving SpeechResult to file."""
        import mlx_audio

        result = mlx_audio.speak("Hello", model="parler-tts-mini")
        output_path = tmp_path / "test_output.wav"
        result.save(output_path)

        assert output_path.exists()
        assert output_path.stat().st_size > 0
