"""Tests for enhance() functional API."""

import numpy as np
import pytest

import mlx.core as mx


class TestEnhanceSpectral:
    """Tests for spectral gating enhancement method."""

    def test_basic_enhance(self, random_audio):
        """Test basic spectral enhancement."""
        from mlx_audio.functional.enhance import enhance

        result = enhance(random_audio, method="spectral", sample_rate=22050)

        assert hasattr(result, "array")
        assert result.array.shape == (len(random_audio),)

    def test_enhance_returns_result_type(self, random_audio):
        """Test that enhance returns EnhancementResult."""
        from mlx_audio.functional.enhance import enhance
        from mlx_audio.types.results import EnhancementResult

        result = enhance(random_audio, method="spectral", sample_rate=22050)

        assert isinstance(result, EnhancementResult)

    def test_keep_original(self, random_audio):
        """Test keep_original stores original audio."""
        from mlx_audio.functional.enhance import enhance

        result = enhance(
            random_audio,
            method="spectral",
            sample_rate=22050,
            keep_original=True,
        )

        assert "original" in result.metadata
        original, enhanced = result.before_after
        assert original is not None

    def test_spectral_parameters(self, random_audio):
        """Test passing spectral gating parameters."""
        from mlx_audio.functional.enhance import enhance

        result = enhance(
            random_audio,
            method="spectral",
            sample_rate=22050,
            threshold_db=-30,
            n_fft=1024,
        )

        assert result.array.shape == (len(random_audio),)

    def test_mlx_array_input(self, random_audio):
        """Test with MLX array input."""
        from mlx_audio.functional.enhance import enhance

        audio_mx = mx.array(random_audio)
        result = enhance(audio_mx, method="spectral", sample_rate=22050)

        assert result.array.shape == audio_mx.shape


class TestEnhanceAdaptive:
    """Tests for adaptive spectral gating."""

    def test_adaptive_method(self, random_audio):
        """Test spectral gating with non-stationary noise."""
        from mlx_audio.functional.enhance import enhance

        # Adaptive (non-stationary) spectral gating
        result = enhance(
            random_audio,
            method="spectral",
            sample_rate=22050,
            stationary=False,
        )

        assert result.array.shape == (len(random_audio),)


class TestEnhanceResultType:
    """Tests for EnhancementResult type."""

    def test_result_has_sample_rate(self, random_audio):
        """Test result has sample_rate attribute."""
        from mlx_audio.functional.enhance import enhance

        result = enhance(random_audio, method="spectral", sample_rate=22050)

        assert result.sample_rate == 22050

    def test_result_has_model_name(self, random_audio):
        """Test result has model_name attribute."""
        from mlx_audio.functional.enhance import enhance

        result = enhance(random_audio, method="spectral", sample_rate=22050)

        assert result.model_name == "spectral_gate"

    def test_result_to_numpy(self, random_audio):
        """Test result can be converted to numpy."""
        from mlx_audio.functional.enhance import enhance

        result = enhance(random_audio, method="spectral", sample_rate=22050)

        np_audio = result.to_numpy()
        assert isinstance(np_audio, np.ndarray)
        assert np_audio.shape == result.array.shape
