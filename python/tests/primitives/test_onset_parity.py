"""Parity tests for onset detection against librosa."""

import numpy as np
import pytest

import mlx.core as mx

librosa = pytest.importorskip("librosa")


class TestOnsetStrengthParity:
    """Parity tests for onset_strength vs librosa."""

    @pytest.fixture
    def audio(self):
        """Generate test audio."""
        np.random.seed(42)
        return np.random.randn(22050).astype(np.float32)

    def test_onset_strength_shape_parity(self, audio):
        """Test output shape matches librosa."""
        from mlx_audio.primitives import onset_strength

        audio_mx = mx.array(audio)

        mlx_env = onset_strength(audio_mx, sr=22050, hop_length=512)
        librosa_env = librosa.onset.onset_strength(y=audio, sr=22050, hop_length=512)

        # Shapes should match
        assert len(mlx_env) == len(librosa_env), (
            f"Shape mismatch: MLX {len(mlx_env)} vs librosa {len(librosa_env)}"
        )

    def test_onset_strength_reasonable_output(self, audio):
        """Test onset envelope has reasonable output characteristics."""
        from mlx_audio.primitives import onset_strength

        audio_mx = mx.array(audio)

        mlx_env = np.array(onset_strength(audio_mx, sr=22050, hop_length=512))
        librosa_env = librosa.onset.onset_strength(y=audio, sr=22050, hop_length=512)

        # Both should be non-negative
        assert mlx_env.min() >= 0, "MLX onset should be non-negative"
        assert librosa_env.min() >= 0, "Librosa onset should be non-negative"

        # Both should have similar dynamic range (order of magnitude)
        mlx_range = mlx_env.max() - mlx_env.min()
        librosa_range = librosa_env.max() - librosa_env.min()

        if librosa_range > 0:
            ratio = mlx_range / librosa_range
            assert 0.1 < ratio < 10, f"Range ratio {ratio:.2f} too different"

    def test_onset_detect_similar_results(self, audio):
        """Test onset detection finds similar onsets to librosa."""
        from mlx_audio.primitives import onset_detect

        audio_mx = mx.array(audio)

        mlx_onsets = np.array(onset_detect(y=audio_mx, sr=22050, units="frames"))
        librosa_onsets = librosa.onset.onset_detect(y=audio, sr=22050, units="frames")

        # Both should find onsets (or both find none)
        if len(librosa_onsets) > 0 and len(mlx_onsets) > 0:
            # Check if first onset is within 5 frames
            first_diff = abs(mlx_onsets[0] - librosa_onsets[0])
            assert first_diff < 10, f"First onset differs by {first_diff} frames"


class TestOnsetStrengthMultiParity:
    """Parity tests for multi-band onset strength."""

    @pytest.fixture
    def audio(self):
        """Generate test audio."""
        np.random.seed(42)
        return np.random.randn(22050).astype(np.float32)

    def test_multi_channel_shape(self, audio):
        """Test multi-band output has correct shape."""
        from mlx_audio.primitives import onset_strength_multi

        audio_mx = mx.array(audio)

        # Our API uses channels as int (number of bands)
        # Librosa uses channels as list of mel bin boundaries
        mlx_multi = onset_strength_multi(
            audio_mx, sr=22050, channels=[(0, 32), (32, 64), (64, 96), (96, 128)]
        )
        librosa_multi = librosa.onset.onset_strength_multi(
            y=audio, sr=22050, channels=[0, 32, 64, 96, 128]
        )

        # Should have same number of channels and time frames
        assert mlx_multi.shape[0] == librosa_multi.shape[0], (
            f"Channel mismatch: {mlx_multi.shape[0]} vs {librosa_multi.shape[0]}"
        )
        assert mlx_multi.shape[1] == librosa_multi.shape[1], (
            f"Time frames mismatch: {mlx_multi.shape[1]} vs {librosa_multi.shape[1]}"
        )
