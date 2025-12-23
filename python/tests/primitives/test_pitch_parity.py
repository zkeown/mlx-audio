"""Parity tests for pitch detection against librosa."""

import numpy as np
import pytest

import mlx.core as mx

librosa = pytest.importorskip("librosa")


def generate_sine(freq: float, sr: int, duration: float) -> np.ndarray:
    """Generate pure sine wave."""
    t = np.arange(int(sr * duration)) / sr
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


class TestYINParity:
    """Parity tests for YIN vs librosa.yin."""

    def test_yin_pure_tone_440(self):
        """Test YIN on 440 Hz sine wave."""
        from mlx_audio.primitives import yin

        audio = generate_sine(440, 22050, 1.0)
        audio_mx = mx.array(audio)

        mlx_f0, mlx_voiced = yin(audio_mx, sr=22050, fmin=100, fmax=1000)
        librosa_f0 = librosa.yin(audio, sr=22050, fmin=100, fmax=1000)

        mlx_f0 = np.array(mlx_f0)
        mlx_voiced = np.array(mlx_voiced)

        # Get voiced frames
        mlx_voiced_f0 = mlx_f0[mlx_voiced]
        librosa_voiced_f0 = librosa_f0[librosa_f0 > 0]

        if len(mlx_voiced_f0) > 0 and len(librosa_voiced_f0) > 0:
            mlx_median = np.median(mlx_voiced_f0)
            librosa_median = np.median(librosa_voiced_f0)

            # Both should be close to 440 Hz
            assert abs(mlx_median - 440) < 10, f"MLX: {mlx_median:.1f} Hz"
            assert abs(librosa_median - 440) < 10, f"Librosa: {librosa_median:.1f} Hz"

            # Should be close to each other
            assert abs(mlx_median - librosa_median) < 5, (
                f"Mismatch: MLX {mlx_median:.1f} vs librosa {librosa_median:.1f}"
            )

    def test_yin_pure_tone_220(self):
        """Test YIN on 220 Hz sine wave."""
        from mlx_audio.primitives import yin

        audio = generate_sine(220, 22050, 1.0)
        audio_mx = mx.array(audio)

        mlx_f0, mlx_voiced = yin(audio_mx, sr=22050, fmin=100, fmax=500)
        librosa_f0 = librosa.yin(audio, sr=22050, fmin=100, fmax=500)

        mlx_f0 = np.array(mlx_f0)
        mlx_voiced = np.array(mlx_voiced)

        mlx_voiced_f0 = mlx_f0[mlx_voiced]
        librosa_voiced_f0 = librosa_f0[librosa_f0 > 0]

        if len(mlx_voiced_f0) > 0 and len(librosa_voiced_f0) > 0:
            mlx_median = np.median(mlx_voiced_f0)
            librosa_median = np.median(librosa_voiced_f0)

            assert abs(mlx_median - 220) < 10
            assert abs(librosa_median - 220) < 10

    def test_yin_output_length(self):
        """Test YIN output length matches librosa."""
        from mlx_audio.primitives import yin

        audio = generate_sine(440, 22050, 1.0)
        audio_mx = mx.array(audio)

        frame_length = 2048
        hop_length = 512

        mlx_f0, _ = yin(
            audio_mx, sr=22050, frame_length=frame_length, hop_length=hop_length,
            fmin=100, fmax=1000
        )
        librosa_f0 = librosa.yin(
            audio, sr=22050, frame_length=frame_length, hop_length=hop_length,
            fmin=100, fmax=1000
        )

        # Lengths should be close (within 1-2 frames due to padding differences)
        assert abs(len(mlx_f0) - len(librosa_f0)) <= 2, (
            f"Length mismatch: MLX {len(mlx_f0)} vs librosa {len(librosa_f0)}"
        )


class TestPYINParity:
    """Parity tests for PYIN vs librosa.pyin."""

    def test_pyin_pure_tone_440(self):
        """Test PYIN on 440 Hz sine wave."""
        from mlx_audio.primitives import pyin

        audio = generate_sine(440, 22050, 1.0)
        audio_mx = mx.array(audio)

        mlx_f0, mlx_voiced, mlx_prob = pyin(audio_mx, sr=22050, fmin=100, fmax=1000)
        librosa_f0, librosa_voiced, librosa_prob = librosa.pyin(
            audio, sr=22050, fmin=100, fmax=1000
        )

        mlx_f0 = np.array(mlx_f0)
        mlx_voiced = np.array(mlx_voiced)

        # Get voiced frames
        mlx_voiced_f0 = mlx_f0[mlx_voiced]
        librosa_voiced_f0 = librosa_f0[librosa_voiced]

        if len(mlx_voiced_f0) > 0 and len(librosa_voiced_f0) > 0:
            mlx_median = np.median(mlx_voiced_f0)
            librosa_median = np.median(librosa_voiced_f0)

            # Both should be close to 440 Hz
            assert abs(mlx_median - 440) < 15, f"MLX: {mlx_median:.1f} Hz"
            assert abs(librosa_median - 440) < 15, f"Librosa: {librosa_median:.1f} Hz"

    def test_pyin_voiced_probability(self):
        """Test PYIN voiced probability is reasonable."""
        from mlx_audio.primitives import pyin

        # Pure tone should have high voicing probability
        audio = generate_sine(440, 22050, 1.0)
        audio_mx = mx.array(audio)

        _, _, mlx_prob = pyin(audio_mx, sr=22050, fmin=100, fmax=1000)
        _, _, librosa_prob = librosa.pyin(audio, sr=22050, fmin=100, fmax=1000)

        mlx_prob = np.array(mlx_prob)

        # Mean probability should be high for pure tone
        assert np.mean(mlx_prob) > 0.5, f"MLX prob too low: {np.mean(mlx_prob):.3f}"
        assert np.nanmean(librosa_prob) > 0.5, (
            f"Librosa prob too low: {np.nanmean(librosa_prob):.3f}"
        )

    def test_pyin_silence_low_probability(self):
        """Test PYIN gives low probability for silence."""
        from mlx_audio.primitives import pyin

        audio = np.zeros(22050, dtype=np.float32)
        audio_mx = mx.array(audio)

        _, mlx_voiced, mlx_prob = pyin(audio_mx, sr=22050, fmin=100, fmax=1000)

        mlx_prob = np.array(mlx_prob)
        mlx_voiced = np.array(mlx_voiced)

        # Silence should have low/no voicing
        assert np.mean(mlx_prob) < 0.5, f"Prob too high for silence: {np.mean(mlx_prob)}"
        assert np.sum(mlx_voiced) < len(mlx_voiced) * 0.3, "Too many voiced frames"


class TestPitchEdgeCases:
    """Edge case parity tests."""

    def test_frequency_range_respected(self):
        """Test that frequency bounds are respected."""
        from mlx_audio.primitives import yin

        audio = generate_sine(150, 22050, 0.5)
        audio_mx = mx.array(audio)

        # Set fmin above the actual frequency
        f0, voiced = yin(audio_mx, sr=22050, fmin=200, fmax=500)
        f0 = np.array(f0)
        voiced = np.array(voiced)

        voiced_f0 = f0[voiced]
        if len(voiced_f0) > 0:
            # All detected f0 should be >= fmin
            assert np.all(voiced_f0 >= 200), "Detected f0 below fmin"
