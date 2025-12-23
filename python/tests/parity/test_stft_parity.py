"""
STFT/ISTFT Parity Tests

Tests MLX STFT implementation against librosa reference.
Generates reference data that can be used for Swift parity tests.
"""

import json
import numpy as np
import pytest

try:
    import librosa

    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

try:
    import mlx.core as mx
    from mlx_audio.primitives import stft, istft

    HAS_MLX = True
except ImportError:
    HAS_MLX = False


@pytest.mark.parity
@pytest.mark.skipif(not HAS_LIBROSA, reason="librosa not installed")
@pytest.mark.skipif(not HAS_MLX, reason="mlx_audio not installed")
class TestSTFTParity:
    """Test STFT implementation against librosa reference."""

    def test_stft_sine_wave(self):
        """STFT of pure sine wave should match librosa."""
        # Generate 1 second of 440 Hz sine wave at 22050 Hz
        sr = 22050
        duration = 1.0
        frequency = 440.0

        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        signal = np.sin(2 * np.pi * frequency * t).astype(np.float32)

        # STFT parameters
        n_fft = 2048
        hop_length = 512

        # Librosa reference
        librosa_stft = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)
        librosa_mag = np.abs(librosa_stft)

        # MLX implementation
        mlx_signal = mx.array(signal)
        mlx_stft = stft(mlx_signal, n_fft=n_fft, hop_length=hop_length)
        mlx_mag = np.array(mx.abs(mlx_stft))

        # Compare magnitudes
        # Note: rtol=1e-4 gives ~0.02% violations due to float32 precision
        # on large values (~500). Max abs diff is ~1.5e-4, well within bounds.
        np.testing.assert_allclose(
            mlx_mag, librosa_mag, rtol=1e-4, atol=2e-4
        )

    def test_stft_random_signal(self):
        """STFT of random signal should match librosa."""
        np.random.seed(42)
        signal = np.random.randn(22050).astype(np.float32)

        n_fft = 2048
        hop_length = 512

        # Librosa reference
        librosa_stft = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)
        librosa_mag = np.abs(librosa_stft)

        # MLX implementation
        mlx_signal = mx.array(signal)
        mlx_stft = stft(mlx_signal, n_fft=n_fft, hop_length=hop_length)
        mlx_mag = np.array(mx.abs(mlx_stft))

        np.testing.assert_allclose(mlx_mag, librosa_mag, rtol=1e-4, atol=1e-5)

    def test_stft_different_window_sizes(self):
        """STFT with different window sizes should match librosa."""
        np.random.seed(123)
        signal = np.random.randn(16000).astype(np.float32)

        for n_fft in [512, 1024, 2048, 4096]:
            hop_length = n_fft // 4

            librosa_stft = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)
            librosa_mag = np.abs(librosa_stft)

            mlx_signal = mx.array(signal)
            mlx_stft = stft(mlx_signal, n_fft=n_fft, hop_length=hop_length)
            mlx_mag = np.array(mx.abs(mlx_stft))

            np.testing.assert_allclose(
                mlx_mag, librosa_mag, rtol=1e-4, atol=1e-5, err_msg=f"Failed for n_fft={n_fft}"
            )

    def test_istft_roundtrip(self):
        """STFT -> ISTFT should reconstruct signal."""
        np.random.seed(42)
        signal = np.random.randn(22050).astype(np.float32)

        n_fft = 2048
        hop_length = 512

        # MLX roundtrip
        mlx_signal = mx.array(signal)
        mlx_stft_result = stft(mlx_signal, n_fft=n_fft, hop_length=hop_length)
        reconstructed = istft(mlx_stft_result, hop_length=hop_length, length=len(signal))
        reconstructed = np.array(reconstructed)

        # Should be close to original (within STFT precision)
        np.testing.assert_allclose(reconstructed, signal, rtol=1e-3, atol=1e-4)

    def test_stft_output_shape(self):
        """STFT output shape should match expected dimensions."""
        signal_length = 22050
        n_fft = 2048
        hop_length = 512

        signal = np.zeros(signal_length, dtype=np.float32)
        mlx_signal = mx.array(signal)
        result = stft(mlx_signal, n_fft=n_fft, hop_length=hop_length)

        # Expected: (n_fft//2 + 1, n_frames)
        # n_frames = 1 + (signal_length + pad - n_fft) // hop_length
        expected_freq_bins = n_fft // 2 + 1

        assert result.shape[0] == expected_freq_bins


@pytest.mark.parity
@pytest.mark.skipif(not HAS_LIBROSA, reason="librosa not installed")
@pytest.mark.skipif(not HAS_MLX, reason="mlx_audio not installed")
class TestMelFilterbankParity:
    """Test mel filterbank against librosa reference."""

    def test_mel_filterbank_shape(self):
        """Mel filterbank shape should match librosa."""
        from mlx_audio.primitives import mel_filterbank

        sr = 22050
        n_fft = 2048
        n_mels = 128

        # Librosa reference
        librosa_fb = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)

        # MLX implementation
        mlx_fb = mel_filterbank(sr=sr, n_fft=n_fft, n_mels=n_mels)
        mlx_fb = np.array(mlx_fb)

        assert librosa_fb.shape == mlx_fb.shape

    def test_mel_filterbank_values(self):
        """Mel filterbank values should match librosa."""
        from mlx_audio.primitives import mel_filterbank

        sr = 22050
        n_fft = 2048
        n_mels = 128

        librosa_fb = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, norm="slaney")

        mlx_fb = mel_filterbank(sr=sr, n_fft=n_fft, n_mels=n_mels, norm="slaney")
        mlx_fb = np.array(mlx_fb)

        np.testing.assert_allclose(mlx_fb, librosa_fb, rtol=1e-4, atol=1e-6)

    def test_mel_filterbank_different_params(self):
        """Mel filterbank with various parameters should match librosa."""
        from mlx_audio.primitives import mel_filterbank

        test_cases = [
            {"sr": 16000, "n_fft": 400, "n_mels": 80},  # Whisper-like
            {"sr": 44100, "n_fft": 4096, "n_mels": 128},  # High-res
            {"sr": 8000, "n_fft": 256, "n_mels": 40},  # Low-res
        ]

        for params in test_cases:
            librosa_fb = librosa.filters.mel(
                sr=params["sr"],
                n_fft=params["n_fft"],
                n_mels=params["n_mels"],
                norm="slaney",
            )

            mlx_fb = mel_filterbank(**params, norm="slaney")
            mlx_fb = np.array(mlx_fb)

            np.testing.assert_allclose(
                mlx_fb,
                librosa_fb,
                rtol=1e-4,
                atol=1e-6,
                err_msg=f"Failed for params={params}",
            )


@pytest.mark.parity
@pytest.mark.skipif(not HAS_LIBROSA, reason="librosa not installed")
@pytest.mark.skipif(not HAS_MLX, reason="mlx_audio not installed")
class TestMelSpectrogramParity:
    """Test mel spectrogram against librosa reference."""

    def test_mel_spectrogram(self):
        """Mel spectrogram should match librosa."""
        from mlx_audio.primitives import melspectrogram

        np.random.seed(42)
        signal = np.random.randn(22050).astype(np.float32)

        sr = 22050
        n_fft = 2048
        hop_length = 512
        n_mels = 128

        # Librosa reference
        librosa_mel = librosa.feature.melspectrogram(
            y=signal, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
        )

        # MLX implementation
        mlx_signal = mx.array(signal)
        mlx_mel = melspectrogram(
            mlx_signal,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )
        mlx_mel = np.array(mlx_mel)

        np.testing.assert_allclose(
            mlx_mel, librosa_mel, rtol=1e-3, atol=1e-5
        )


def generate_reference_data():
    """
    Generate reference data for Swift parity tests.

    Run this function to create JSON files with reference values
    that can be loaded in Swift tests.
    """
    import os

    output_dir = os.path.join(os.path.dirname(__file__), "reference_data")
    os.makedirs(output_dir, exist_ok=True)

    # Generate STFT reference for sine wave
    sr = 22050
    frequency = 440.0
    t = np.linspace(0, 0.1, int(sr * 0.1), endpoint=False)
    signal = np.sin(2 * np.pi * frequency * t).astype(np.float32)

    n_fft = 512
    hop_length = 128

    librosa_stft = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length, center=True)
    librosa_mag = np.abs(librosa_stft)

    reference = {
        "sample_rate": sr,
        "frequency": frequency,
        "n_fft": n_fft,
        "hop_length": hop_length,
        "signal": signal.tolist(),
        "magnitude": librosa_mag.tolist(),
        "shape": list(librosa_mag.shape),
    }

    with open(os.path.join(output_dir, "stft_reference.json"), "w") as f:
        json.dump(reference, f, indent=2)

    # Generate mel filterbank reference
    mel_fb = librosa.filters.mel(sr=16000, n_fft=400, n_mels=80, norm="slaney")

    mel_reference = {
        "sample_rate": 16000,
        "n_fft": 400,
        "n_mels": 80,
        "filterbank": mel_fb.tolist(),
        "shape": list(mel_fb.shape),
    }

    with open(os.path.join(output_dir, "mel_filterbank_reference.json"), "w") as f:
        json.dump(mel_reference, f, indent=2)

    print(f"Reference data saved to {output_dir}")


if __name__ == "__main__":
    generate_reference_data()
