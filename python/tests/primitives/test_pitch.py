"""Tests for pitch detection primitives (YIN and PYIN)."""

import numpy as np
import pytest

import mlx.core as mx
from mlx_audio.primitives import yin, pyin


def generate_sine_wave(freq: float, sr: int, duration: float) -> np.ndarray:
    """Generate a sine wave at the specified frequency."""
    t = np.arange(int(sr * duration)) / sr
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


class TestYIN:
    """Tests for YIN pitch detection."""

    def test_basic_yin(self, random_audio):
        """Test basic YIN returns f0 and voiced flag."""
        audio = mx.array(random_audio)
        f0, voiced = yin(audio, sr=22050, fmin=50, fmax=2000)

        assert isinstance(f0, mx.array)
        assert isinstance(voiced, mx.array)
        assert f0.shape == voiced.shape

    def test_pure_tone_detection(self):
        """Test YIN detects known frequency."""
        # Generate 440 Hz sine wave
        audio = generate_sine_wave(440, 22050, 1.0)
        audio_mx = mx.array(audio)

        f0, voiced = yin(audio_mx, sr=22050, fmin=100, fmax=1000)

        # Find voiced frames
        voiced_np = np.array(voiced)
        f0_np = np.array(f0)

        voiced_f0 = f0_np[voiced_np]
        if len(voiced_f0) > 0:
            median_f0 = np.median(voiced_f0)
            # Should be close to 440 Hz (tightened from 20 Hz - measured: 0.67 Hz)
            assert abs(median_f0 - 440) < 5, f"Expected ~440 Hz, got {median_f0}"

    def test_frequency_range(self, random_audio):
        """Test that f0 respects frequency bounds."""
        audio = mx.array(random_audio)
        fmin, fmax = 100, 500
        f0, voiced = yin(audio, sr=22050, fmin=fmin, fmax=fmax)

        f0_np = np.array(f0)
        voiced_np = np.array(voiced)

        # Voiced frames should have f0 in range
        voiced_f0 = f0_np[voiced_np]
        if len(voiced_f0) > 0:
            assert np.all(voiced_f0 >= fmin)
            assert np.all(voiced_f0 <= fmax)

    def test_frame_length(self, random_audio):
        """Test different frame lengths."""
        audio = mx.array(random_audio)

        f0_short, _ = yin(audio, sr=22050, frame_length=1024, hop_length=256)
        f0_long, _ = yin(audio, sr=22050, frame_length=2048, hop_length=256)

        # Both should produce output
        assert len(f0_short) > 0
        assert len(f0_long) > 0

    def test_hop_length(self, random_audio):
        """Test different hop lengths."""
        audio = mx.array(random_audio)

        f0_small_hop, _ = yin(audio, sr=22050, hop_length=256)
        f0_large_hop, _ = yin(audio, sr=22050, hop_length=512)

        # Smaller hop = more output frames
        assert len(f0_small_hop) > len(f0_large_hop)

    def test_threshold(self, random_audio):
        """Test threshold affects voicing decision."""
        audio = mx.array(random_audio)

        # Lower threshold = more strict voicing
        _, voiced_strict = yin(audio, sr=22050, threshold=0.1)
        _, voiced_loose = yin(audio, sr=22050, threshold=0.5)

        # Strict threshold may have fewer voiced frames
        assert isinstance(voiced_strict, mx.array)
        assert isinstance(voiced_loose, mx.array)


class TestPYIN:
    """Tests for PYIN probabilistic pitch detection."""

    def test_basic_pyin(self, random_audio):
        """Test basic PYIN returns f0, voiced_flag, and probabilities."""
        audio = mx.array(random_audio)
        f0, voiced_flag, voiced_prob = pyin(audio, sr=22050, fmin=50, fmax=2000)

        assert isinstance(f0, mx.array)
        assert isinstance(voiced_flag, mx.array)
        assert isinstance(voiced_prob, mx.array)
        assert f0.shape == voiced_flag.shape
        assert f0.shape == voiced_prob.shape

    def test_voiced_probability_range(self, random_audio):
        """Test voiced probability is between 0 and 1."""
        audio = mx.array(random_audio)
        _, _, voiced_prob = pyin(audio, sr=22050, fmin=50, fmax=2000)

        assert mx.all(voiced_prob >= 0)
        assert mx.all(voiced_prob <= 1)

    def test_pure_tone_detection(self):
        """Test PYIN detects known frequency."""
        # Generate 220 Hz sine wave
        audio = generate_sine_wave(220, 22050, 1.0)
        audio_mx = mx.array(audio)

        f0, voiced_flag, voiced_prob = pyin(audio_mx, sr=22050, fmin=100, fmax=500)

        # Find high-confidence voiced frames
        voiced_np = np.array(voiced_flag)
        f0_np = np.array(f0)

        voiced_f0 = f0_np[voiced_np]
        if len(voiced_f0) > 0:
            median_f0 = np.median(voiced_f0)
            # Should be close to 220 Hz (tightened from 20 Hz - measured: 0.50 Hz)
            assert abs(median_f0 - 220) < 5, f"Expected ~220 Hz, got {median_f0}"

    def test_resolution_parameter(self, random_audio):
        """Test resolution affects pitch candidate grid."""
        audio = mx.array(random_audio)

        # Higher resolution = finer pitch grid
        f0_coarse, _, _ = pyin(audio, sr=22050, resolution=0.5)
        f0_fine, _, _ = pyin(audio, sr=22050, resolution=0.1)

        assert isinstance(f0_coarse, mx.array)
        assert isinstance(f0_fine, mx.array)

    def test_fill_na(self, random_audio):
        """Test fill_na for unvoiced frames."""
        audio = mx.array(random_audio)

        f0_nan, _, _ = pyin(audio, sr=22050, fill_na=None)
        f0_zero, _, _ = pyin(audio, sr=22050, fill_na=0.0)

        # Both should be valid arrays
        assert isinstance(f0_nan, mx.array)
        assert isinstance(f0_zero, mx.array)


class TestPitchEdgeCases:
    """Edge case tests for pitch detection."""

    def test_short_audio(self):
        """Test with very short audio."""
        audio = mx.array(np.random.randn(512).astype(np.float32))

        # Should not crash
        f0_yin, voiced_yin = yin(audio, sr=22050, fmin=50, fmax=2000)
        f0_pyin, _, _ = pyin(audio, sr=22050, fmin=50, fmax=2000)

        assert isinstance(f0_yin, mx.array)
        assert isinstance(f0_pyin, mx.array)

    def test_silence(self):
        """Test with silent audio."""
        audio = mx.zeros((22050,))  # 1 second of silence

        f0_yin, voiced_yin = yin(audio, sr=22050)
        f0_pyin, voiced_pyin, _ = pyin(audio, sr=22050)

        # Should have low voicing in silence
        assert isinstance(f0_yin, mx.array)
        assert isinstance(f0_pyin, mx.array)

    def test_high_frequency(self):
        """Test detecting higher frequency."""
        audio = generate_sine_wave(880, 22050, 0.5)  # A5
        audio_mx = mx.array(audio)

        f0, voiced = yin(audio_mx, sr=22050, fmin=400, fmax=1000)

        voiced_np = np.array(voiced)
        f0_np = np.array(f0)
        voiced_f0 = f0_np[voiced_np]

        if len(voiced_f0) > 0:
            median_f0 = np.median(voiced_f0)
            # Tightened from 30 Hz - measured: 1.24 Hz
            assert abs(median_f0 - 880) < 5, f"Expected ~880 Hz, got {median_f0}"
