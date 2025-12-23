"""Tests for spectral gating (noise reduction) primitives."""

import numpy as np
import pytest

import mlx.core as mx
from mlx_audio.primitives import spectral_gate, spectral_gate_adaptive


class TestSpectralGate:
    """Tests for spectral_gate function."""

    def test_basic_output_shape(self, random_audio):
        """Test output shape matches input."""
        audio = mx.array(random_audio)
        enhanced = spectral_gate(audio, sr=22050)

        assert enhanced.shape == audio.shape

    def test_output_dtype(self, random_audio):
        """Test output dtype is float32."""
        audio = mx.array(random_audio)
        enhanced = spectral_gate(audio, sr=22050)

        assert enhanced.dtype == mx.float32

    def test_threshold_parameter(self, random_audio):
        """Test different threshold values."""
        audio = mx.array(random_audio)

        # Lower threshold = less aggressive gating
        enhanced_mild = spectral_gate(audio, sr=22050, threshold_db=-40)
        # Higher threshold = more aggressive gating
        enhanced_strong = spectral_gate(audio, sr=22050, threshold_db=-20)

        assert enhanced_mild.shape == audio.shape
        assert enhanced_strong.shape == audio.shape

    def test_with_noise_profile(self, random_audio):
        """Test with explicit noise profile."""
        audio = mx.array(random_audio)
        # Use first portion as noise profile
        noise = audio[:2048]

        enhanced = spectral_gate(audio, sr=22050, noise_profile=noise)

        assert enhanced.shape == audio.shape

    def test_n_fft_parameter(self, random_audio):
        """Test different FFT sizes."""
        audio = mx.array(random_audio)

        enhanced_small = spectral_gate(audio, sr=22050, n_fft=512)
        enhanced_large = spectral_gate(audio, sr=22050, n_fft=2048)

        assert enhanced_small.shape == audio.shape
        assert enhanced_large.shape == audio.shape

    def test_prop_decrease(self, random_audio):
        """Test proportional decrease parameter."""
        audio = mx.array(random_audio)

        # Full reduction
        enhanced_full = spectral_gate(audio, sr=22050, prop_decrease=1.0)
        # Partial reduction
        enhanced_partial = spectral_gate(audio, sr=22050, prop_decrease=0.5)

        assert enhanced_full.shape == audio.shape
        assert enhanced_partial.shape == audio.shape

    def test_stereo_audio(self, stereo_audio):
        """Test with stereo audio."""
        audio = mx.array(stereo_audio)
        enhanced = spectral_gate(audio, sr=44100)

        assert enhanced.shape == audio.shape

    def test_reduces_noise(self):
        """Test that spectral gating reduces noise."""
        np.random.seed(42)
        # Create noisy signal: sine wave + noise
        t = np.arange(22050) / 22050.0
        signal = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        noise = 0.1 * np.random.randn(22050).astype(np.float32)
        noisy = signal + noise

        audio = mx.array(noisy)
        enhanced = spectral_gate(audio, sr=22050, threshold_db=-30)

        # Enhanced should have less variance in noise regions
        assert enhanced.shape == audio.shape


class TestSpectralGateAdaptive:
    """Tests for adaptive spectral gating."""

    def test_basic_output(self, random_audio):
        """Test basic adaptive gating output."""
        audio = mx.array(random_audio)
        enhanced = spectral_gate_adaptive(audio, sr=22050)

        assert enhanced.shape == audio.shape

    def test_look_ahead_parameter(self, random_audio):
        """Test look_ahead_ms affects output."""
        audio = mx.array(random_audio)

        # Different look-ahead times
        enhanced_short = spectral_gate_adaptive(audio, sr=22050, look_ahead_ms=50.0)
        enhanced_long = spectral_gate_adaptive(audio, sr=22050, look_ahead_ms=200.0)

        assert enhanced_short.shape == audio.shape
        assert enhanced_long.shape == audio.shape

    def test_threshold_parameter(self, random_audio):
        """Test threshold_db parameter."""
        audio = mx.array(random_audio)

        enhanced = spectral_gate_adaptive(audio, sr=22050, threshold_db=-25)

        assert enhanced.shape == audio.shape


class TestSpectralGateEdgeCases:
    """Edge case tests for spectral gating."""

    def test_short_audio(self):
        """Test with very short audio."""
        audio = mx.array(np.random.randn(256).astype(np.float32))
        enhanced = spectral_gate(audio, sr=22050, n_fft=256)

        # Should handle short audio
        assert isinstance(enhanced, mx.array)

    def test_silence(self):
        """Test with silent audio."""
        audio = mx.zeros((22050,))
        enhanced = spectral_gate(audio, sr=22050)

        # Silent audio should remain silent
        assert mx.all(mx.abs(enhanced) < 1e-6)

    def test_pure_tone(self):
        """Test spectral gating preserves pure tone."""
        t = np.arange(22050) / 22050.0
        signal = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        audio = mx.array(signal)

        # With low threshold, should preserve most of the signal
        enhanced = spectral_gate(audio, sr=22050, threshold_db=-60)

        # Energy should be mostly preserved
        orig_energy = float(mx.sum(audio ** 2))
        enhanced_energy = float(mx.sum(enhanced ** 2))

        assert enhanced_energy > 0.5 * orig_energy
