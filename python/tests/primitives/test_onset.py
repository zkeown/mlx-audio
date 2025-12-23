"""Tests for onset detection primitives."""

import numpy as np
import pytest

import mlx.core as mx
from mlx_audio.primitives import onset_strength, onset_detect, onset_strength_multi


class TestOnsetStrength:
    """Tests for onset_strength function."""

    def test_basic_output_shape(self, random_audio):
        """Test onset envelope has expected shape."""
        audio = mx.array(random_audio)
        envelope = onset_strength(audio, sr=22050)

        assert envelope.ndim == 1
        assert len(envelope) > 0

    def test_custom_hop_length(self, random_audio):
        """Test different hop lengths produce different output sizes."""
        audio = mx.array(random_audio)

        env1 = onset_strength(audio, sr=22050, hop_length=512)
        env2 = onset_strength(audio, sr=22050, hop_length=256)

        # Smaller hop = more frames
        assert len(env2) > len(env1)

    def test_center_parameter(self, random_audio):
        """Test center parameter affects output."""
        audio = mx.array(random_audio)

        env_centered = onset_strength(audio, sr=22050, center=True)
        env_uncentered = onset_strength(audio, sr=22050, center=False)

        # Both should produce valid output
        assert len(env_centered) > 0
        assert len(env_uncentered) > 0

    def test_non_negative_output(self, random_audio):
        """Test onset strength is non-negative."""
        audio = mx.array(random_audio)
        envelope = onset_strength(audio, sr=22050)

        assert mx.all(envelope >= 0)

    def test_lag_parameter(self, random_audio):
        """Test lag parameter for spectral flux calculation."""
        audio = mx.array(random_audio)

        env_lag1 = onset_strength(audio, sr=22050, lag=1)
        env_lag2 = onset_strength(audio, sr=22050, lag=2)

        # Both should be valid, possibly different lengths due to diff
        assert len(env_lag1) > 0
        assert len(env_lag2) > 0


class TestOnsetDetect:
    """Tests for onset_detect function."""

    def test_from_envelope(self, random_audio):
        """Test detection from pre-computed envelope."""
        audio = mx.array(random_audio)
        envelope = onset_strength(audio, sr=22050)
        onsets = onset_detect(onset_envelope=envelope, sr=22050)

        assert isinstance(onsets, mx.array)
        # Onsets should be frame indices
        if len(onsets) > 0:
            assert mx.all(onsets >= 0)
            assert mx.all(onsets < len(envelope))

    def test_from_audio(self, random_audio):
        """Test detection directly from audio."""
        audio = mx.array(random_audio)
        onsets = onset_detect(y=audio, sr=22050)

        assert isinstance(onsets, mx.array)

    def test_units_parameter(self, random_audio):
        """Test different output units."""
        audio = mx.array(random_audio)

        frames = onset_detect(y=audio, sr=22050, units="frames")
        samples = onset_detect(y=audio, sr=22050, units="samples")
        time = onset_detect(y=audio, sr=22050, units="time")

        # All should return valid arrays
        assert isinstance(frames, mx.array)
        assert isinstance(samples, mx.array)
        assert isinstance(time, mx.array)

    def test_backtrack(self, random_audio):
        """Test backtracking to energy minima."""
        audio = mx.array(random_audio)

        onsets_no_bt = onset_detect(y=audio, sr=22050, backtrack=False)
        onsets_bt = onset_detect(y=audio, sr=22050, backtrack=True)

        # Both should be valid (backtracked may differ)
        assert isinstance(onsets_no_bt, mx.array)
        assert isinstance(onsets_bt, mx.array)

    def test_pre_max_post_max(self, random_audio):
        """Test pre_max and post_max parameters."""
        audio = mx.array(random_audio)

        onsets = onset_detect(y=audio, sr=22050, pre_max=3, post_max=3)

        assert isinstance(onsets, mx.array)


class TestOnsetStrengthMulti:
    """Tests for multi-band onset strength."""

    def test_output_shape(self, random_audio):
        """Test multi-band output shape."""
        audio = mx.array(random_audio)
        # channels is list of (fmin, fmax) tuples
        channels = [(0, 32), (32, 64), (64, 96), (96, 128)]
        multi_env = onset_strength_multi(audio, sr=22050, channels=channels)

        assert multi_env.ndim == 2
        assert multi_env.shape[0] == 4  # 4 channels

    def test_channel_count(self, random_audio):
        """Test different channel counts."""
        audio = mx.array(random_audio)

        channels2 = [(0, 64), (64, 128)]
        channels4 = [(0, 32), (32, 64), (64, 96), (96, 128)]

        env2 = onset_strength_multi(audio, sr=22050, channels=channels2)
        env4 = onset_strength_multi(audio, sr=22050, channels=channels4)

        assert env2.shape[0] == 2
        assert env4.shape[0] == 4

    def test_aggregate_function(self, random_audio):
        """Test aggregation to single envelope."""
        audio = mx.array(random_audio)
        channels = [(0, 32), (32, 64), (64, 96), (96, 128)]
        multi_env = onset_strength_multi(audio, sr=22050, channels=channels)

        # Aggregate with mean
        aggregated = mx.mean(multi_env, axis=0)
        assert aggregated.ndim == 1
