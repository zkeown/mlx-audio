"""Tests for beat tracking and tempo estimation primitives."""

import numpy as np
import pytest

import mlx.core as mx
from mlx_audio.primitives import tempo, beat_track, plp


class TestTempo:
    """Tests for tempo estimation."""

    def test_basic_tempo(self, random_audio):
        """Test basic tempo estimation returns valid BPM."""
        audio = mx.array(random_audio)
        bpm = tempo(audio, sr=22050)

        # tempo returns array of tempos
        assert isinstance(bpm, mx.array)
        if len(bpm) > 0:
            # BPM should be in reasonable range
            assert mx.all(bpm > 0)
            assert mx.all(bpm < 500)  # Reasonable upper bound

    def test_from_onset_envelope(self, random_audio):
        """Test tempo from pre-computed onset envelope."""
        from mlx_audio.primitives import onset_strength

        audio = mx.array(random_audio)
        envelope = onset_strength(audio, sr=22050)
        bpm = tempo(onset_envelope=envelope, sr=22050)

        assert isinstance(bpm, mx.array)

    def test_aggregate_single(self, random_audio):
        """Test aggregate=True returns single tempo."""
        audio = mx.array(random_audio)
        bpm = tempo(audio, sr=22050, aggregate=True)

        # With aggregate, should be single value or single-element array
        assert bpm.size >= 1

    def test_ac_size(self, random_audio):
        """Test ac_size parameter."""
        audio = mx.array(random_audio)

        # Test with different autocorrelation sizes
        bpm = tempo(audio, sr=22050, ac_size=4.0)

        assert isinstance(bpm, mx.array)

    def test_max_tempo(self, random_audio):
        """Test max_tempo parameter."""
        audio = mx.array(random_audio)
        bpm = tempo(audio, sr=22050, max_tempo=200)

        assert isinstance(bpm, mx.array)


class TestBeatTrack:
    """Tests for beat tracking."""

    def test_basic_beat_track(self, random_audio):
        """Test basic beat tracking returns tempo and beats."""
        audio = mx.array(random_audio)
        estimated_tempo, beats = beat_track(audio, sr=22050)

        assert isinstance(estimated_tempo, (float, mx.array))
        assert isinstance(beats, mx.array)

    def test_beats_increasing(self, random_audio):
        """Test that beats are in increasing order."""
        audio = mx.array(random_audio)
        _, beats = beat_track(audio, sr=22050)

        if len(beats) > 1:
            diffs = beats[1:] - beats[:-1]
            assert mx.all(diffs > 0)

    def test_units_frames(self, random_audio):
        """Test beat output in frames."""
        audio = mx.array(random_audio)
        _, beats = beat_track(audio, sr=22050, units="frames")

        assert isinstance(beats, mx.array)
        if len(beats) > 0:
            assert mx.all(beats >= 0)

    def test_units_time(self, random_audio):
        """Test beat output in seconds."""
        audio = mx.array(random_audio)
        _, beats = beat_track(audio, sr=22050, units="time")

        assert isinstance(beats, mx.array)
        if len(beats) > 0:
            # Times should be positive and less than audio duration
            assert mx.all(beats >= 0)
            duration = len(random_audio) / 22050
            assert mx.all(beats <= duration + 0.1)  # Small tolerance

    def test_units_samples(self, random_audio):
        """Test beat output in samples."""
        audio = mx.array(random_audio)
        _, beats = beat_track(audio, sr=22050, units="samples")

        assert isinstance(beats, mx.array)
        if len(beats) > 0:
            assert mx.all(beats >= 0)
            assert mx.all(beats < len(random_audio) + 512)

    def test_tightness(self, random_audio):
        """Test tightness parameter for beat tracking."""
        audio = mx.array(random_audio)
        bpm, beats = beat_track(audio, sr=22050, tightness=100)

        assert isinstance(beats, mx.array)
        # Beats should be in order
        if len(beats) > 2:
            diffs = np.array(beats[1:] - beats[:-1])
            # Allow some variation
            assert np.all(diffs > 0)

    def test_trim(self, random_audio):
        """Test trim parameter for leading/trailing beats."""
        audio = mx.array(random_audio)
        _, beats_trimmed = beat_track(audio, sr=22050, trim=True)
        _, beats_untrimmed = beat_track(audio, sr=22050, trim=False)

        assert isinstance(beats_trimmed, mx.array)
        assert isinstance(beats_untrimmed, mx.array)


class TestPLP:
    """Tests for predominant local pulse estimation."""

    def test_basic_plp(self, random_audio):
        """Test basic PLP computation."""
        audio = mx.array(random_audio)
        pulse = plp(audio, sr=22050)

        assert isinstance(pulse, mx.array)
        assert pulse.ndim == 1
        assert len(pulse) > 0

    def test_plp_from_envelope(self, random_audio):
        """Test PLP from pre-computed onset envelope."""
        from mlx_audio.primitives import onset_strength

        audio = mx.array(random_audio)
        envelope = onset_strength(audio, sr=22050)
        pulse = plp(onset_envelope=envelope, sr=22050)

        assert isinstance(pulse, mx.array)

    def test_tempo_range(self, random_audio):
        """Test PLP with constrained tempo range."""
        audio = mx.array(random_audio)
        pulse = plp(audio, sr=22050, tempo_min=60, tempo_max=180)

        assert isinstance(pulse, mx.array)
        assert len(pulse) > 0

    def test_plp_normalized(self, random_audio):
        """Test PLP output range."""
        audio = mx.array(random_audio)
        pulse = plp(audio, sr=22050)

        # PLP should be roughly normalized
        assert mx.all(pulse >= 0)
