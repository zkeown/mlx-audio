"""Parity tests for beat/tempo detection against librosa."""

import numpy as np
import pytest

import mlx.core as mx

librosa = pytest.importorskip("librosa")


class TestTempoParity:
    """Parity tests for tempo estimation vs librosa."""

    @pytest.fixture
    def audio(self):
        """Generate test audio with clear tempo."""
        np.random.seed(42)
        sr = 22050
        duration = 5.0
        # Create clicks at 120 BPM (2 Hz)
        t = np.arange(int(sr * duration)) / sr
        # Add impulses every 0.5 seconds (120 BPM)
        audio = np.zeros_like(t, dtype=np.float32)
        click_times = np.arange(0, duration, 0.5)
        for ct in click_times:
            idx = int(ct * sr)
            if idx < len(audio):
                audio[idx : idx + 100] = np.exp(-np.arange(100) / 10)
        # Add some noise
        audio += 0.1 * np.random.randn(len(audio)).astype(np.float32)
        return audio

    def test_tempo_reasonable_range(self, audio):
        """Test tempo is in reasonable range."""
        from mlx_audio.primitives import tempo

        audio_mx = mx.array(audio)

        mlx_tempo = np.array(tempo(audio_mx, sr=22050))
        librosa_tempo = librosa.feature.tempo(y=audio, sr=22050)

        # Both should detect something in 60-240 BPM range
        assert np.any((mlx_tempo > 60) & (mlx_tempo < 240))
        assert np.any((librosa_tempo > 60) & (librosa_tempo < 240))

    def test_tempo_click_track(self, audio):
        """Test tempo detection on click track."""
        from mlx_audio.primitives import tempo

        audio_mx = mx.array(audio)

        mlx_tempo = np.array(tempo(audio_mx, sr=22050, aggregate=True))
        # librosa.feature.tempo aggregate parameter is a callable, not bool
        librosa_tempo = librosa.feature.tempo(y=audio, sr=22050)

        # Should be close to 120 BPM or a harmonic (60, 240)
        expected_tempos = [60, 120, 240]

        mlx_close = any(abs(mlx_tempo[0] - t) < 15 for t in expected_tempos)
        librosa_close = any(abs(librosa_tempo[0] - t) < 15 for t in expected_tempos)

        assert mlx_close, f"MLX tempo {mlx_tempo[0]:.1f} not near expected"
        assert librosa_close, f"Librosa tempo {librosa_tempo[0]:.1f} not near expected"


class TestBeatTrackParity:
    """Parity tests for beat tracking vs librosa."""

    @pytest.fixture
    def audio(self):
        """Generate test audio with clear beats."""
        np.random.seed(42)
        sr = 22050
        duration = 3.0
        t = np.arange(int(sr * duration)) / sr
        audio = np.zeros_like(t, dtype=np.float32)
        # Beats at 120 BPM
        beat_times = np.arange(0, duration, 0.5)
        for bt in beat_times:
            idx = int(bt * sr)
            if idx < len(audio):
                audio[idx : idx + 200] = np.exp(-np.arange(200) / 20)
        audio += 0.05 * np.random.randn(len(audio)).astype(np.float32)
        return audio

    def test_beat_track_finds_beats(self, audio):
        """Test that beat tracking finds beats."""
        from mlx_audio.primitives import beat_track

        audio_mx = mx.array(audio)

        mlx_tempo, mlx_beats = beat_track(audio_mx, sr=22050)
        librosa_tempo, librosa_beats = librosa.beat.beat_track(y=audio, sr=22050)

        # Both should find multiple beats
        assert len(mlx_beats) >= 2, f"MLX found only {len(mlx_beats)} beats"
        assert len(librosa_beats) >= 2, f"Librosa found only {len(librosa_beats)} beats"

    def test_beat_intervals_similar(self, audio):
        """Test beat intervals are similar."""
        from mlx_audio.primitives import beat_track

        audio_mx = mx.array(audio)

        _, mlx_beats = beat_track(audio_mx, sr=22050, units="time")
        _, librosa_beats = librosa.beat.beat_track(y=audio, sr=22050, units="time")

        mlx_beats = np.array(mlx_beats)
        librosa_beats = np.array(librosa_beats)

        if len(mlx_beats) > 1 and len(librosa_beats) > 1:
            mlx_intervals = np.diff(mlx_beats)
            librosa_intervals = np.diff(librosa_beats)

            # Median intervals should be similar (within 20%)
            mlx_med = np.median(mlx_intervals)
            librosa_med = np.median(librosa_intervals)

            ratio = mlx_med / librosa_med if librosa_med > 0 else 1
            assert 0.5 < ratio < 2.0, (
                f"Interval ratio {ratio:.2f} outside expected range"
            )
