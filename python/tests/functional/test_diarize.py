"""Tests for diarize() functional API."""

import numpy as np
import pytest


class TestDiarizeBasic:
    """Basic tests for diarize function."""

    @pytest.fixture
    def audio_5sec(self):
        """5 seconds of random audio at 16kHz."""
        np.random.seed(42)
        return np.random.randn(80000).astype(np.float32)

    def test_basic_diarize(self, audio_5sec):
        """Test basic diarization call."""
        from mlx_audio.functional.diarize import diarize

        result = diarize(audio_5sec, sample_rate=16000)

        assert hasattr(result, "segments")
        assert hasattr(result, "num_speakers")
        assert isinstance(result.segments, list)

    def test_returns_diarization_result(self, audio_5sec):
        """Test that diarize returns DiarizationResult."""
        from mlx_audio.functional.diarize import diarize
        from mlx_audio.types.results import DiarizationResult

        result = diarize(audio_5sec, sample_rate=16000)

        assert isinstance(result, DiarizationResult)

    def test_fixed_num_speakers(self, audio_5sec):
        """Test diarization with fixed number of speakers."""
        from mlx_audio.functional.diarize import diarize

        result = diarize(audio_5sec, sample_rate=16000, num_speakers=2)

        # Should have at most 2 speakers
        assert result.num_speakers <= 2

    def test_segments_have_speaker_labels(self, audio_5sec):
        """Test that segments have speaker labels."""
        from mlx_audio.functional.diarize import diarize

        result = diarize(audio_5sec, sample_rate=16000)

        for seg in result.segments:
            assert hasattr(seg, "speaker")
            assert seg.speaker.startswith("SPEAKER_")

    def test_segments_have_timing(self, audio_5sec):
        """Test that segments have timing information."""
        from mlx_audio.functional.diarize import diarize

        result = diarize(audio_5sec, sample_rate=16000)

        for seg in result.segments:
            assert hasattr(seg, "start")
            assert hasattr(seg, "end")
            assert seg.start < seg.end
            assert seg.start >= 0


class TestDiarizeWithEmbeddings:
    """Tests for diarization with speaker embeddings."""

    @pytest.fixture
    def audio_5sec(self):
        """5 seconds of random audio at 16kHz."""
        np.random.seed(42)
        return np.random.randn(80000).astype(np.float32)

    def test_return_embeddings(self, audio_5sec):
        """Test returning speaker embeddings."""
        from mlx_audio.functional.diarize import diarize

        result = diarize(audio_5sec, sample_rate=16000, return_embeddings=True)

        assert result.speaker_embeddings is not None
        assert isinstance(result.speaker_embeddings, dict)


class TestDiarizationResult:
    """Tests for DiarizationResult type."""

    @pytest.fixture
    def audio_5sec(self):
        """5 seconds of random audio at 16kHz."""
        np.random.seed(42)
        return np.random.randn(80000).astype(np.float32)

    def test_result_speakers_property(self, audio_5sec):
        """Test speakers property lists unique speakers."""
        from mlx_audio.functional.diarize import diarize

        result = diarize(audio_5sec, sample_rate=16000)

        speakers = result.speakers
        assert isinstance(speakers, list)
        # Should be sorted
        assert speakers == sorted(speakers)

    def test_result_total_duration(self, audio_5sec):
        """Test total_duration property."""
        from mlx_audio.functional.diarize import diarize

        result = diarize(audio_5sec, sample_rate=16000)

        duration = result.total_duration
        assert duration >= 0
        # Should not exceed audio duration
        audio_duration = len(audio_5sec) / 16000
        assert duration <= audio_duration + 0.1

    def test_result_to_rttm(self, audio_5sec):
        """Test RTTM export."""
        from mlx_audio.functional.diarize import diarize

        result = diarize(audio_5sec, sample_rate=16000)

        rttm = result.to_rttm(filename="test")
        assert isinstance(rttm, str)
        # RTTM format has SPEAKER lines
        if result.segments:
            assert "SPEAKER" in rttm

    def test_get_speaker_segments(self, audio_5sec):
        """Test getting segments for specific speaker."""
        from mlx_audio.functional.diarize import diarize

        result = diarize(audio_5sec, sample_rate=16000)

        if result.speakers:
            speaker = result.speakers[0]
            segs = result.get_speaker_segments(speaker)
            assert isinstance(segs, list)
            for seg in segs:
                assert seg.speaker == speaker


class TestSpeakerSegment:
    """Tests for SpeakerSegment type."""

    def test_segment_duration_property(self):
        """Test duration property calculation."""
        from mlx_audio.types.results import SpeakerSegment

        seg = SpeakerSegment(
            speaker="SPEAKER_00",
            start=1.0,
            end=3.5,
        )

        assert seg.duration == 2.5

    def test_segment_with_text(self):
        """Test segment with optional text."""
        from mlx_audio.types.results import SpeakerSegment

        seg = SpeakerSegment(
            speaker="SPEAKER_01",
            start=0.0,
            end=1.0,
            text="Hello world",
        )

        assert seg.text == "Hello world"
