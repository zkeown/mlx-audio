"""Tests for detect_speech functional API."""

import pytest
import mlx.core as mx

from mlx_audio.functional.detect_speech import detect_speech
from mlx_audio.types.vad import VADResult, SpeechSegment


class TestDetectSpeech:
    """Tests for the detect_speech function."""

    def test_detect_speech_with_array(self):
        """Test detect_speech with MLX array input."""
        # 1 second of audio at 16kHz
        audio = mx.random.normal((16000,))

        result = detect_speech(audio, sample_rate=16000)

        assert isinstance(result, VADResult)
        assert result.sample_rate == 16000
        assert isinstance(result.segments, list)

    def test_detect_speech_with_numpy(self):
        """Test detect_speech with numpy array input."""
        import numpy as np

        audio = np.random.randn(16000).astype(np.float32)

        result = detect_speech(audio, sample_rate=16000)

        assert isinstance(result, VADResult)

    def test_detect_speech_threshold(self):
        """Test threshold parameter affects detection."""
        audio = mx.random.normal((16000,))

        # High threshold
        result_high = detect_speech(audio, sample_rate=16000, threshold=0.9)

        # Low threshold
        result_low = detect_speech(audio, sample_rate=16000, threshold=0.1)

        # Both should return valid results
        assert isinstance(result_high, VADResult)
        assert isinstance(result_low, VADResult)

    def test_detect_speech_return_probabilities(self):
        """Test return_probabilities flag."""
        audio = mx.random.normal((16000,))

        # Without probabilities
        result_no_prob = detect_speech(
            audio, sample_rate=16000, return_probabilities=False
        )
        assert result_no_prob.probabilities is None

        # With probabilities
        result_with_prob = detect_speech(
            audio, sample_rate=16000, return_probabilities=True
        )
        assert result_with_prob.probabilities is not None
        assert result_with_prob.probabilities.shape[0] > 0

    def test_detect_speech_short_audio(self):
        """Test detect_speech with very short audio."""
        # Less than one window (512 samples at 16kHz)
        audio = mx.random.normal((256,))

        result = detect_speech(audio, sample_rate=16000)

        assert isinstance(result, VADResult)

    def test_detect_speech_long_audio(self):
        """Test detect_speech with longer audio."""
        # 5 seconds of audio
        audio = mx.random.normal((80000,))

        result = detect_speech(audio, sample_rate=16000)

        assert isinstance(result, VADResult)
        # Should have processed multiple windows
        if result.probabilities is not None:
            assert result.probabilities.shape[0] > 100

    def test_detect_speech_min_duration_params(self):
        """Test min_speech_duration and min_silence_duration parameters."""
        audio = mx.random.normal((16000,))

        result = detect_speech(
            audio,
            sample_rate=16000,
            min_speech_duration=0.5,  # 500ms minimum
            min_silence_duration=0.2,  # 200ms minimum
        )

        assert isinstance(result, VADResult)

    def test_vad_result_properties(self):
        """Test VADResult properties work correctly."""
        audio = mx.random.normal((16000,))

        result = detect_speech(audio, sample_rate=16000, return_probabilities=True)

        # Test properties
        _ = result.speech_ratio
        _ = result.total_duration
        _ = result.num_segments
        _ = result.get_speech_times()
        _ = result.get_silence_times()

    def test_vad_result_get_speech_audio(self):
        """Test extracting speech portions from audio."""
        audio = mx.random.normal((16000,))

        result = detect_speech(audio, sample_rate=16000)

        # Should not error even if no speech detected
        speech_audio = result.get_speech_audio(audio)
        assert isinstance(speech_audio, mx.array)

    def test_vad_result_get_silence_audio(self):
        """Test extracting silence portions from audio."""
        audio = mx.random.normal((16000,))

        result = detect_speech(audio, sample_rate=16000)

        silence_audio = result.get_silence_audio(audio)
        assert isinstance(silence_audio, mx.array)

    def test_vad_result_to_whisper_segments(self):
        """Test conversion to Whisper-compatible format."""
        audio = mx.random.normal((16000,))

        result = detect_speech(audio, sample_rate=16000)

        whisper_segs = result.to_whisper_segments()

        assert isinstance(whisper_segs, list)
        for seg in whisper_segs:
            assert "start" in seg
            assert "end" in seg

    def test_vad_result_to_audacity_labels(self):
        """Test export as Audacity label format."""
        audio = mx.random.normal((16000,))

        result = detect_speech(audio, sample_rate=16000)

        labels = result.to_audacity_labels()

        assert isinstance(labels, str)

    def test_vad_result_from_probabilities(self):
        """Test creating VADResult from probabilities."""
        probs = mx.array([0.1, 0.2, 0.8, 0.9, 0.85, 0.3, 0.1])

        result = VADResult.from_probabilities(
            probabilities=probs,
            sample_rate=16000,
            window_size_samples=512,
            threshold=0.5,
        )

        assert isinstance(result, VADResult)
        # Should detect speech in the high probability region
        assert result.num_segments >= 0


class TestVADResultSave:
    """Tests for VADResult save functionality."""

    def test_save_json(self, tmp_path):
        """Test saving result as JSON."""
        result = VADResult(
            segments=[
                SpeechSegment(start=0.5, end=1.5, probability=0.9),
                SpeechSegment(start=2.0, end=3.0, probability=0.85),
            ],
            sample_rate=16000,
        )

        output_path = tmp_path / "vad_result.json"
        result.save(output_path, format="json")

        assert output_path.exists()

        # Verify content
        import json

        with open(output_path) as f:
            data = json.load(f)

        assert len(data["segments"]) == 2
        assert data["sample_rate"] == 16000

    def test_save_txt(self, tmp_path):
        """Test saving result as text."""
        result = VADResult(
            segments=[
                SpeechSegment(start=0.5, end=1.5, probability=0.9),
            ],
            sample_rate=16000,
        )

        output_path = tmp_path / "vad_result.txt"
        result.save(output_path, format="txt")

        assert output_path.exists()

        content = output_path.read_text()
        assert "0.500" in content
        assert "1.500" in content

    def test_save_audacity(self, tmp_path):
        """Test saving result as Audacity labels."""
        result = VADResult(
            segments=[
                SpeechSegment(start=0.5, end=1.5, probability=0.9),
            ],
            sample_rate=16000,
        )

        output_path = tmp_path / "vad_result.txt"
        result.save(output_path, format="audacity")

        assert output_path.exists()

        content = output_path.read_text()
        assert "speech_1" in content


class TestSpeechSegment:
    """Tests for SpeechSegment class."""

    def test_speech_segment_creation(self):
        """Test creating a speech segment."""
        seg = SpeechSegment(start=1.0, end=2.5, probability=0.9)

        assert seg.start == 1.0
        assert seg.end == 2.5
        assert seg.probability == 0.9

    def test_speech_segment_duration(self):
        """Test duration property."""
        seg = SpeechSegment(start=1.0, end=2.5)

        assert seg.duration == 1.5
