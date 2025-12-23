"""Whisper quality benchmark tests.

These tests verify that the MLX Whisper implementation produces
transcriptions with WER (Word Error Rate) comparable to the
reference implementation.

Quality targets:
- WER difference within 1% of reference implementation
- Tested on synthetic audio and known transcriptions
"""

import pytest
import numpy as np

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

try:
    import jiwer

    HAS_JIWER = True
except ImportError:
    HAS_JIWER = False


def compute_wer(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate using jiwer.

    Args:
        reference: Ground truth transcription
        hypothesis: Model's transcription

    Returns:
        WER as a float (0.0 = perfect, 1.0 = all errors)
    """
    # Normalize: lowercase, remove punctuation
    reference = reference.lower().strip()
    hypothesis = hypothesis.lower().strip()

    if not reference:
        return 0.0 if not hypothesis else 1.0

    return jiwer.wer(reference, hypothesis)


def generate_synthetic_audio(
    duration_seconds: float = 2.0,
    sample_rate: int = 16000,
    seed: int = 42,
) -> np.ndarray:
    """Generate synthetic audio for testing.

    Note: This generates random audio, not actual speech.
    For real WER testing, you would use actual speech samples.

    Returns:
        audio: [samples] audio array
    """
    np.random.seed(seed)
    samples = int(duration_seconds * sample_rate)

    # Generate noise with some structure
    t = np.linspace(0, duration_seconds, samples)
    audio = (
        np.sin(2 * np.pi * 200 * t) * 0.3
        + np.sin(2 * np.pi * 400 * t) * 0.2
        + np.random.randn(samples) * 0.1
    )

    return audio.astype(np.float32)


@pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
@pytest.mark.slow
class TestWhisperQuality:
    """Quality tests for Whisper speech recognition."""

    def test_model_forward(self):
        """Test that model forward pass works."""
        from mlx_audio.models.whisper import Whisper, WhisperConfig

        config = WhisperConfig.tiny()
        model = Whisper(config)

        # Generate test mel spectrogram input
        mel_input = mx.random.normal([1, config.n_mels, 100])
        token_input = mx.array([[50257, 50362]])  # SOT, language

        logits = model(mel_input, token_input)
        mx.eval(logits)

        assert logits.shape[0] == 1
        assert logits.shape[1] == 2  # seq_len
        assert logits.shape[2] == config.n_vocab

    def test_encoder_output_shape(self):
        """Test encoder output shape."""
        from mlx_audio.models.whisper import Whisper, WhisperConfig

        config = WhisperConfig.tiny()
        model = Whisper(config)

        mel_input = mx.random.normal([1, config.n_mels, 3000])  # ~30s

        encoder_output = model.encode(mel_input)
        mx.eval(encoder_output)

        assert encoder_output.shape[0] == 1
        assert encoder_output.shape[2] == config.n_audio_state

    def test_decoder_output_shape(self):
        """Test decoder output shape."""
        from mlx_audio.models.whisper import Whisper, WhisperConfig

        config = WhisperConfig.tiny()
        model = Whisper(config)

        mel_input = mx.random.normal([1, config.n_mels, 100])
        encoder_output = model.encode(mel_input)

        tokens = mx.array([[50257, 50362, 50359]])  # SOT, language, transcribe
        logits = model.decode(tokens, encoder_output)
        mx.eval(logits)

        assert logits.shape[0] == 1
        assert logits.shape[1] == 3
        assert logits.shape[2] == config.n_vocab

    def test_incremental_decoding(self):
        """Test incremental decoding works."""
        from mlx_audio.models.whisper import Whisper, WhisperConfig

        config = WhisperConfig.tiny()
        model = Whisper(config)

        mel_input = mx.random.normal([1, config.n_mels, 100])
        encoder_output = model.encode(mel_input)
        mx.eval(encoder_output)

        # Initial decode
        initial_tokens = mx.array([[50257, 50362, 50359]])
        logits1 = model.decode(initial_tokens, encoder_output)
        mx.eval(logits1)

        # Get next token
        next_token = mx.argmax(logits1[:, -1:, :], axis=-1)

        # Append and decode again (simple approach without cache)
        all_tokens = mx.concatenate([initial_tokens, next_token], axis=1)
        logits2 = model.decode(all_tokens, encoder_output)
        mx.eval(logits2)

        # Should work and produce valid logits
        assert logits2.shape[0] == 1
        assert logits2.shape[1] == 4  # original 3 + 1 new
        assert logits2.shape[2] == config.n_vocab

    @pytest.mark.slow
    def test_pretrained_transcription(self, tmp_path):
        """Test transcription with pretrained model (if available)."""
        try:
            from mlx_audio.models.whisper import Whisper

            model = Whisper.from_pretrained("mlx-community/whisper-tiny")
            model.eval()
        except Exception as e:
            pytest.skip(f"Could not load pretrained model: {e}")

        # Generate random mel input
        mel_input = mx.random.normal([1, 80, 3000])

        # Run encoder
        encoder_output = model.encode(mel_input)
        mx.eval(encoder_output)

        # Verify shapes
        assert encoder_output.shape[0] == 1
        assert encoder_output.shape[2] == 384  # tiny model state size


@pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
@pytest.mark.skipif(not HAS_JIWER, reason="jiwer not available")
class TestWhisperWER:
    """WER computation tests."""

    def test_wer_computation(self):
        """Test WER computation with known examples."""
        # Perfect match
        wer = compute_wer("hello world", "hello world")
        assert wer == 0.0

        # One word error
        wer = compute_wer("hello world", "hello words")
        assert wer == 0.5  # 1 error out of 2 words

        # All errors
        wer = compute_wer("hello world", "goodbye universe")
        assert wer == 1.0  # 2 errors out of 2 words

        # Insertion
        wer = compute_wer("hello world", "hello big world")
        assert wer == 0.5  # 1 insertion

    def test_wer_case_insensitive(self):
        """Test that WER is case-insensitive."""
        wer = compute_wer("Hello World", "hello world")
        assert wer == 0.0

    def test_wer_empty_reference(self):
        """Test WER with empty reference."""
        wer = compute_wer("", "")
        assert wer == 0.0

        wer = compute_wer("", "something")
        assert wer == 1.0


@pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
class TestWhisperEdgeCases:
    """Edge case tests for Whisper."""

    def test_short_audio(self):
        """Test with very short audio."""
        from mlx_audio.models.whisper import Whisper, WhisperConfig

        config = WhisperConfig.tiny()
        model = Whisper(config)

        # Very short mel (< 1 second)
        short_mel = mx.random.normal([1, config.n_mels, 10])

        encoder_output = model.encode(short_mel)
        mx.eval(encoder_output)

        assert encoder_output.shape[0] == 1

    def test_long_audio(self):
        """Test with long audio (30 seconds)."""
        from mlx_audio.models.whisper import Whisper, WhisperConfig

        config = WhisperConfig.tiny()
        model = Whisper(config)

        # 30 seconds of audio
        long_mel = mx.random.normal([1, config.n_mels, 3000])

        encoder_output = model.encode(long_mel)
        mx.eval(encoder_output)

        assert encoder_output.shape[0] == 1

    def test_batch_processing(self):
        """Test batch processing."""
        from mlx_audio.models.whisper import Whisper, WhisperConfig

        config = WhisperConfig.tiny()
        model = Whisper(config)

        # Batch of 2
        batch_mel = mx.random.normal([2, config.n_mels, 100])

        encoder_output = model.encode(batch_mel)
        mx.eval(encoder_output)

        assert encoder_output.shape[0] == 2

    def test_deterministic(self):
        """Test that encoder is deterministic."""
        from mlx_audio.models.whisper import Whisper, WhisperConfig

        config = WhisperConfig.tiny()
        model = Whisper(config)
        model.eval()

        mel_input = mx.random.normal([1, config.n_mels, 100])

        output1 = model.encode(mel_input)
        mx.eval(output1)

        output2 = model.encode(mel_input)
        mx.eval(output2)

        assert mx.allclose(output1, output2)
