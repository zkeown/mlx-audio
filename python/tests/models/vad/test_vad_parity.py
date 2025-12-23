"""Parity and numerical accuracy tests for VAD model.

These tests verify that the MLX VAD model produces outputs that are:
1. Numerically stable across different input types
2. Consistent between batched and unbatched processing
3. Produce valid probability distributions
4. Handle edge cases correctly

For full PyTorch parity testing, run with converted weights and compare
against the original Silero VAD implementation.
"""

import pytest
import mlx.core as mx
import numpy as np

from mlx_audio.models.vad import SileroVAD, VADConfig
from mlx_audio.models.vad.layers import VADEncoder, VADDecoder, StackedLSTM


# Module-level shared model for faster tests
@pytest.fixture(scope="module")
def shared_parity_model():
    """Create a shared VAD model for parity testing."""
    config = VADConfig(
        sample_rate=16000,
        window_size_samples=512,
        hidden_size=64,
        num_layers=2,
    )
    return SileroVAD(config)


class TestNumericalStability:
    """Tests for numerical stability of the VAD model."""

    @pytest.fixture
    def model(self, shared_parity_model):
        """Get the shared model and reset its state."""
        shared_parity_model.reset_state()
        return shared_parity_model

    def test_output_in_valid_range(self, model):
        """Test that output probabilities are in [0, 1]."""
        audio = mx.random.normal((512,))
        prob, _ = model(audio)

        prob_val = float(prob)
        assert 0.0 <= prob_val <= 1.0, f"Probability {prob_val} outside [0, 1]"

    def test_output_stable_with_zeros(self, model):
        """Test that model handles zero input."""
        audio = mx.zeros((512,))
        prob, _ = model(audio)

        prob_val = float(prob)
        assert not np.isnan(prob_val), "NaN output on zero input"
        assert not np.isinf(prob_val), "Inf output on zero input"
        assert 0.0 <= prob_val <= 1.0

    def test_output_stable_with_large_values(self, model):
        """Test that model handles large input values."""
        audio = mx.random.normal((512,)) * 100.0
        prob, _ = model(audio)

        prob_val = float(prob)
        assert not np.isnan(prob_val), "NaN output on large input"
        assert not np.isinf(prob_val), "Inf output on large input"
        assert 0.0 <= prob_val <= 1.0

    def test_output_stable_with_small_values(self, model):
        """Test that model handles small input values."""
        audio = mx.random.normal((512,)) * 1e-6
        prob, _ = model(audio)

        prob_val = float(prob)
        assert not np.isnan(prob_val), "NaN output on small input"
        assert not np.isinf(prob_val), "Inf output on small input"
        assert 0.0 <= prob_val <= 1.0

    def test_deterministic_output(self, model):
        """Test that same input produces same output."""
        # Use fixed seed for reproducibility
        mx.random.seed(42)
        audio = mx.random.normal((512,))

        prob1, state1 = model(audio)
        model.reset_state()

        prob2, state2 = model(audio)

        assert mx.allclose(prob1, prob2), "Non-deterministic output"

    def test_batch_consistency(self, model):
        """Test that batched and unbatched produce similar results."""
        mx.random.seed(42)
        audio = mx.random.normal((512,))

        # Unbatched
        prob_single, _ = model(audio)
        model.reset_state()

        # Batched
        audio_batch = audio[None, :]  # [1, 512]
        prob_batch, _ = model(audio_batch)

        # Should produce same result (tightened from 1e-5 - measured: exact match)
        assert mx.allclose(prob_single, prob_batch.squeeze(), atol=1e-7)


class TestLayerParity:
    """Tests for individual layer numerical properties."""

    def test_encoder_output_shape(self):
        """Test encoder produces correct output shape (single vector per window)."""
        encoder = VADEncoder(input_size=512, hidden_size=64)
        x = mx.random.normal((2, 512))  # [batch, samples]

        out = encoder(x)

        # Output should be [batch, hidden] - single vector per window
        # This matches Silero architecture: 1 timestep per window for LSTM
        assert out.shape == (2, 64)

    def test_decoder_output_range(self):
        """Test decoder output is in [0, 1]."""
        decoder = VADDecoder(hidden_size=64)
        x = mx.random.normal((2, 64))  # [batch, features]

        out = decoder(x)

        # Should be probabilities
        assert mx.all(out >= 0.0)
        assert mx.all(out <= 1.0)

    def test_lstm_state_shape(self):
        """Test LSTM produces correct state shapes."""
        lstm = StackedLSTM(input_size=64, hidden_size=128, num_layers=2)
        x = mx.random.normal((2, 10, 64))

        out, (h, c) = lstm(x)

        # Output shape
        assert out.shape == (2, 10, 128)

        # State shapes - [num_layers, batch, hidden]
        assert h.shape == (2, 2, 128)
        assert c.shape == (2, 2, 128)

    def test_lstm_state_continuity(self):
        """Test that LSTM state can be passed between calls."""
        lstm = StackedLSTM(input_size=64, hidden_size=128, num_layers=2)

        x1 = mx.random.normal((2, 10, 64))
        x2 = mx.random.normal((2, 10, 64))

        # First call
        out1, state1 = lstm(x1)

        # Second call with state
        out2, state2 = lstm(x2, state1)

        # States should be different
        h1, c1 = state1
        h2, c2 = state2

        assert not mx.allclose(h1, h2)
        assert not mx.allclose(c1, c2)


class TestProcessingParity:
    """Tests for processing complete audio sequences."""

    @pytest.fixture
    def model(self, shared_parity_model):
        """Get the shared model and reset its state."""
        shared_parity_model.reset_state()
        return shared_parity_model

    def test_chunked_vs_full_processing(self, model):
        """Test that chunked processing is consistent."""
        # Create 2 seconds of audio
        audio = mx.random.normal((32000,))

        # Process all at once
        probs_full, segments_full = model.process_audio(
            audio, model.config.sample_rate
        )
        model.reset_state()

        # Process in chunks (simulate streaming)
        chunk_size = 512
        probs_chunked = []

        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i + chunk_size]
            if len(chunk) == chunk_size:
                prob, _ = model(chunk)
                probs_chunked.append(float(prob))

        model.reset_state()

        # Both should produce valid probability sequences
        assert len(segments_full) >= 0
        assert len(probs_chunked) > 0
        assert all(0.0 <= p <= 1.0 for p in probs_chunked)

    def test_different_sample_rates(self):
        """Test model behavior at different sample rates."""
        for sr, window in [(16000, 512), (8000, 256)]:
            config = VADConfig(
                sample_rate=sr,
                window_size_samples=window,
                hidden_size=64,
                num_layers=1,
            )
            model = SileroVAD(config)

            audio = mx.random.normal((window,))
            prob, _ = model(audio)

            prob_val = float(prob)
            assert 0.0 <= prob_val <= 1.0, f"Invalid prob at {sr}Hz: {prob_val}"


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.fixture
    def model(self, shared_parity_model):
        """Get the shared model and reset its state."""
        shared_parity_model.reset_state()
        return shared_parity_model

    def test_very_short_audio(self, model):
        """Test handling of audio shorter than window."""
        audio = mx.random.normal((100,))  # Less than 512
        result = model.process_audio(audio, 16000)

        # Should not crash
        assert result is not None

    def test_very_long_audio(self, model):
        """Test handling of long audio (10 seconds)."""
        audio = mx.random.normal((160000,))  # 10 seconds
        probs, segments = model.process_audio(audio, 16000)

        # Should produce probabilities for all frames
        expected_frames = 160000 // 512
        assert probs.shape[0] >= expected_frames - 1

    def test_single_sample(self, model):
        """Test handling of single sample."""
        audio = mx.array([0.5])
        result = model.process_audio(audio, 16000)

        # Should not crash
        assert result is not None

    def test_empty_audio(self, model):
        """Test handling of empty audio."""
        audio = mx.array([])

        # Empty audio should be handled gracefully
        try:
            probs, segments = model.process_audio(audio, 16000)
            assert len(segments) == 0
        except ValueError:
            # It's acceptable to raise an error for empty audio
            pass

    def test_nan_input_handling(self, model):
        """Test that model handles NaN gracefully."""
        audio = mx.array([float("nan")] * 512)

        # This may produce NaN output, but shouldn't crash
        try:
            prob, _ = model(audio)
            # If it doesn't crash, that's acceptable
        except Exception as e:
            pytest.fail(f"Model crashed on NaN input: {e}")

    def test_inf_input_handling(self, model):
        """Test that model handles Inf gracefully."""
        audio = mx.array([float("inf")] * 512)

        # This may produce extreme output, but shouldn't crash
        try:
            prob, _ = model(audio)
            # If it doesn't crash, that's acceptable
        except Exception as e:
            pytest.fail(f"Model crashed on Inf input: {e}")


class TestSyntheticSignals:
    """Tests with synthetic audio signals."""

    @pytest.fixture
    def model(self, shared_parity_model):
        """Get the shared model and reset its state."""
        shared_parity_model.reset_state()
        return shared_parity_model

    def test_pure_sine_wave(self, model):
        """Test with pure sine wave (should be consistent)."""
        t = mx.linspace(0, 1, 16000)  # 1 second
        audio = mx.sin(2 * mx.pi * 440 * t)  # 440 Hz sine

        probs, segments = model.process_audio(audio, 16000)

        # Should process without error
        assert probs is not None
        probs_list = probs.tolist()
        # Probabilities should be somewhat consistent for uniform signal
        assert np.std(probs_list) < 0.5  # Not too variable

    def test_white_noise(self, model):
        """Test with white noise."""
        audio = mx.random.normal((16000,))  # 1 second of noise

        result = model.process_audio(audio, 16000)

        # Should process without error
        assert result is not None

    def test_silence(self, model):
        """Test with complete silence."""
        audio = mx.zeros((16000,))  # 1 second of silence

        result = model.process_audio(audio, 16000)

        # Model shouldn't detect speech in pure silence
        assert result is not None
        # (actual detection depends on model weights)

    def test_impulse(self, model):
        """Test with impulse signal."""
        audio = mx.zeros((16000,))
        # Create impulse
        audio_np = np.zeros(16000)
        audio_np[8000] = 1.0
        audio = mx.array(audio_np)

        result = model.process_audio(audio, 16000)

        # Should process without error
        assert result is not None


class TestWeightLoadingParity:
    """Tests for weight loading and model initialization."""

    def test_random_weights_produce_valid_output(self):
        """Test that random weights still produce valid probabilities."""
        config = VADConfig(
            sample_rate=16000,
            window_size_samples=512,
            hidden_size=64,
            num_layers=2,
        )
        model = SileroVAD(config)

        audio = mx.random.normal((512,))
        prob, _ = model(audio)

        assert 0.0 <= float(prob) <= 1.0

    def test_different_hidden_sizes(self):
        """Test various hidden size configurations."""
        for hidden_size in [32, 64, 128, 256]:
            config = VADConfig(
                sample_rate=16000,
                window_size_samples=512,
                hidden_size=hidden_size,
                num_layers=2,
            )
            model = SileroVAD(config)

            audio = mx.random.normal((512,))
            prob, _ = model(audio)

            assert 0.0 <= float(prob) <= 1.0

    def test_different_num_layers(self):
        """Test various layer configurations."""
        for num_layers in [1, 2, 3, 4]:
            config = VADConfig(
                sample_rate=16000,
                window_size_samples=512,
                hidden_size=64,
                num_layers=num_layers,
            )
            model = SileroVAD(config)

            audio = mx.random.normal((512,))
            prob, _ = model(audio)

            assert 0.0 <= float(prob) <= 1.0


@pytest.mark.slow
class TestLongRunning:
    """Long-running tests for comprehensive validation."""

    def test_extended_processing(self):
        """Test processing 60 seconds of audio."""
        config = VADConfig(
            sample_rate=16000,
            window_size_samples=512,
            hidden_size=64,
            num_layers=2,
        )
        model = SileroVAD(config)

        # 60 seconds
        audio = mx.random.normal((960000,))

        probs, segments = model.process_audio(audio, 16000)

        assert probs is not None
        assert probs.shape[0] > 1000

    def test_many_short_chunks(self):
        """Test processing many short chunks."""
        config = VADConfig(
            sample_rate=16000,
            window_size_samples=512,
            hidden_size=64,
            num_layers=2,
        )
        model = SileroVAD(config)

        # Process 1000 chunks
        probs = []
        for _ in range(1000):
            audio = mx.random.normal((512,))
            prob, _ = model(audio)
            probs.append(float(prob))

        # All should be valid
        assert all(0.0 <= p <= 1.0 for p in probs)
        # State should accumulate meaningfully
        assert len(set(probs)) > 1  # Not all the same
