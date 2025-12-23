"""Tests for VAD model."""

import pytest
import mlx.core as mx

from mlx_audio.models.vad import SileroVAD, VADConfig


# Module-level fixtures for faster tests
@pytest.fixture(scope="module")
def shared_model():
    """Create a shared VAD model for testing."""
    config = VADConfig.silero_vad_16k()
    return SileroVAD(config)


@pytest.fixture(scope="module")
def shared_model_8k():
    """Create a shared 8kHz VAD model for testing."""
    config = VADConfig.silero_vad_8k()
    return SileroVAD(config)


class TestSileroVAD:
    """Tests for SileroVAD model."""

    @pytest.fixture
    def model(self, shared_model):
        """Get the shared model and reset its state."""
        shared_model.reset_state()
        return shared_model

    @pytest.fixture
    def model_8k(self, shared_model_8k):
        """Get the shared 8kHz model and reset its state."""
        shared_model_8k.reset_state()
        return shared_model_8k

    def test_model_creation(self, model):
        """Test model can be created."""
        assert model is not None
        assert model.config.sample_rate == 16000
        assert model.config.hidden_size == 128

    def test_model_creation_8k(self, model_8k):
        """Test 8kHz model can be created."""
        assert model_8k is not None
        assert model_8k.config.sample_rate == 8000
        assert model_8k.config.window_size_samples == 256

    def test_model_default_config(self):
        """Test model with default config."""
        model = SileroVAD()
        assert model.config.sample_rate == 16000

    def test_forward_single_sample(self, model):
        """Test forward pass with single sample (no batch)."""
        audio = mx.random.normal((512,))  # 32ms at 16kHz

        prob, state = model(audio)

        # Check output shape
        assert prob.shape == (1,)
        # Check probability is in valid range
        prob_val = float(prob)
        assert 0.0 <= prob_val <= 1.0

        # Check state structure
        assert "h" in state
        assert "c" in state
        assert "context" in state

    def test_forward_batched(self, model):
        """Test forward pass with batched input."""
        batch_size = 4
        audio = mx.random.normal((batch_size, 512))

        prob, state = model(audio)

        # Check output shape
        assert prob.shape == (batch_size, 1)

        # Check all probabilities are valid
        for i in range(batch_size):
            prob_val = float(prob[i])
            assert 0.0 <= prob_val <= 1.0

        # Check state shapes
        assert state["h"].shape == (2, batch_size, 128)  # num_layers, batch, hidden
        assert state["c"].shape == (2, batch_size, 128)
        assert state["context"].shape == (batch_size, 64)

    def test_forward_8k(self, model_8k):
        """Test forward pass with 8kHz model."""
        audio = mx.random.normal((256,))  # 32ms at 8kHz

        prob, state = model_8k(audio)

        assert prob.shape == (1,)
        prob_val = float(prob)
        assert 0.0 <= prob_val <= 1.0

    def test_streaming_inference(self, model):
        """Test streaming inference maintains state."""
        num_chunks = 5
        probs = []
        state = None

        for _ in range(num_chunks):
            audio = mx.random.normal((512,))
            prob, state = model(audio, state=state)
            probs.append(float(prob))

        # Should have all valid probabilities
        assert len(probs) == num_chunks
        for p in probs:
            assert 0.0 <= p <= 1.0

        # State should be maintained
        assert state is not None
        assert state["h"] is not None
        assert state["c"] is not None

    def test_reset_state(self, model):
        """Test state reset."""
        state = model.reset_state(batch_size=1)

        assert state["h"].shape == (2, 1, 128)
        assert state["c"].shape == (2, 1, 128)
        assert state["context"].shape == (1, 64)

        # All values should be zero
        assert mx.all(state["h"] == 0)
        assert mx.all(state["c"] == 0)
        assert mx.all(state["context"] == 0)

    def test_reset_state_batched(self, model):
        """Test batched state reset."""
        batch_size = 8
        state = model.reset_state(batch_size=batch_size)

        assert state["h"].shape == (2, batch_size, 128)
        assert state["c"].shape == (2, batch_size, 128)
        assert state["context"].shape == (batch_size, 64)

    def test_process_audio_complete(self, model):
        """Test processing complete audio."""
        # 1 second of audio at 16kHz
        audio = mx.random.normal((16000,))

        probs, segments = model.process_audio(audio)

        # Should have probabilities for each window
        expected_windows = 16000 // 512 + (1 if 16000 % 512 else 0)
        assert probs.shape[0] == expected_windows

        # Segments should be list of tuples
        assert isinstance(segments, list)
        for seg in segments:
            assert len(seg) == 2
            start, end = seg
            assert start >= 0
            assert end > start

    def test_process_audio_short(self, model):
        """Test processing short audio."""
        # Less than one window
        audio = mx.random.normal((256,))

        probs, segments = model.process_audio(audio)

        # Should still produce output (padded)
        assert probs.shape[0] >= 1

    def test_process_audio_threshold(self, model):
        """Test process_audio with custom threshold."""
        audio = mx.random.normal((16000,))

        # High threshold should produce fewer segments
        _, segments_high = model.process_audio(audio, threshold=0.9)
        _, segments_low = model.process_audio(audio, threshold=0.1)

        # Note: Can't guarantee which has more segments with random audio,
        # but both should return valid results
        assert isinstance(segments_high, list)
        assert isinstance(segments_low, list)

    def test_output_probability_range(self, model):
        """Test output is always in [0, 1] range."""
        # Test with various input magnitudes
        for scale in [0.001, 0.1, 1.0, 10.0, 100.0]:
            audio = mx.random.normal((512,)) * scale
            prob, _ = model(audio)
            prob_val = float(prob)
            assert 0.0 <= prob_val <= 1.0, f"Probability {prob_val} out of range for scale {scale}"

    def test_deterministic_with_same_input(self, model):
        """Test model produces same output for same input."""
        audio = mx.array([0.1] * 512)

        prob1, state1 = model(audio)
        prob2, state2 = model(audio)

        # Should produce identical results
        assert float(prob1) == float(prob2)

    def test_model_components(self, model):
        """Test model has expected components."""
        assert hasattr(model, "encoder")
        assert hasattr(model, "lstm")
        assert hasattr(model, "decoder")
        assert hasattr(model, "config")


class TestVADEncoder:
    """Tests for VAD encoder layer."""

    def test_encoder_output_shape(self):
        """Test encoder produces correct output shape."""
        from mlx_audio.models.vad.layers import VADEncoder

        encoder = VADEncoder(input_size=512, hidden_size=128)
        audio = mx.random.normal((512,))

        output = encoder(audio)

        # Output should be [hidden_size] - single vector per window
        # Matches Silero architecture: 1 timestep per window for LSTM
        assert output.ndim == 1
        assert output.shape[0] == 128

    def test_encoder_batched(self):
        """Test encoder with batched input."""
        from mlx_audio.models.vad.layers import VADEncoder

        encoder = VADEncoder(input_size=512, hidden_size=128)
        audio = mx.random.normal((4, 512))

        output = encoder(audio)

        # Output should be [batch, hidden] - single vector per window
        assert output.ndim == 2
        assert output.shape == (4, 128)


class TestVADDecoder:
    """Tests for VAD decoder layer."""

    def test_decoder_output_shape(self):
        """Test decoder produces correct output shape."""
        from mlx_audio.models.vad.layers import VADDecoder

        decoder = VADDecoder(hidden_size=128)
        features = mx.random.normal((128,))

        output = decoder(features)

        assert output.shape == (1,)

    def test_decoder_probability_range(self):
        """Test decoder output is in [0, 1]."""
        from mlx_audio.models.vad.layers import VADDecoder

        decoder = VADDecoder(hidden_size=128)

        for _ in range(10):
            features = mx.random.normal((128,)) * 10
            output = decoder(features)
            prob = float(output)
            assert 0.0 <= prob <= 1.0


class TestStackedLSTM:
    """Tests for stacked LSTM layer."""

    def test_stacked_lstm_output_shape(self):
        """Test stacked LSTM produces correct output shape."""
        from mlx_audio.models.vad.layers import StackedLSTM

        lstm = StackedLSTM(input_size=128, hidden_size=128, num_layers=2)
        x = mx.random.normal((10, 128))  # [seq_len, input_size]

        output, (h, c) = lstm(x)

        assert output.shape == (10, 128)  # [seq_len, hidden_size]
        assert h.shape == (2, 128)  # [num_layers, hidden_size]
        assert c.shape == (2, 128)

    def test_stacked_lstm_batched(self):
        """Test stacked LSTM with batched input."""
        from mlx_audio.models.vad.layers import StackedLSTM

        lstm = StackedLSTM(input_size=128, hidden_size=128, num_layers=2)
        x = mx.random.normal((4, 10, 128))  # [batch, seq_len, input_size]

        output, (h, c) = lstm(x)

        assert output.shape == (4, 10, 128)  # [batch, seq_len, hidden_size]
        assert h.shape == (2, 4, 128)  # [num_layers, batch, hidden_size]
        assert c.shape == (2, 4, 128)

    def test_stacked_lstm_state_passing(self):
        """Test stacked LSTM maintains state across calls."""
        from mlx_audio.models.vad.layers import StackedLSTM

        lstm = StackedLSTM(input_size=128, hidden_size=128, num_layers=2)

        # Use batched input to test state passing
        x1 = mx.random.normal((1, 5, 128))  # [batch, seq, features]
        output1, (h1, c1) = lstm(x1)

        x2 = mx.random.normal((1, 5, 128))
        output2, (h2, c2) = lstm(x2, hidden=(h1, c1))

        # States should be different
        assert not mx.allclose(h1, h2)
        assert not mx.allclose(c1, c2)
