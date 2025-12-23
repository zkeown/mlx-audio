"""Tests for MusicGen model."""

import pytest
import mlx.core as mx

from mlx_audio.models.musicgen.config import MusicGenConfig
from mlx_audio.models.musicgen.model import MusicGen


class TestMusicGenModel:
    """Tests for MusicGen model."""

    @pytest.fixture
    def small_config(self):
        """Create a minimal config for fast testing."""
        return MusicGenConfig(
            num_codebooks=2,
            codebook_size=256,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=128,
            text_hidden_size=64,
            max_duration=2.0,
            frame_rate=25,
        )

    @pytest.fixture
    def small_model(self, small_config):
        """Create a small model for testing."""
        return MusicGen(small_config)

    def test_model_initialization(self, small_model, small_config):
        """Test model initializes correctly."""
        assert small_model.config == small_config
        assert small_model.embeddings is not None
        assert small_model.decoder is not None
        assert small_model.lm_head is not None

    def test_forward_output_shape(self, small_model, small_config):
        """Test forward pass output shape."""
        batch_size = 2
        seq_length = 10
        text_length = 5

        # Input tokens [B, K, T]
        input_ids = mx.zeros(
            (batch_size, small_config.num_codebooks, seq_length),
            dtype=mx.int32
        )

        # Text conditioning [B, S, D]
        encoder_states = mx.random.normal(
            (batch_size, text_length, small_config.text_hidden_size)
        )

        logits, kv_cache = small_model.forward(
            input_ids,
            encoder_hidden_states=encoder_states,
        )

        # Output should be [B, K, T, V]
        expected_vocab = small_config.codebook_size + 1
        assert logits.shape == (
            batch_size,
            small_config.num_codebooks,
            seq_length,
            expected_vocab
        )

    def test_call_returns_logits(self, small_model, small_config):
        """Test __call__ returns logits only."""
        input_ids = mx.zeros((1, small_config.num_codebooks, 5), dtype=mx.int32)
        encoder_states = mx.random.normal((1, 3, small_config.text_hidden_size))

        logits = small_model(input_ids, encoder_hidden_states=encoder_states)

        assert logits.ndim == 4  # [B, K, T, V]

    def test_text_projection(self, small_model, small_config):
        """Test text embedding projection."""
        text_embeds = mx.random.normal((1, 10, small_config.text_hidden_size))
        projected = small_model.project_text_embeddings(text_embeds)

        assert projected.shape == (1, 10, small_config.hidden_size)

    def test_incremental_decoding(self, small_model, small_config):
        """Test incremental decoding with KV cache."""
        encoder_states = mx.random.normal((1, 5, small_config.text_hidden_size))

        # First step
        input_ids = mx.zeros((1, small_config.num_codebooks, 1), dtype=mx.int32)
        logits1, cache1 = small_model.forward(
            input_ids,
            encoder_hidden_states=encoder_states,
        )

        assert cache1 is not None
        assert len(cache1) == small_config.num_hidden_layers

        # Second step with cache
        next_input = mx.zeros((1, small_config.num_codebooks, 1), dtype=mx.int32)
        logits2, cache2 = small_model.forward(
            next_input,
            encoder_hidden_states=encoder_states,
            kv_cache=cache1,
        )

        assert logits2.shape == logits1.shape
        # Cache should have grown
        assert cache2[0][0].shape[1] == 2  # 1 + 1

    def test_generate_codes_without_codec(self, small_model):
        """Test generate produces codes even without audio codec."""
        encoder_states = mx.random.normal((1, 5, 64))

        # generate() only produces discrete codes, doesn't need codec
        codes = small_model.generate(
            encoder_states,
            max_new_tokens=10,
        )
        assert codes.ndim == 3  # [B, K, T]
        assert codes.shape[0] == 1
        assert codes.shape[1] == small_model.config.num_codebooks

    def test_delay_pattern_scheduler(self, small_model, small_config):
        """Test delay pattern scheduler is properly initialized."""
        scheduler = small_model.delay_pattern

        assert scheduler.num_codebooks == small_config.num_codebooks
        assert scheduler.pad_token_id == small_config.pad_token_id


class TestMusicGenGeneration:
    """Tests for MusicGen generation (with mocked codec)."""

    @pytest.fixture
    def small_config(self):
        return MusicGenConfig(
            num_codebooks=2,
            codebook_size=256,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=128,
            text_hidden_size=64,
            max_duration=2.0,
            frame_rate=25,
        )

    @pytest.fixture
    def model_with_mock_codec(self, small_config):
        """Create model with mocked audio codec."""

        class MockCodec:
            def decode(self, codes):
                batch_size, num_codebooks, num_frames = codes.shape
                # Return mock audio: [B, C, samples]
                num_samples = num_frames * 640  # hop_length
                return mx.zeros((batch_size, 1, num_samples))

        model = MusicGen(small_config)
        model._audio_codec = MockCodec()
        return model

    def test_generate_output_shape(self, model_with_mock_codec, small_config):
        """Test generate produces correct output shape."""
        encoder_states = mx.random.normal((1, 5, small_config.text_hidden_size))

        codes = model_with_mock_codec.generate(
            encoder_states,
            max_new_tokens=10,
            temperature=1.0,
        )

        assert codes.shape == (1, small_config.num_codebooks, 10)

    def test_generate_with_duration(self, model_with_mock_codec, small_config):
        """Test generate with duration parameter."""
        encoder_states = mx.random.normal((1, 5, small_config.text_hidden_size))

        codes = model_with_mock_codec.generate(
            encoder_states,
            duration=0.4,  # 0.4s * 25 fps = 10 tokens
            temperature=1.0,
        )

        assert codes.shape[2] == 10

    def test_generate_deterministic_with_seed(self, model_with_mock_codec, small_config):
        """Test generation is deterministic with same seed."""
        encoder_states = mx.random.normal((1, 5, small_config.text_hidden_size))

        codes1 = model_with_mock_codec.generate(
            encoder_states,
            max_new_tokens=5,
            seed=42,
            temperature=1.0,
        )

        codes2 = model_with_mock_codec.generate(
            encoder_states,
            max_new_tokens=5,
            seed=42,
            temperature=1.0,
        )

        assert mx.array_equal(codes1, codes2)

    def test_decode_audio(self, model_with_mock_codec):
        """Test decoding codes to audio."""
        codes = mx.zeros((1, 2, 10), dtype=mx.int32)
        audio = model_with_mock_codec.decode_audio(codes)

        assert audio.ndim == 3  # [B, C, T]
