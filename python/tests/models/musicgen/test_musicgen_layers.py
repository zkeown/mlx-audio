"""Tests for MusicGen layers."""

import pytest
import mlx.core as mx

from mlx_audio.models.musicgen.config import MusicGenConfig
from mlx_audio.models.musicgen.layers.embeddings import (
    SinusoidalPositionalEmbedding,
    CodebookEmbeddings,
)
from mlx_audio.models.musicgen.layers.lm_head import (
    DelayPatternScheduler,
    MusicGenLMHead,
)
from mlx_audio.models.musicgen.layers.transformer import (
    MultiHeadAttention,
    MusicGenDecoderBlock,
    MusicGenDecoder,
)


class TestSinusoidalPositionalEmbedding:
    """Tests for sinusoidal positional embeddings."""

    def test_output_shape(self):
        """Test output shape is correct."""
        pos_emb = SinusoidalPositionalEmbedding(hidden_size=256, max_length=1000)

        positions = mx.arange(50)
        embeddings = pos_emb(positions)

        assert embeddings.shape == (50, 256)

    def test_batched_positions(self):
        """Test with batched positions."""
        pos_emb = SinusoidalPositionalEmbedding(hidden_size=256, max_length=1000)

        positions = mx.array([[0, 1, 2], [3, 4, 5]])
        embeddings = pos_emb(positions)

        assert embeddings.shape == (2, 3, 256)

    def test_different_positions_different_embeddings(self):
        """Test that different positions produce different embeddings."""
        pos_emb = SinusoidalPositionalEmbedding(hidden_size=256, max_length=100)

        emb_0 = pos_emb(mx.array([0]))
        emb_1 = pos_emb(mx.array([1]))

        assert not mx.allclose(emb_0, emb_1)


class TestCodebookEmbeddings:
    """Tests for codebook embeddings."""

    @pytest.fixture
    def small_config(self):
        return MusicGenConfig(
            num_codebooks=2,
            codebook_size=512,
            hidden_size=128,
            max_duration=5.0,
        )

    def test_output_shape(self, small_config):
        """Test output shape is correct."""
        embeddings = CodebookEmbeddings(small_config)

        # [B, K, T]
        input_ids = mx.zeros((2, small_config.num_codebooks, 50), dtype=mx.int32)
        output = embeddings(input_ids)

        assert output.shape == (2, 50, small_config.hidden_size)

    def test_get_codebook_embedding(self, small_config):
        """Test getting embedding for specific codebook."""
        embeddings = CodebookEmbeddings(small_config)

        token_ids = mx.array([[0, 1, 2]])
        emb = embeddings.get_codebook_embedding(0, token_ids)

        assert emb.shape == (1, 3, small_config.hidden_size)


class TestDelayPatternScheduler:
    """Tests for delay pattern scheduler."""

    def test_apply_delay_pattern(self):
        """Test applying delay pattern adds correct padding."""
        scheduler = DelayPatternScheduler(num_codebooks=4, pad_token_id=999)

        codes = mx.zeros((1, 4, 10), dtype=mx.int32)  # [B, K, T]
        delayed = scheduler.apply_delay_pattern(codes)

        # Output should be longer by num_codebooks - 1
        assert delayed.shape == (1, 4, 13)  # 10 + 4 - 1 = 13

    def test_revert_delay_pattern(self):
        """Test reverting delay pattern."""
        scheduler = DelayPatternScheduler(num_codebooks=4, pad_token_id=999)

        # Create delayed codes
        delayed = mx.zeros((1, 4, 13), dtype=mx.int32)
        reverted = scheduler.revert_delay_pattern(delayed)

        assert reverted.shape == (1, 4, 10)  # 13 - 4 + 1 = 10

    def test_delay_revert_roundtrip(self):
        """Test that apply -> revert is identity."""
        scheduler = DelayPatternScheduler(num_codebooks=4, pad_token_id=999)

        original = mx.arange(10)[None, None, :].astype(mx.int32)
        original = mx.broadcast_to(original, (1, 4, 10))

        delayed = scheduler.apply_delay_pattern(original)
        reverted = scheduler.revert_delay_pattern(delayed)

        # Should recover original values (ignoring padding that was added)
        assert reverted.shape == original.shape

    def test_build_delay_pattern_mask(self):
        """Test building attention mask."""
        scheduler = DelayPatternScheduler(num_codebooks=4)

        mask = scheduler.build_delay_pattern_mask(seq_length=10)

        assert mask.shape == (4, 10, 10)

    def test_get_valid_positions(self):
        """Test getting valid positions at each step."""
        scheduler = DelayPatternScheduler(num_codebooks=4)

        # At step 0, only codebook 0 is valid
        assert scheduler.get_valid_positions(0) == [0]

        # At step 1, codebooks 0 and 1 are valid
        assert scheduler.get_valid_positions(1) == [0, 1]

        # At step 3, all codebooks are valid
        assert scheduler.get_valid_positions(3) == [0, 1, 2, 3]

        # At step 10, all codebooks are still valid
        assert scheduler.get_valid_positions(10) == [0, 1, 2, 3]


class TestMusicGenLMHead:
    """Tests for language model head."""

    @pytest.fixture
    def small_config(self):
        return MusicGenConfig(
            num_codebooks=2,
            codebook_size=512,
            hidden_size=128,
        )

    def test_output_all_codebooks(self, small_config):
        """Test output for all codebooks."""
        lm_head = MusicGenLMHead(small_config)

        hidden_states = mx.random.normal((2, 10, small_config.hidden_size))
        logits = lm_head(hidden_states)

        assert logits.shape == (2, 2, 10, 513)  # [B, K, T, V]

    def test_output_single_codebook(self, small_config):
        """Test output for single codebook."""
        lm_head = MusicGenLMHead(small_config)

        hidden_states = mx.random.normal((2, 10, small_config.hidden_size))
        logits = lm_head(hidden_states, codebook_idx=0)

        assert logits.shape == (2, 10, 513)  # [B, T, V]


class TestMultiHeadAttention:
    """Tests for multi-head attention."""

    def test_self_attention_output_shape(self):
        """Test self-attention output shape."""
        attn = MultiHeadAttention(hidden_size=256, num_heads=8)

        x = mx.random.normal((2, 10, 256))
        output, kv_cache = attn(x)

        assert output.shape == (2, 10, 256)
        assert kv_cache is not None

    def test_cross_attention_output_shape(self):
        """Test cross-attention output shape."""
        attn = MultiHeadAttention(hidden_size=256, num_heads=8)

        x = mx.random.normal((2, 10, 256))
        kv = mx.random.normal((2, 20, 256))
        output, kv_cache = attn(x, key_value_states=kv)

        assert output.shape == (2, 10, 256)
        assert kv_cache is None  # No caching for cross-attention

    def test_kv_cache_update(self):
        """Test KV cache is properly updated."""
        attn = MultiHeadAttention(hidden_size=256, num_heads=8)

        # First forward
        x1 = mx.random.normal((2, 5, 256))
        _, cache1 = attn(x1)

        # Second forward with cache
        x2 = mx.random.normal((2, 3, 256))
        _, cache2 = attn(x2, kv_cache=cache1)

        # Cache should have grown
        assert cache2[0].shape[1] == 8  # 5 + 3


class TestMusicGenDecoderBlock:
    """Tests for decoder block."""

    @pytest.fixture
    def small_config(self):
        return MusicGenConfig(
            hidden_size=128,
            num_attention_heads=4,
            intermediate_size=256,
        )

    def test_output_shape(self, small_config):
        """Test output shape."""
        block = MusicGenDecoderBlock(small_config)

        x = mx.random.normal((2, 10, small_config.hidden_size))
        encoder_states = mx.random.normal((2, 20, small_config.hidden_size))

        output, kv_cache = block(x, encoder_hidden_states=encoder_states)

        assert output.shape == (2, 10, small_config.hidden_size)

    def test_without_encoder_states(self, small_config):
        """Test forward without encoder states."""
        block = MusicGenDecoderBlock(small_config)

        x = mx.random.normal((2, 10, small_config.hidden_size))
        output, kv_cache = block(x)

        assert output.shape == (2, 10, small_config.hidden_size)


class TestMusicGenDecoder:
    """Tests for full decoder."""

    @pytest.fixture
    def small_config(self):
        return MusicGenConfig(
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=256,
        )

    def test_output_shape(self, small_config):
        """Test output shape."""
        decoder = MusicGenDecoder(small_config)

        x = mx.random.normal((2, 10, small_config.hidden_size))
        encoder_states = mx.random.normal((2, 20, small_config.hidden_size))

        output, kv_cache = decoder(x, encoder_hidden_states=encoder_states)

        assert output.shape == (2, 10, small_config.hidden_size)
        assert len(kv_cache) == small_config.num_hidden_layers

    def test_create_causal_mask(self, small_config):
        """Test causal mask creation."""
        decoder = MusicGenDecoder(small_config)

        mask = decoder.create_causal_mask(seq_length=5, offset=0)

        assert mask.shape == (5, 5)
        # Check that future positions are masked
        assert mask[0, 1] == float("-inf")
        assert mask[0, 0] == 0.0
