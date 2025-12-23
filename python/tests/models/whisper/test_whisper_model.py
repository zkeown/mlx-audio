"""Tests for Whisper model architecture."""

import pytest

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

from mlx_audio.models.whisper.config import WhisperConfig


@pytest.fixture
def small_config():
    """Small config for fast testing."""
    return WhisperConfig(
        n_mels=80,
        n_audio_ctx=100,
        n_audio_state=64,
        n_audio_head=4,
        n_audio_layer=2,
        n_text_ctx=50,
        n_text_state=64,
        n_text_head=4,
        n_text_layer=2,
        n_vocab=1000,
    )


@pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
class TestWhisperEncoder:
    """Tests for Whisper audio encoder."""

    def test_encoder_forward_shape(self, small_config):
        """Test encoder produces correct output shape."""
        from mlx_audio.models.whisper.layers.encoder import AudioEncoder

        encoder = AudioEncoder(small_config)

        # Input: [B, n_mels, T]
        mel = mx.random.normal((2, 80, 200))

        # Output: [B, T//2, n_state] (due to stride-2 conv)
        output = encoder(mel)

        assert output.shape == (2, 100, 64)

    def test_encoder_unbatched(self, small_config):
        """Test encoder handles unbatched input."""
        from mlx_audio.models.whisper.layers.encoder import AudioEncoder

        encoder = AudioEncoder(small_config)

        # Input: [n_mels, T] (no batch dim)
        mel = mx.random.normal((80, 200))

        output = encoder(mel)

        # Should add batch dim
        assert output.shape == (1, 100, 64)


@pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
class TestWhisperDecoder:
    """Tests for Whisper text decoder."""

    def test_decoder_forward_shape(self, small_config):
        """Test decoder produces correct output shape."""
        from mlx_audio.models.whisper.layers.decoder import TextDecoder

        decoder = TextDecoder(small_config)

        # Tokens: [B, L]
        tokens = mx.array([[1, 2, 3, 4, 5]])

        # Audio features: [B, S, D]
        audio_features = mx.random.normal((1, 100, 64))

        logits = decoder(tokens, audio_features)

        assert logits.shape == (1, 5, 1000)

    def test_decoder_incremental(self, small_config):
        """Test decoder with KV cache for incremental decoding."""
        from mlx_audio.models.whisper.layers.decoder import TextDecoder
        from mlx_audio.models.whisper.kv_cache import KVCache

        decoder = TextDecoder(small_config)

        audio_features = mx.random.normal((1, 100, 64))

        # Create pre-allocated cache
        cache = KVCache(
            max_length=16,
            n_layers=small_config.n_text_layer,
            hidden_dim=small_config.n_text_state,
            batch_size=1,
        )

        # First step: process initial tokens
        tokens1 = mx.array([[1, 2, 3]])
        logits1 = decoder(tokens1, audio_features, cache)

        assert logits1.shape == (1, 3, 1000)
        assert cache.length == 3

        # Second step: process single new token with cache
        tokens2 = mx.array([[4]])
        logits2 = decoder(tokens2, audio_features, cache)

        assert logits2.shape == (1, 1, 1000)
        assert cache.length == 4  # 3 + 1 tokens


@pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
class TestWhisperModel:
    """Tests for full Whisper model."""

    def test_model_forward(self, small_config):
        """Test full model forward pass."""
        from mlx_audio.models.whisper.model import Whisper

        model = Whisper(small_config)

        mel = mx.random.normal((1, 80, 200))
        tokens = mx.array([[1, 2, 3, 4, 5]])

        logits = model(mel, tokens)

        assert logits.shape == (1, 5, 1000)

    def test_encode(self, small_config):
        """Test encoder-only forward pass."""
        from mlx_audio.models.whisper.model import Whisper

        model = Whisper(small_config)

        mel = mx.random.normal((1, 80, 200))
        features = model.encode(mel)

        assert features.shape == (1, 100, 64)

    def test_decode_with_cache(self, small_config):
        """Test decoder with KV caching."""
        from mlx_audio.models.whisper.model import Whisper
        from mlx_audio.models.whisper.kv_cache import KVCache

        model = Whisper(small_config)

        mel = mx.random.normal((1, 80, 200))
        features = model.encode(mel)

        # Create pre-allocated cache
        cache = KVCache(
            max_length=16,
            n_layers=small_config.n_text_layer,
            hidden_dim=small_config.n_text_state,
            batch_size=1,
        )

        # First decode
        tokens1 = mx.array([[1, 2, 3]])
        logits1 = model.decode(tokens1, features, cache)

        # Incremental decode
        tokens2 = mx.array([[4]])
        logits2 = model.decode(tokens2, features, cache)

        assert logits2.shape == (1, 1, 1000)
        assert cache.length == 4

    def test_dims_property(self, small_config):
        """Test dims property returns correct values."""
        from mlx_audio.models.whisper.model import Whisper

        model = Whisper(small_config)
        dims = model.dims

        assert dims["n_mels"] == 80
        assert dims["n_audio_state"] == 64
        assert dims["n_vocab"] == 1000


@pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
class TestMultiHeadAttention:
    """Tests for multi-head attention layer."""

    def test_self_attention(self):
        """Test self-attention forward pass."""
        from mlx_audio.models.whisper.layers.attention import MultiHeadAttention

        attn = MultiHeadAttention(n_state=64, n_head=4)

        x = mx.random.normal((2, 10, 64))
        output, _ = attn(x)

        assert output.shape == (2, 10, 64)

    def test_cross_attention(self):
        """Test cross-attention forward pass."""
        from mlx_audio.models.whisper.layers.attention import MultiHeadAttention

        attn = MultiHeadAttention(n_state=64, n_head=4)

        x = mx.random.normal((2, 5, 64))  # Query
        xa = mx.random.normal((2, 20, 64))  # Key/Value source

        output, _ = attn(x, xa=xa)

        assert output.shape == (2, 5, 64)

    def test_kv_cache_accumulation(self):
        """Test KV cache grows correctly with pre-allocated cache."""
        from mlx_audio.models.whisper.layers.attention import MultiHeadAttention
        from mlx_audio.models.whisper.kv_cache import KVCache

        attn = MultiHeadAttention(n_state=64, n_head=4)

        # Create pre-allocated cache
        cache = KVCache(
            max_length=16,
            n_layers=1,
            hidden_dim=64,
            batch_size=1,
        )

        # First step
        x1 = mx.random.normal((1, 3, 64))
        attn(x1, kv_cache=cache, layer_idx=0)
        cache.step(3)

        assert cache.length == 3  # 3 cached tokens

        # Second step with cache
        x2 = mx.random.normal((1, 2, 64))
        attn(x2, kv_cache=cache, layer_idx=0)
        cache.step(2)

        assert cache.length == 5  # 3 + 2 cached tokens

    def test_causal_mask(self):
        """Test causal masking prevents attending to future."""
        from mlx_audio.models.whisper.layers.attention import MultiHeadAttention

        attn = MultiHeadAttention(n_state=64, n_head=4)

        x = mx.random.normal((1, 5, 64))

        # Create causal mask
        mask = mx.triu(mx.full((5, 5), float("-inf")), k=1)

        output, _ = attn(x, mask=mask)
        assert output.shape == (1, 5, 64)


@pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
class TestResidualAttentionBlock:
    """Tests for residual attention block."""

    def test_encoder_block(self):
        """Test encoder block (no cross-attention)."""
        from mlx_audio.models.whisper.layers.attention import ResidualAttentionBlock

        block = ResidualAttentionBlock(n_state=64, n_head=4, cross_attention=False)

        x = mx.random.normal((2, 10, 64))
        output = block(x)

        assert output.shape == (2, 10, 64)

    def test_decoder_block(self):
        """Test decoder block (with cross-attention)."""
        from mlx_audio.models.whisper.layers.attention import ResidualAttentionBlock

        block = ResidualAttentionBlock(n_state=64, n_head=4, cross_attention=True)

        x = mx.random.normal((2, 5, 64))
        xa = mx.random.normal((2, 20, 64))

        output = block(x, xa=xa)

        assert output.shape == (2, 5, 64)
