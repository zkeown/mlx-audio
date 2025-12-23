"""Tests for CLAP model."""

import pytest
import numpy as np

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False


@pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
class TestHTSAT:
    """Tests for HTSAT audio encoder."""

    def test_output_shape(self):
        """Test HTSAT output shape."""
        from mlx_audio.models.clap.config import CLAPAudioConfig
        from mlx_audio.models.clap.layers.htsat import HTSAT

        config = CLAPAudioConfig(
            embed_dim=24,  # Small for testing
            depths=(1, 1, 1, 1),
            num_heads=(2, 2, 4, 4),
            hidden_size=192,  # Must match final_dim = embed_dim * 2^(num_stages-1)
        )
        model = HTSAT(config)

        # Input: [B, 1, F, T] mel spectrogram
        x = mx.random.normal((2, 1, 64, 256))
        out = model(x)

        # Output is final_dim = 24 * 2^3 = 192 (no fc layer, goes directly to projection)
        assert out.shape == (2, 192)  # [B, final_dim]

    def test_forward_features(self):
        """Test feature extraction before projection."""
        from mlx_audio.models.clap.config import CLAPAudioConfig
        from mlx_audio.models.clap.layers.htsat import HTSAT

        config = CLAPAudioConfig(
            embed_dim=24,
            depths=(1, 1, 1, 1),
            num_heads=(2, 2, 4, 4),
            hidden_size=64,
        )
        model = HTSAT(config)

        x = mx.random.normal((2, 1, 64, 256))
        features = model.forward_features(x)

        # Features before final projection
        assert features.ndim == 2
        assert features.shape[0] == 2


@pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
class TestCLAPTextEncoder:
    """Tests for CLAP text encoder."""

    def test_output_shape(self):
        """Test text encoder output shape."""
        from mlx_audio.models.clap.config import CLAPTextConfig
        from mlx_audio.models.clap.layers.text_encoder import CLAPTextEncoder

        config = CLAPTextConfig(
            vocab_size=1000,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=128,
        )
        encoder = CLAPTextEncoder(config, projection_dim=32)

        # Input: [B, L] token IDs
        input_ids = mx.array(np.random.randint(0, 1000, (2, 20)))
        out = encoder(input_ids)

        assert out.shape == (2, 32)  # [B, projection_dim]

    def test_with_attention_mask(self):
        """Test text encoder with attention mask."""
        from mlx_audio.models.clap.config import CLAPTextConfig
        from mlx_audio.models.clap.layers.text_encoder import CLAPTextEncoder

        config = CLAPTextConfig(
            vocab_size=1000,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=128,
        )
        encoder = CLAPTextEncoder(config, projection_dim=32)

        input_ids = mx.array(np.random.randint(0, 1000, (2, 20)))
        # Create attention mask with last 5 tokens masked
        mask_np = np.ones((2, 20), dtype=np.float32)
        mask_np[:, 15:] = 0
        attention_mask = mx.array(mask_np)

        out = encoder(input_ids, attention_mask=attention_mask)

        assert out.shape == (2, 32)
        assert mx.all(mx.isfinite(out))

    def test_normalization(self):
        """Test output normalization."""
        from mlx_audio.models.clap.config import CLAPTextConfig
        from mlx_audio.models.clap.layers.text_encoder import CLAPTextEncoder

        config = CLAPTextConfig(
            vocab_size=1000,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=128,
        )
        encoder = CLAPTextEncoder(config, projection_dim=32)

        input_ids = mx.array(np.random.randint(0, 1000, (2, 20)))
        out = encoder(input_ids, normalize=True)

        # Check L2 norm is approximately 1
        norms = mx.linalg.norm(out, axis=-1)
        assert mx.all(mx.abs(norms - 1.0) < 1e-5)


@pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
class TestCLAP:
    """Tests for main CLAP model."""

    def test_model_creation(self):
        """Test model instantiation."""
        from mlx_audio.models.clap import CLAP, CLAPConfig

        config = CLAPConfig()
        config.audio.embed_dim = 24
        config.audio.depths = (1, 1, 1, 1)
        config.audio.num_heads = (2, 2, 4, 4)
        config.audio.hidden_size = 64
        config.text.hidden_size = 64
        config.text.num_hidden_layers = 2
        config.text.num_attention_heads = 4
        config.text.intermediate_size = 128

        model = CLAP(config)
        assert model is not None

    def test_encode_audio(self):
        """Test audio encoding."""
        from mlx_audio.models.clap import CLAP, CLAPConfig

        config = CLAPConfig()
        config.audio.embed_dim = 24
        config.audio.depths = (1, 1, 1, 1)
        config.audio.num_heads = (2, 2, 4, 4)
        config.audio.hidden_size = 64
        config.projection_dim = 32

        model = CLAP(config)

        # Mel spectrogram input
        mel = mx.random.normal((2, 1, 64, 256))
        audio_embeds = model.encode_audio(mel)

        assert audio_embeds.shape == (2, 32)

    def test_encode_audio_normalized(self):
        """Test audio encoding with normalization."""
        from mlx_audio.models.clap import CLAP, CLAPConfig

        config = CLAPConfig()
        config.audio.embed_dim = 24
        config.audio.depths = (1, 1, 1, 1)
        config.audio.num_heads = (2, 2, 4, 4)
        config.audio.hidden_size = 64
        config.projection_dim = 32

        model = CLAP(config)

        mel = mx.random.normal((2, 1, 64, 256))
        audio_embeds = model.encode_audio(mel, normalize=True)

        # Check L2 norm is approximately 1
        norms = mx.linalg.norm(audio_embeds, axis=-1)
        assert mx.all(mx.abs(norms - 1.0) < 1e-5)

    def test_encode_text(self):
        """Test text encoding."""
        from mlx_audio.models.clap import CLAP, CLAPConfig

        config = CLAPConfig()
        config.audio.embed_dim = 24
        config.audio.depths = (1, 1, 1, 1)
        config.audio.num_heads = (2, 2, 4, 4)
        config.audio.hidden_size = 64
        config.text.vocab_size = 1000
        config.text.hidden_size = 64
        config.text.num_hidden_layers = 2
        config.text.num_attention_heads = 4
        config.text.intermediate_size = 128
        config.projection_dim = 32

        model = CLAP(config)

        input_ids = mx.array(np.random.randint(0, 1000, (3, 20)))
        text_embeds = model.encode_text(input_ids)

        assert text_embeds.shape == (3, 32)

    def test_similarity(self):
        """Test similarity computation."""
        from mlx_audio.models.clap import CLAP, CLAPConfig

        config = CLAPConfig()
        config.audio.embed_dim = 24
        config.audio.depths = (1, 1, 1, 1)
        config.audio.num_heads = (2, 2, 4, 4)
        config.audio.hidden_size = 64
        config.text.vocab_size = 1000
        config.text.hidden_size = 64
        config.text.num_hidden_layers = 2
        config.text.num_attention_heads = 4
        config.text.intermediate_size = 128
        config.projection_dim = 32

        model = CLAP(config)

        # Create embeddings
        mel = mx.random.normal((2, 1, 64, 256))
        input_ids = mx.array(np.random.randint(0, 1000, (3, 20)))

        audio_embeds = model.encode_audio(mel)
        text_embeds = model.encode_text(input_ids)

        similarity = model.similarity(audio_embeds, text_embeds)

        # [B_audio, B_text]
        assert similarity.shape == (2, 3)

    def test_forward(self):
        """Test full forward pass."""
        from mlx_audio.models.clap import CLAP, CLAPConfig

        config = CLAPConfig()
        config.audio.embed_dim = 24
        config.audio.depths = (1, 1, 1, 1)
        config.audio.num_heads = (2, 2, 4, 4)
        config.audio.hidden_size = 64
        config.text.vocab_size = 1000
        config.text.hidden_size = 64
        config.text.num_hidden_layers = 2
        config.text.num_attention_heads = 4
        config.text.intermediate_size = 128
        config.projection_dim = 32

        model = CLAP(config)

        mel = mx.random.normal((2, 1, 64, 256))
        input_ids = mx.array(np.random.randint(0, 1000, (2, 20)))

        result = model(audio=mel, input_ids=input_ids)

        assert "audio_embeds" in result
        assert "text_embeds" in result
        assert "logits_per_audio" in result
        assert "logits_per_text" in result

        assert result["audio_embeds"].shape == (2, 32)
        assert result["text_embeds"].shape == (2, 32)
        assert result["logits_per_audio"].shape == (2, 2)
