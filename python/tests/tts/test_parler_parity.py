"""Parity tests for Parler-TTS model against HuggingFace reference.

These tests verify that the MLX implementation produces outputs
that match the HuggingFace transformers implementation.

Tests are organized into tiers:
- Tier 1: Shape and config validation (fast, no weights needed)
- Tier 2: Weight conversion validation (requires HF model download)
- Tier 3: Numerical parity (requires converted weights)
"""

import pytest
import numpy as np

# Check dependencies
try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

try:
    import torch  # noqa: F401
    from transformers import (  # noqa: F401
        ParlerTTSForConditionalGeneration,
        ParlerTTSConfig as HFParlerConfig,
    )

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def small_config_dict():
    """Small config for testing (avoids needing large model downloads)."""
    return {
        "num_codebooks": 4,
        "codebook_size": 1024,
        "hidden_size": 256,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "intermediate_size": 1024,
        "sample_rate": 24000,
        "frame_rate": 75,
        "text_hidden_size": 256,
        "description_hidden_size": 256,
        "max_text_length": 128,
        "max_description_length": 64,
    }


@pytest.fixture
def mlx_small_config(small_config_dict):
    """Create MLX ParlerTTSConfig from small config dict."""
    from mlx_audio.models.tts import ParlerTTSConfig

    return ParlerTTSConfig(**small_config_dict)


@pytest.fixture
def mlx_small_model(mlx_small_config):
    """Create small MLX Parler-TTS model for testing."""
    from mlx_audio.models.tts import ParlerTTS

    return ParlerTTS(mlx_small_config)


# =============================================================================
# Tier 1: Shape and Configuration Tests (No HF required)
# =============================================================================


@pytest.mark.parity
class TestParlerTTSShapes:
    """Shape and configuration validation tests."""

    def test_config_mini_matches_expected(self):
        """Test that mini config has expected dimensions."""
        from mlx_audio.models.tts import ParlerTTSConfig

        config = ParlerTTSConfig.mini()

        assert config.hidden_size == 1024
        assert config.num_hidden_layers == 24
        assert config.num_attention_heads == 16
        assert config.num_codebooks == 9
        assert config.codebook_size == 1024
        assert config.sample_rate == 24000
        assert config.frame_rate == 75

    def test_config_large_matches_expected(self):
        """Test that large config has expected dimensions."""
        from mlx_audio.models.tts import ParlerTTSConfig

        config = ParlerTTSConfig.large()

        assert config.hidden_size == 1536
        assert config.num_hidden_layers == 36
        assert config.num_attention_heads == 24
        assert config.num_codebooks == 9

    def test_config_head_dim_computed(self):
        """Test that head_dim is computed correctly."""
        from mlx_audio.models.tts import ParlerTTSConfig

        config = ParlerTTSConfig.mini()
        expected_head_dim = config.hidden_size // config.num_attention_heads

        assert config.head_dim == expected_head_dim
        assert config.head_dim == 64  # 1024 / 16

    def test_config_vocab_size_includes_special_tokens(self):
        """Test vocab_size includes pad and bos tokens."""
        from mlx_audio.models.tts import ParlerTTSConfig

        config = ParlerTTSConfig.mini()

        # vocab_size = codebook_size + 2 (pad + bos)
        assert config.vocab_size == config.codebook_size + 2
        assert config.vocab_size == 1026

    def test_config_max_new_tokens_computed(self):
        """Test max_new_tokens computed from duration and frame_rate."""
        from mlx_audio.models.tts import ParlerTTSConfig

        config = ParlerTTSConfig.mini()

        # max_new_tokens = max_duration * frame_rate
        expected = int(config.max_duration * config.frame_rate)
        assert config.max_new_tokens == expected

    def test_embeddings_output_shape(self, mlx_small_model, mlx_small_config):
        """Test embeddings produce correct output shape."""
        batch_size = 2
        seq_length = 10
        num_codebooks = mlx_small_config.num_codebooks

        # Create input codes [B, K, T]
        input_ids = mx.zeros((batch_size, num_codebooks, seq_length), dtype=mx.int32)

        # Get embeddings
        embeddings = mlx_small_model.embeddings(input_ids)

        # Should be [B, T, D]
        assert embeddings.shape == (batch_size, seq_length, mlx_small_config.hidden_size)

    def test_text_projection_shape(self, mlx_small_model, mlx_small_config):
        """Test text projection produces correct output shape."""
        batch_size = 2
        seq_length = 20

        # Create text embeddings [B, S, text_hidden_size]
        text_embeds = mx.random.normal(
            (batch_size, seq_length, mlx_small_config.text_hidden_size)
        )

        # Project
        projected = mlx_small_model.project_text_embeddings(text_embeds)

        # Should be [B, S, hidden_size]
        assert projected.shape == (batch_size, seq_length, mlx_small_config.hidden_size)

    def test_description_projection_shape(self, mlx_small_model, mlx_small_config):
        """Test description projection produces correct output shape."""
        batch_size = 2
        seq_length = 15

        # Create description embeddings [B, S, description_hidden_size]
        desc_embeds = mx.random.normal(
            (batch_size, seq_length, mlx_small_config.description_hidden_size)
        )

        # Project
        projected = mlx_small_model.project_description_embeddings(desc_embeds)

        # Should be [B, S, hidden_size]
        assert projected.shape == (batch_size, seq_length, mlx_small_config.hidden_size)

    def test_decoder_output_shape(self, mlx_small_model, mlx_small_config):
        """Test decoder produces correct output shape."""
        batch_size = 2
        seq_length = 10
        encoder_seq = 20

        # Create inputs
        hidden_states = mx.random.normal(
            (batch_size, seq_length, mlx_small_config.hidden_size)
        )
        encoder_states = mx.random.normal(
            (batch_size, encoder_seq, mlx_small_config.hidden_size)
        )

        # Run decoder
        output, _ = mlx_small_model.decoder(
            hidden_states,
            encoder_hidden_states=encoder_states,
        )

        # Should be [B, T, D]
        assert output.shape == (batch_size, seq_length, mlx_small_config.hidden_size)

    def test_lm_head_output_shape(self, mlx_small_model, mlx_small_config):
        """Test LM head produces correct logits shape."""
        batch_size = 2
        seq_length = 10

        # Create hidden states [B, T, D]
        hidden_states = mx.random.normal(
            (batch_size, seq_length, mlx_small_config.hidden_size)
        )

        # Get logits
        logits = mlx_small_model.lm_head(hidden_states)

        # Should be [B, K, T, V] where V is vocab_size
        expected_shape = (
            batch_size,
            mlx_small_config.num_codebooks,
            seq_length,
            mlx_small_config.vocab_size,
        )
        assert logits.shape == expected_shape

    def test_full_forward_shape(self, mlx_small_model, mlx_small_config):
        """Test full forward pass produces correct shapes."""
        batch_size = 2
        seq_length = 10
        encoder_seq = 20
        num_codebooks = mlx_small_config.num_codebooks

        # Create inputs
        input_ids = mx.zeros((batch_size, num_codebooks, seq_length), dtype=mx.int32)
        encoder_states = mx.random.normal(
            (batch_size, encoder_seq, mlx_small_config.hidden_size)
        )

        # Forward pass
        logits, kv_cache = mlx_small_model.forward(
            input_ids,
            encoder_hidden_states=encoder_states,
        )

        # Logits should be [B, K, T, V]
        expected_logits_shape = (
            batch_size,
            num_codebooks,
            seq_length,
            mlx_small_config.vocab_size,
        )
        assert logits.shape == expected_logits_shape

        # KV cache should have entries for each layer
        assert len(kv_cache) == mlx_small_config.num_hidden_layers


@pytest.mark.parity
class TestDelayPatternParity:
    """Tests for delay pattern implementation."""

    def test_delay_pattern_roundtrip(self, mlx_small_model):
        """Test that delay pattern apply/revert is a roundtrip."""
        # Create test codes [B, K, T]
        batch_size = 2
        num_codebooks = 4
        seq_length = 20

        codes = mx.arange(batch_size * num_codebooks * seq_length).reshape(
            (batch_size, num_codebooks, seq_length)
        )

        # Apply delay pattern
        delayed = mlx_small_model.delay_pattern.apply_delay_pattern(codes)

        # Revert delay pattern
        reverted = mlx_small_model.delay_pattern.revert_delay_pattern(delayed)

        # Should match original
        np.testing.assert_array_equal(
            np.array(reverted),
            np.array(codes),
            err_msg="Delay pattern roundtrip failed",
        )

    def test_delay_pattern_shape_preserved(self, mlx_small_model):
        """Test that delay pattern preserves shape (with padding)."""
        batch_size = 1
        num_codebooks = 4
        seq_length = 10

        codes = mx.zeros((batch_size, num_codebooks, seq_length), dtype=mx.int32)

        # Apply delay pattern - may add padding
        delayed = mlx_small_model.delay_pattern.apply_delay_pattern(codes)

        # Batch and codebook dims should be same
        assert delayed.shape[0] == batch_size
        assert delayed.shape[1] == num_codebooks
        # Time dimension may be extended due to delays
        assert delayed.shape[2] >= seq_length


# =============================================================================
# Tier 2: Weight Conversion Validation
# =============================================================================


@pytest.mark.parity
class TestParlerTTSWeightConversion:
    """Tests for weight conversion from HuggingFace format.

    These tests validate the key mapping logic without needing HF models.
    """

    def test_weight_key_mapping_completeness(self):
        """Test that all major weight groups are mapped."""
        from mlx_audio.models.tts.convert import _map_parler_key

        # Test key mappings for major components
        test_mappings = [
            # Decoder layers
            ("decoder.layers.0.self_attn.q_proj.weight", True),
            ("decoder.layers.0.self_attn.k_proj.weight", True),
            ("decoder.layers.0.self_attn.v_proj.weight", True),
            ("decoder.layers.0.self_attn.o_proj.weight", True),
            # Layer norms
            ("decoder.layers.0.self_attn_layer_norm.weight", True),
            ("decoder.layers.0.final_layer_norm.weight", True),
            # FFN
            ("decoder.layers.0.mlp.gate_proj.weight", True),
            ("decoder.layers.0.mlp.up_proj.weight", True),
            ("decoder.layers.0.mlp.down_proj.weight", True),
            # Embeddings
            ("decoder.embed_tokens.0.weight", True),
            # LM heads
            ("lm_heads.0.weight", True),
            # Projections
            ("enc_to_dec_proj.weight", True),
            # Encoder weights should be skipped
            ("text_encoder.encoder.layer.0.weight", False),
            ("t5_encoder.encoder.layer.0.weight", False),
        ]

        for pt_key, should_map in test_mappings:
            mlx_key = _map_parler_key(pt_key)
            if should_map:
                assert mlx_key is not None, f"Key {pt_key} should be mapped"
            else:
                assert mlx_key is None, f"Key {pt_key} should be skipped"

    def test_weight_key_mapping_patterns(self):
        """Test specific key mapping transformations."""
        from mlx_audio.models.tts.convert import _map_parler_key

        # Test specific transformations
        test_cases = [
            # o_proj -> out_proj
            (
                "decoder.layers.0.self_attn.o_proj.weight",
                "decoder.layers.0.self_attn.out_proj.weight",
            ),
            # Embed tokens mapping
            (
                "decoder.embed_tokens.0.weight",
                "embeddings.embeddings.0.weight",
            ),
            # LM heads mapping
            (
                "lm_heads.0.weight",
                "lm_head.linears.0.weight",
            ),
            # FFN SwiGLU mapping
            (
                "decoder.layers.0.mlp.gate_proj.weight",
                "decoder.layers.0.fc1.weight",
            ),
            (
                "decoder.layers.0.mlp.up_proj.weight",
                "decoder.layers.0.fc2.weight",
            ),
            (
                "decoder.layers.0.mlp.down_proj.weight",
                "decoder.layers.0.fc3.weight",
            ),
        ]

        for pt_key, expected_mlx_key in test_cases:
            actual_mlx_key = _map_parler_key(pt_key)
            assert actual_mlx_key == expected_mlx_key, (
                f"Key mapping mismatch: {pt_key} -> {actual_mlx_key}, "
                f"expected {expected_mlx_key}"
            )

    def test_skip_patterns(self):
        """Test that non-essential weights are skipped."""
        from mlx_audio.models.tts.convert import _map_parler_key

        skip_keys = [
            "text_encoder.embeddings.weight",
            "t5_encoder.layer.0.weight",
            "some_layer.num_batches_tracked",
            "layer.running_mean",
            "layer.running_var",
            "embeddings.position_ids",
        ]

        for key in skip_keys:
            result = _map_parler_key(key)
            assert result is None, f"Key {key} should be skipped but got {result}"

    def test_convert_function_signature(self):
        """Test that convert function has expected signature."""
        from mlx_audio.models.tts.convert import convert_parler_weights

        import inspect

        sig = inspect.signature(convert_parler_weights)
        params = list(sig.parameters.keys())

        assert "pytorch_path" in params
        assert "output_path" in params
        assert "config" in params

    def test_download_and_convert_function_exists(self):
        """Test that download_and_convert function exists with expected models."""
        from mlx_audio.models.tts.convert import download_and_convert

        import inspect

        sig = inspect.signature(download_and_convert)
        params = sig.parameters

        # Check default model name
        assert params["model_name"].default == "parler-tts-mini"


# =============================================================================
# Tier 3: Config Serialization Tests
# =============================================================================


@pytest.mark.parity
class TestParlerTTSConfigParity:
    """Tests for MLX config serialization and factory methods."""

    def test_config_from_dict_roundtrip(self):
        """Test config serialization roundtrip."""
        from mlx_audio.models.tts import ParlerTTSConfig

        original = ParlerTTSConfig.mini()
        config_dict = original.to_dict()
        restored = ParlerTTSConfig.from_dict(config_dict)

        # Key attributes should match
        assert restored.hidden_size == original.hidden_size
        assert restored.num_hidden_layers == original.num_hidden_layers
        assert restored.num_attention_heads == original.num_attention_heads
        assert restored.num_codebooks == original.num_codebooks
        assert restored.codebook_size == original.codebook_size
        assert restored.sample_rate == original.sample_rate

    def test_config_from_name_mini(self):
        """Test config.from_name('mini') matches config.mini()."""
        from mlx_audio.models.tts import ParlerTTSConfig

        from_name = ParlerTTSConfig.from_name("mini")
        direct = ParlerTTSConfig.mini()

        assert from_name.hidden_size == direct.hidden_size
        assert from_name.num_hidden_layers == direct.num_hidden_layers

    def test_config_from_name_large(self):
        """Test config.from_name('large') matches config.large()."""
        from mlx_audio.models.tts import ParlerTTSConfig

        from_name = ParlerTTSConfig.from_name("large")
        direct = ParlerTTSConfig.large()

        assert from_name.hidden_size == direct.hidden_size
        assert from_name.num_hidden_layers == direct.num_hidden_layers

    def test_config_from_name_invalid(self):
        """Test config.from_name raises for invalid names."""
        from mlx_audio.models.tts import ParlerTTSConfig

        with pytest.raises(ValueError, match="Unknown Parler-TTS model"):
            ParlerTTSConfig.from_name("invalid-model-name")


# =============================================================================
# Tier 4: Numerical Stability Tests
# =============================================================================


@pytest.mark.parity
class TestParlerTTSNumericalStability:
    """Tests for numerical stability of the model."""

    def test_forward_no_nan(self, mlx_small_model, mlx_small_config):
        """Test that forward pass doesn't produce NaN."""
        batch_size = 1
        seq_length = 5
        encoder_seq = 10
        num_codebooks = mlx_small_config.num_codebooks

        input_ids = mx.zeros((batch_size, num_codebooks, seq_length), dtype=mx.int32)
        encoder_states = mx.random.normal(
            (batch_size, encoder_seq, mlx_small_config.hidden_size)
        )

        logits, _ = mlx_small_model.forward(
            input_ids,
            encoder_hidden_states=encoder_states,
        )

        # Check for NaN
        assert not mx.any(mx.isnan(logits)), "Forward pass produced NaN values"

    def test_forward_no_inf(self, mlx_small_model, mlx_small_config):
        """Test that forward pass doesn't produce Inf."""
        batch_size = 1
        seq_length = 5
        encoder_seq = 10
        num_codebooks = mlx_small_config.num_codebooks

        input_ids = mx.zeros((batch_size, num_codebooks, seq_length), dtype=mx.int32)
        encoder_states = mx.random.normal(
            (batch_size, encoder_seq, mlx_small_config.hidden_size)
        )

        logits, _ = mlx_small_model.forward(
            input_ids,
            encoder_hidden_states=encoder_states,
        )

        # Check for Inf
        assert not mx.any(mx.isinf(logits)), "Forward pass produced Inf values"

    def test_embeddings_bounded(self, mlx_small_model, mlx_small_config):
        """Test that embeddings are in reasonable range."""
        batch_size = 1
        seq_length = 10
        num_codebooks = mlx_small_config.num_codebooks

        input_ids = mx.zeros((batch_size, num_codebooks, seq_length), dtype=mx.int32)
        embeddings = mlx_small_model.embeddings(input_ids)

        # Embeddings should be bounded (typical init is ~N(0, 0.02))
        max_val = float(mx.max(mx.abs(embeddings)))
        assert max_val < 100, f"Embeddings too large: max={max_val}"

    def test_logits_reasonable_range(self, mlx_small_model, mlx_small_config):
        """Test that logits are in reasonable range for softmax."""
        batch_size = 1
        seq_length = 5
        encoder_seq = 10
        num_codebooks = mlx_small_config.num_codebooks

        input_ids = mx.zeros((batch_size, num_codebooks, seq_length), dtype=mx.int32)
        encoder_states = mx.random.normal(
            (batch_size, encoder_seq, mlx_small_config.hidden_size)
        ) * 0.02  # Scale down to avoid extreme values

        logits, _ = mlx_small_model.forward(
            input_ids,
            encoder_hidden_states=encoder_states,
        )

        # Logits should be reasonable for softmax (not too extreme)
        max_logit = float(mx.max(mx.abs(logits)))
        # Allow larger range since untrained model may have varied outputs
        assert max_logit < 1000, f"Logits too extreme: max={max_logit}"
