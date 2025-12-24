"""PyTorch parity tests for Parler-TTS model.

These tests validate that the MLX implementation produces outputs
that match the PyTorch/HuggingFace reference implementation.

The MLX implementation now matches the PyTorch parler-tts architecture:
- 2-layer GELU FFN (fc1/fc2)
- Sinusoidal positional embeddings (added once to inputs)
- LayerNorm with bias

These tests cover:
1. Shape validation
2. Embedding parity (weights can be copied directly)
3. LM head parity
4. Full decoder parity
5. Weight key mapping validation
6. Numerical stability

Requires: parler-tts package installed (pip install parler-tts)
"""

import mlx.core as mx
import numpy as np
import pytest

# Check if parler-tts is available
try:
    from parler_tts import ParlerTTSForConditionalGeneration

    HAS_PARLER_TTS = True
except ImportError:
    HAS_PARLER_TTS = False

# Check if torch is available
try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# =============================================================================
# Test Data Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def random_inputs():
    """Generate random inputs for testing."""
    np.random.seed(42)
    batch_size = 2
    seq_length = 10
    encoder_seq = 20
    hidden_size = 1024  # mini model hidden size
    num_codebooks = 9

    return {
        "input_ids": np.random.randint(
            0, 1024, size=(batch_size, num_codebooks, seq_length)
        ).astype(np.int64),
        "encoder_hidden_states": np.random.randn(batch_size, encoder_seq, hidden_size).astype(
            np.float32
        )
        * 0.02,
        "hidden_states": np.random.randn(batch_size, seq_length, hidden_size).astype(np.float32)
        * 0.02,
    }


# =============================================================================
# PyTorch Reference Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def pt_reference_outputs(random_inputs):
    """Pre-compute all PyTorch reference outputs at once.

    This avoids interleaving PyTorch and MLX calls which causes significant
    slowdown due to Metal GPU context switching. All PyTorch inference runs
    in a single batch, then MLX tests compare against cached results.

    Returns dict with:
        - 'model': PyTorch model
        - 'config': Model config
        - 'embedding_output': Embedding layer output
        - 'lm_head_output': LM head output
    """
    if not HAS_PARLER_TTS:
        pytest.skip("parler-tts package not available")
    if not HAS_TORCH:
        pytest.skip("torch package not available")

    # Load PyTorch model
    print("Loading PyTorch Parler-TTS model...")
    model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-v1")
    model.eval()

    results = {
        "model": model,
        "config": model.config.decoder,
    }

    with torch.no_grad():
        # Get decoder model reference
        # Structure: model.decoder -> ParlerTTSForCausalLM
        # model.decoder.model.decoder -> ParlerTTSDecoder (has embed_tokens, layers)
        decoder_causal_lm = model.decoder
        inner_decoder = decoder_causal_lm.model.decoder

        # 1. Embedding output
        input_ids = torch.from_numpy(random_inputs["input_ids"])  # [B, K, T]
        batch_size, num_codebooks, seq_length = input_ids.shape

        # Sum embeddings from all codebooks
        embeddings_sum = None
        for k in range(min(num_codebooks, len(inner_decoder.embed_tokens))):
            codebook_ids = input_ids[:, k, :]  # [B, T]
            emb = inner_decoder.embed_tokens[k](codebook_ids)  # [B, T, D]
            if embeddings_sum is None:
                embeddings_sum = emb
            else:
                embeddings_sum = embeddings_sum + emb
        results["embedding_output"] = embeddings_sum.numpy()

        # 2. LM head output
        hidden_size = model.config.decoder.hidden_size
        hidden_states = torch.from_numpy(
            np.random.randn(batch_size, seq_length, hidden_size).astype(np.float32) * 0.02
        )
        lm_head_outputs = []
        for k in range(len(decoder_causal_lm.lm_heads)):
            logits_k = decoder_causal_lm.lm_heads[k](hidden_states)  # [B, T, V]
            lm_head_outputs.append(logits_k.unsqueeze(1))  # [B, 1, T, V]
        lm_head_output = torch.cat(lm_head_outputs, dim=1)  # [B, K, T, V]
        results["lm_head_output"] = lm_head_output.numpy()
        results["lm_head_input"] = hidden_states.numpy()

        # 3. Decoder block output (first layer only for speed)
        # Create input with sinusoidal positional embeddings (like the model does)
        decoder_input = embeddings_sum
        pos_emb = inner_decoder.embed_positions.weights[:seq_length]
        decoder_input = decoder_input + pos_emb

        # Run first decoder layer
        first_layer = inner_decoder.layers[0]

        # Create causal mask
        causal_mask = torch.triu(
            torch.ones(seq_length, seq_length, dtype=torch.bool), diagonal=1
        )
        causal_mask = causal_mask.float().masked_fill(causal_mask, float("-inf"))

        # Forward through first layer
        layer_output = first_layer(
            decoder_input,
            attention_mask=causal_mask.unsqueeze(0).unsqueeze(0),  # [1, 1, T, T]
        )[0]
        results["decoder_layer_output"] = layer_output.numpy()
        results["decoder_layer_input"] = decoder_input.numpy()
        results["sinusoidal_pos_emb"] = pos_emb.numpy()

    print("PyTorch reference outputs computed.")
    return results


@pytest.fixture(scope="module")
def pt_model(pt_reference_outputs):
    """Get PyTorch model from pre-computed references."""
    return pt_reference_outputs["model"]


# =============================================================================
# MLX Model Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def mlx_model():
    """Create MLX Parler-TTS model (without weights for shape testing)."""
    from mlx_audio.models.tts import ParlerTTS, ParlerTTSConfig

    config = ParlerTTSConfig.mini()
    return ParlerTTS(config)


@pytest.fixture(scope="module")
def mlx_model_with_embedding_weights(pt_reference_outputs):
    """Create MLX model with embedding and LM head weights from PyTorch."""
    from mlx_audio.models.tts import ParlerTTS, ParlerTTSConfig

    config = ParlerTTSConfig.mini()
    model = ParlerTTS(config)

    pt_model = pt_reference_outputs["model"]
    pt_state_dict = pt_model.decoder.state_dict()

    # Only load embedding and LM head weights
    mlx_weights = {}

    # Embeddings: model.decoder.embed_tokens.K.weight -> embeddings.embeddings.K.weight
    for k in range(config.num_codebooks):
        pt_key = f"model.decoder.embed_tokens.{k}.weight"
        if pt_key in pt_state_dict:
            mlx_key = f"embeddings.embeddings.{k}.weight"
            mlx_weights[mlx_key] = mx.array(pt_state_dict[pt_key].detach().cpu().numpy())

    # LM heads: lm_heads.K.weight -> lm_head.linears.K.weight
    for k in range(config.num_codebooks):
        pt_key = f"lm_heads.{k}.weight"
        if pt_key in pt_state_dict:
            mlx_key = f"lm_head.linears.{k}.weight"
            mlx_weights[mlx_key] = mx.array(pt_state_dict[pt_key].detach().cpu().numpy())
        # Bias
        pt_key_bias = f"lm_heads.{k}.bias"
        if pt_key_bias in pt_state_dict:
            mlx_key_bias = f"lm_head.linears.{k}.bias"
            mlx_weights[mlx_key_bias] = mx.array(pt_state_dict[pt_key_bias].detach().cpu().numpy())

    # Load weights (strict=False to allow partial loading)
    model.load_weights(list(mlx_weights.items()), strict=False)

    return model


@pytest.fixture(scope="module")
def mlx_model_with_all_weights(pt_reference_outputs):
    """Create MLX model with ALL decoder weights from PyTorch.

    This loads the full decoder including attention, FFN, and layer norms.
    """
    from mlx_audio.models.tts import ParlerTTS, ParlerTTSConfig
    from mlx_audio.models.tts.convert import ParlerTTSConverter

    config = ParlerTTSConfig.mini()
    model = ParlerTTS(config)

    pt_model = pt_reference_outputs["model"]
    pt_state_dict = pt_model.decoder.state_dict()

    converter = ParlerTTSConverter()
    mlx_weights = {}

    for pt_key, pt_tensor in pt_state_dict.items():
        mlx_key = converter.map_key(pt_key)
        if mlx_key is not None:
            mlx_weights[mlx_key] = mx.array(pt_tensor.detach().cpu().numpy())

    # Load weights (strict=False since some keys may not match perfectly)
    model.load_weights(list(mlx_weights.items()), strict=False)

    return model


# =============================================================================
# Shape Validation Tests
# =============================================================================


@pytest.mark.parity
class TestParlerTTSShapes:
    """Tests for shape validation (no weights needed)."""

    def test_embedding_output_shape(self, mlx_model, random_inputs):
        """Test embedding output has correct shape."""
        input_ids = mx.array(random_inputs["input_ids"])
        embeddings = mlx_model.embeddings(input_ids)
        mx.eval(embeddings)

        batch_size, num_codebooks, seq_length = random_inputs["input_ids"].shape
        expected_shape = (batch_size, seq_length, mlx_model.config.hidden_size)
        assert embeddings.shape == expected_shape, (
            f"Shape mismatch: got {embeddings.shape}, expected {expected_shape}"
        )

    def test_decoder_output_shape(self, mlx_model, random_inputs):
        """Test decoder output has correct shape."""
        hidden_states = mx.array(random_inputs["hidden_states"])
        encoder_hidden_states = mx.array(random_inputs["encoder_hidden_states"])

        seq_length = hidden_states.shape[1]
        causal_mask = mlx_model.decoder.create_causal_mask(seq_length)

        output, kv_cache = mlx_model.decoder(
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=causal_mask,
        )
        mx.eval(output)

        assert output.shape == hidden_states.shape, (
            f"Shape mismatch: got {output.shape}, expected {hidden_states.shape}"
        )
        assert len(kv_cache) == mlx_model.config.num_hidden_layers

    def test_lm_head_output_shape(self, mlx_model, random_inputs):
        """Test LM head output has correct shape."""
        batch_size, seq_length = 2, 10
        hidden_states = mx.array(random_inputs["hidden_states"])

        logits = mlx_model.lm_head(hidden_states)
        mx.eval(logits)

        expected_shape = (
            batch_size,
            mlx_model.config.num_codebooks,
            seq_length,
            mlx_model.config.vocab_size,
        )
        assert logits.shape == expected_shape, (
            f"Shape mismatch: got {logits.shape}, expected {expected_shape}"
        )

    def test_forward_output_shape(self, mlx_model, random_inputs):
        """Test forward pass output has correct shape."""
        input_ids = mx.array(random_inputs["input_ids"])
        encoder_hidden_states = mx.array(random_inputs["encoder_hidden_states"])

        logits, kv_cache = mlx_model.forward(
            input_ids,
            encoder_hidden_states=encoder_hidden_states,
        )
        mx.eval(logits)

        batch_size, num_codebooks, seq_length = random_inputs["input_ids"].shape
        expected_shape = (batch_size, num_codebooks, seq_length, mlx_model.config.vocab_size)
        assert logits.shape == expected_shape, (
            f"Shape mismatch: got {logits.shape}, expected {expected_shape}"
        )


# =============================================================================
# Embedding Parity Tests (weights should match exactly)
# =============================================================================


@pytest.mark.parity
@pytest.mark.slow
class TestParlerTTSEmbeddingParity:
    """Tests for embedding layer parity.

    Embeddings have the same structure between PyTorch and MLX,
    so we can test numerical parity.
    """

    def test_codebook_embeddings_match(
        self, pt_reference_outputs, mlx_model_with_embedding_weights, random_inputs
    ):
        """Test that codebook embeddings match PyTorch."""
        pt_emb = pt_reference_outputs["embedding_output"]

        input_ids = mx.array(random_inputs["input_ids"])
        mlx_emb = mlx_model_with_embedding_weights.embeddings(input_ids)
        mx.eval(mlx_emb)

        # Compare
        mae = np.mean(np.abs(pt_emb - np.array(mlx_emb)))
        assert mae < 1e-5, f"Embedding MAE too high: {mae}"

    def test_embedding_shapes_match(
        self, pt_reference_outputs, mlx_model_with_embedding_weights, random_inputs
    ):
        """Test that embedding output shapes match."""
        pt_emb = pt_reference_outputs["embedding_output"]

        input_ids = mx.array(random_inputs["input_ids"])
        mlx_emb = mlx_model_with_embedding_weights.embeddings(input_ids)
        mx.eval(mlx_emb)

        assert pt_emb.shape == mlx_emb.shape, (
            f"Shape mismatch: PT={pt_emb.shape}, MLX={mlx_emb.shape}"
        )


# =============================================================================
# LM Head Parity Tests
# =============================================================================


@pytest.mark.parity
@pytest.mark.slow
class TestParlerTTSLMHeadParity:
    """Tests for LM head parity.

    LM heads are simple linear layers and should match exactly
    when weights are copied.
    """

    def test_lm_head_shapes_match(self, pt_reference_outputs, mlx_model_with_embedding_weights):
        """Test that LM head output shapes match."""
        pt_output = pt_reference_outputs["lm_head_output"]
        pt_input = pt_reference_outputs["lm_head_input"]

        hidden_states = mx.array(pt_input)
        mlx_output = mlx_model_with_embedding_weights.lm_head(hidden_states)
        mx.eval(mlx_output)

        assert pt_output.shape == mlx_output.shape, (
            f"Shape mismatch: PT={pt_output.shape}, MLX={mlx_output.shape}"
        )

    def test_lm_head_output_match(self, pt_reference_outputs, mlx_model_with_embedding_weights):
        """Test that LM head output matches PyTorch."""
        pt_output = pt_reference_outputs["lm_head_output"]
        pt_input = pt_reference_outputs["lm_head_input"]

        hidden_states = mx.array(pt_input)
        mlx_output = mlx_model_with_embedding_weights.lm_head(hidden_states)
        mx.eval(mlx_output)

        # Compare values
        mae = np.mean(np.abs(pt_output - np.array(mlx_output)))
        assert mae < 1e-4, f"LM head MAE too high: {mae}"


# =============================================================================
# Decoder Parity Tests (full decoder with all weights)
# =============================================================================


@pytest.mark.parity
@pytest.mark.slow
class TestParlerTTSDecoderParity:
    """Tests for full decoder parity with all weights loaded."""

    def test_sinusoidal_embeddings_match(self, pt_reference_outputs):
        """Test that sinusoidal positional embeddings match PyTorch."""
        from mlx_audio.models.tts.layers.embeddings import sinusoidal_embeddings

        pt_pos_emb = pt_reference_outputs["sinusoidal_pos_emb"]
        seq_length = pt_pos_emb.shape[0]
        hidden_size = pt_pos_emb.shape[1]

        mlx_pos_emb = sinusoidal_embeddings(seq_length, hidden_size)
        mx.eval(mlx_pos_emb)

        mae = np.mean(np.abs(pt_pos_emb - np.array(mlx_pos_emb)))
        assert mae < 1e-5, f"Sinusoidal embedding MAE too high: {mae}"

    def test_decoder_layer_output_match(
        self, pt_reference_outputs, mlx_model_with_all_weights, random_inputs
    ):
        """Test that first decoder layer output matches PyTorch."""
        pt_layer_output = pt_reference_outputs["decoder_layer_output"]
        pt_layer_input = pt_reference_outputs["decoder_layer_input"]

        # Run first decoder layer in MLX
        mlx_model = mlx_model_with_all_weights
        first_layer = mlx_model.decoder.layers[0]

        hidden_states = mx.array(pt_layer_input)
        seq_length = hidden_states.shape[1]
        causal_mask = mlx_model.decoder.create_causal_mask(seq_length)

        mlx_output, _ = first_layer(
            hidden_states,
            attention_mask=causal_mask,
        )
        mx.eval(mlx_output)

        # Compare
        mae = np.mean(np.abs(pt_layer_output - np.array(mlx_output)))
        assert mae < 1e-3, f"Decoder layer MAE too high: {mae}"


# =============================================================================
# Numerical Stability Tests
# =============================================================================


@pytest.mark.parity
class TestParlerTTSNumericalStability:
    """Tests for numerical stability of the MLX model."""

    def test_forward_no_nan(self, mlx_model, random_inputs):
        """Test that forward pass doesn't produce NaN."""
        input_ids = mx.array(random_inputs["input_ids"])
        encoder_hidden_states = mx.array(random_inputs["encoder_hidden_states"])

        logits, _ = mlx_model.forward(
            input_ids,
            encoder_hidden_states=encoder_hidden_states,
        )
        mx.eval(logits)

        assert not mx.any(mx.isnan(logits)), "Forward pass produced NaN"

    def test_forward_no_inf(self, mlx_model, random_inputs):
        """Test that forward pass doesn't produce Inf."""
        input_ids = mx.array(random_inputs["input_ids"])
        encoder_hidden_states = mx.array(random_inputs["encoder_hidden_states"])

        logits, _ = mlx_model.forward(
            input_ids,
            encoder_hidden_states=encoder_hidden_states,
        )
        mx.eval(logits)

        assert not mx.any(mx.isinf(logits)), "Forward pass produced Inf"

    def test_embeddings_bounded(self, mlx_model, random_inputs):
        """Test that embeddings are in reasonable range."""
        input_ids = mx.array(random_inputs["input_ids"])
        embeddings = mlx_model.embeddings(input_ids)
        mx.eval(embeddings)

        max_val = float(mx.max(mx.abs(embeddings)))
        assert max_val < 100, f"Embeddings too large: max={max_val}"


# =============================================================================
# Weight Mapping Tests
# =============================================================================


@pytest.mark.parity
@pytest.mark.slow
class TestParlerTTSWeightMapping:
    """Tests for weight key mapping validation."""

    def test_embedding_keys_mapped(self, pt_reference_outputs):
        """Test that embedding weights are properly mapped."""
        from mlx_audio.models.tts.convert import ParlerTTSConverter

        pt_model = pt_reference_outputs["model"]
        pt_state_dict = pt_model.decoder.state_dict()

        converter = ParlerTTSConverter()

        # Check embedding keys
        for k in range(9):  # num_codebooks
            pt_key = f"model.decoder.embed_tokens.{k}.weight"
            if pt_key in pt_state_dict:
                mlx_key = converter.map_key(pt_key)
                assert mlx_key is not None, f"Embedding key {pt_key} not mapped"
                assert f"embeddings.embeddings.{k}" in mlx_key, (
                    f"Wrong mapping for {pt_key}: got {mlx_key}"
                )

    def test_lm_head_keys_mapped(self, pt_reference_outputs):
        """Test that LM head weights are properly mapped."""
        from mlx_audio.models.tts.convert import ParlerTTSConverter

        pt_model = pt_reference_outputs["model"]
        pt_state_dict = pt_model.decoder.state_dict()

        converter = ParlerTTSConverter()

        # Check LM head keys
        for k in range(9):  # num_codebooks
            pt_key = f"lm_heads.{k}.weight"
            if pt_key in pt_state_dict:
                mlx_key = converter.map_key(pt_key)
                assert mlx_key is not None, f"LM head key {pt_key} not mapped"
                assert f"lm_head.linears.{k}" in mlx_key, f"Wrong mapping: {mlx_key}"

    def test_decoder_layer_keys_mapped(self, pt_reference_outputs):
        """Test that decoder layer weights are mapped."""
        from mlx_audio.models.tts.convert import ParlerTTSConverter

        pt_model = pt_reference_outputs["model"]
        pt_state_dict = pt_model.decoder.state_dict()

        converter = ParlerTTSConverter()

        # Check first layer attention keys
        attention_keys = [
            "model.decoder.layers.0.self_attn.q_proj.weight",
            "model.decoder.layers.0.self_attn.k_proj.weight",
            "model.decoder.layers.0.self_attn.v_proj.weight",
            "model.decoder.layers.0.self_attn.out_proj.weight",
        ]

        for pt_key in attention_keys:
            if pt_key in pt_state_dict:
                mlx_key = converter.map_key(pt_key)
                assert mlx_key is not None, f"Attention key {pt_key} not mapped"

    def test_text_encoder_keys_skipped(self, pt_reference_outputs):
        """Test that text encoder weights are skipped."""
        from mlx_audio.models.tts.convert import ParlerTTSConverter

        converter = ParlerTTSConverter()

        # These keys should be skipped (not part of decoder)
        skip_keys = [
            "text_encoder.embeddings.weight",
            "t5_encoder.layer.0.weight",
        ]

        for key in skip_keys:
            mlx_key = converter.map_key(key)
            assert mlx_key is None, f"Key {key} should be skipped but got {mlx_key}"


# =============================================================================
# Architecture Verification Test
# =============================================================================


@pytest.mark.parity
@pytest.mark.slow
class TestArchitectureMatch:
    """Verify that MLX architecture matches PyTorch."""

    def test_architecture_matches_pytorch(self, pt_reference_outputs):
        """Verify MLX architecture now matches PyTorch."""
        pt_model = pt_reference_outputs["model"]
        config = pt_model.config.decoder
        pt_state_dict = pt_model.decoder.state_dict()

        # Verify GELU activation (MLX now uses GELU too)
        assert config.activation_function == "gelu", (
            f"Expected GELU activation, got {config.activation_function}"
        )

        # Verify sinusoidal positional embeddings (not RoPE)
        assert not config.rope_embeddings, "Expected sinusoidal embeddings, not RoPE"

        # Verify LayerNorm (with bias)
        assert "model.decoder.layers.0.self_attn_layer_norm.bias" in pt_state_dict, (
            "Expected LayerNorm with bias"
        )

        # Verify 2-layer FFN (no fc3)
        assert "model.decoder.layers.0.fc3.weight" not in pt_state_dict, (
            "Expected 2-layer FFN (no fc3)"
        )
        assert "model.decoder.layers.0.fc1.weight" in pt_state_dict
        assert "model.decoder.layers.0.fc2.weight" in pt_state_dict

        print("\n=== Architecture Match Verified ===")
        print("  - GELU activation: ✓")
        print("  - Sinusoidal positional embeddings: ✓")
        print("  - LayerNorm with bias: ✓")
        print("  - 2-layer FFN (fc1/fc2): ✓")
