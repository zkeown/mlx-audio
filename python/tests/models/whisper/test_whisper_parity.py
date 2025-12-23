"""Parity tests for Whisper model against HuggingFace implementation.

These tests verify that the MLX implementation produces numerically
equivalent results to the HuggingFace transformers implementation.
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
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperConfig as HFConfig
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


@pytest.fixture
def small_config_dict():
    """Small config for testing (avoids downloading large models)."""
    return {
        "d_model": 64,
        "encoder_layers": 2,
        "decoder_layers": 2,
        "encoder_attention_heads": 4,
        "decoder_attention_heads": 4,
        "encoder_ffn_dim": 256,
        "decoder_ffn_dim": 256,
        "vocab_size": 51865,  # Must be >= pad_token_id (50257)
        "num_mel_bins": 80,
        "max_source_positions": 100,
        "max_target_positions": 50,
        "pad_token_id": 50256,  # Explicit padding token
        "bos_token_id": 50257,
        "eos_token_id": 50256,
        "decoder_start_token_id": 50258,
    }


@pytest.mark.skipif(not HAS_MLX or not HAS_TRANSFORMERS, reason="MLX or transformers not available")
@pytest.mark.slow
@pytest.mark.parity
class TestWhisperEncoderParity:
    """Parity tests for Whisper encoder."""

    def test_encoder_output_shape_matches(self, small_config_dict):
        """Test encoder output shapes match between implementations."""
        from mlx_audio.models.whisper import Whisper, WhisperConfig

        # Create HuggingFace model
        hf_config = HFConfig(**small_config_dict)
        hf_model = WhisperForConditionalGeneration(hf_config)
        hf_model.eval()

        # Create MLX model with matching config
        mlx_config = WhisperConfig(
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
        mlx_model = Whisper(mlx_config)

        # Create random input
        np.random.seed(42)
        mel_np = np.random.randn(1, 80, 200).astype(np.float32)

        # HuggingFace forward
        with torch.no_grad():
            hf_mel = torch.tensor(mel_np)
            hf_output = hf_model.model.encoder(hf_mel).last_hidden_state
            hf_shape = hf_output.shape

        # MLX forward
        mlx_mel = mx.array(mel_np)
        mlx_output = mlx_model.encode(mlx_mel)
        mlx_shape = mlx_output.shape

        # Shapes should match
        assert mlx_shape[0] == hf_shape[0], "Batch dimension mismatch"
        assert mlx_shape[2] == hf_shape[2], "Hidden dimension mismatch"


@pytest.mark.skipif(not HAS_MLX or not HAS_TRANSFORMERS, reason="MLX or transformers not available")
@pytest.mark.slow
@pytest.mark.parity
class TestWhisperDecoderParity:
    """Parity tests for Whisper decoder."""

    def test_decoder_output_shape_matches(self, small_config_dict):
        """Test decoder output shapes match between implementations."""
        from mlx_audio.models.whisper import Whisper, WhisperConfig

        # Create HuggingFace model
        hf_config = HFConfig(**small_config_dict)
        hf_model = WhisperForConditionalGeneration(hf_config)
        hf_model.eval()

        # Create MLX model with matching vocab size
        mlx_config = WhisperConfig(
            n_mels=80,
            n_audio_ctx=100,
            n_audio_state=64,
            n_audio_head=4,
            n_audio_layer=2,
            n_text_ctx=50,
            n_text_state=64,
            n_text_head=4,
            n_text_layer=2,
            n_vocab=51865,  # Match HuggingFace config
        )
        mlx_model = Whisper(mlx_config)

        # Create random inputs
        np.random.seed(42)
        mel_np = np.random.randn(1, 80, 200).astype(np.float32)
        tokens_np = np.array([[1, 2, 3, 4, 5]], dtype=np.int64)

        # HuggingFace forward
        with torch.no_grad():
            hf_mel = torch.tensor(mel_np)
            hf_tokens = torch.tensor(tokens_np)
            hf_output = hf_model(hf_mel, decoder_input_ids=hf_tokens)
            hf_logits_shape = hf_output.logits.shape

        # MLX forward
        mlx_mel = mx.array(mel_np)
        mlx_tokens = mx.array(tokens_np)
        mlx_logits = mlx_model(mlx_mel, mlx_tokens)
        mlx_logits_shape = mlx_logits.shape

        # Shapes should match
        assert mlx_logits_shape == hf_logits_shape


@pytest.mark.skipif(not HAS_MLX or not HAS_TRANSFORMERS, reason="MLX or transformers not available")
@pytest.mark.slow
@pytest.mark.parity
class TestWhisperWeightConversion:
    """Tests for weight conversion from HuggingFace."""

    def test_weight_conversion_keys(self, small_config_dict, tmp_path):
        """Test that converted weights have expected keys."""
        from mlx_audio.models.whisper.convert import _convert_encoder, _convert_decoder

        # Create HuggingFace model
        hf_config = HFConfig(**small_config_dict)
        hf_model = WhisperForConditionalGeneration(hf_config)

        state_dict = hf_model.state_dict()

        # Convert encoder weights
        encoder_weights = _convert_encoder(state_dict)

        # Check some expected keys exist
        assert any("encoder.conv1" in k for k in encoder_weights.keys())
        assert any("encoder.conv2" in k for k in encoder_weights.keys())
        assert any("encoder.blocks" in k for k in encoder_weights.keys())

        # Convert decoder weights
        decoder_weights = _convert_decoder(state_dict)

        # Check some expected keys exist
        assert any("decoder.token_embedding" in k for k in decoder_weights.keys())
        assert any("decoder.positional_embedding" in k for k in decoder_weights.keys())
        assert any("decoder.blocks" in k for k in decoder_weights.keys())


@pytest.mark.skipif(not HAS_MLX or not HAS_TRANSFORMERS, reason="MLX or transformers not available")
@pytest.mark.slow
@pytest.mark.parity
class TestWhisperNumericalParity:
    """Numerical parity tests - verify MLX outputs match HuggingFace within tolerance."""

    def test_encoder_numerical_parity(self, small_config_dict):
        """Test encoder outputs match numerically after weight transfer."""
        from mlx_audio.models.whisper import Whisper, WhisperConfig
        from mlx_audio.models.whisper.convert import _convert_encoder

        # Create HuggingFace model
        hf_config = HFConfig(**small_config_dict)
        hf_model = WhisperForConditionalGeneration(hf_config)
        hf_model.eval()

        # Create MLX model with matching config
        mlx_config = WhisperConfig(
            n_mels=80,
            n_audio_ctx=100,
            n_audio_state=64,
            n_audio_head=4,
            n_audio_layer=2,
            n_text_ctx=50,
            n_text_state=64,
            n_text_head=4,
            n_text_layer=2,
            n_vocab=51865,
        )
        mlx_model = Whisper(mlx_config)

        # Convert and load weights
        state_dict = hf_model.state_dict()
        encoder_weights = _convert_encoder(state_dict)

        # Load only encoder weights into MLX model (strict=False for partial load)
        mlx_weights = {}
        for k, v in encoder_weights.items():
            mlx_weights[k] = mx.array(v)
        mlx_model.load_weights(list(mlx_weights.items()), strict=False)

        # Create deterministic input
        np.random.seed(42)
        mel_np = np.random.randn(1, 80, 200).astype(np.float32)

        # HuggingFace forward
        with torch.no_grad():
            hf_mel = torch.tensor(mel_np)
            hf_output = hf_model.model.encoder(hf_mel).last_hidden_state
            hf_output_np = hf_output.numpy()

        # MLX forward
        mlx_mel = mx.array(mel_np)
        mlx_output = mlx_model.encode(mlx_mel)
        mx.eval(mlx_output)
        mlx_output_np = np.array(mlx_output)

        # Check numerical parity
        # Tightened atol from 1e-5 to 1e-6 (measured: 7.15e-7, headroom: 14x)
        # rtol kept at 1e-4 as differences are dominated by atol for small values
        np.testing.assert_allclose(
            mlx_output_np,
            hf_output_np,
            rtol=1e-4,
            atol=1e-6,
            err_msg="Encoder outputs do not match within tolerance"
        )

    def test_full_model_numerical_parity(self, small_config_dict):
        """Test full model logits match numerically after weight transfer."""
        from mlx_audio.models.whisper import Whisper, WhisperConfig
        from mlx_audio.models.whisper.convert import convert_weights

        # Create HuggingFace model
        hf_config = HFConfig(**small_config_dict)
        hf_model = WhisperForConditionalGeneration(hf_config)
        hf_model.eval()

        # Create MLX model with matching config
        mlx_config = WhisperConfig(
            n_mels=80,
            n_audio_ctx=100,
            n_audio_state=64,
            n_audio_head=4,
            n_audio_layer=2,
            n_text_ctx=50,
            n_text_state=64,
            n_text_head=4,
            n_text_layer=2,
            n_vocab=51865,
        )
        mlx_model = Whisper(mlx_config)

        # Convert and load all weights
        state_dict = hf_model.state_dict()
        all_weights = convert_weights(state_dict)

        mlx_weights = {}
        for k, v in all_weights.items():
            mlx_weights[k] = mx.array(v)
        mlx_model.load_weights(list(mlx_weights.items()))

        # Create deterministic inputs
        np.random.seed(42)
        mel_np = np.random.randn(1, 80, 200).astype(np.float32)
        tokens_np = np.array([[1, 2, 3, 4, 5]], dtype=np.int64)

        # HuggingFace forward
        with torch.no_grad():
            hf_mel = torch.tensor(mel_np)
            hf_tokens = torch.tensor(tokens_np)
            hf_output = hf_model(hf_mel, decoder_input_ids=hf_tokens)
            hf_logits_np = hf_output.logits.numpy()

        # MLX forward
        mlx_mel = mx.array(mel_np)
        mlx_tokens = mx.array(tokens_np)
        mlx_logits = mlx_model(mlx_mel, mlx_tokens)
        mx.eval(mlx_logits)
        mlx_logits_np = np.array(mlx_logits)

        # Check numerical parity
        # Tightened atol from 1e-5 to 1e-6 (measured: 4.02e-7, headroom: 25x)
        # rtol kept at 1e-4 as differences are dominated by atol for small values
        np.testing.assert_allclose(
            mlx_logits_np,
            hf_logits_np,
            rtol=1e-4,
            atol=1e-6,
            err_msg="Full model logits do not match within tolerance"
        )

    def test_decoder_cross_attention_parity(self, small_config_dict):
        """Test decoder cross-attention produces same results."""
        from mlx_audio.models.whisper import Whisper, WhisperConfig
        from mlx_audio.models.whisper.convert import convert_weights

        # Create HuggingFace model
        hf_config = HFConfig(**small_config_dict)
        hf_model = WhisperForConditionalGeneration(hf_config)
        hf_model.eval()

        # Create MLX model
        mlx_config = WhisperConfig(
            n_mels=80,
            n_audio_ctx=100,
            n_audio_state=64,
            n_audio_head=4,
            n_audio_layer=2,
            n_text_ctx=50,
            n_text_state=64,
            n_text_head=4,
            n_text_layer=2,
            n_vocab=51865,
        )
        mlx_model = Whisper(mlx_config)

        # Convert and load weights
        state_dict = hf_model.state_dict()
        all_weights = convert_weights(state_dict)
        mlx_weights = {k: mx.array(v) for k, v in all_weights.items()}
        mlx_model.load_weights(list(mlx_weights.items()))

        # Test with multiple token sequences to verify cross-attention
        np.random.seed(42)
        mel_np = np.random.randn(1, 80, 200).astype(np.float32)

        for seq_len in [1, 3, 5, 10]:
            tokens_np = np.arange(1, seq_len + 1, dtype=np.int64).reshape(1, -1)

            with torch.no_grad():
                hf_logits = hf_model(
                    torch.tensor(mel_np),
                    decoder_input_ids=torch.tensor(tokens_np)
                ).logits.numpy()

            mlx_logits = mlx_model(mx.array(mel_np), mx.array(tokens_np))
            mx.eval(mlx_logits)

            # Tightened atol from 1e-5 to 1e-6 (consistent with other parity tests)
            np.testing.assert_allclose(
                np.array(mlx_logits),
                hf_logits,
                rtol=1e-4,
                atol=1e-6,
                err_msg=f"Mismatch at sequence length {seq_len}"
            )


@pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
class TestWhisperInference:
    """Tests for inference utilities."""

    def test_log_mel_spectrogram_shape(self):
        """Test log mel spectrogram computation."""
        from mlx_audio.models.whisper.inference import compute_log_mel_spectrogram

        # Create random audio (16kHz, 3 seconds)
        audio = mx.random.normal((48000,))

        mel = compute_log_mel_spectrogram(
            audio,
            n_mels=80,
            n_fft=400,
            hop_length=160,
            sample_rate=16000,
        )

        # Check shape: [n_mels, T]
        assert mel.shape[0] == 80
        assert mel.shape[1] > 0

    def test_pad_or_trim(self):
        """Test audio padding and trimming."""
        from mlx_audio.models.whisper.inference import pad_or_trim

        # Test padding
        short_audio = mx.random.normal((10000,))
        padded = pad_or_trim(short_audio, 480000)
        assert padded.shape[-1] == 480000

        # Test trimming
        long_audio = mx.random.normal((500000,))
        trimmed = pad_or_trim(long_audio, 480000)
        assert trimmed.shape[-1] == 480000

        # Test exact length
        exact_audio = mx.random.normal((480000,))
        result = pad_or_trim(exact_audio, 480000)
        assert result.shape[-1] == 480000


@pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
class TestDecodingOptions:
    """Tests for DecodingOptions."""

    def test_default_options(self):
        """Test default decoding options."""
        from mlx_audio.models.whisper.inference import DecodingOptions

        options = DecodingOptions()

        assert options.language is None
        assert options.task == "transcribe"
        assert options.temperature == 0.0
        assert options.beam_size == 1
        assert options.is_greedy is True

    def test_beam_search_options(self):
        """Test beam search options."""
        from mlx_audio.models.whisper.inference import DecodingOptions

        options = DecodingOptions(
            beam_size=5,
            temperature=0.0,
        )

        assert options.is_greedy is False

    def test_sampling_options(self):
        """Test sampling options."""
        from mlx_audio.models.whisper.inference import DecodingOptions

        options = DecodingOptions(
            temperature=0.7,
            beam_size=1,
        )

        assert options.is_greedy is False
