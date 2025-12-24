"""Numerical parity tests for CLAP against HuggingFace transformers.

These tests verify that the MLX CLAP implementation produces outputs
that match the HuggingFace reference implementation.
"""

import pytest
import numpy as np

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

try:
    from transformers import ClapModel, ClapProcessor
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


@pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers not available")
@pytest.mark.slow
class TestCLAPParity:
    """Parity tests comparing MLX CLAP to HuggingFace CLAP."""

    @pytest.fixture
    def hf_model(self):
        """Load HuggingFace CLAP model."""
        return ClapModel.from_pretrained("laion/clap-htsat-fused")

    @pytest.fixture
    def hf_processor(self):
        """Load HuggingFace CLAP processor."""
        return ClapProcessor.from_pretrained("laion/clap-htsat-fused")

    @pytest.fixture
    def mlx_model(self, tmp_path):
        """Load MLX CLAP model with converted weights."""
        from mlx_audio.models.clap.convert import convert_clap_weights
        from mlx_audio.models.clap import CLAP

        # Convert weights
        path = convert_clap_weights("laion/clap-htsat-fused", tmp_path / "clap")

        # Load model and set to eval mode
        model = CLAP.from_pretrained(path)
        model.eval()
        return model

    def test_audio_embedding_parity(self, hf_model, hf_processor, mlx_model):
        """Verify audio embeddings match within tolerance."""
        import torch

        # Generate deterministic test audio
        np.random.seed(42)
        audio = np.random.randn(48000).astype(np.float32)  # 1 second

        # HuggingFace embedding
        inputs = hf_processor(
            audio=audio,
            return_tensors="pt",
            sampling_rate=48000,
        )
        with torch.no_grad():
            hf_embed = hf_model.get_audio_features(**inputs).numpy()

        # MLX embedding - convert HF features to MLX format
        # HF gives [B, 4, T, F] = [1, 4, 1001, 64]
        # We need [B, 4, F, T] for fusion (matching HF's is_longer=True)
        mel_features = inputs["input_features"].numpy()  # [1, 4, 1001, 64]
        # Permute to [B, 4, F, T] - keep all 4 channels for fusion
        mel_features = np.transpose(mel_features, (0, 1, 3, 2))  # [1, 4, 64, 1001]
        mel = mx.array(mel_features)

        # Pass is_longer flag to match HF processing
        is_longer = inputs["is_longer"].numpy()  # [1, 1] with True
        is_longer = mx.array(is_longer.flatten())  # [1]

        mlx_embed = mlx_model.encode_audio(mel, is_longer=is_longer)
        mlx_embed_np = np.array(mlx_embed)

        # Check cosine similarity
        hf_norm = hf_embed / np.linalg.norm(hf_embed, axis=-1, keepdims=True)
        mlx_norm = mlx_embed_np / np.linalg.norm(mlx_embed_np, axis=-1, keepdims=True)
        cos_sim = np.dot(hf_norm[0], mlx_norm[0])

        assert cos_sim > 0.99, f"Cosine similarity {cos_sim} below threshold"

    def test_text_embedding_parity(self, hf_model, hf_processor, mlx_model):
        """Verify text embeddings match within tolerance."""
        import torch

        # Test text
        texts = ["a dog barking", "music playing"]

        # HuggingFace embedding
        inputs = hf_processor(text=texts, return_tensors="pt", padding=True)
        with torch.no_grad():
            hf_embed = hf_model.get_text_features(**inputs).numpy()

        # MLX embedding
        input_ids = mx.array(inputs["input_ids"].numpy())
        attention_mask = mx.array(inputs["attention_mask"].numpy())

        mlx_embed = mlx_model.encode_text(input_ids, attention_mask=attention_mask)
        mlx_embed_np = np.array(mlx_embed)

        # Check cosine similarity for each text
        for i in range(len(texts)):
            hf_norm = hf_embed[i] / np.linalg.norm(hf_embed[i])
            mlx_norm = mlx_embed_np[i] / np.linalg.norm(mlx_embed_np[i])
            cos_sim = np.dot(hf_norm, mlx_norm)

            # Tightened from 0.99 to 0.9999 (measured: 1.0000 exact match)
            assert cos_sim > 0.9999, f"Text {i} cosine similarity {cos_sim} below threshold"

    def test_similarity_parity(self, hf_model, hf_processor, mlx_model):
        """Verify audio-text similarity matches."""
        import torch

        # Test data
        np.random.seed(42)
        audio = np.random.randn(48000).astype(np.float32)
        texts = ["a dog barking", "music playing", "car engine"]

        # HuggingFace
        audio_inputs = hf_processor(
            audio=audio,
            return_tensors="pt",
            sampling_rate=48000,
        )
        text_inputs = hf_processor(text=texts, return_tensors="pt", padding=True)

        with torch.no_grad():
            hf_audio = hf_model.get_audio_features(**audio_inputs)
            hf_text = hf_model.get_text_features(**text_inputs)
            hf_sim = (hf_audio @ hf_text.T).numpy()

        # MLX - convert HF features to MLX format
        # HF gives [B, 4, T, F], we need [B, 4, F, T] for fusion
        mel_features = audio_inputs["input_features"].numpy()
        mel_features = np.transpose(mel_features, (0, 1, 3, 2))  # [B, 4, F, T]
        mel = mx.array(mel_features)
        input_ids = mx.array(text_inputs["input_ids"].numpy())
        attention_mask = mx.array(text_inputs["attention_mask"].numpy())

        # Pass is_longer flag to match HF processing
        is_longer = audio_inputs["is_longer"].numpy()
        is_longer = mx.array(is_longer.flatten())

        mlx_audio = mlx_model.encode_audio(mel, is_longer=is_longer)
        mlx_text = mlx_model.encode_text(input_ids, attention_mask=attention_mask)
        mlx_sim = np.array(mlx_model.similarity(mlx_audio, mlx_text))

        # Compare ranking (order of similarity scores)
        hf_order = np.argsort(hf_sim[0])[::-1]
        mlx_order = np.argsort(mlx_sim[0])[::-1]

        assert np.array_equal(hf_order, mlx_order), "Similarity ranking doesn't match"

    def test_embedding_deterministic(self, mlx_model):
        """Verify embeddings are deterministic."""
        mel = mx.random.normal((1, 1, 64, 256))

        emb1 = mlx_model.encode_audio(mel)
        emb2 = mlx_model.encode_audio(mel)

        assert mx.allclose(emb1, emb2), "Embeddings are not deterministic"


@pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
class TestCLAPEmbedAPI:
    """Tests for the embed() functional API."""

    def test_embed_raises_without_input(self):
        """Test that embed raises without audio or text."""
        from mlx_audio.functional.embed import embed
        from mlx_audio.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError, match="At least one of audio or text"):
            embed()

    @pytest.mark.skipif(
        not HAS_TRANSFORMERS, reason="transformers needed for tokenizer"
    )
    @pytest.mark.slow
    def test_embed_text_only(self, tmp_path):
        """Test text-only embedding with the embed() API."""
        from mlx_audio.functional.embed import CLAPEmbeddingResult, _tokenize_text
        from mlx_audio.models.clap.convert import convert_clap_weights
        from mlx_audio.models.clap import CLAP

        # Convert weights and load model
        path = convert_clap_weights("laion/clap-htsat-fused", tmp_path / "clap")
        model = CLAP.from_pretrained(path)

        # Test single text
        input_ids, attention_mask = _tokenize_text(["a dog barking"])
        result_embeds = model.encode_text(input_ids, attention_mask)
        assert result_embeds.shape == (1, 512)

        # Test batch text
        input_ids, attention_mask = _tokenize_text(
            ["dog barking", "cat meowing", "music playing"]
        )
        result_embeds = model.encode_text(input_ids, attention_mask)
        assert result_embeds.shape == (3, 512)

    @pytest.mark.skipif(
        not HAS_TRANSFORMERS, reason="transformers needed for tokenizer"
    )
    @pytest.mark.slow
    def test_embed_audio_only(self, tmp_path):
        """Test audio-only embedding with the embed() API."""
        from mlx_audio.models.clap.convert import convert_clap_weights
        from mlx_audio.models.clap import CLAP

        # Convert weights and load model
        path = convert_clap_weights("laion/clap-htsat-fused", tmp_path / "clap")
        model = CLAP.from_pretrained(path)

        # Generate test audio (1 second at 48kHz)
        np.random.seed(42)
        audio = mx.array(np.random.randn(1, 1, 64, 256).astype(np.float32))

        result_embeds = model.encode_audio(audio)
        assert result_embeds.shape == (1, 512)

    @pytest.mark.skipif(
        not HAS_TRANSFORMERS, reason="transformers needed for tokenizer"
    )
    @pytest.mark.slow
    def test_embed_zero_shot_classification(self, tmp_path):
        """Test zero-shot classification with embed() API."""
        from mlx_audio.functional.embed import CLAPEmbeddingResult, _tokenize_text
        from mlx_audio.models.clap.convert import convert_clap_weights
        from mlx_audio.models.clap import CLAP

        # Convert weights and load model
        path = convert_clap_weights("laion/clap-htsat-fused", tmp_path / "clap")
        model = CLAP.from_pretrained(path)

        # Generate test audio
        np.random.seed(42)
        audio = mx.array(np.random.randn(1, 1, 64, 256).astype(np.float32))
        labels = ["dog barking", "cat meowing", "bird singing"]

        # Get embeddings
        audio_embeds = model.encode_audio(audio)
        input_ids, attention_mask = _tokenize_text(labels)
        text_embeds = model.encode_text(input_ids, attention_mask)
        similarity = model.similarity(audio_embeds, text_embeds)

        # Create result for testing best_match
        result = CLAPEmbeddingResult(
            audio_embeds=audio_embeds,
            text_embeds=text_embeds,
            similarity=similarity,
            text_labels=labels,
        )

        assert result.audio_embeds is not None
        assert result.text_embeds is not None
        assert result.similarity is not None
        assert result.similarity.shape == (1, 3)
        assert result.text_labels == labels

        # Test best_match
        best = result.best_match()
        assert best in labels

        # Test top-k
        top2 = result.best_match(top_k=2)
        assert len(top2) == 2
        assert all(label in labels for label in top2)

    def test_clap_embedding_result(self):
        """Test CLAPEmbeddingResult dataclass."""
        from mlx_audio.functional.embed import CLAPEmbeddingResult

        # Test with audio embeddings only
        audio_embeds = mx.random.normal((1, 512))
        result = CLAPEmbeddingResult(
            audio_embeds=audio_embeds,
            model_name="test",
        )

        assert result.vectors.shape == (1, 512)
        assert result.dimension == 512

    def test_clap_embedding_result_similarity(self):
        """Test CLAPEmbeddingResult similarity methods."""
        from mlx_audio.functional.embed import CLAPEmbeddingResult

        # Create normalized embeddings
        emb1 = mx.random.normal((1, 512))
        emb1 = emb1 / mx.linalg.norm(emb1)
        emb2 = mx.random.normal((1, 512))
        emb2 = emb2 / mx.linalg.norm(emb2)

        result1 = CLAPEmbeddingResult(audio_embeds=emb1)
        result2 = CLAPEmbeddingResult(audio_embeds=emb2)

        similarity = result1.cosine_similarity(result2)
        assert -1 <= similarity <= 1

    def test_clap_embedding_result_best_match(self):
        """Test CLAPEmbeddingResult best_match method."""
        from mlx_audio.functional.embed import CLAPEmbeddingResult

        # Create similarity scores
        similarity = mx.array([[0.8, 0.3, 0.5]])
        text_labels = ["dog", "cat", "bird"]

        result = CLAPEmbeddingResult(
            similarity=similarity,
            text_labels=text_labels,
        )

        assert result.best_match() == "dog"
        assert result.best_match(top_k=2) == ["dog", "bird"]
