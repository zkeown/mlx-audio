"""Parler-TTS text-to-speech model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import mlx.core as mx
import mlx.nn as nn

from mlx_audio.models.tts.config import ParlerTTSConfig
from mlx_audio.models.tts.layers.embeddings import CodebookEmbeddings
from mlx_audio.models.tts.layers.transformer import ParlerTTSDecoder
from mlx_audio.models.tts.layers.lm_head import ParlerTTSLMHead, DelayPatternScheduler

if TYPE_CHECKING:
    from mlx_audio.models.encodec import EnCodec


class ParlerTTS(nn.Module):
    """Parler-TTS text-to-speech model.

    Parler-TTS generates natural speech from text descriptions using:
    1. T5 text encoder for text prompt conditioning
    2. T5 encoder for voice description conditioning
    3. Transformer decoder with delay pattern for multi-codebook generation
    4. DAC/EnCodec for audio tokenization/detokenization

    The model uses a delay pattern where different codebooks are offset
    in time, allowing parallel generation while respecting dependencies.

    Attributes:
        config: Model configuration
        embeddings: Codebook token embeddings
        decoder: Transformer decoder
        lm_head: Language model head for each codebook
        delay_pattern: Delay pattern scheduler

    Example:
        >>> config = ParlerTTSConfig.mini()
        >>> model = ParlerTTS(config)
        >>> # Generate from text conditioning
        >>> prompt_embeds = model.encode_text("Hello, how are you today?")
        >>> desc_embeds = model.encode_description("A warm female voice")
        >>> audio = model.generate(prompt_embeds, desc_embeds, duration=5.0)
    """

    def __init__(self, config: ParlerTTSConfig | None = None):
        """Initialize Parler-TTS model.

        Args:
            config: Model configuration. If None, uses mini config.
        """
        super().__init__()
        if config is None:
            config = ParlerTTSConfig.mini()
        self.config = config

        # Codebook embeddings
        self.embeddings = CodebookEmbeddings(config)

        # Text conditioning projection (T5 hidden size -> decoder hidden size)
        self.text_projection = nn.Linear(
            config.text_hidden_size,
            config.hidden_size,
            bias=False,
        )

        # Description conditioning projection
        self.description_projection = nn.Linear(
            config.description_hidden_size,
            config.hidden_size,
            bias=False,
        )

        # Transformer decoder
        self.decoder = ParlerTTSDecoder(config)

        # Language model head (one per codebook)
        self.lm_head = ParlerTTSLMHead(config)

        # Delay pattern scheduler
        self.delay_pattern = DelayPatternScheduler(
            num_codebooks=config.num_codebooks,
            pad_token_id=config.pad_token_id,
        )

        # Audio codec (loaded separately)
        self._audio_codec: "EnCodec" | None = None

    @property
    def audio_codec(self) -> "EnCodec":
        """Get audio codec (lazy loaded)."""
        if self._audio_codec is None:
            raise RuntimeError(
                "Audio codec not set. Use model.set_audio_codec() or load with from_pretrained()."
            )
        return self._audio_codec

    def set_audio_codec(self, codec: "EnCodec") -> None:
        """Set the audio codec for encoding/decoding audio.

        Args:
            codec: EnCodec/DAC model instance
        """
        self._audio_codec = codec

    def project_text_embeddings(
        self,
        text_embeddings: mx.array,
    ) -> mx.array:
        """Project text embeddings to decoder hidden size.

        Args:
            text_embeddings: Text encoder outputs [B, S, text_hidden_size]

        Returns:
            Projected embeddings [B, S, hidden_size]
        """
        return self.text_projection(text_embeddings)

    def project_description_embeddings(
        self,
        description_embeddings: mx.array,
    ) -> mx.array:
        """Project description embeddings to decoder hidden size.

        Args:
            description_embeddings: Description encoder outputs [B, S, desc_hidden_size]

        Returns:
            Projected embeddings [B, S, hidden_size]
        """
        return self.description_projection(description_embeddings)

    def forward(
        self,
        input_ids: mx.array,
        encoder_hidden_states: mx.array | None = None,
        attention_mask: mx.array | None = None,
        encoder_attention_mask: mx.array | None = None,
        kv_cache: list[tuple[mx.array, mx.array]] | None = None,
        position_offset: int = 0,
    ) -> tuple[mx.array, list[tuple[mx.array, mx.array]]]:
        """Forward pass through the model.

        Args:
            input_ids: Codebook token IDs [B, K, T]
            encoder_hidden_states: Conditioning (projected text + description) [B, S, D]
            attention_mask: Causal self-attention mask
            encoder_attention_mask: Cross-attention mask
            kv_cache: Cached key/values for incremental decoding
            position_offset: Position offset for RoPE

        Returns:
            Tuple of:
                - Logits [B, K, T, V] for each codebook
                - Updated KV cache
        """
        # Compute embeddings from codebook tokens
        hidden_states = self.embeddings(input_ids)

        # Create causal mask if not provided
        if attention_mask is None:
            seq_length = input_ids.shape[-1]
            attention_mask = self.decoder.create_causal_mask(seq_length, position_offset)

        # Run decoder
        hidden_states, new_kv_cache = self.decoder(
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            encoder_attention_mask=encoder_attention_mask,
            kv_cache=kv_cache,
            position_offset=position_offset,
        )

        # Compute logits for all codebooks
        logits = self.lm_head(hidden_states)  # [B, K, T, V]

        return logits, new_kv_cache

    def __call__(
        self,
        input_ids: mx.array,
        encoder_hidden_states: mx.array | None = None,
        **kwargs,
    ) -> mx.array:
        """Forward pass returning logits only.

        Args:
            input_ids: Codebook token IDs [B, K, T]
            encoder_hidden_states: Conditioning [B, S, D]
            **kwargs: Additional arguments passed to forward()

        Returns:
            Logits [B, K, T, V]
        """
        logits, _ = self.forward(
            input_ids,
            encoder_hidden_states=encoder_hidden_states,
            **kwargs,
        )
        return logits

    def generate(
        self,
        prompt_hidden_states: mx.array,
        description_hidden_states: mx.array | None = None,
        max_new_tokens: int | None = None,
        duration: float | None = None,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.0,
        seed: int | None = None,
        progress_callback: Callable[[float], None] | None = None,
    ) -> mx.array:
        """Generate audio tokens from text conditioning.

        Uses autoregressive generation with the delay pattern to produce
        tokens for all codebooks.

        Args:
            prompt_hidden_states: Text prompt conditioning [B, S, D]
            description_hidden_states: Voice description conditioning [B, S, D]
            max_new_tokens: Maximum tokens to generate (overrides duration)
            duration: Generation duration in seconds
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling parameter (0 to disable)
            top_p: Nucleus sampling threshold (0 to disable)
            seed: Random seed for reproducibility
            progress_callback: Optional callback for progress updates

        Returns:
            Generated codes [B, K, T] where K is num_codebooks
        """
        from mlx_audio.models.tts.inference import generate_tokens

        # Determine max tokens
        if max_new_tokens is None:
            if duration is not None:
                max_new_tokens = int(duration * self.config.frame_rate)
            else:
                max_new_tokens = self.config.max_new_tokens

        # Project and combine conditioning
        prompt_projected = self.project_text_embeddings(prompt_hidden_states)

        if description_hidden_states is not None:
            description_projected = self.project_description_embeddings(
                description_hidden_states
            )
            # Concatenate prompt and description conditioning
            encoder_hidden_states = mx.concatenate(
                [description_projected, prompt_projected],
                axis=1,
            )
        else:
            encoder_hidden_states = prompt_projected

        # Generate tokens
        codes = generate_tokens(
            model=self,
            encoder_hidden_states=encoder_hidden_states,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=seed,
            progress_callback=progress_callback,
        )

        return codes

    def decode_audio(self, codes: mx.array) -> mx.array:
        """Decode audio tokens to waveform.

        Args:
            codes: Audio codes [B, K, T]

        Returns:
            Audio waveform [B, C, samples]
        """
        return self.audio_codec.decode(codes)

    @classmethod
    def from_pretrained(
        cls,
        path: str | Path,
        **kwargs,
    ) -> "ParlerTTS":
        """Load pretrained Parler-TTS model.

        Args:
            path: Path to model directory containing:
                  - config.json
                  - model.safetensors (or weights.npz)
                  - Optionally: dac/ subdirectory with codec weights
            **kwargs: Additional arguments to override config

        Returns:
            Loaded Parler-TTS model
        """
        path = Path(path)

        # Load config
        config_path = path / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config_dict = json.load(f)
            config = ParlerTTSConfig.from_dict(config_dict)
        else:
            config = ParlerTTSConfig.mini()

        # Override config with kwargs
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        # Create model
        model = cls(config)

        # Load weights
        weights_path = path / "model.safetensors"
        if weights_path.exists():
            model.load_weights(str(weights_path))
        else:
            npz_path = path / "weights.npz"
            if npz_path.exists():
                model.load_weights(str(npz_path))

        # Try to load audio codec if present
        dac_path = path / "dac"
        if dac_path.exists():
            from mlx_audio.models.encodec import EnCodec

            codec = EnCodec.from_pretrained(dac_path)
            model.set_audio_codec(codec)

        return model

    def save_pretrained(self, path: str | Path) -> None:
        """Save model to directory.

        Args:
            path: Output directory
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save config
        config_path = path / "config.json"
        with open(config_path, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        # Save weights
        weights_path = path / "model.safetensors"
        self.save_weights(str(weights_path))

        # Save audio codec if set
        if self._audio_codec is not None:
            dac_path = path / "dac"
            self._audio_codec.save_pretrained(dac_path)
