"""Audio generation API."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    import mlx.core as mx

    from mlx_audio.types.results import GenerationResult


def generate(
    prompt: str,
    *,
    model: str = "musicgen-medium",
    duration: float = 10.0,
    temperature: float = 1.0,
    top_k: int = 250,
    top_p: float = 0.0,
    cfg_scale: float = 3.0,
    seed: int | None = None,
    output_file: str | Path | None = None,
    progress_callback: Callable[[float], None] | None = None,
    **kwargs,
) -> "GenerationResult":
    """Generate audio from text description.

    This is the main entry point for text-to-audio generation using MusicGen.

    Args:
        prompt: Text description of desired audio (e.g., "jazz piano, upbeat mood")
        model: Model name or path (e.g., "musicgen-small", "musicgen-medium", "musicgen-large")
        duration: Output duration in seconds (max 30s)
        temperature: Sampling temperature (higher = more random, 0 = greedy)
        top_k: Top-k sampling parameter (0 to disable)
        top_p: Nucleus sampling threshold (0 to disable)
        cfg_scale: Classifier-free guidance scale (1.0 = no guidance, higher = more prompt adherence)
        seed: Random seed for reproducibility
        output_file: Optional path to save generated audio
        progress_callback: Optional callback for progress updates [0, 1]
        **kwargs: Additional model-specific parameters

    Returns:
        GenerationResult with generated audio data

    Example:
        >>> audio = mlx_audio.generate("jazz piano, upbeat mood")
        >>> audio.play()  # Play the audio
        >>> audio.save("output.wav")  # Save to file

        >>> # With custom parameters
        >>> audio = mlx_audio.generate(
        ...     "electronic dance music",
        ...     duration=15.0,
        ...     temperature=0.8,
        ...     seed=42,
        ... )

        >>> # With progress callback
        >>> def on_progress(p):
        ...     print(f"Progress: {p:.1%}")
        >>> audio = mlx_audio.generate("ambient soundscape", progress_callback=on_progress)

    Note:
        This function requires a text encoder (T5) to be available.
        On first run, models will be downloaded from HuggingFace Hub.
    """
    import mlx.core as mx
    import numpy as np

    from mlx_audio.models.musicgen import MusicGen
    from mlx_audio.models.encodec import EnCodec
    from mlx_audio.hub.cache import get_cache
    from mlx_audio.types.results import GenerationResult

    # Clamp duration to valid range
    max_duration = 30.0
    if duration > max_duration:
        import warnings
        warnings.warn(
            f"Duration {duration}s exceeds maximum {max_duration}s. "
            f"Clamping to {max_duration}s."
        )
        duration = max_duration

    # Load model with caching
    cache = get_cache()
    musicgen = cache.get_model(model, MusicGen)

    # Ensure audio codec is set up
    if musicgen._audio_codec is None:
        # Load default EnCodec
        encodec = cache.get_model("encodec-32khz", EnCodec)
        musicgen.set_audio_codec(encodec)

    # Encode text prompt using T5
    text_embeddings = _encode_text_prompt(prompt, musicgen.config)

    # Generate audio tokens
    codes = musicgen.generate(
        encoder_hidden_states=text_embeddings,
        duration=duration,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        cfg_scale=cfg_scale,
        seed=seed,
        progress_callback=progress_callback,
    )

    # Decode to audio waveform
    audio = musicgen.decode_audio(codes)

    # Convert to numpy for result
    audio_np = np.array(audio)

    # Create result
    result = GenerationResult(
        array=mx.array(audio_np),
        sample_rate=musicgen.config.sample_rate,
        prompt=prompt,
        model_name=model,
        generation_params={
            "duration": duration,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "cfg_scale": cfg_scale,
            "seed": seed,
        },
    )

    # Save to file if requested
    if output_file is not None:
        output_file = Path(output_file)
        result.save(output_file)

    return result


def _encode_text_prompt(
    prompt: str,
    config,
) -> "mx.array":
    """Encode text prompt using T5.

    Args:
        prompt: Text description
        config: MusicGen config with text encoder settings

    Returns:
        Text embeddings [1, S, D]
    """
    import mlx.core as mx

    try:
        # Try to use transformers for T5
        from transformers import T5Tokenizer, T5EncoderModel
        import torch

        tokenizer = T5Tokenizer.from_pretrained(config.text_encoder_name)
        encoder = T5EncoderModel.from_pretrained(config.text_encoder_name)

        # Tokenize
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config.max_text_length,
        )

        # Encode
        with torch.no_grad():
            outputs = encoder(**inputs)
            embeddings = outputs.last_hidden_state

        # Convert to MLX
        embeddings_np = embeddings.numpy()
        return mx.array(embeddings_np)

    except ImportError:
        # Fallback: create dummy embeddings (for testing without transformers)
        import warnings
        warnings.warn(
            "transformers library not available. "
            "Using dummy text embeddings. Install transformers for proper text encoding: "
            "pip install transformers"
        )

        # Create random embeddings as placeholder
        seq_length = min(len(prompt.split()) + 2, config.max_text_length)
        embeddings = mx.random.normal((1, seq_length, config.text_hidden_size))
        return embeddings
