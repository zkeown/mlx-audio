"""Text-to-speech synthesis API."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    import mlx.core as mx

    from mlx_audio.types.results import SpeechResult


def speak(
    text: str,
    *,
    model: str = "parler-tts-mini",
    description: str | None = None,
    duration: float | None = None,
    speed: float = 1.0,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.0,
    seed: int | None = None,
    output_file: str | Path | None = None,
    progress_callback: Callable[[float], None] | None = None,
    **kwargs,
) -> "SpeechResult":
    """Convert text to speech.

    This is the main entry point for text-to-speech synthesis using Parler-TTS.

    Args:
        text: Text to synthesize into speech
        model: Model name or path (e.g., "parler-tts-mini", "parler-tts-large")
        description: Voice description for controlling speaker characteristics
                    (e.g., "A warm female voice, speaking clearly and slowly")
        duration: Optional target duration in seconds (None = automatic)
        speed: Speech speed multiplier (0.5 = half speed, 2.0 = double speed)
        temperature: Sampling temperature (higher = more random, 0 = greedy)
        top_k: Top-k sampling parameter (0 to disable)
        top_p: Nucleus sampling threshold (0 to disable)
        seed: Random seed for reproducibility
        output_file: Optional path to save generated audio
        progress_callback: Optional callback for progress updates [0, 1]
        **kwargs: Additional model-specific parameters

    Returns:
        SpeechResult with generated audio data

    Example:
        >>> audio = mlx_audio.speak("Hello, how are you today?")
        >>> audio.play()  # Play the audio
        >>> audio.save("output.wav")  # Save to file

        >>> # With voice description
        >>> audio = mlx_audio.speak(
        ...     "Welcome to the presentation.",
        ...     description="A professional male voice, clear and authoritative",
        ... )

        >>> # With custom parameters
        >>> audio = mlx_audio.speak(
        ...     "This is a test.",
        ...     speed=1.2,
        ...     temperature=0.8,
        ...     seed=42,
        ... )

        >>> # With progress callback
        >>> def on_progress(p):
        ...     print(f"Progress: {p:.1%}")
        >>> audio = mlx_audio.speak("Hello world", progress_callback=on_progress)

    Note:
        This function requires a text encoder (T5) to be available.
        On first run, models will be downloaded from HuggingFace Hub.
    """
    import mlx.core as mx
    import numpy as np

    from mlx_audio.models.tts import ParlerTTS
    from mlx_audio.hub.cache import get_cache
    from mlx_audio.types.results import SpeechResult

    # Load model with caching
    cache = get_cache()
    tts_model = cache.get_model(model, ParlerTTS)

    # Ensure audio codec is set up
    if tts_model._audio_codec is None:
        # Load DAC codec (Parler-TTS uses DAC 24kHz)
        from mlx_audio.models.encodec import EnCodec

        codec = cache.get_model("dac-24khz", EnCodec)
        tts_model.set_audio_codec(codec)

    # Encode text prompt using T5
    prompt_embeddings = _encode_text(text, tts_model.config, encoder_type="prompt")

    # Encode voice description if provided
    description_embeddings = None
    if description is not None:
        description_embeddings = _encode_text(
            description, tts_model.config, encoder_type="description"
        )

    # Calculate duration if not specified
    if duration is None:
        # Estimate duration based on text length (~150 words per minute)
        word_count = len(text.split())
        duration = max(1.0, word_count / 2.5)  # ~2.5 words per second
        duration = min(duration, tts_model.config.max_duration)

    # Adjust for speed
    effective_duration = duration / speed

    # Generate audio tokens
    codes = tts_model.generate(
        prompt_hidden_states=prompt_embeddings,
        description_hidden_states=description_embeddings,
        duration=effective_duration,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        seed=seed,
        progress_callback=progress_callback,
    )

    # Decode to audio waveform
    audio = tts_model.decode_audio(codes)

    # Apply speed adjustment via resampling if needed
    if speed != 1.0:
        audio = _adjust_speed(audio, speed, tts_model.config.sample_rate)

    # Convert to numpy for result
    audio_np = np.array(audio)

    # Create result
    result = SpeechResult(
        array=mx.array(audio_np),
        sample_rate=tts_model.config.sample_rate,
        text=text,
        description=description,
        model_name=model,
        generation_params={
            "duration": duration,
            "speed": speed,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "seed": seed,
        },
    )

    # Save to file if requested
    if output_file is not None:
        output_file = Path(output_file)
        result.save(output_file)

    return result


def _encode_text(
    text: str,
    config,
    encoder_type: str = "prompt",
) -> "mx.array":
    """Encode text using T5.

    Args:
        text: Text to encode
        config: ParlerTTS config with encoder settings
        encoder_type: "prompt" or "description"

    Returns:
        Text embeddings [1, S, D]
    """
    import mlx.core as mx

    if encoder_type == "description":
        encoder_name = config.description_encoder_name
        max_length = config.max_description_length
    else:
        encoder_name = config.text_encoder_name
        max_length = config.max_text_length

    try:
        # Try to use transformers for T5
        from transformers import T5Tokenizer, T5EncoderModel
        import torch

        tokenizer = T5Tokenizer.from_pretrained(encoder_name)
        encoder = T5EncoderModel.from_pretrained(encoder_name)

        # Tokenize
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
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
        hidden_size = (
            config.description_hidden_size
            if encoder_type == "description"
            else config.text_hidden_size
        )
        seq_length = min(len(text.split()) + 2, max_length)
        embeddings = mx.random.normal((1, seq_length, hidden_size))
        return embeddings


def _adjust_speed(
    audio: "mx.array",
    speed: float,
    sample_rate: int,
) -> "mx.array":
    """Adjust audio playback speed via resampling.

    Args:
        audio: Audio waveform [B, C, samples]
        speed: Speed multiplier
        sample_rate: Original sample rate

    Returns:
        Speed-adjusted audio
    """
    import mlx.core as mx

    if speed == 1.0:
        return audio

    # Simple linear interpolation for speed adjustment
    # speed > 1.0 means faster playback (fewer samples)
    # speed < 1.0 means slower playback (more samples)

    original_length = audio.shape[-1]
    new_length = int(original_length / speed)

    if new_length == original_length:
        return audio

    # Create new sample indices
    indices = mx.linspace(0, original_length - 1, new_length)
    indices_floor = mx.floor(indices).astype(mx.int32)
    indices_ceil = mx.minimum(indices_floor + 1, original_length - 1)
    weights = indices - indices_floor.astype(mx.float32)

    # Interpolate
    # audio shape: [B, C, samples]
    audio_floor = audio[..., indices_floor]
    audio_ceil = audio[..., indices_ceil]
    audio_resampled = audio_floor * (1 - weights) + audio_ceil * weights

    return audio_resampled
