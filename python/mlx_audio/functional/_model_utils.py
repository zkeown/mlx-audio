"""Model detection and utility functions for functional API.

This module provides shared utilities for model type detection
and other common model-related operations used across functional modules.
"""

from __future__ import annotations

# Known CLAP model identifiers
CLAP_MODELS = frozenset({
    "clap-htsat-fused",
    "clap-htsat-unfused",
    "clap-laion-audio-630k",
    "clap-laion-music",
})


def is_clap_model(model: str) -> bool:
    """Check if model name refers to a CLAP model.

    CLAP models are used for zero-shot classification and embedding
    via audio-text similarity.

    Args:
        model: Model name or path

    Returns:
        True if this is a CLAP model
    """
    # Check exact matches first
    if model in CLAP_MODELS:
        return True

    # Check if "clap" appears in the model name (case-insensitive)
    return "clap" in model.lower()


def is_whisper_model(model: str) -> bool:
    """Check if model name refers to a Whisper model.

    Args:
        model: Model name or path

    Returns:
        True if this is a Whisper model
    """
    whisper_prefixes = ("whisper-", "whisper_", "openai/whisper")
    model_lower = model.lower()

    return any(model_lower.startswith(p) for p in whisper_prefixes)


def is_musicgen_model(model: str) -> bool:
    """Check if model name refers to a MusicGen model.

    Args:
        model: Model name or path

    Returns:
        True if this is a MusicGen model
    """
    return "musicgen" in model.lower()


def is_demucs_model(model: str) -> bool:
    """Check if model name refers to a Demucs/HTDemucs model.

    Args:
        model: Model name or path

    Returns:
        True if this is a Demucs model
    """
    demucs_names = {"htdemucs", "htdemucs_ft", "htdemucs_6s", "demucs"}
    return model.lower() in demucs_names or "demucs" in model.lower()


def is_encodec_model(model: str) -> bool:
    """Check if model name refers to an EnCodec model.

    Args:
        model: Model name or path

    Returns:
        True if this is an EnCodec model
    """
    return "encodec" in model.lower()


def is_parler_tts_model(model: str) -> bool:
    """Check if model name refers to a Parler TTS model.

    Args:
        model: Model name or path

    Returns:
        True if this is a Parler TTS model
    """
    return "parler" in model.lower() or "tts" in model.lower()


def get_model_type(model: str) -> str | None:
    """Get the type of model from its name.

    Args:
        model: Model name or path

    Returns:
        Model type string ('clap', 'whisper', 'musicgen', 'demucs',
        'encodec', 'parler-tts') or None if unknown
    """
    if is_clap_model(model):
        return "clap"
    elif is_whisper_model(model):
        return "whisper"
    elif is_musicgen_model(model):
        return "musicgen"
    elif is_demucs_model(model):
        return "demucs"
    elif is_encodec_model(model):
        return "encodec"
    elif is_parler_tts_model(model):
        return "parler-tts"
    else:
        return None


__all__ = [
    "CLAP_MODELS",
    "is_clap_model",
    "is_whisper_model",
    "is_musicgen_model",
    "is_demucs_model",
    "is_encodec_model",
    "is_parler_tts_model",
    "get_model_type",
]
