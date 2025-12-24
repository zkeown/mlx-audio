"""Model detection and utility functions for functional API.

This module provides shared utilities for model type detection
and other common model-related operations used across functional modules.

Model detection uses explicit registry lookups to avoid false positives
from substring matching on user paths (e.g., "/path/to/my_clap_stuff/").
"""

from __future__ import annotations

import os

# Known CLAP model identifiers (exact matches)
CLAP_MODELS = frozenset({
    "clap-htsat-fused",
    "clap-htsat-unfused",
    "clap-laion-audio-630k",
    "clap-laion-music",
    "larger_clap_general",
    "larger_clap_music",
})

# Known Whisper model identifiers (exact matches)
WHISPER_MODELS = frozenset({
    "whisper-tiny",
    "whisper-tiny.en",
    "whisper-base",
    "whisper-base.en",
    "whisper-small",
    "whisper-small.en",
    "whisper-medium",
    "whisper-medium.en",
    "whisper-large",
    "whisper-large-v2",
    "whisper-large-v3",
    "whisper-large-v3-turbo",
})

# Known MusicGen model identifiers (exact matches)
MUSICGEN_MODELS = frozenset({
    "musicgen-small",
    "musicgen-medium",
    "musicgen-large",
    "musicgen-melody",
    "musicgen-melody-large",
    "musicgen-stereo-small",
    "musicgen-stereo-medium",
    "musicgen-stereo-large",
})

# Known Demucs model identifiers (exact matches)
DEMUCS_MODELS = frozenset({
    "htdemucs",
    "htdemucs_ft",
    "htdemucs_6s",
    "htdemucs_mmi",
    "demucs",
})

# Known EnCodec model identifiers (exact matches)
ENCODEC_MODELS = frozenset({
    "encodec-24khz",
    "encodec-48khz",
    "encodec_24khz",
    "encodec_48khz",
})

# Known Parler TTS model identifiers (exact matches)
PARLER_TTS_MODELS = frozenset({
    "parler-tts-mini",
    "parler-tts-mini-v1",
    "parler-tts-large",
    "parler-tts-large-v1",
})


def _extract_model_name(model: str) -> str:
    """Extract the model name from a path or identifier.

    For paths like "/path/to/models/htdemucs_ft", extracts "htdemucs_ft".
    For HuggingFace paths like "org/model-name", extracts "model-name".

    Args:
        model: Model name, path, or HuggingFace identifier

    Returns:
        Extracted model name for matching
    """
    # Handle file system paths
    if os.sep in model or "/" in model:
        # Get the last component of the path
        name = os.path.basename(model.rstrip(os.sep + "/"))
        # Handle HuggingFace org/model format
        if not name and "/" in model:
            name = model.rsplit("/", 1)[-1]
        return name.lower()
    return model.lower()


def is_clap_model(model: str) -> bool:
    """Check if model name refers to a CLAP model.

    CLAP models are used for zero-shot classification and embedding
    via audio-text similarity.

    Args:
        model: Model name or path

    Returns:
        True if this is a CLAP model
    """
    # Check exact matches first (case-insensitive)
    if model.lower() in {m.lower() for m in CLAP_MODELS}:
        return True

    # Extract model name from path
    name = _extract_model_name(model)
    if name in {m.lower() for m in CLAP_MODELS}:
        return True

    # Check for CLAP-prefixed models (handles "clap-custom-variant")
    return name.startswith("clap-") or name.startswith("clap_")


def is_whisper_model(model: str) -> bool:
    """Check if model name refers to a Whisper model.

    Args:
        model: Model name or path

    Returns:
        True if this is a Whisper model
    """
    # Check exact matches first
    model_lower = model.lower()
    if model_lower in {m.lower() for m in WHISPER_MODELS}:
        return True

    # Extract model name from path
    name = _extract_model_name(model)
    if name in {m.lower() for m in WHISPER_MODELS}:
        return True

    # Check for whisper-prefixed models
    return name.startswith("whisper-") or name.startswith("whisper_")


def is_musicgen_model(model: str) -> bool:
    """Check if model name refers to a MusicGen model.

    Args:
        model: Model name or path

    Returns:
        True if this is a MusicGen model
    """
    # Check exact matches first
    model_lower = model.lower()
    if model_lower in {m.lower() for m in MUSICGEN_MODELS}:
        return True

    # Extract model name from path
    name = _extract_model_name(model)
    if name in {m.lower() for m in MUSICGEN_MODELS}:
        return True

    # Check for musicgen-prefixed models
    return name.startswith("musicgen-") or name.startswith("musicgen_")


def is_demucs_model(model: str) -> bool:
    """Check if model name refers to a Demucs/HTDemucs model.

    Args:
        model: Model name or path

    Returns:
        True if this is a Demucs model
    """
    # Check exact matches first
    model_lower = model.lower()
    if model_lower in {m.lower() for m in DEMUCS_MODELS}:
        return True

    # Extract model name from path
    name = _extract_model_name(model)
    if name in {m.lower() for m in DEMUCS_MODELS}:
        return True

    # Check for demucs/htdemucs-prefixed models
    return (
        name.startswith("demucs-") or name.startswith("demucs_") or
        name.startswith("htdemucs-") or name.startswith("htdemucs_")
    )


def is_encodec_model(model: str) -> bool:
    """Check if model name refers to an EnCodec model.

    Args:
        model: Model name or path

    Returns:
        True if this is an EnCodec model
    """
    # Check exact matches first
    model_lower = model.lower()
    if model_lower in {m.lower() for m in ENCODEC_MODELS}:
        return True

    # Extract model name from path
    name = _extract_model_name(model)
    if name in {m.lower() for m in ENCODEC_MODELS}:
        return True

    # Check for encodec-prefixed models
    return name.startswith("encodec-") or name.startswith("encodec_")


def is_parler_tts_model(model: str) -> bool:
    """Check if model name refers to a Parler TTS model.

    Args:
        model: Model name or path

    Returns:
        True if this is a Parler TTS model
    """
    # Check exact matches first
    model_lower = model.lower()
    if model_lower in {m.lower() for m in PARLER_TTS_MODELS}:
        return True

    # Extract model name from path
    name = _extract_model_name(model)
    if name in {m.lower() for m in PARLER_TTS_MODELS}:
        return True

    # Check for parler-tts-prefixed models
    return name.startswith("parler-tts-") or name.startswith("parler_tts_")


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
    # Model registries
    "CLAP_MODELS",
    "WHISPER_MODELS",
    "MUSICGEN_MODELS",
    "DEMUCS_MODELS",
    "ENCODEC_MODELS",
    "PARLER_TTS_MODELS",
    # Detection functions
    "is_clap_model",
    "is_whisper_model",
    "is_musicgen_model",
    "is_demucs_model",
    "is_encodec_model",
    "is_parler_tts_model",
    "get_model_type",
]
