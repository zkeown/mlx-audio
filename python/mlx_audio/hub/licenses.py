"""License information and commercial use warnings for mlx-audio models.

This module provides license metadata for pre-trained models and runtime
warnings when loading models with commercial use restrictions.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Callable


class LicenseType(Enum):
    """License types for model weights."""

    MIT = "MIT"
    APACHE_2_0 = "Apache-2.0"
    CC_BY_NC_4_0 = "CC-BY-NC-4.0"  # Non-commercial only
    ISC = "ISC"
    UNKNOWN = "Unknown"


@dataclass(frozen=True)
class LicenseInfo:
    """License information for a model.

    Attributes:
        license_type: The license type
        commercial_use: Whether commercial use is permitted
        attribution_required: Whether attribution is required
        source_url: URL to license or model page
        notes: Additional notes about usage
    """

    license_type: LicenseType
    commercial_use: bool
    attribution_required: bool
    source_url: str = ""
    notes: str = ""


# Model license database
# Reference: THIRD_PARTY_LICENSES.md
MODEL_LICENSES: dict[str, LicenseInfo] = {
    # Separation models - MIT (commercial OK)
    "htdemucs": LicenseInfo(
        license_type=LicenseType.MIT,
        commercial_use=True,
        attribution_required=True,
        source_url="https://github.com/facebookresearch/demucs",
        notes="Meta's HTDemucs under MIT license",
    ),
    # Transcription models - MIT (commercial OK)
    "whisper-large-v3-turbo": LicenseInfo(
        license_type=LicenseType.MIT,
        commercial_use=True,
        attribution_required=True,
        source_url="https://github.com/openai/whisper",
    ),
    "whisper-large-v3": LicenseInfo(
        license_type=LicenseType.MIT,
        commercial_use=True,
        attribution_required=True,
        source_url="https://github.com/openai/whisper",
    ),
    "whisper-large-v2": LicenseInfo(
        license_type=LicenseType.MIT,
        commercial_use=True,
        attribution_required=True,
        source_url="https://github.com/openai/whisper",
    ),
    "whisper-medium": LicenseInfo(
        license_type=LicenseType.MIT,
        commercial_use=True,
        attribution_required=True,
        source_url="https://github.com/openai/whisper",
    ),
    "whisper-small": LicenseInfo(
        license_type=LicenseType.MIT,
        commercial_use=True,
        attribution_required=True,
        source_url="https://github.com/openai/whisper",
    ),
    "whisper-base": LicenseInfo(
        license_type=LicenseType.MIT,
        commercial_use=True,
        attribution_required=True,
        source_url="https://github.com/openai/whisper",
    ),
    "whisper-tiny": LicenseInfo(
        license_type=LicenseType.MIT,
        commercial_use=True,
        attribution_required=True,
        source_url="https://github.com/openai/whisper",
    ),
    # Embedding models - Apache 2.0 (commercial OK)
    "clap-htsat-fused": LicenseInfo(
        license_type=LicenseType.APACHE_2_0,
        commercial_use=True,
        attribution_required=True,
        source_url="https://github.com/LAION-AI/CLAP",
    ),
    "clap-htsat-unfused": LicenseInfo(
        license_type=LicenseType.APACHE_2_0,
        commercial_use=True,
        attribution_required=True,
        source_url="https://github.com/LAION-AI/CLAP",
    ),
    # Generation models - CC-BY-NC-4.0 (NON-COMMERCIAL ONLY)
    "musicgen-small": LicenseInfo(
        license_type=LicenseType.CC_BY_NC_4_0,
        commercial_use=False,
        attribution_required=True,
        source_url="https://github.com/facebookresearch/audiocraft",
        notes="Meta's MusicGen - non-commercial use only. "
        "Commercial use requires separate license from Meta.",
    ),
    "musicgen-medium": LicenseInfo(
        license_type=LicenseType.CC_BY_NC_4_0,
        commercial_use=False,
        attribution_required=True,
        source_url="https://github.com/facebookresearch/audiocraft",
        notes="Meta's MusicGen - non-commercial use only. "
        "Commercial use requires separate license from Meta.",
    ),
    "musicgen-large": LicenseInfo(
        license_type=LicenseType.CC_BY_NC_4_0,
        commercial_use=False,
        attribution_required=True,
        source_url="https://github.com/facebookresearch/audiocraft",
        notes="Meta's MusicGen - non-commercial use only. "
        "Commercial use requires separate license from Meta.",
    ),
    "musicgen-melody": LicenseInfo(
        license_type=LicenseType.CC_BY_NC_4_0,
        commercial_use=False,
        attribution_required=True,
        source_url="https://github.com/facebookresearch/audiocraft",
        notes="Meta's MusicGen - non-commercial use only. "
        "Commercial use requires separate license from Meta.",
    ),
    # Codec models - CC-BY-NC-4.0 (NON-COMMERCIAL ONLY)
    "encodec-32khz": LicenseInfo(
        license_type=LicenseType.CC_BY_NC_4_0,
        commercial_use=False,
        attribution_required=True,
        source_url="https://github.com/facebookresearch/encodec",
        notes="Meta's EnCodec - non-commercial use only. "
        "Commercial use requires separate license from Meta.",
    ),
    "encodec-24khz": LicenseInfo(
        license_type=LicenseType.CC_BY_NC_4_0,
        commercial_use=False,
        attribution_required=True,
        source_url="https://github.com/facebookresearch/encodec",
        notes="Meta's EnCodec - non-commercial use only. "
        "Commercial use requires separate license from Meta.",
    ),
    # VAD models - MIT (commercial OK)
    "silero-vad": LicenseInfo(
        license_type=LicenseType.MIT,
        commercial_use=True,
        attribution_required=True,
        source_url="https://github.com/snakers4/silero-vad",
    ),
    "silero-vad-8k": LicenseInfo(
        license_type=LicenseType.MIT,
        commercial_use=True,
        attribution_required=True,
        source_url="https://github.com/snakers4/silero-vad",
    ),
    # Enhancement models - ISC (commercial OK)
    "deepfilternet2": LicenseInfo(
        license_type=LicenseType.ISC,
        commercial_use=True,
        attribution_required=True,
        source_url="https://github.com/Rikorose/DeepFilterNet",
    ),
    "deepfilternet2-16k": LicenseInfo(
        license_type=LicenseType.ISC,
        commercial_use=True,
        attribution_required=True,
        source_url="https://github.com/Rikorose/DeepFilterNet",
    ),
    # Diarization models - Apache 2.0 (commercial OK)
    "ecapa-tdnn": LicenseInfo(
        license_type=LicenseType.APACHE_2_0,
        commercial_use=True,
        attribution_required=True,
        source_url="https://github.com/speechbrain/speechbrain",
    ),
    # TTS models - Apache 2.0 (commercial OK)
    "parler-tts-mini": LicenseInfo(
        license_type=LicenseType.APACHE_2_0,
        commercial_use=True,
        attribution_required=True,
        source_url="https://github.com/huggingface/parler-tts",
    ),
    "parler-tts-large": LicenseInfo(
        license_type=LicenseType.APACHE_2_0,
        commercial_use=True,
        attribution_required=True,
        source_url="https://github.com/huggingface/parler-tts",
    ),
}


class NonCommercialModelWarning(UserWarning):
    """Warning issued when loading a model with commercial use restrictions."""

    pass


# Track which models we've already warned about (warn once per session)
_warned_models: set[str] = set()

# Environment variable to suppress warnings
_SUPPRESS_LICENSE_WARNINGS = "MLX_AUDIO_SUPPRESS_LICENSE_WARNINGS"


def get_license_info(model_name: str) -> LicenseInfo | None:
    """Get license information for a model.

    Args:
        model_name: Name of the model (e.g., "musicgen-medium")

    Returns:
        LicenseInfo if known, None otherwise
    """
    return MODEL_LICENSES.get(model_name)


def is_commercial_safe(model_name: str) -> bool:
    """Check if a model is safe for commercial use.

    Args:
        model_name: Name of the model

    Returns:
        True if commercial use is permitted, False otherwise.
        Returns True for unknown models (optimistic default).
    """
    info = get_license_info(model_name)
    if info is None:
        return True  # Optimistic default for unknown models
    return info.commercial_use


def check_license_and_warn(model_name: str) -> None:
    """Check model license and issue warning if non-commercial.

    This function is called during model loading to alert users about
    commercial use restrictions.

    Args:
        model_name: Name of the model being loaded
    """
    # Check if warnings are suppressed
    if os.environ.get(_SUPPRESS_LICENSE_WARNINGS, "").lower() in ("1", "true", "yes"):
        return

    # Only warn once per model per session
    if model_name in _warned_models:
        return

    info = get_license_info(model_name)
    if info is None:
        return

    if not info.commercial_use:
        _warned_models.add(model_name)
        warning_msg = (
            f"\n{'=' * 70}\n"
            f"LICENSE WARNING: '{model_name}' is licensed under {info.license_type.value}\n"
            f"This model is for NON-COMMERCIAL USE ONLY.\n"
            f"\n"
            f"Commercial use of this model requires a separate license.\n"
        )
        if info.notes:
            warning_msg += f"\n{info.notes}\n"
        if info.source_url:
            warning_msg += f"\nMore info: {info.source_url}\n"
        warning_msg += (
            f"\nTo suppress this warning, set environment variable:\n"
            f"  export {_SUPPRESS_LICENSE_WARNINGS}=1\n"
            f"{'=' * 70}"
        )
        warnings.warn(warning_msg, NonCommercialModelWarning, stacklevel=3)


def list_commercial_safe_models() -> list[str]:
    """List all models that are safe for commercial use.

    Returns:
        List of model names with permissive licenses
    """
    return [name for name, info in MODEL_LICENSES.items() if info.commercial_use]


def list_non_commercial_models() -> list[str]:
    """List all models with commercial use restrictions.

    Returns:
        List of model names with non-commercial licenses
    """
    return [name for name, info in MODEL_LICENSES.items() if not info.commercial_use]
