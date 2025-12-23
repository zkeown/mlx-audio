"""HTDemucs source separation model for MLX."""

from mlx_audio.models.demucs.config import HTDemucsConfig
from mlx_audio.models.demucs.model import HTDemucs
from mlx_audio.models.demucs.bag import BagOfModels
from mlx_audio.models.demucs.inference import apply_model
from mlx_audio.models.demucs.convert import (
    convert_htdemucs_weights,
    convert_from_demucs_package,
    convert_bag_from_demucs_package,
    download_and_convert,
)

__all__ = [
    "HTDemucs",
    "HTDemucsConfig",
    "BagOfModels",
    "apply_model",
    "convert_htdemucs_weights",
    "convert_from_demucs_package",
    "convert_bag_from_demucs_package",
    "download_and_convert",
]
