"""Banquet query-based source separation model for MLX.

Banquet uses a reference audio query to extract matching sounds from a mixture.
Based on the ISMIR 2024 paper by Watcharasupat et al.

Pipeline:
    Query Audio → PaSST → 768-dim embedding
                              ↓
    Mixture → STFT → BandSplit → SeqBand (24 LSTMs) → FiLM → MaskEstimation → iSTFT → Output
"""

from mlx_audio.models.banquet.bandsplit import BandSplitModule, NormFC
from mlx_audio.models.banquet.banquet import Banquet, BanquetOutput
from mlx_audio.models.banquet.config import BandSpec, BanquetConfig, PaSSTConfig
from mlx_audio.models.banquet.film import FiLM
from mlx_audio.models.banquet.inference import (
    apply_model,
    prepare_query_mel,
    separate,
)
from mlx_audio.models.banquet.maskestim import (
    GLU,
    NormMLP,
    OverlappingMaskEstimationModule,
)
from mlx_audio.models.banquet.passt import PaSST
from mlx_audio.models.banquet.tfmodel import ResidualRNN, SeqBandModellingModule
from mlx_audio.models.banquet.utils import MusicalBandsplitSpecification

__all__ = [
    # Main model
    "Banquet",
    "BanquetOutput",
    # Query encoder
    "PaSST",
    # Components
    "BandSplitModule",
    "NormFC",
    "SeqBandModellingModule",
    "ResidualRNN",
    "FiLM",
    "OverlappingMaskEstimationModule",
    "NormMLP",
    "GLU",
    # Config
    "BanquetConfig",
    "PaSSTConfig",
    "BandSpec",
    # Inference
    "apply_model",
    "separate",
    "prepare_query_mel",
    # Utils
    "MusicalBandsplitSpecification",
]
