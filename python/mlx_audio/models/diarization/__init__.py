"""Speaker diarization models."""

from .config import DiarizationConfig, EcapaTDNNConfig
from .model import ECAPATDNN, SpeakerDiarization

__all__ = [
    "ECAPATDNN",
    "EcapaTDNNConfig",
    "SpeakerDiarization",
    "DiarizationConfig",
]
