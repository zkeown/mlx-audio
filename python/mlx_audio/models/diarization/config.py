"""Configuration for speaker diarization models."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EcapaTDNNConfig:
    """ECAPA-TDNN speaker embedding model configuration.

    Attributes:
        input_dim: Number of input features (80 for mel filterbanks)
        channels: Channel sizes for TDNN layers
        kernel_sizes: Kernel sizes for TDNN layers
        dilations: Dilation rates for SE-Res2Net blocks
        attention_channels: Channels for SE attention
        res2net_scale: Number of scale paths in Res2Net
        se_channels: Squeeze-Excitation reduction channels
        lin_neurons: Output embedding dimension (typically 192 or 512)
        global_context: Use global context in attention pooling
    """

    input_dim: int = 80
    channels: tuple[int, ...] = (1024, 1024, 1024, 1024, 3072)
    kernel_sizes: tuple[int, ...] = (5, 3, 3, 3, 1)
    dilations: tuple[int, ...] = (1, 2, 3, 4, 1)
    attention_channels: int = 128
    res2net_scale: int = 8
    se_channels: int = 128
    lin_neurons: int = 192
    global_context: bool = True

    @classmethod
    def ecapa_tdnn_512(cls) -> EcapaTDNNConfig:
        """Standard ECAPA-TDNN with 512-dim embeddings."""
        return cls(lin_neurons=512)

    @classmethod
    def ecapa_tdnn_192(cls) -> EcapaTDNNConfig:
        """Compact ECAPA-TDNN with 192-dim embeddings."""
        return cls(lin_neurons=192)

    @classmethod
    def from_dict(cls, d: dict) -> EcapaTDNNConfig:
        """Create config from dictionary."""
        # Handle tuple fields
        result = {}
        for k, v in d.items():
            if k in cls.__dataclass_fields__:
                if k in ("channels", "kernel_sizes", "dilations"):
                    result[k] = tuple(v) if isinstance(v, list) else v
                else:
                    result[k] = v
        return cls(**result)


@dataclass
class DiarizationConfig:
    """Full diarization pipeline configuration.

    Attributes:
        embedding: Speaker embedding model config
        sample_rate: Audio sample rate
        segment_duration: Duration of segments for embedding (seconds)
        segment_step: Step size between segments (seconds)
        cluster_threshold: Agglomerative clustering distance threshold
        min_segment_duration: Minimum speaker segment duration
        min_speakers: Minimum number of speakers
        max_speakers: Maximum number of speakers (None = auto-detect)
        vad_onset: VAD onset threshold
        vad_offset: VAD offset threshold
    """

    embedding: EcapaTDNNConfig = field(default_factory=EcapaTDNNConfig)
    sample_rate: int = 16000
    segment_duration: float = 1.5
    segment_step: float = 0.75
    cluster_threshold: float = 0.5
    min_segment_duration: float = 0.5
    min_speakers: int = 1
    max_speakers: int | None = None
    vad_onset: float = 0.5
    vad_offset: float = 0.3

    # Feature extraction
    n_mels: int = 80
    n_fft: int = 400
    hop_length: int = 160
    fmin: float = 20.0
    fmax: float = 7600.0

    @classmethod
    def default(cls) -> DiarizationConfig:
        """Default configuration."""
        return cls()

    @classmethod
    def from_dict(cls, d: dict) -> DiarizationConfig:
        """Create config from dictionary."""
        embedding_dict = d.pop("embedding", {})
        embedding = EcapaTDNNConfig.from_dict(embedding_dict)

        result = {
            k: v for k, v in d.items()
            if k in cls.__dataclass_fields__
        }
        result["embedding"] = embedding
        return cls(**result)
