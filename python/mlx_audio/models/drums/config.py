"""Configuration for DrumTranscriber model."""

from dataclasses import dataclass


NUM_CLASSES = 14  # Number of drum classes


@dataclass
class DrumTranscriberConfig:
    """Configuration for DrumTranscriber model."""

    # Input
    n_mels: int = 128
    num_classes: int = NUM_CLASSES

    # Encoder
    encoder_type: str = "standard"  # "standard" or "lightweight"
    base_channels: int = 32
    embed_dim: int = 512

    # Transformer
    num_layers: int = 4
    num_heads: int = 8
    mlp_ratio: float = 4.0
    use_local_attention: bool = False
    window_size: int = 64  # For local attention
    max_seq_len: int = 2048

    # Heads
    head_hidden_dim: int | None = None  # Defaults to embed_dim // 2
    share_head_layers: bool = False

    # Regularization
    dropout: float = 0.1
    attention_dropout: float = 0.1
    encoder_dropout: float = 0.1


@dataclass
class SpectrogramConfig:
    """Configuration for mel-spectrogram computation.

    Tuned for cymbal discrimination with high frequency resolution.
    """

    sample_rate: int = 44100
    n_fft: int = 2048
    hop_length: int = 441  # 10ms frames at 44.1kHz (~100 fps)
    n_mels: int = 128
    fmin: float = 20.0
    fmax: float = 20000.0

    @property
    def frame_rate(self) -> float:
        """Frames per second."""
        return self.sample_rate / self.hop_length

    @property
    def frame_duration_ms(self) -> float:
        """Duration of each frame in milliseconds."""
        return 1000.0 * self.hop_length / self.sample_rate

    def time_to_frame(self, time_sec: float) -> int:
        """Convert time in seconds to frame index."""
        return int(round(time_sec * self.frame_rate))

    def frame_to_time(self, frame: int) -> float:
        """Convert frame index to time in seconds."""
        return frame / self.frame_rate
