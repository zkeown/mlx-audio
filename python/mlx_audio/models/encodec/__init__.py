"""EnCodec neural audio codec for audio tokenization."""

from mlx_audio.models.encodec.config import EnCodecConfig
from mlx_audio.models.encodec.model import EnCodec

__all__ = ["EnCodec", "EnCodecConfig"]
