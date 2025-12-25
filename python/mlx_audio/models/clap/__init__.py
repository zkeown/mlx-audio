"""CLAP (Contrastive Language-Audio Pretraining) model for mlx-audio.

CLAP learns a joint embedding space for audio and text, enabling:
- Audio embeddings for similarity search
- Text embeddings for audio-text matching
- Zero-shot audio classification
- Audio-text retrieval

Example:
    >>> from mlx_audio.models.clap import CLAP
    >>> model = CLAP.from_pretrained("path/to/clap")
    >>> audio_emb = model.encode_audio(mel_spectrogram)
    >>> text_emb = model.encode_text(token_ids)
    >>> similarity = model.similarity(audio_emb, text_emb)
"""

from mlx_audio.models.clap.config import CLAPAudioConfig, CLAPConfig, CLAPTextConfig
from mlx_audio.models.clap.model import CLAP

__all__ = [
    "CLAP",
    "CLAPConfig",
    "CLAPAudioConfig",
    "CLAPTextConfig",
]
