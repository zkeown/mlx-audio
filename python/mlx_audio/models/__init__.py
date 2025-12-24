"""Pre-built model implementations for mlx-audio.

This module provides access to all pre-trained audio models:

Source Separation:
    - HTDemucs: Hybrid Transformer Demucs for music source separation
    - Banquet: Query-based source separation using audio references

Speech Processing:
    - Whisper: OpenAI's speech recognition model
    - SileroVAD: Voice activity detection
    - ECAPATDNN: Speaker embedding and diarization

Audio-Text:
    - CLAP: Contrastive Language-Audio Pretraining for embeddings

Audio Generation:
    - MusicGen: Text-to-music generation
    - ParlerTTS: Text-to-speech synthesis
    - EnCodec: Neural audio codec

Audio Analysis:
    - DeepFilterNet: Audio enhancement and denoising
    - CLAPClassifier: Audio classification

Example:
    >>> from mlx_audio.models import HTDemucs, CLAP, Whisper
    >>> htdemucs = HTDemucs.from_pretrained("htdemucs_ft")
    >>> clap = CLAP.from_pretrained("clap-htsat-fused")
"""

# Source Separation
from mlx_audio.models.demucs import (
    HTDemucs,
    HTDemucsConfig,
    BagOfModels,
    apply_model as apply_demucs,
)
from mlx_audio.models.banquet import (
    Banquet,
    BanquetConfig,
    apply_model as apply_banquet,
)

# Speech Processing
from mlx_audio.models.whisper import (
    Whisper,
    WhisperConfig,
    WhisperTokenizer,
    apply_model as apply_whisper,
)
from mlx_audio.models.vad import (
    SileroVAD,
    VADConfig,
)
from mlx_audio.models.diarization import (
    ECAPATDNN,
    EcapaTDNNConfig,
    SpeakerDiarization,
    DiarizationConfig,
)

# Audio-Text
from mlx_audio.models.clap import (
    CLAP,
    CLAPConfig,
    CLAPAudioConfig,
    CLAPTextConfig,
)

# Audio Generation
from mlx_audio.models.musicgen import (
    MusicGen,
    MusicGenConfig,
)
from mlx_audio.models.tts import (
    ParlerTTS,
    ParlerTTSConfig,
)
from mlx_audio.models.encodec import (
    EnCodec,
    EnCodecConfig,
)

# Audio Analysis
from mlx_audio.models.enhance import (
    DeepFilterNet,
    DeepFilterNetConfig,
)
from mlx_audio.models.classifier import (
    CLAPClassifier,
    ClassifierConfig,
)

__all__ = [
    # Source Separation
    "HTDemucs",
    "HTDemucsConfig",
    "BagOfModels",
    "apply_demucs",
    "Banquet",
    "BanquetConfig",
    "apply_banquet",
    # Speech Processing
    "Whisper",
    "WhisperConfig",
    "WhisperTokenizer",
    "apply_whisper",
    "SileroVAD",
    "VADConfig",
    "ECAPATDNN",
    "EcapaTDNNConfig",
    "SpeakerDiarization",
    "DiarizationConfig",
    # Audio-Text
    "CLAP",
    "CLAPConfig",
    "CLAPAudioConfig",
    "CLAPTextConfig",
    # Audio Generation
    "MusicGen",
    "MusicGenConfig",
    "ParlerTTS",
    "ParlerTTSConfig",
    "EnCodec",
    "EnCodecConfig",
    # Audio Analysis
    "DeepFilterNet",
    "DeepFilterNetConfig",
    "CLAPClassifier",
    "ClassifierConfig",
]
