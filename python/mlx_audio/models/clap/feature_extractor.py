"""CLAP feature extractor for audio preprocessing.

Implements audio preprocessing compatible with HuggingFace's ClapFeatureExtractor.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mlx.core as mx
import numpy as np

if TYPE_CHECKING:
    from mlx_audio.models.clap.config import CLAPAudioConfig


class CLAPFeatureExtractor:
    """Audio feature extractor for CLAP models.

    Uses HuggingFace's ClapFeatureExtractor when available for exact
    compatibility. Falls back to a numpy implementation otherwise.

    Args:
        sample_rate: Target sample rate (default: 48000)
        n_fft: FFT window size (default: 1024)
        hop_length: Hop length (default: 480)
        n_mels: Number of mel bins (default: 64)
        max_length_s: Maximum audio length in seconds (default: 10)
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        n_fft: int = 1024,
        hop_length: int = 480,
        n_mels: int = 64,
        max_length_s: float = 10.0,
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.max_length_s = max_length_s
        self.nb_max_samples = int(max_length_s * sample_rate)

        # Try to use HuggingFace feature extractor for exact compatibility
        self._hf_processor = None
        try:
            from transformers import ClapProcessor
            self._hf_processor = ClapProcessor.from_pretrained(
                "laion/clap-htsat-fused"
            )
        except (ImportError, Exception):
            pass

    @classmethod
    def from_config(cls, config: "CLAPAudioConfig") -> "CLAPFeatureExtractor":
        """Create feature extractor from CLAP audio config."""
        return cls(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
        )

    def __call__(
        self,
        audio: np.ndarray | mx.array,
        sample_rate: int | None = None,
        return_tensors: str = "mx",
    ) -> dict:
        """Process audio into CLAP-compatible features.

        Args:
            audio: Audio waveform [T] or [B, T]
            sample_rate: Audio sample rate (uses self.sample_rate if None)
            return_tensors: Output format ("mx" or "np")

        Returns:
            Dictionary with:
                - input_features: Mel spectrogram [B, 4, n_frames, n_mels]
                - is_longer: Boolean tensor [B, 1]
        """
        if isinstance(audio, mx.array):
            audio = np.array(audio)

        # Handle batched input
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]

        sr = sample_rate if sample_rate is not None else self.sample_rate

        # Use HuggingFace processor if available (exact compatibility)
        if self._hf_processor is not None:
            return self._process_with_hf(audio, sr, return_tensors)
        else:
            return self._process_fallback(audio, sr, return_tensors)

    def _process_with_hf(
        self,
        audio: np.ndarray,
        sample_rate: int,
        return_tensors: str,
    ) -> dict:
        """Process audio using HuggingFace's feature extractor."""
        batch_size = audio.shape[0]

        # Process each sample
        all_features = []
        all_is_longer = []

        for i in range(batch_size):
            inputs = self._hf_processor(
                audio=audio[i],
                sampling_rate=sample_rate,
                return_tensors="np",
            )
            all_features.append(inputs["input_features"])
            all_is_longer.append(inputs["is_longer"])

        # Stack batch
        input_features = np.concatenate(all_features, axis=0)
        is_longer = np.concatenate(all_is_longer, axis=0)

        if return_tensors == "mx":
            input_features = mx.array(input_features)
            is_longer = mx.array(is_longer)

        return {
            "input_features": input_features,
            "is_longer": is_longer,
        }

    def _process_fallback(
        self,
        audio: np.ndarray,
        sample_rate: int,
        return_tensors: str,
    ) -> dict:
        """Fallback processing without HuggingFace (less accurate)."""
        try:
            import librosa
        except ImportError:
            raise ImportError(
                "librosa is required for audio processing without transformers. "
                "Install with: pip install librosa"
            )

        batch_size = audio.shape[0]
        mel_spectrograms = []
        is_longer_list = []

        for i in range(batch_size):
            sample = audio[i]

            # Resample if needed
            if sample_rate != self.sample_rate:
                sample = librosa.resample(
                    sample,
                    orig_sr=sample_rate,
                    target_sr=self.sample_rate
                )

            # Check if longer than max
            is_longer = len(sample) > self.nb_max_samples
            is_longer_list.append([is_longer])

            # Compute mel spectrogram
            mel = self._compute_mel(sample)

            # For fusion mode, always produce 4 channels
            if is_longer:
                mel = self._compute_fusion_mel(sample)
            else:
                # Pad/truncate and repeat 4 times
                mel = self._compute_padded_mel(sample)
                mel = np.stack([mel, mel, mel, mel], axis=0)

            mel_spectrograms.append(mel)

        input_features = np.stack(mel_spectrograms, axis=0)
        is_longer_arr = np.array(is_longer_list)

        if return_tensors == "mx":
            input_features = mx.array(input_features)
            is_longer_arr = mx.array(is_longer_arr)

        return {
            "input_features": input_features,
            "is_longer": is_longer_arr,
        }

    def _compute_mel(self, audio: np.ndarray) -> np.ndarray:
        """Compute log-mel spectrogram."""
        import librosa

        # Compute mel spectrogram
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=50,
            fmax=14000,
            htk=True,
        )

        # Convert to dB
        mel_db = librosa.power_to_db(mel, ref=1.0, top_db=None)

        return mel_db.T  # [T, F]

    def _compute_padded_mel(self, audio: np.ndarray) -> np.ndarray:
        """Compute mel for audio shorter than max length."""
        # Pad with repeat if needed
        if len(audio) < self.nb_max_samples:
            n_repeat = int(self.nb_max_samples / len(audio))
            audio = np.tile(audio, n_repeat)

        # Pad to exact length
        if len(audio) < self.nb_max_samples:
            audio = np.pad(
                audio,
                (0, self.nb_max_samples - len(audio)),
                mode='constant'
            )
        else:
            audio = audio[:self.nb_max_samples]

        return self._compute_mel(audio)

    def _compute_fusion_mel(self, audio: np.ndarray) -> np.ndarray:
        """Compute fusion mel for long audio (4 channels)."""
        import torch

        # Compute full mel
        mel = self._compute_mel(audio)  # [T, F]

        chunk_frames = self.nb_max_samples // self.hop_length + 1
        total_frames = mel.shape[0]

        if chunk_frames >= total_frames:
            # Audio is actually short enough, just repeat
            return np.stack([mel, mel, mel, mel], axis=0)

        # Get 3 random crops
        ranges = np.array_split(
            list(range(0, total_frames - chunk_frames + 1)), 3
        )
        if len(ranges[1]) == 0:
            ranges[1] = [0]
        if len(ranges[2]) == 0:
            ranges[2] = [0]

        idx_front = np.random.choice(ranges[0])
        idx_middle = np.random.choice(ranges[1])
        idx_back = np.random.choice(ranges[2])

        mel_front = mel[idx_front:idx_front + chunk_frames, :]
        mel_middle = mel[idx_middle:idx_middle + chunk_frames, :]
        mel_back = mel[idx_back:idx_back + chunk_frames, :]

        # Downsampled version
        mel_tensor = torch.tensor(mel[None, None, :])
        mel_shrink = torch.nn.functional.interpolate(
            mel_tensor,
            size=[chunk_frames, self.n_mels],
            mode="bilinear",
            align_corners=False
        )
        mel_shrink = mel_shrink[0][0].numpy()

        return np.stack(
            [mel_shrink, mel_front, mel_middle, mel_back],
            axis=0
        )
