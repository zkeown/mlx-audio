"""ECAPA-TDNN speaker embedding and diarization models."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .config import DiarizationConfig, EcapaTDNNConfig
from .layers.pooling import AttentiveStatisticsPooling
from .layers.se_res2net import SERes2NetBlock

if TYPE_CHECKING:
    pass


class ECAPATDNN(nn.Module):
    """ECAPA-TDNN speaker embedding model.

    Extracts fixed-dimensional speaker embeddings from variable-length audio.
    Architecture follows SpeechBrain's implementation.

    Parameters
    ----------
    config : EcapaTDNNConfig
        Model configuration.
    """

    def __init__(self, config: EcapaTDNNConfig | None = None):
        super().__init__()
        self.config = config or EcapaTDNNConfig()

        # Initial TDNN layer (1D conv with large kernel)
        self.layer1 = nn.Conv1d(
            self.config.input_dim,
            self.config.channels[0],
            self.config.kernel_sizes[0],
            padding=self.config.kernel_sizes[0] // 2,
        )
        self.bn1 = nn.BatchNorm(self.config.channels[0])

        # SE-Res2Net blocks
        self.layer2 = SERes2NetBlock(
            self.config.channels[0],
            self.config.channels[1],
            kernel_size=self.config.kernel_sizes[1],
            dilation=self.config.dilations[1],
            scale=self.config.res2net_scale,
            se_channels=self.config.se_channels,
        )

        self.layer3 = SERes2NetBlock(
            self.config.channels[1],
            self.config.channels[2],
            kernel_size=self.config.kernel_sizes[2],
            dilation=self.config.dilations[2],
            scale=self.config.res2net_scale,
            se_channels=self.config.se_channels,
        )

        self.layer4 = SERes2NetBlock(
            self.config.channels[2],
            self.config.channels[3],
            kernel_size=self.config.kernel_sizes[3],
            dilation=self.config.dilations[3],
            scale=self.config.res2net_scale,
            se_channels=self.config.se_channels,
        )

        # Multi-layer feature aggregation (MFA)
        # Concatenate outputs of layers 2, 3, 4
        mfa_channels = sum(self.config.channels[1:4])
        self.mfa = nn.Conv1d(mfa_channels, self.config.channels[4], 1)
        self.bn_mfa = nn.BatchNorm(self.config.channels[4])

        # Attentive statistics pooling
        self.asp = AttentiveStatisticsPooling(
            self.config.channels[4],
            self.config.attention_channels,
            self.config.global_context,
        )

        # Final batch norm
        self.bn_asp = nn.BatchNorm(self.config.channels[4] * 2)

        # Output linear to embedding dimension
        self.fc = nn.Linear(
            self.config.channels[4] * 2,
            self.config.lin_neurons,
        )

    def __call__(self, x: mx.array) -> mx.array:
        """Extract speaker embedding.

        Parameters
        ----------
        x : mx.array
            Input features of shape (batch, time, features) or (time, features).
            Typically 80-dim mel filterbank features.

        Returns
        -------
        mx.array
            Speaker embedding of shape (batch, lin_neurons) or (lin_neurons,).
        """
        # Handle unbatched input
        input_is_2d = x.ndim == 2
        if input_is_2d:
            x = x[None, :]

        # Input is (batch, time, features) which matches MLX conv1d (batch, length, channels)

        # Initial layer
        out1 = nn.relu(self.bn1(self.layer1(x)))

        # SE-Res2Net blocks
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)

        # Multi-layer feature aggregation (concatenate along channels axis)
        mfa_in = mx.concatenate([out2, out3, out4], axis=2)
        mfa_out = nn.relu(self.bn_mfa(self.mfa(mfa_in)))

        # Attentive statistics pooling
        pooled = self.asp(mfa_out)  # (batch, channels * 2)
        pooled = self.bn_asp(pooled)

        # Final embedding
        embedding = self.fc(pooled)

        if input_is_2d:
            embedding = embedding[0]

        return embedding

    @classmethod
    def from_pretrained(
        cls,
        path: str | Path | None = None,
        config: EcapaTDNNConfig | None = None,
    ) -> "ECAPATDNN":
        """Load pretrained model.

        Parameters
        ----------
        path : str or Path, optional
            Path to model weights.
        config : EcapaTDNNConfig, optional
            Model configuration.

        Returns
        -------
        ECAPATDNN
            Loaded model.
        """
        if config is None:
            config = EcapaTDNNConfig()

        model = cls(config)

        if path is not None:
            path = Path(path)
            weights_path = path / "weights.npz" if path.is_dir() else path

            if weights_path.exists():
                weights = mx.load(str(weights_path))
                model.load_weights(list(weights.items()))

        return model


class SpeakerDiarization(nn.Module):
    """Speaker diarization pipeline.

    Combines VAD, speaker embedding, and clustering for end-to-end
    diarization ("who spoke when").

    Parameters
    ----------
    config : DiarizationConfig
        Pipeline configuration.
    """

    def __init__(self, config: DiarizationConfig | None = None):
        super().__init__()
        self.config = config or DiarizationConfig()
        self.embedding_model = ECAPATDNN(self.config.embedding)

    def _compute_mel_features(self, audio: mx.array) -> mx.array:
        """Compute mel filterbank features for embedding extraction.

        Parameters
        ----------
        audio : mx.array
            Audio waveform (samples,) or (batch, samples).

        Returns
        -------
        mx.array
            Mel features of shape (batch, n_frames, n_mels).
        """
        from mlx_audio.primitives import melspectrogram

        mel = melspectrogram(
            audio,
            sr=self.config.sample_rate,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            n_mels=self.config.n_mels,
            fmin=self.config.fmin,
            fmax=self.config.fmax,
        )

        # Transpose to (batch, n_frames, n_mels)
        if mel.ndim == 2:
            mel = mel[None, :]
        mel = mx.transpose(mel, (0, 2, 1))

        # Log compression
        mel = mx.log(mel + 1e-8)

        return mel

    def extract_embeddings(
        self,
        audio: mx.array,
        segments: list[tuple[float, float]] | None = None,
    ) -> tuple[mx.array, list[tuple[float, float]]]:
        """Extract speaker embeddings from audio segments.

        Parameters
        ----------
        audio : mx.array
            Audio waveform.
        segments : list of (start, end), optional
            Pre-computed speech segments in seconds.
            If None, uses fixed-window segmentation.

        Returns
        -------
        tuple
            (embeddings, segments) where:
            - embeddings: (n_segments, embed_dim)
            - segments: List of (start, end) times
        """
        sr = self.config.sample_rate
        audio_np = np.array(audio)

        if segments is None:
            # Fixed-window segmentation
            seg_samples = int(self.config.segment_duration * sr)
            step_samples = int(self.config.segment_step * sr)

            segments = []
            start = 0
            while start + seg_samples <= len(audio_np):
                end = start + seg_samples
                segments.append((start / sr, end / sr))
                start += step_samples

        # Extract embedding for each segment
        embeddings = []
        for start_sec, end_sec in segments:
            start_sample = int(start_sec * sr)
            end_sample = int(end_sec * sr)

            segment_audio = mx.array(audio_np[start_sample:end_sample])

            # Compute mel features
            mel_feat = self._compute_mel_features(segment_audio)

            # Extract embedding
            emb = self.embedding_model(mel_feat[0])
            embeddings.append(emb)

        embeddings = mx.stack(embeddings, axis=0)
        return embeddings, segments

    def cluster_embeddings(
        self,
        embeddings: mx.array,
        num_speakers: int | None = None,
    ) -> np.ndarray:
        """Cluster embeddings into speaker groups.

        Uses agglomerative hierarchical clustering with cosine distance.

        Parameters
        ----------
        embeddings : mx.array
            Speaker embeddings of shape (n_segments, embed_dim).
        num_speakers : int, optional
            Number of speakers. If None, auto-detect.

        Returns
        -------
        np.ndarray
            Cluster labels of shape (n_segments,).
        """
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import pdist

        emb_np = np.array(embeddings)

        # Normalize embeddings
        emb_norm = emb_np / (np.linalg.norm(emb_np, axis=1, keepdims=True) + 1e-8)

        # Compute pairwise cosine distances
        distances = pdist(emb_norm, metric="cosine")

        # Hierarchical clustering
        linkage_matrix = linkage(distances, method="average")

        if num_speakers is not None:
            # Fixed number of clusters
            labels = fcluster(linkage_matrix, num_speakers, criterion="maxclust")
        else:
            # Threshold-based clustering
            labels = fcluster(
                linkage_matrix,
                self.config.cluster_threshold,
                criterion="distance"
            )

        return labels - 1  # Convert to 0-indexed

    def __call__(
        self,
        audio: mx.array,
        num_speakers: int | None = None,
    ) -> list[tuple[str, float, float]]:
        """Run full diarization pipeline.

        Parameters
        ----------
        audio : mx.array
            Input audio waveform.
        num_speakers : int, optional
            Number of speakers (None = auto-detect).

        Returns
        -------
        list
            List of (speaker_id, start_time, end_time) tuples.
        """
        # Extract embeddings
        embeddings, segments = self.extract_embeddings(audio)

        # Cluster
        labels = self.cluster_embeddings(embeddings, num_speakers)

        # Build output
        results = []
        for i, (start, end) in enumerate(segments):
            speaker_id = f"SPEAKER_{labels[i]:02d}"
            results.append((speaker_id, start, end))

        # Merge consecutive segments with same speaker
        results = self._merge_consecutive(results)

        return results

    def _merge_consecutive(
        self,
        segments: list[tuple[str, float, float]],
    ) -> list[tuple[str, float, float]]:
        """Merge consecutive segments with the same speaker."""
        if len(segments) <= 1:
            return segments

        merged = [segments[0]]
        for speaker, start, end in segments[1:]:
            prev_speaker, prev_start, prev_end = merged[-1]

            # If same speaker and close in time, merge
            if speaker == prev_speaker and start - prev_end < 0.5:
                merged[-1] = (speaker, prev_start, end)
            else:
                merged.append((speaker, start, end))

        return merged

    @classmethod
    def from_pretrained(
        cls,
        path: str | Path | None = None,
        config: DiarizationConfig | None = None,
    ) -> "SpeakerDiarization":
        """Load pretrained model.

        Parameters
        ----------
        path : str or Path, optional
            Path to model weights.
        config : DiarizationConfig, optional
            Pipeline configuration.

        Returns
        -------
        SpeakerDiarization
            Loaded model.
        """
        if config is None:
            config = DiarizationConfig()

        model = cls(config)

        if path is not None:
            path = Path(path)
            weights_path = path / "weights.npz" if path.is_dir() else path

            if weights_path.exists():
                weights = mx.load(str(weights_path))
                model.embedding_model.load_weights(list(weights.items()))

        return model
