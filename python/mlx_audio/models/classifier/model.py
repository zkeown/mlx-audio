"""CLAP-based audio classifier model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import mlx.core as mx
import mlx.nn as nn

from mlx_audio.models.classifier.config import (
    ClassifierConfig,
    FreezeMode,
    MLPHeadConfig,
    TaskMode,
)
from mlx_audio.models.classifier.layers.mlp_head import MLPHead

if TYPE_CHECKING:
    from mlx_audio.models.clap import CLAP


class CLAPClassifier(nn.Module):
    """Audio classifier built on CLAP embeddings.

    Combines a CLAP audio encoder with an MLP classification head for
    audio classification (single-label) or tagging (multi-label) tasks.

    Architecture:
        Audio -> CLAP Encoder -> 512-dim embedding -> MLP Head -> Logits

    The CLAP encoder can be frozen, partially frozen, or fully fine-tuned
    depending on the freeze_mode setting.

    Args:
        config: ClassifierConfig specifying model architecture
        clap: Optional pre-loaded CLAP model (loads from config if not provided)

    Example:
        >>> config = ClassifierConfig(
        ...     head=MLPHeadConfig(num_classes=50),
        ...     freeze_mode=FreezeMode.FROZEN,
        ... )
        >>> model = CLAPClassifier(config)
        >>> logits = model(audio)  # [B, 50]
    """

    def __init__(
        self,
        config: ClassifierConfig | None = None,
        clap: "CLAP | None" = None,
    ) -> None:
        super().__init__()
        if config is None:
            config = ClassifierConfig()
        self.config = config

        # Load or use provided CLAP model
        if clap is not None:
            self.clap = clap
        else:
            from mlx_audio.models.clap import CLAP
            from mlx_audio.hub.cache import get_cache

            cache = get_cache()
            self.clap = cache.get_model(config.clap_model, CLAP)

        # Apply freezing based on config
        self._apply_freeze_mode(config.freeze_mode)

        # Create classifier head
        self.head = MLPHead(config.head)

        # Build label mappings for inference
        self._label_to_idx: dict[str, int] = {}
        self._idx_to_label: dict[int, str] = {}
        if config.label_names:
            self._build_label_mapping(config.label_names)

    def _apply_freeze_mode(self, mode: FreezeMode) -> None:
        """Apply parameter freezing based on mode.

        Args:
            mode: Freezing mode to apply
        """
        if mode == FreezeMode.FROZEN:
            # Freeze entire CLAP encoder
            self.clap.freeze()
        elif mode == FreezeMode.PROJECTION_ONLY:
            # Freeze audio encoder, allow projection fine-tuning
            self.clap.audio_encoder.freeze()
            if self.clap.audio_fusion is not None:
                self.clap.audio_fusion.freeze()
            # audio_projection remains trainable
        # FreezeMode.FULL: nothing frozen

    def _build_label_mapping(self, label_names: list[str]) -> None:
        """Build label-to-index and index-to-label mappings.

        Args:
            label_names: List of class/tag names
        """
        for idx, name in enumerate(label_names):
            self._label_to_idx[name] = idx
            self._idx_to_label[idx] = name

    def encode(
        self,
        audio: mx.array,
        normalize: bool = True,
    ) -> mx.array:
        """Encode audio to CLAP embeddings.

        Args:
            audio: Mel spectrogram [B, 1, F, T], [B, F, T], or waveform [B, T]
            normalize: L2 normalize embeddings

        Returns:
            Audio embeddings of shape [B, 512]
        """
        return self.clap.encode_audio(audio, normalize=normalize)

    def forward(
        self,
        audio: mx.array,
        normalize: bool = True,
    ) -> mx.array:
        """Full forward pass from audio to logits.

        Args:
            audio: Mel spectrogram [B, 1, F, T], [B, F, T], or waveform [B, T]
            normalize: L2 normalize CLAP embeddings

        Returns:
            Logits of shape [B, num_classes]
        """
        embeddings = self.encode(audio, normalize=normalize)
        logits = self.head(embeddings)
        return logits

    def __call__(self, audio: mx.array, **kwargs) -> mx.array:
        """Forward pass."""
        return self.forward(audio, **kwargs)

    def predict(
        self,
        audio: mx.array,
        return_probabilities: bool = False,
    ) -> mx.array | tuple[mx.array, mx.array]:
        """Make predictions from audio.

        Args:
            audio: Input audio
            return_probabilities: Also return probability scores

        Returns:
            For classification: class indices [B] or (indices, probs)
            For tagging: binary labels [B, num_classes] or (labels, probs)
        """
        logits = self.forward(audio)

        if self.config.task == TaskMode.CLASSIFICATION:
            probs = mx.softmax(logits, axis=-1)
            predictions = mx.argmax(probs, axis=-1)
        else:  # TAGGING
            probs = mx.sigmoid(logits)
            predictions = (probs >= self.config.threshold).astype(mx.int32)

        if return_probabilities:
            return predictions, probs
        return predictions

    def predict_labels(
        self,
        audio: mx.array,
        top_k: int = 1,
    ) -> list[str] | list[list[str]]:
        """Predict with string labels.

        Args:
            audio: Input audio
            top_k: Number of top predictions per sample (classification only)

        Returns:
            For classification: list of label(s) per sample
            For tagging: list of active labels per sample

        Raises:
            ValueError: If no label mapping is available
        """
        if not self._idx_to_label:
            raise ValueError(
                "No label mapping available. Set label_names in config or load "
                "a model that includes label mappings."
            )

        logits = self.forward(audio)

        if self.config.task == TaskMode.CLASSIFICATION:
            probs = mx.softmax(logits, axis=-1)
            # Get top-k indices per sample
            top_indices = mx.argsort(probs, axis=-1)[:, ::-1][:, :top_k]
            results = []
            for sample_indices in top_indices:
                labels = [self._idx_to_label[int(idx)] for idx in sample_indices]
                results.append(labels[0] if top_k == 1 else labels)
            return results
        else:  # TAGGING
            probs = mx.sigmoid(logits)
            active = probs >= self.config.threshold
            results = []
            for sample_active in active:
                indices = mx.where(sample_active)[0]
                labels = [self._idx_to_label[int(idx)] for idx in indices]
                results.append(labels)
            return results

    def get_probabilities(
        self,
        audio: mx.array,
        labels: list[str] | None = None,
    ) -> dict[str, float] | list[dict[str, float]]:
        """Get probabilities for specific labels or all labels.

        Args:
            audio: Input audio [B, ...] or [...]
            labels: Specific labels to query (None = all labels)

        Returns:
            Dictionary mapping label names to probabilities
        """
        if not self._idx_to_label:
            raise ValueError("No label mapping available.")

        # Handle unbatched input
        if audio.ndim == 1 or (audio.ndim == 3 and audio.shape[0] != 1):
            audio = audio[None, ...]
            single_sample = True
        else:
            single_sample = audio.shape[0] == 1

        logits = self.forward(audio)

        if self.config.task == TaskMode.CLASSIFICATION:
            probs = mx.softmax(logits, axis=-1)
        else:
            probs = mx.sigmoid(logits)

        # Convert to list of dicts
        results = []
        for sample_probs in probs:
            prob_dict = {}
            for idx, prob in enumerate(sample_probs):
                label = self._idx_to_label.get(idx, str(idx))
                if labels is None or label in labels:
                    prob_dict[label] = float(prob)
            results.append(prob_dict)

        return results[0] if single_sample else results

    @classmethod
    def from_pretrained(
        cls,
        path: str | Path,
        **kwargs,
    ) -> CLAPClassifier:
        """Load pretrained classifier from directory.

        The directory should contain:
        - config.json: Classifier configuration
        - head.safetensors: MLP head weights
        - (optional) clap/: Fine-tuned CLAP weights

        Args:
            path: Path to model directory
            **kwargs: Override config parameters

        Returns:
            Loaded CLAPClassifier model
        """
        from mlx_audio.models.clap import CLAP

        path = Path(path)

        # Load config
        config_path = path / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config_dict = json.load(f)
            config = ClassifierConfig.from_dict(config_dict)
        else:
            config = ClassifierConfig()

        # Override with kwargs
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
            elif hasattr(config.head, key):
                setattr(config.head, key, value)

        # Check for fine-tuned CLAP
        clap_path = path / "clap"
        clap = None
        if clap_path.exists():
            clap = CLAP.from_pretrained(clap_path)

        # Create model
        model = cls(config, clap=clap)

        # Load head weights
        head_path = path / "head.safetensors"
        if head_path.exists():
            model.head.load_weights(str(head_path))

        model.eval()
        return model

    def save_pretrained(
        self,
        path: str | Path,
        save_clap: bool = False,
    ) -> None:
        """Save model to directory.

        Args:
            path: Output directory
            save_clap: Whether to save CLAP encoder (use if fine-tuned)
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save config
        with open(path / "config.json", "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        # Save head weights
        self.head.save_weights(str(path / "head.safetensors"))

        # Optionally save CLAP
        if save_clap:
            self.clap.save_pretrained(path / "clap")

    @property
    def num_classes(self) -> int:
        """Number of output classes."""
        return self.config.head.num_classes

    @property
    def label_names(self) -> list[str] | None:
        """List of class/tag names if available."""
        return self.config.label_names
