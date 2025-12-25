"""Configuration for audio classification models."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TaskMode(Enum):
    """Classification task mode.

    Attributes:
        CLASSIFICATION: Single-label classification with softmax
        TAGGING: Multi-label classification with sigmoid
    """

    CLASSIFICATION = "classification"
    TAGGING = "tagging"


class FreezeMode(Enum):
    """Encoder freezing mode for fine-tuning.

    Attributes:
        FROZEN: Freeze entire CLAP encoder (fastest training)
        PROJECTION_ONLY: Freeze audio encoder, fine-tune projection head
        FULL: Fine-tune entire model (slowest, potentially best performance)
    """

    FROZEN = "frozen"
    PROJECTION_ONLY = "projection"
    FULL = "full"


@dataclass
class MLPHeadConfig:
    """Configuration for the MLP classifier head.

    Attributes:
        input_dim: Input dimension from CLAP embeddings (default: 512)
        num_classes: Number of output classes/tags
        hidden_dims: List of hidden layer dimensions (e.g., [256] for one layer)
        dropout: Dropout probability applied after each hidden layer
        activation: Activation function ("relu", "gelu", "silu")
        use_batch_norm: Whether to use batch normalization
    """

    input_dim: int = 512
    num_classes: int = 10
    hidden_dims: list[int] = field(default_factory=lambda: [256])
    dropout: float = 0.2
    activation: str = "relu"
    use_batch_norm: bool = False

    def __post_init__(self) -> None:
        if self.num_classes < 1:
            raise ValueError(f"num_classes must be >= 1, got {self.num_classes}")
        if not 0 <= self.dropout < 1:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.activation not in ("relu", "gelu", "silu"):
            raise ValueError(
                f"activation must be one of 'relu', 'gelu', 'silu', got {self.activation}"
            )


@dataclass
class ClassifierConfig:
    """Full configuration for CLAP-based audio classifier.

    Attributes:
        clap_model: CLAP model identifier or path (e.g., "clap-htsat-fused")
        head: MLP head configuration
        task: Classification mode (single-label or multi-label)
        freeze_mode: How to freeze the CLAP encoder during training
        label_names: Optional list of class/tag names for inference
        threshold: Probability threshold for tagging mode (default: 0.5)
    """

    clap_model: str = "clap-htsat-fused"
    head: MLPHeadConfig = field(default_factory=MLPHeadConfig)
    task: TaskMode = TaskMode.CLASSIFICATION
    freeze_mode: FreezeMode = FreezeMode.FROZEN
    label_names: list[str] | None = None
    threshold: float = 0.5

    def __post_init__(self) -> None:
        if not 0 < self.threshold < 1:
            raise ValueError(f"threshold must be in (0, 1), got {self.threshold}")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ClassifierConfig:
        """Create config from dictionary.

        Args:
            d: Configuration dictionary

        Returns:
            ClassifierConfig instance
        """
        d = d.copy()

        # Parse head config
        head_dict = d.pop("head", {})
        head = MLPHeadConfig(**head_dict) if isinstance(head_dict, dict) else head_dict

        # Parse task mode
        task = d.pop("task", "classification")
        if isinstance(task, str):
            task = TaskMode(task)

        # Parse freeze mode
        freeze_mode = d.pop("freeze_mode", "frozen")
        if isinstance(freeze_mode, str):
            freeze_mode = FreezeMode(freeze_mode)

        return cls(head=head, task=task, freeze_mode=freeze_mode, **d)

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary.

        Returns:
            Configuration as dictionary
        """
        return {
            "clap_model": self.clap_model,
            "head": {
                "input_dim": self.head.input_dim,
                "num_classes": self.head.num_classes,
                "hidden_dims": self.head.hidden_dims,
                "dropout": self.head.dropout,
                "activation": self.head.activation,
                "use_batch_norm": self.head.use_batch_norm,
            },
            "task": self.task.value,
            "freeze_mode": self.freeze_mode.value,
            "label_names": self.label_names,
            "threshold": self.threshold,
        }

    @classmethod
    def for_esc50(cls, freeze_mode: FreezeMode = FreezeMode.FROZEN) -> ClassifierConfig:
        """Create config for ESC-50 dataset (50 classes, single-label).

        Args:
            freeze_mode: Encoder freezing strategy

        Returns:
            ClassifierConfig for ESC-50
        """
        return cls(
            head=MLPHeadConfig(num_classes=50, hidden_dims=[256]),
            task=TaskMode.CLASSIFICATION,
            freeze_mode=freeze_mode,
        )

    @classmethod
    def for_audioset(cls, freeze_mode: FreezeMode = FreezeMode.FROZEN) -> ClassifierConfig:
        """Create config for AudioSet (527 classes, multi-label).

        Args:
            freeze_mode: Encoder freezing strategy

        Returns:
            ClassifierConfig for AudioSet
        """
        return cls(
            head=MLPHeadConfig(num_classes=527, hidden_dims=[512, 256]),
            task=TaskMode.TAGGING,
            freeze_mode=freeze_mode,
            threshold=0.3,
        )

    @classmethod
    def for_fsd50k(cls, freeze_mode: FreezeMode = FreezeMode.FROZEN) -> ClassifierConfig:
        """Create config for FSD50K (200 classes, multi-label).

        Args:
            freeze_mode: Encoder freezing strategy

        Returns:
            ClassifierConfig for FSD50K
        """
        return cls(
            head=MLPHeadConfig(num_classes=200, hidden_dims=[256]),
            task=TaskMode.TAGGING,
            freeze_mode=freeze_mode,
            threshold=0.3,
        )
