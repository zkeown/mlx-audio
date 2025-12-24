"""TrainModule wrapper for drum transcription model.

Integrates DrumTranscriber with mlx-audio's training framework.
"""

import mlx.core as mx
import mlx.optimizers as optim

from mlx_audio.models.drums.config import DrumTranscriberConfig
from mlx_audio.models.drums.loss import DrumTranscriptionLoss
from mlx_audio.models.drums.model import DrumTranscriber
from mlx_audio.train.module import OptimizerConfig, TrainModule
from mlx_audio.train.schedulers import WarmupCosineScheduler


class DrumTranscriberModule(TrainModule):
    """TrainModule for drum transcription.

    Wraps DrumTranscriber model with training logic for use with
    mlx-audio's Trainer.
    """

    def __init__(
        self,
        config: DrumTranscriberConfig | None = None,
        onset_weight: float = 1.0,
        velocity_weight: float = 0.5,
        pos_weight: float | mx.array = 150.0,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 500,
        total_steps: int | None = None,
    ):
        """Initialize training module.

        Args:
            config: Model configuration
            onset_weight: Weight for onset loss
            velocity_weight: Weight for velocity loss
            pos_weight: Positive class weight for BCE loss
            learning_rate: Peak learning rate
            weight_decay: AdamW weight decay
            warmup_steps: Number of warmup steps for LR schedule
            total_steps: Total training steps (for cosine schedule)
        """
        super().__init__()

        self.model = DrumTranscriber(config)
        self.criterion = DrumTranscriptionLoss(
            onset_weight=onset_weight,
            velocity_weight=velocity_weight,
            pos_weight=pos_weight,
        )

        # Store optimizer config
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay
        self._warmup_steps = warmup_steps
        self._total_steps = total_steps

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        """Forward pass through model."""
        return self.model(x)

    def compute_loss(self, batch: dict) -> tuple[mx.array, dict[str, mx.array]]:
        """Compute loss for a batch.

        Args:
            batch: Dict with 'spectrogram', 'onset_target', 'velocity_target'

        Returns:
            Tuple of (loss, metrics_dict)
        """
        spec = batch["spectrogram"]
        onset_target = batch["onset_target"]
        velocity_target = batch["velocity_target"]

        # Forward pass
        onset_logits, velocity = self.model(spec)

        # Compute losses
        losses = self.criterion(onset_logits, velocity, onset_target, velocity_target)

        return losses["loss"], {
            "onset_loss": losses["onset_loss"],
            "velocity_loss": losses["velocity_loss"],
        }

    def configure_optimizers(self) -> OptimizerConfig:
        """Configure optimizer with warmup + cosine LR schedule."""
        total_steps = self._total_steps
        if total_steps is None:
            # Try to get from trainer
            try:
                total_steps = self.trainer.max_steps or 10000
            except RuntimeError:
                total_steps = 10000

        schedule = WarmupCosineScheduler(
            peak_lr=self._learning_rate,
            warmup_steps=self._warmup_steps,
            total_steps=total_steps,
            min_lr=1e-6,
        )

        return OptimizerConfig(
            optimizer=optim.AdamW(
                learning_rate=schedule,
                weight_decay=self._weight_decay,
            ),
            lr_schedule_name="warmup_cosine",
        )

    def validation_step(self, batch: dict) -> dict[str, mx.array]:
        """Perform validation step.

        Args:
            batch: Dict with 'spectrogram', 'onset_target', 'velocity_target'

        Returns:
            Dict of validation metrics
        """
        spec = batch["spectrogram"]
        onset_target = batch["onset_target"]
        velocity_target = batch["velocity_target"]

        # Forward pass
        onset_logits, velocity = self.model(spec)

        # Compute losses
        losses = self.criterion(onset_logits, velocity, onset_target, velocity_target)

        # Compute additional metrics
        onset_probs = mx.sigmoid(onset_logits)
        onset_preds = (onset_probs > 0.5).astype(mx.float32)

        # Precision, Recall, F1
        tp = mx.sum(onset_preds * onset_target)
        fp = mx.sum(onset_preds * (1 - onset_target))
        fn = mx.sum((1 - onset_preds) * onset_target)

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        return {
            "val_loss": losses["loss"],
            "val_onset_loss": losses["onset_loss"],
            "val_velocity_loss": losses["velocity_loss"],
            "val_precision": precision,
            "val_recall": recall,
            "val_f1": f1,
        }
