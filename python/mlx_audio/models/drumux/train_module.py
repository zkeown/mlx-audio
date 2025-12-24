"""TrainModule for Drumux drum transcription.

Integrates with mlx-audio's training framework.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from mlx_audio.train.module import TrainModule, OptimizerConfig
from mlx_audio.train.schedulers import WarmupCosineScheduler

from .model import DrumTranscriber, DrumTranscriberConfig, create_model


class PositiveWeightedBCELoss:
    """Binary cross-entropy with class-specific positive weights.
    
    Handles extreme class imbalance in drum transcription where
    positives are ~0.3% of all frames.
    """

    def __init__(
        self,
        pos_weight: mx.array | float = 150.0,
        label_smoothing: float = 0.0,
    ):
        """Initialize loss.
        
        Args:
            pos_weight: Weight for positive class. Can be:
                - float: Same weight for all classes
                - mx.array of shape (num_classes,): Per-class weights
            label_smoothing: Label smoothing factor
        """
        self.pos_weight = pos_weight if isinstance(pos_weight, mx.array) else mx.array(pos_weight)
        self.label_smoothing = label_smoothing

    def __call__(self, logits: mx.array, targets: mx.array) -> mx.array:
        """Compute weighted BCE loss.
        
        Args:
            logits: Raw logits (batch, time, num_classes)
            targets: Binary targets (batch, time, num_classes)
            
        Returns:
            Scalar loss
        """
        # Apply label smoothing
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        # Numerically stable BCE with logits
        # BCE = max(logits, 0) - logits * targets + log(1 + exp(-|logits|))
        max_val = mx.maximum(logits, 0)
        loss = max_val - logits * targets + mx.log(1 + mx.exp(-mx.abs(logits)))

        # Apply positive weighting
        # Weight positive examples more heavily
        weight = mx.where(targets > 0.5, self.pos_weight, mx.array(1.0))
        loss = loss * weight

        return mx.mean(loss)


class MaskedMSELoss:
    """MSE loss computed only where mask is positive.
    
    For velocity prediction - only compute loss where there are onsets.
    """

    def __call__(
        self,
        pred: mx.array,
        target: mx.array,
        mask: mx.array,
    ) -> mx.array:
        """Compute masked MSE.

        Args:
            pred: Predictions (batch, time, num_classes)
            target: Targets (batch, time, num_classes)
            mask: Binary mask (batch, time, num_classes)

        Returns:
            Scalar loss (or 0 if no positives)
        """
        # Only compute loss where mask is positive
        # Use mx.maximum to avoid division by zero without triggering eval
        mask_sum = mx.sum(mask)
        squared_error = (pred - target) ** 2
        masked_error = squared_error * mask
        # Safe division: if mask_sum is 0, result will be 0/1 = 0
        return mx.sum(masked_error) / mx.maximum(mask_sum, mx.array(1.0))


class DrumuxTrainModule(TrainModule):
    """Training module for Drumux drum transcription model."""

    def __init__(
        self,
        config: DrumTranscriberConfig | None = None,
        preset: str = "standard",
        pos_weight: mx.array | float = 150.0,
        onset_loss_weight: float = 1.0,
        velocity_loss_weight: float = 0.5,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 500,
        total_steps: int | None = None,
    ):
        """Initialize training module.
        
        Args:
            config: Model configuration (overrides preset if provided)
            preset: Model preset if config not provided
            pos_weight: Positive class weight for BCE loss
            onset_loss_weight: Weight for onset detection loss
            velocity_loss_weight: Weight for velocity regression loss
            learning_rate: Peak learning rate
            weight_decay: AdamW weight decay
            warmup_steps: LR warmup steps
            total_steps: Total training steps (for LR schedule)
        """
        super().__init__()

        # Build model
        if config is not None:
            self.model = DrumTranscriber(config)
        else:
            self.model = create_model(preset)

        # Loss functions
        self.onset_criterion = PositiveWeightedBCELoss(pos_weight=pos_weight)
        self.velocity_criterion = MaskedMSELoss()

        # Loss weights
        self.onset_loss_weight = onset_loss_weight
        self.velocity_loss_weight = velocity_loss_weight

        # Optimizer config
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        """Forward pass - delegates to model."""
        return self.model(x)

    def training_step(
        self, batch: dict, batch_idx: int
    ) -> dict[str, mx.array]:
        """Compute training loss (Lightning-compatible API).

        Args:
            batch: Dict with keys:
                - spectrogram: (batch, time, freq, 1)
                - onset_target: (batch, time, num_classes)
                - velocity_target: (batch, time, num_classes)
            batch_idx: Current batch index

        Returns:
            Dict with 'loss' and additional metrics
        """
        spec = batch["spectrogram"]
        onset_target = batch["onset_target"]
        velocity_target = batch["velocity_target"]

        # Forward pass
        onset_logits, velocity = self.model(spec)

        # Onset loss
        onset_loss = self.onset_criterion(onset_logits, onset_target)

        # Velocity loss (only where there are onsets)
        velocity_loss = self.velocity_criterion(velocity, velocity_target, onset_target)

        # Combined loss
        loss = (
            self.onset_loss_weight * onset_loss
            + self.velocity_loss_weight * velocity_loss
        )

        return {
            "loss": loss,
            "onset_loss": onset_loss,
            "velocity_loss": velocity_loss,
        }

    def validation_step(
        self, batch: dict, batch_idx: int = 0
    ) -> dict[str, mx.array]:
        """Validation step with additional metrics."""
        result = self.training_step(batch, batch_idx)

        # Compute F1 metrics
        spec = batch["spectrogram"]
        onset_target = batch["onset_target"]

        onset_logits, _ = self.model(spec)
        onset_probs = mx.sigmoid(onset_logits)
        onset_preds = (onset_probs > 0.5).astype(mx.float32)

        # Precision, recall, F1
        tp = mx.sum(onset_preds * onset_target)
        fp = mx.sum(onset_preds * (1 - onset_target))
        fn = mx.sum((1 - onset_preds) * onset_target)

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        return {
            "val_loss": result["loss"],
            "val_onset_loss": result["onset_loss"],
            "val_velocity_loss": result["velocity_loss"],
            "val_precision": precision,
            "val_recall": recall,
            "val_f1": f1,
        }

    def predict_step(
        self, batch: dict, batch_idx: int = 0
    ) -> tuple[mx.array, mx.array]:
        """Prediction step - returns onset logits and velocities.

        Args:
            batch: Dict with 'spectrogram' key

        Returns:
            Tuple of (onset_logits, velocity)
        """
        spec = batch["spectrogram"]
        return self.model(spec)

    def configure_optimizers(self) -> OptimizerConfig:
        """Configure optimizer with warmup + cosine schedule."""
        total_steps = self.total_steps or 10000

        schedule = WarmupCosineScheduler(
            peak_lr=self.learning_rate,
            warmup_steps=self.warmup_steps,
            total_steps=total_steps,
        )

        optimizer = optim.AdamW(
            learning_rate=schedule,
            weight_decay=self.weight_decay,
        )

        return OptimizerConfig(
            optimizer=optimizer,
            lr_schedule_name="warmup_cosine",
        )

    @property
    def num_parameters(self) -> int:
        """Total model parameters."""
        return self.model.num_parameters
