"""Training module for CLAP-based audio classifier."""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from mlx_audio.models.classifier.config import ClassifierConfig, TaskMode
from mlx_audio.models.classifier.model import CLAPClassifier
from mlx_audio.train import OptimizerConfig, TrainModule


class CLAPClassifierModule(TrainModule):
    """Training module for CLAP audio classifier.

    Implements compute_loss() and configure_optimizers() for the
    mlx-train framework. Supports both single-label classification
    and multi-label tagging with appropriate loss functions.

    Args:
        config: ClassifierConfig for the model
        learning_rate: Peak learning rate
        weight_decay: AdamW weight decay
        warmup_ratio: Fraction of total steps for warmup
        total_steps: Total training steps (for scheduler, optional)
        class_weights: Optional per-class weights for imbalanced data

    Example:
        >>> config = ClassifierConfig.for_esc50()
        >>> module = CLAPClassifierModule(config, learning_rate=3e-4)
        >>> trainer = Trainer(max_epochs=20)
        >>> trainer.fit(module, train_loader, val_loader)
    """

    def __init__(
        self,
        config: ClassifierConfig,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        total_steps: int | None = None,
        class_weights: mx.array | None = None,
    ) -> None:
        super().__init__()
        self.classifier = CLAPClassifier(config)
        self.config = config

        # Training hyperparameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self._total_steps = total_steps
        self.class_weights = class_weights

    def __call__(self, audio: mx.array) -> mx.array:
        """Forward pass."""
        return self.classifier(audio)

    def compute_loss(
        self,
        batch: tuple[mx.array, mx.array],
    ) -> tuple[mx.array, dict[str, mx.array]]:
        """Compute loss and metrics for a batch.

        Args:
            batch: Tuple of (audio, labels)
                audio: [B, 1, F, T] mel spectrogram or [B, T] waveform
                labels: [B] for classification, [B, num_classes] for tagging

        Returns:
            Tuple of (loss, metrics_dict)
        """
        audio, labels = batch
        logits = self.classifier(audio)

        if self.config.task == TaskMode.CLASSIFICATION:
            return self._compute_classification_loss(logits, labels)
        else:
            return self._compute_tagging_loss(logits, labels)

    def _compute_classification_loss(
        self,
        logits: mx.array,
        labels: mx.array,
    ) -> tuple[mx.array, dict[str, mx.array]]:
        """Compute loss for single-label classification.

        Args:
            logits: Model logits [B, num_classes]
            labels: Class indices [B]

        Returns:
            Tuple of (loss, metrics)
        """
        # Cross-entropy loss
        if self.class_weights is not None:
            # Weighted cross-entropy
            log_probs = mx.log(mx.softmax(logits, axis=-1) + 1e-8)
            one_hot = mx.zeros_like(logits)
            one_hot = one_hot.at[mx.arange(labels.shape[0]), labels].set(1.0)
            weighted_loss = -mx.sum(one_hot * log_probs * self.class_weights, axis=-1)
            loss = mx.mean(weighted_loss)
        else:
            loss = mx.mean(nn.losses.cross_entropy(logits, labels))

        # Accuracy metric
        predictions = mx.argmax(logits, axis=-1)
        accuracy = mx.mean(predictions == labels)

        return loss, {"accuracy": accuracy}

    def _compute_tagging_loss(
        self,
        logits: mx.array,
        labels: mx.array,
    ) -> tuple[mx.array, dict[str, mx.array]]:
        """Compute loss for multi-label tagging.

        Args:
            logits: Model logits [B, num_classes]
            labels: Binary labels [B, num_classes]

        Returns:
            Tuple of (loss, metrics)
        """
        labels = labels.astype(mx.float32)

        # Binary cross-entropy with optional class weighting
        probs = mx.sigmoid(logits)

        if self.class_weights is not None:
            # Weighted BCE
            bce = -(
                labels * mx.log(probs + 1e-8) * self.class_weights
                + (1 - labels) * mx.log(1 - probs + 1e-8)
            )
        else:
            bce = -(
                labels * mx.log(probs + 1e-8)
                + (1 - labels) * mx.log(1 - probs + 1e-8)
            )
        loss = mx.mean(bce)

        # Metrics
        predictions = (probs >= self.config.threshold).astype(mx.int32)
        labels_int = labels.astype(mx.int32)

        # Exact match accuracy (all labels correct)
        exact_match = mx.mean(mx.all(predictions == labels_int, axis=-1))

        # Micro precision/recall/F1
        true_positives = mx.sum(predictions * labels_int)
        predicted_positives = mx.sum(predictions)
        actual_positives = mx.sum(labels_int)

        precision = true_positives / (predicted_positives + 1e-8)
        recall = true_positives / (actual_positives + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        return loss, {
            "exact_match": exact_match,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    def configure_optimizers(self) -> OptimizerConfig:
        """Configure optimizer with warmup cosine schedule."""
        from mlx_audio.train.schedulers import WarmupCosineScheduler

        # Determine total steps
        total_steps = self._total_steps
        if total_steps is None:
            if self._trainer is not None and self._trainer.max_steps is not None:
                total_steps = self._trainer.max_steps
            else:
                total_steps = 10000  # Default fallback

        warmup_steps = int(total_steps * self.warmup_ratio)

        schedule = WarmupCosineScheduler(
            peak_lr=self.learning_rate,
            warmup_steps=warmup_steps,
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

    def validation_step(self, batch: Any) -> dict[str, mx.array]:
        """Validation step with additional metrics.

        Args:
            batch: Validation batch

        Returns:
            Validation metrics
        """
        loss, metrics = self.compute_loss(batch)
        return {"val_loss": loss, **{f"val_{k}": v for k, v in metrics.items()}}

    @property
    def model(self) -> CLAPClassifier:
        """Access the underlying classifier model."""
        return self.classifier
