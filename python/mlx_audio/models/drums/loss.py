"""Loss functions for drum transcription training.

Multi-task loss combining:
1. Onset detection (binary cross-entropy with positive class weighting for class imbalance)
2. Velocity prediction (MSE, masked to only count where there are actual onsets)

Ported from PyTorch implementation.
"""

import mlx.core as mx


def positive_weighted_bce_loss(
    logits: mx.array,
    targets: mx.array,
    pos_weight: float | mx.array = 150.0,
) -> mx.array:
    """Binary cross-entropy loss with positive class weighting.

    For drum transcription, only ~0.3% of frames have onsets. Standard BCE
    allows the model to predict all zeros and still get low loss. This loss
    heavily penalizes missing positive examples.

    Args:
        logits: Predicted logits (B, T, C) before sigmoid
        targets: Binary targets (B, T, C)
        pos_weight: Weight for positive class. Can be:
            - float: Same weight for all classes
            - mx.array of shape (C,): Per-class weights

    Returns:
        Scalar loss value
    """
    # Numerically stable sigmoid + BCE
    # BCE = -y * log(sigmoid(x)) - (1-y) * log(1 - sigmoid(x))
    # With pos_weight: -pos_weight * y * log(sigmoid(x)) - (1-y) * log(1 - sigmoid(x))

    # Use log-sum-exp trick for numerical stability
    # log(sigmoid(x)) = x - log(1 + exp(x)) = -softplus(-x)
    # log(1 - sigmoid(x)) = -log(1 + exp(x)) = -softplus(x)

    # softplus(x) = log(1 + exp(x))
    # For numerical stability: softplus(x) = max(x, 0) + log(1 + exp(-abs(x)))

    max_val = mx.maximum(-logits, mx.zeros_like(logits))
    loss = (
        (1 - targets) * logits
        + max_val
        + mx.log(mx.exp(-max_val) + mx.exp(-logits - max_val))
    )

    # Apply positive weight
    # pos_weight should already be an mx.array (converted at init time)
    # This ensures compile can capture it properly
    if not isinstance(pos_weight, mx.array):
        pos_weight = mx.array(pos_weight)
    weight = mx.where(targets > 0.5, pos_weight, mx.ones_like(logits))

    loss = loss * weight

    return mx.mean(loss)


def masked_mse_loss(
    pred: mx.array,
    target: mx.array,
    mask: mx.array,
) -> mx.array:
    """MSE loss masked to only count where mask is 1.

    Used for velocity prediction - we only care about velocity
    where there's actually an onset.

    Args:
        pred: Predicted values (B, T, C)
        target: Target values (B, T, C)
        mask: Binary mask (B, T, C), 1 where to compute loss

    Returns:
        Scalar loss value
    """
    # Only compute loss where mask is 1
    masked_pred = pred * mask
    masked_target = target * mask

    mse = (masked_pred - masked_target) ** 2

    # Average over non-zero mask elements
    # Use float instead of mx.array to avoid compile issues
    num_elements = mx.maximum(mx.sum(mask), 1.0)
    return mx.sum(mse) / num_elements


class DrumTranscriptionLoss:
    """Combined loss for drum transcription.

    Combines onset detection loss and velocity regression loss.
    """

    def __init__(
        self,
        onset_weight: float = 1.0,
        velocity_weight: float = 0.5,
        pos_weight: float | mx.array = 150.0,
    ):
        """Initialize combined loss.

        Args:
            onset_weight: Weight for onset loss
            velocity_weight: Weight for velocity loss
            pos_weight: Positive class weight for BCE. Can be:
                - float: Same weight for all classes
                - mx.array of shape (num_classes,): Per-class weights
        """
        self.onset_weight = onset_weight
        self.velocity_weight = velocity_weight
        # Convert to mx.array at init time so it can be captured by compile
        if isinstance(pos_weight, mx.array):
            self.pos_weight = pos_weight
        else:
            self.pos_weight = mx.array(pos_weight)

    def __call__(
        self,
        onset_logits: mx.array,
        velocity_pred: mx.array,
        onset_target: mx.array,
        velocity_target: mx.array,
    ) -> dict[str, mx.array]:
        """Compute combined loss.

        Args:
            onset_logits: Predicted onset logits (B, T, C)
            velocity_pred: Predicted velocities (B, T, C) in [0, 1]
            onset_target: Target onsets (B, T, C) binary
            velocity_target: Target velocities (B, T, C) in [0, 1]

        Returns:
            Dict with 'loss', 'onset_loss', 'velocity_loss'
        """
        # Onset loss (pos-weighted BCE)
        onset_loss = positive_weighted_bce_loss(
            onset_logits, onset_target, self.pos_weight
        )

        # Velocity loss (masked MSE)
        velocity_loss = masked_mse_loss(
            velocity_pred,
            velocity_target,
            onset_target,  # Only compute where there are actual onsets
        )

        # Combined loss
        total_loss = self.onset_weight * onset_loss + self.velocity_weight * velocity_loss

        return {
            "loss": total_loss,
            "onset_loss": onset_loss,
            "velocity_loss": velocity_loss,
        }


def compute_class_weights_from_loader(
    dataloader,
    num_classes: int = 14,
    min_weight: float = 50.0,
    max_weight: float = 2000.0,
    max_batches: int | None = 100,
    verbose: bool = True,
) -> mx.array:
    """Compute per-class pos_weights from a DataLoader.

    Rare classes get higher weights to counteract the model's tendency
    to predict "no onset" for them.

    Args:
        dataloader: DataLoader to compute weights from
        num_classes: Number of classes
        min_weight: Minimum pos_weight (for most common class)
        max_weight: Maximum pos_weight (cap to prevent instability)
        max_batches: Maximum batches to process (None = all)
        verbose: Print class frequencies and weights

    Returns:
        mx.array of shape (num_classes,) with per-class pos_weights
    """
    import numpy as np

    onset_counts = np.zeros(num_classes)
    total_frames = 0

    for i, batch in enumerate(dataloader):
        if max_batches and i >= max_batches:
            break

        onset_target = batch["onset_target"]

        # Convert to numpy if MLX array
        if isinstance(onset_target, mx.array):
            onset_target = np.array(onset_target)

        # Sum across batch and time dimensions
        onset_counts += onset_target.sum(axis=(0, 1))
        total_frames += onset_target.shape[0] * onset_target.shape[1]

    # Compute per-class positive rates
    pos_rates = onset_counts / total_frames

    # Inverse frequency weighting
    weights = 1.0 / (pos_rates + 1e-8)

    # Scale so the most common class gets min_weight
    weights = weights / weights.min() * min_weight

    # Cap at max_weight to prevent instability
    weights = np.clip(weights, min_weight, max_weight)

    if verbose:
        print("\nPer-class weight statistics:")
        print(f"{'Class':<10} {'Count':>10} {'Rate':>10} {'Weight':>10}")
        print("-" * 45)
        for i in range(num_classes):
            count = int(onset_counts[i])
            rate = pos_rates[i]
            weight = weights[i]
            capped = " (capped)" if weight >= max_weight else ""
            print(f"{i:<10} {count:>10,} {rate:>10.4%} {weight:>10.1f}{capped}")
        print()

    return mx.array(weights.astype(np.float32))
