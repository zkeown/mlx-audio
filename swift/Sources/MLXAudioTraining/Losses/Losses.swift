// Losses.swift
// Loss functions for audio model training.

import Foundation
import MLX
import MLXNN

// MARK: - Loss Reduction

/// How to reduce loss over batch dimension.
public enum LossReduction: Sendable {
    /// No reduction, return per-sample losses.
    case none

    /// Mean over all elements.
    case mean

    /// Sum over all elements.
    case sum
}

/// Apply reduction to a loss tensor.
public func applyReduction(_ loss: MLXArray, reduction: LossReduction) -> MLXArray {
    switch reduction {
    case .none:
        return loss
    case .mean:
        return MLX.mean(loss)
    case .sum:
        return MLX.sum(loss)
    }
}

// MARK: - Helper Functions

/// Create one-hot encoding.
///
/// - Parameters:
///   - indices: Integer indices [B]
///   - numClasses: Number of classes
/// - Returns: One-hot tensor [B, numClasses]
private func oneHot(_ indices: MLXArray, numClasses: Int) -> MLXArray {
    let batchSize = indices.dim(0)
    // Create identity matrix and gather rows
    let identity = MLXArray.eye(numClasses)
    return identity[indices]
}

// MARK: - Cross Entropy Loss

/// Cross-entropy loss for classification.
///
/// Computes the cross-entropy between logits and target class indices.
///
/// - Parameters:
///   - logits: Predicted logits [B, C] where C is number of classes
///   - targets: Target class indices [B] as integers
///   - weights: Optional per-class weights [C]
///   - labelSmoothing: Label smoothing factor (default: 0.0)
///   - reduction: How to reduce the loss
/// - Returns: Cross-entropy loss
public func crossEntropyLoss(
    logits: MLXArray,
    targets: MLXArray,
    weights: MLXArray? = nil,
    labelSmoothing: Float = 0.0,
    reduction: LossReduction = .mean
) -> MLXArray {
    // Compute log softmax
    let logProbs = logSoftmax(logits, axis: -1)

    // Get number of classes
    let numClasses = logits.dim(-1)

    // Convert targets to one-hot
    let oneHotTargets = oneHot(targets, numClasses: numClasses)

    // Apply label smoothing if specified
    var smoothedTargets = oneHotTargets
    if labelSmoothing > 0 {
        let smooth = labelSmoothing / Float(numClasses)
        smoothedTargets = oneHotTargets * (1 - labelSmoothing) + smooth
    }

    // Compute loss: -sum(targets * log_probs)
    var loss = -MLX.sum(smoothedTargets * logProbs, axis: -1)

    // Apply class weights if provided
    if let weights = weights {
        let sampleWeights = weights[targets]
        loss = loss * sampleWeights
    }

    return applyReduction(loss, reduction: reduction)
}

/// Binary cross-entropy loss.
///
/// For multi-label classification where each class is independent.
///
/// - Parameters:
///   - logits: Predicted logits [B, C]
///   - targets: Target labels [B, C] as 0 or 1
///   - posWeight: Optional positive class weight
///   - reduction: How to reduce the loss
/// - Returns: Binary cross-entropy loss
public func binaryCrossEntropyLoss(
    logits: MLXArray,
    targets: MLXArray,
    posWeight: Float? = nil,
    reduction: LossReduction = .mean
) -> MLXArray {
    // Compute sigmoid
    let probs = sigmoid(logits)

    // BCE: -[y * log(p) + (1-y) * log(1-p)]
    let eps: Float = 1e-7
    var loss = -(
        targets * MLX.log(probs + eps) +
        (1 - targets) * MLX.log(1 - probs + eps)
    )

    // Apply positive weight if provided
    if let pw = posWeight {
        let weight = targets * pw + (1 - targets)
        loss = loss * weight
    }

    // Reduce over classes first
    loss = MLX.mean(loss, axis: -1)

    return applyReduction(loss, reduction: reduction)
}

// MARK: - Contrastive Loss

/// Contrastive loss for CLAP-style audio-text learning.
///
/// Implements InfoNCE loss for matching audio and text embeddings.
/// Assumes audio[i] should match text[i] (diagonal is positive).
///
/// - Parameters:
///   - audioEmbeddings: Audio embeddings [B, D]
///   - textEmbeddings: Text embeddings [B, D]
///   - temperature: Temperature for softmax scaling
///   - reduction: How to reduce the loss
/// - Returns: Contrastive loss
public func contrastiveLoss(
    audioEmbeddings: MLXArray,
    textEmbeddings: MLXArray,
    temperature: Float = 0.07,
    reduction: LossReduction = .mean
) -> MLXArray {
    // Compute similarity matrix
    let logitsPerAudio = MLX.matmul(audioEmbeddings, textEmbeddings.T) / temperature
    let logitsPerText = logitsPerAudio.T

    // Labels are diagonal (audio[i] matches text[i])
    let batchSize = audioEmbeddings.dim(0)
    let labels = MLXArray(0..<Int32(batchSize))

    // Cross entropy in both directions
    let lossAudio = crossEntropyLoss(
        logits: logitsPerAudio,
        targets: labels,
        reduction: reduction
    )
    let lossText = crossEntropyLoss(
        logits: logitsPerText,
        targets: labels,
        reduction: reduction
    )

    return (lossAudio + lossText) / 2
}

/// Triplet loss with hard negative mining.
///
/// - Parameters:
///   - anchor: Anchor embeddings [B, D]
///   - positive: Positive embeddings [B, D]
///   - negative: Negative embeddings [B, D]
///   - margin: Margin for triplet loss
///   - reduction: How to reduce the loss
/// - Returns: Triplet loss
public func tripletLoss(
    anchor: MLXArray,
    positive: MLXArray,
    negative: MLXArray,
    margin: Float = 0.2,
    reduction: LossReduction = .mean
) -> MLXArray {
    // Compute distances
    let posDistance = MLX.sqrt(MLX.sum((anchor - positive) * (anchor - positive), axis: -1))
    let negDistance = MLX.sqrt(MLX.sum((anchor - negative) * (anchor - negative), axis: -1))

    // Triplet loss: max(0, pos_dist - neg_dist + margin)
    let loss = MLX.maximum(posDistance - negDistance + margin, MLXArray(Float(0)))

    return applyReduction(loss, reduction: reduction)
}

// MARK: - SDR Loss

/// Signal-to-Distortion Ratio loss for source separation.
///
/// Computes negative SDR (minimizing loss maximizes SDR).
/// SDR = 10 * log10(||s||^2 / ||s - s_hat||^2)
///
/// - Parameters:
///   - predictions: Predicted sources [B, S, C, T] or [B, S, T]
///   - targets: Target sources (same shape)
///   - eps: Small constant for numerical stability
///   - reduction: How to reduce the loss
/// - Returns: Negative SDR loss
public func sdrLoss(
    predictions: MLXArray,
    targets: MLXArray,
    eps: Float = 1e-8,
    reduction: LossReduction = .mean
) -> MLXArray {
    // Flatten to [B, S, -1] for easier computation
    let ndim = predictions.ndim
    let predFlat: MLXArray
    let targetFlat: MLXArray

    if ndim == 4 {
        // [B, S, C, T] -> [B, S, C*T]
        let B = predictions.dim(0)
        let S = predictions.dim(1)
        predFlat = predictions.reshaped([B, S, -1])
        targetFlat = targets.reshaped([B, S, -1])
    } else if ndim == 3 {
        // Already [B, S, T]
        predFlat = predictions
        targetFlat = targets
    } else {
        // [B, T] -> [B, 1, T]
        predFlat = predictions.expandedDimensions(axis: 1)
        targetFlat = targets.expandedDimensions(axis: 1)
    }

    // Compute signal power: ||s||^2
    let signalPower = MLX.sum(targetFlat * targetFlat, axis: -1)

    // Compute noise power: ||s - s_hat||^2
    let noise = targetFlat - predFlat
    let noisePower = MLX.sum(noise * noise, axis: -1) + eps

    // SDR = 10 * log10(signal / noise)
    let sdr = 10 * MLX.log10(signalPower / noisePower)

    // Return negative SDR (minimize loss = maximize SDR)
    let loss = -sdr

    // Flatten source dimension for reduction
    let lossFlat = MLX.mean(loss, axis: -1)

    return applyReduction(lossFlat, reduction: reduction)
}

/// Scale-Invariant SDR loss.
///
/// More robust to scaling differences between prediction and target.
///
/// - Parameters:
///   - predictions: Predicted sources
///   - targets: Target sources
///   - eps: Numerical stability constant
///   - reduction: How to reduce the loss
/// - Returns: Negative SI-SDR loss
public func siSdrLoss(
    predictions: MLXArray,
    targets: MLXArray,
    eps: Float = 1e-8,
    reduction: LossReduction = .mean
) -> MLXArray {
    // Zero-mean
    let predZeroMean = predictions - MLX.mean(predictions, axis: -1, keepDims: true)
    let targetZeroMean = targets - MLX.mean(targets, axis: -1, keepDims: true)

    // Compute scale: <s, s_hat> / ||s||^2
    let dotProduct = MLX.sum(targetZeroMean * predZeroMean, axis: -1, keepDims: true)
    let targetPower = MLX.sum(targetZeroMean * targetZeroMean, axis: -1, keepDims: true) + eps
    let scale = dotProduct / targetPower

    // Scaled target
    let scaledTarget = scale * targetZeroMean

    // SI-SDR = 10 * log10(||s_target||^2 / ||e_noise||^2)
    let signalPower = MLX.sum(scaledTarget * scaledTarget, axis: -1)
    let noise = predZeroMean - scaledTarget
    let noisePower = MLX.sum(noise * noise, axis: -1) + eps

    let siSdr = 10 * MLX.log10(signalPower / noisePower)

    return applyReduction(-siSdr, reduction: reduction)
}

// MARK: - MSE Loss

/// Mean Squared Error loss.
///
/// - Parameters:
///   - predictions: Predicted values
///   - targets: Target values
///   - reduction: How to reduce the loss
/// - Returns: MSE loss
public func mseLoss(
    predictions: MLXArray,
    targets: MLXArray,
    reduction: LossReduction = .mean
) -> MLXArray {
    let diff = predictions - targets
    let loss = diff * diff
    return applyReduction(loss, reduction: reduction)
}

/// L1 (Mean Absolute Error) loss.
///
/// - Parameters:
///   - predictions: Predicted values
///   - targets: Target values
///   - reduction: How to reduce the loss
/// - Returns: L1 loss
public func l1Loss(
    predictions: MLXArray,
    targets: MLXArray,
    reduction: LossReduction = .mean
) -> MLXArray {
    let loss = MLX.abs(predictions - targets)
    return applyReduction(loss, reduction: reduction)
}

/// Huber loss (smooth L1).
///
/// Combines MSE and L1 for robustness to outliers.
///
/// - Parameters:
///   - predictions: Predicted values
///   - targets: Target values
///   - delta: Threshold for switching between L1 and L2
///   - reduction: How to reduce the loss
/// - Returns: Huber loss
public func huberLoss(
    predictions: MLXArray,
    targets: MLXArray,
    delta: Float = 1.0,
    reduction: LossReduction = .mean
) -> MLXArray {
    let diff = predictions - targets
    let absDiff = MLX.abs(diff)

    // L2 for small errors, L1 for large errors
    let quadratic = 0.5 * diff * diff
    let linear = delta * (absDiff - 0.5 * delta)

    let loss = MLX.where(absDiff .<= delta, quadratic, linear)
    return applyReduction(loss, reduction: reduction)
}

// MARK: - CTC Loss (Placeholder)

/// CTC loss for sequence-to-sequence (ASR).
///
/// Note: This requires a custom MLX implementation for efficient
/// forward-backward algorithm. This is a placeholder API.
public func ctcLoss(
    logits: MLXArray,      // [T, B, V]
    targets: MLXArray,     // [B, S]
    inputLengths: MLXArray,
    targetLengths: MLXArray,
    blankIndex: Int = 0,
    reduction: LossReduction = .mean
) -> MLXArray {
    // CTC requires dynamic programming (forward-backward algorithm)
    // This would need a custom MLX kernel for efficiency
    fatalError("CTC loss is not yet implemented. Use external CTC library.")
}

// MARK: - Focal Loss

/// Focal loss for handling class imbalance.
///
/// Reduces the loss for well-classified examples, focusing on hard examples.
/// FL(p) = -alpha * (1-p)^gamma * log(p)
///
/// - Parameters:
///   - logits: Predicted logits [B, C]
///   - targets: Target class indices [B]
///   - alpha: Weighting factor (default: 0.25)
///   - gamma: Focusing parameter (default: 2.0)
///   - reduction: How to reduce the loss
/// - Returns: Focal loss
public func focalLoss(
    logits: MLXArray,
    targets: MLXArray,
    alpha: Float = 0.25,
    gamma: Float = 2.0,
    reduction: LossReduction = .mean
) -> MLXArray {
    let probs = softmax(logits, axis: -1)
    let numClasses = logits.dim(-1)
    let oneHotTargets = oneHot(targets, numClasses: numClasses)

    // Get probability of true class
    let pt = MLX.sum(probs * oneHotTargets, axis: -1)

    // Focal weight: (1 - p_t)^gamma
    let focalWeight = MLX.pow(1 - pt, gamma)

    // Focal loss
    let eps: Float = 1e-7
    let loss = -alpha * focalWeight * MLX.log(pt + eps)

    return applyReduction(loss, reduction: reduction)
}
