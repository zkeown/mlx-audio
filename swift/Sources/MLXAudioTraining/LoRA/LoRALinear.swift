// LoRALinear.swift
// Low-Rank Adaptation (LoRA) linear layer.
//
// Implements "LoRA: Low-Rank Adaptation of Large Language Models"
// (Hu et al., 2021) for parameter-efficient fine-tuning.

import Foundation
import MLX
import MLXNN

// MARK: - LoRA Linear Layer

/// Low-Rank Adapted Linear layer.
///
/// LoRA reduces the number of trainable parameters by learning low-rank
/// decomposition matrices instead of updating all weights. The forward
/// pass computes:
///
/// ```
/// output = base(x) + scale * (x @ A^T @ B^T)
/// ```
///
/// Where:
/// - `base` is the frozen original Linear layer
/// - `A` is [rank, in_features] initialized with Kaiming uniform
/// - `B` is [out_features, rank] initialized with zeros
/// - `scale` = alpha / rank
///
/// This means the output starts identical to the base layer (since B=0),
/// and gradually learns adaptations during training.
public class LoRALinear: Module, @unchecked Sendable {
    /// Original linear layer (frozen).
    private let base: Linear

    /// Low-rank matrix A: [rank, in_features].
    /// Initialized with scaled random values.
    @ParameterInfo(key: "lora_A") var loraA: MLXArray

    /// Low-rank matrix B: [out_features, rank].
    /// Initialized with zeros so initial output equals base output.
    @ParameterInfo(key: "lora_B") var loraB: MLXArray

    /// Rank of the LoRA adaptation.
    public let rank: Int

    /// Alpha parameter for scaling.
    public let alpha: Float

    /// Scaling factor (alpha / rank).
    public var scale: Float {
        alpha / Float(rank)
    }

    /// Input features.
    public var inFeatures: Int {
        loraA.dim(1)
    }

    /// Output features.
    public var outFeatures: Int {
        loraB.dim(0)
    }

    /// Dropout probability for LoRA path.
    public let dropout: Float

    /// Creates a LoRA-adapted linear layer.
    ///
    /// - Parameters:
    ///   - base: The original Linear layer to adapt
    ///   - rank: Rank of the LoRA matrices (typically 4-64)
    ///   - alpha: Scaling factor (typically equal to rank)
    ///   - dropout: Dropout probability for regularization (default: 0.0)
    public init(
        base: Linear,
        rank: Int = 8,
        alpha: Float? = nil,
        dropout: Float = 0.0
    ) {
        self.base = base
        self.rank = rank
        self.alpha = alpha ?? Float(rank)
        self.dropout = dropout

        // Get dimensions from base layer
        let weight = base.weight
        let inFeatures = weight.dim(1)
        let outFeatures = weight.dim(0)

        // Initialize A with Kaiming uniform (fan_in mode)
        let stdA = sqrt(1.0 / Float(inFeatures))
        self._loraA.wrappedValue = MLXRandom.uniform(
            low: MLXArray(-stdA),
            high: MLXArray(stdA),
            [rank, inFeatures]
        )

        // Initialize B with zeros (output unchanged initially)
        self._loraB.wrappedValue = MLXArray.zeros([outFeatures, rank])

        super.init()

        // Freeze the base layer - its parameters won't be updated
        base.freeze()
    }

    /// Creates a LoRA linear layer from dimensions (without base layer).
    ///
    /// Use this when you want to create a fresh LoRA layer rather than
    /// adapting an existing one.
    ///
    /// - Parameters:
    ///   - inFeatures: Input feature dimension
    ///   - outFeatures: Output feature dimension
    ///   - rank: LoRA rank
    ///   - alpha: Scaling factor
    ///   - bias: Whether to include bias
    public convenience init(
        inFeatures: Int,
        outFeatures: Int,
        rank: Int = 8,
        alpha: Float? = nil,
        bias: Bool = true
    ) {
        let base = Linear(inFeatures, outFeatures, bias: bias)
        self.init(base: base, rank: rank, alpha: alpha)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Base layer output (frozen)
        let output = base(x)

        // LoRA path: x @ A^T @ B^T * scale
        var loraOut = MLX.matmul(x, loraA.T)

        // Apply dropout during training (if configured)
        if dropout > 0 && training {
            let mask = MLXRandom.uniform(low: 0.0, high: 1.0, [1]) .> MLXArray(dropout)
            loraOut = loraOut * mask.asType(loraOut.dtype) / (1 - dropout)
        }

        loraOut = MLX.matmul(loraOut, loraB.T) * scale

        return output + loraOut
    }

    /// Get the number of trainable parameters (just LoRA matrices).
    public var trainableParameterCount: Int {
        rank * inFeatures + outFeatures * rank
    }

    /// Get the total number of parameters (base + LoRA).
    public var totalParameterCount: Int {
        inFeatures * outFeatures + (base.bias != nil ? outFeatures : 0) + trainableParameterCount
    }

    /// Compression ratio (total / trainable).
    public var compressionRatio: Float {
        Float(totalParameterCount) / Float(trainableParameterCount)
    }

    /// Merge LoRA weights into base layer.
    ///
    /// After merging, the layer behaves like a regular Linear with
    /// updated weights. This is useful for inference deployment.
    ///
    /// - Returns: A new Linear layer with merged weights
    public func merge() -> Linear {
        // Compute merged weight: W' = W + scale * B @ A
        let deltaW = scale * MLX.matmul(loraB, loraA)
        let mergedWeight = base.weight + deltaW

        // Create new Linear with merged weights
        if let bias = base.bias {
            return Linear(weight: mergedWeight, bias: bias)
        } else {
            return Linear(weight: mergedWeight)
        }
    }

    /// Reset LoRA weights to initial state.
    ///
    /// This resets B to zeros and reinitializes A, effectively
    /// starting fresh while keeping the base weights.
    public func resetLoRA() {
        let stdA = sqrt(1.0 / Float(inFeatures))
        _loraA.wrappedValue = MLXRandom.uniform(
            low: MLXArray(-stdA),
            high: MLXArray(stdA),
            [rank, inFeatures]
        )
        _loraB.wrappedValue = MLXArray.zeros([outFeatures, rank])
    }
}

// MARK: - Linear Extension

extension Linear {
    /// Create a LoRA-adapted version of this Linear layer.
    ///
    /// - Parameters:
    ///   - rank: LoRA rank
    ///   - alpha: Scaling factor
    ///   - dropout: Dropout probability
    /// - Returns: LoRALinear wrapping this layer
    public func withLoRA(
        rank: Int = 8,
        alpha: Float? = nil,
        dropout: Float = 0.0
    ) -> LoRALinear {
        LoRALinear(base: self, rank: rank, alpha: alpha, dropout: dropout)
    }
}
