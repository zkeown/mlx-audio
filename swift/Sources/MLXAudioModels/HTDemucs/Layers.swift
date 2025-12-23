// Layers.swift
// Custom layers for HTDemucs not included in mlx-swift MLXNN.

import Foundation
@preconcurrency import MLX
import MLXNN

// MARK: - LayerScale

/// Learnable per-channel scaling (from ConvNeXt/Demucs).
///
/// Weight key: `.scale` (not `.weight`)
public class LayerScale: Module, @unchecked Sendable {
    @ParameterInfo(key: "scale") var scale: MLXArray

    /// Creates a LayerScale layer.
    /// - Parameters:
    ///   - channels: Number of channels to scale.
    ///   - init: Initial scale value (default: 1e-4).
    public init(channels: Int, init initValue: Float = 1e-4) {
        self._scale.wrappedValue = MLXArray.ones([channels]) * initValue
    }

    /// Apply per-channel scaling.
    /// - Parameter x: Input tensor in NLC format `[B, T, C]`.
    /// - Returns: Scaled tensor.
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        x * scale
    }
}

// MARK: - ScaledEmbedding

/// Embedding with a learned scale factor.
///
/// Used for frequency embeddings in HTDemucs.
public class ScaledEmbedding: Module, @unchecked Sendable {
    let embedding: Embedding
    let scale: Float

    /// Creates a ScaledEmbedding.
    /// - Parameters:
    ///   - numEmbeddings: Size of the embedding dictionary.
    ///   - embeddingDim: Dimension of each embedding vector.
    ///   - scale: Scale factor applied during forward pass.
    public init(numEmbeddings: Int, embeddingDim: Int, scale: Float = 10.0) {
        self.embedding = Embedding(embeddingCount: numEmbeddings, dimensions: embeddingDim)
        self.scale = scale
    }

    /// Look up embeddings and scale.
    /// - Parameter indices: Integer indices to look up.
    /// - Returns: Scaled embedding vectors.
    public func callAsFunction(_ indices: MLXArray) -> MLXArray {
        embedding(indices) * scale
    }
}

// MARK: - MyGroupNorm

/// Custom GroupNorm that normalizes over axes (1, 2) for NLC tensors.
///
/// This matches the PyTorch demucs `MyGroupNorm` which normalizes
/// over the time and channel dimensions together (groups=1).
public class MyGroupNorm: Module, @unchecked Sendable {
    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "bias") var bias: MLXArray
    let eps: Float

    /// Creates a MyGroupNorm layer.
    /// - Parameters:
    ///   - numChannels: Number of channels (last dimension).
    ///   - eps: Small constant for numerical stability.
    public init(numChannels: Int, eps: Float = 1e-5) {
        self._weight.wrappedValue = MLXArray.ones([numChannels])
        self._bias.wrappedValue = MLXArray.zeros([numChannels])
        self.eps = eps
    }

    /// Apply group normalization over axes (1, 2).
    /// - Parameter x: Input tensor `[B, T, C]`.
    /// - Returns: Normalized tensor.
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Normalize over T and C dimensions (axes 1 and 2)
        let mean = x.mean(axes: [1, 2], keepDims: true)
        let variance = x.variance(axes: [1, 2], keepDims: true)
        let normalized = (x - mean) / MLX.sqrt(variance + eps)
        // Apply learnable scale and bias
        return normalized * weight + bias
    }
}

// MARK: - GLU

/// Gated Linear Unit activation.
///
/// Splits input along the last dimension and applies sigmoid gating.
public func glu(_ x: MLXArray, axis: Int = -1) -> MLXArray {
    let parts = x.split(parts: 2, axis: axis)
    return parts[0] * sigmoid(parts[1])
}
