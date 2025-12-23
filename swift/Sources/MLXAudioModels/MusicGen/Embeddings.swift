// Embeddings.swift
// Embedding layers for MusicGen audio decoder.

import Foundation
import MLX
import MLXFast
import MLXNN

/// Sinusoidal positional embedding for transformer models.
/// Pre-computes sine/cosine embeddings up to a maximum length.
public class SinusoidalPositionalEmbedding: Module, UnaryLayer {

    let embeddingDim: Int
    let maxLength: Int
    let weights: MLXArray

    public init(embeddingDim: Int, maxLength: Int = 8192) {
        self.embeddingDim = embeddingDim
        self.maxLength = maxLength

        // Pre-compute positional embeddings
        let positions = MLXArray(0 ..< maxLength).expandedDimensions(axis: 1)
        let dimIndices = MLXArray(0 ..< embeddingDim)

        // Compute frequencies: 10000^(-2i/d)
        let freqs = exp(-log(Float(10000)) * (dimIndices / Float(embeddingDim)))
        let angles = positions.asType(.float32) * freqs

        // Interleave sin and cos: [sin(0), cos(0), sin(1), cos(1), ...]
        var embedding = MLXArray.zeros([maxLength, embeddingDim])
        let sinValues = sin(angles[0..., .stride(to: embeddingDim, by: 2)])
        let cosValues = cos(angles[0..., .stride(to: embeddingDim, by: 2)])

        // Build embedding by interleaving
        embedding[0..., .stride(to: embeddingDim, by: 2)] = sinValues
        embedding[0..., .stride(from: 1, to: embeddingDim, by: 2)] = cosValues

        self.weights = embedding

        super.init()
    }

    public func callAsFunction(_ positions: MLXArray) -> MLXArray {
        // positions: [B, T] or [T]
        // Returns: [B, T, D] or [T, D]
        return weights[positions]
    }
}

/// Codebook embeddings for MusicGen.
/// Maintains K separate embedding tables (one per codebook) and sums them.
public class CodebookEmbeddings: Module {

    @ModuleInfo(key: "embeddings") var embeddingTables: [Embedding]
    @ModuleInfo(key: "position_embedding") var positionEmbedding: SinusoidalPositionalEmbedding

    let numCodebooks: Int
    let vocabSize: Int
    let embeddingDim: Int

    public init(config: MusicGenConfig) {
        self.numCodebooks = config.numCodebooks
        self.vocabSize = config.vocabSize
        self.embeddingDim = config.hiddenSize

        // Create K separate embedding tables
        var tables: [Embedding] = []
        for _ in 0 ..< numCodebooks {
            tables.append(Embedding(embeddingCount: vocabSize, dimensions: embeddingDim))
        }
        self._embeddingTables.wrappedValue = tables

        // Positional embedding
        self._positionEmbedding.wrappedValue = SinusoidalPositionalEmbedding(
            embeddingDim: embeddingDim
        )

        super.init()
    }

    /// Forward pass for codebook embeddings.
    /// - Parameter inputIds: Token IDs of shape [B, K, T]
    /// - Returns: Embedded tokens of shape [B, T, D]
    public func callAsFunction(_ inputIds: MLXArray) -> MLXArray {
        // inputIds: [B, K, T]
        let batchSize = inputIds.dim(0)
        let seqLength = inputIds.dim(2)

        // Sum embeddings from all K codebooks
        var hidden = MLXArray.zeros([batchSize, seqLength, embeddingDim])

        for k in 0 ..< numCodebooks {
            // Get tokens for codebook k: [B, T]
            let tokens = inputIds[0..., k, 0...]
            // Embed and add
            hidden = hidden + embeddingTables[k](tokens)
        }

        // Add positional embeddings
        let positions = MLXArray(0 ..< seqLength)
        let posEmbed = positionEmbedding(positions)  // [T, D]

        return hidden + posEmbed
    }
}
