// ParlerTTSEmbeddings.swift
// Embedding layers for Parler-TTS.

import Foundation
@preconcurrency import MLX
import MLXNN

// MARK: - Rotary Positional Embedding

/// Rotary Position Embedding (RoPE).
///
/// Applies rotary position embeddings to query and key tensors.
/// This allows the model to learn relative positions through
/// the attention mechanism.
public class RotaryPositionalEmbedding: Module, @unchecked Sendable {
    public let headDim: Int
    public let maxLength: Int
    public let base: Float

    private let invFreq: MLXArray

    public init(
        headDim: Int,
        maxLength: Int = 8192,
        base: Float = 10000.0
    ) {
        self.headDim = headDim
        self.maxLength = maxLength
        self.base = base

        // Compute frequencies
        let indices = MLXArray(stride(from: 0, to: headDim, by: 2).map { Float($0) })
        let invFreqValues = 1.0 / pow(MLXArray(base), indices / Float(headDim))
        self.invFreq = invFreqValues

        super.init()
    }

    /// Compute cos and sin for rotary embedding.
    private func computeCosSin(seqLength: Int, offset: Int = 0) -> (MLXArray, MLXArray) {
        let positions = MLXArray(
            (offset..<(offset + seqLength)).map { Float($0) }
        )
        // [seqLength, headDim // 2]
        let freqs = positions.expandedDimensions(axis: 1) * invFreq.expandedDimensions(axis: 0)
        // [seqLength, headDim]
        let emb = MLX.concatenated([freqs, freqs], axis: -1)
        return (MLX.cos(emb), MLX.sin(emb))
    }

    /// Apply rotary embeddings to query and key.
    ///
    /// - Parameters:
    ///   - query: Query tensor [B, numHeads, T, headDim]
    ///   - key: Key tensor [B, numHeads, S, headDim]
    ///   - offset: Position offset for cached positions
    /// - Returns: Tuple of rotated (query, key)
    public func callAsFunction(
        query: MLXArray,
        key: MLXArray,
        offset: Int = 0
    ) -> (MLXArray, MLXArray) {
        let seqLength = query.dim(2)
        let (cos, sin) = computeCosSin(seqLength: seqLength, offset: offset)

        // Reshape for broadcasting: [1, 1, T, headDim]
        let cosReshaped = cos.expandedDimensions(axes: [0, 1])
        let sinReshaped = sin.expandedDimensions(axes: [0, 1])

        // Apply rotation to query
        let queryRotated = rotate(query, cos: cosReshaped, sin: sinReshaped)

        // Apply rotation to key (may have different length)
        let keySeqLength = key.dim(2)
        let keyRotated: MLXArray
        if keySeqLength != seqLength {
            let (cosK, sinK) = computeCosSin(seqLength: keySeqLength, offset: 0)
            let cosKReshaped = cosK.expandedDimensions(axes: [0, 1])
            let sinKReshaped = sinK.expandedDimensions(axes: [0, 1])
            keyRotated = rotate(key, cos: cosKReshaped, sin: sinKReshaped)
        } else {
            keyRotated = rotate(key, cos: cosReshaped, sin: sinReshaped)
        }

        return (queryRotated, keyRotated)
    }

    /// Apply rotation to tensor.
    private func rotate(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let halfDim = headDim / 2

        // Split into halves
        let x1 = x[.ellipsis, 0..<halfDim]
        let x2 = x[.ellipsis, halfDim...]

        // Split cos/sin
        let cos1 = cos[.ellipsis, 0..<halfDim]
        let cos2 = cos[.ellipsis, halfDim...]
        let sin1 = sin[.ellipsis, 0..<halfDim]
        let sin2 = sin[.ellipsis, halfDim...]

        // Rotate: [x1, x2] -> [x1*cos - x2*sin, x2*cos + x1*sin]
        let rotated = MLX.concatenated(
            [x1 * cos1 - x2 * sin1, x2 * cos2 + x1 * sin2],
            axis: -1
        )
        return rotated
    }
}

// MARK: - Codebook Embeddings

/// Embeddings for multiple audio codebooks in Parler-TTS.
///
/// Each codebook has its own embedding table. The embeddings from
/// all codebooks are summed together to form the input representation.
public class ParlerTTSCodebookEmbeddings: Module, @unchecked Sendable {
    public let numCodebooks: Int
    public let hiddenSize: Int

    @ModuleInfo(key: "embeddings") var embeddingsList: [Embedding]
    @ModuleInfo(key: "input_proj") var inputProj: Linear

    public init(config: ParlerTTSConfig) {
        self.numCodebooks = config.numCodebooks
        self.hiddenSize = config.hiddenSize

        // Separate embedding table for each codebook
        // +2 for special tokens (pad, bos)
        let vocabSize = config.codebookSize + 2

        var embeddings: [Embedding] = []
        for _ in 0..<config.numCodebooks {
            embeddings.append(Embedding(embeddingCount: vocabSize, dimensions: config.hiddenSize))
        }
        self._embeddingsList.wrappedValue = embeddings

        // Input projection
        self._inputProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize)

        super.init()
    }

    /// Compute embeddings for codebook tokens.
    ///
    /// - Parameter inputIds: Token IDs [B, K, T] where K is numCodebooks
    /// - Returns: Token embeddings [B, T, D]
    public func callAsFunction(_ inputIds: MLXArray) -> MLXArray {
        let batchSize = inputIds.dim(0)
        let seqLength = inputIds.dim(2)

        // Sum embeddings from all codebooks
        var embeddings = MLXArray.zeros([batchSize, seqLength, hiddenSize])

        for k in 0..<min(inputIds.dim(1), numCodebooks) {
            let codebookIds = inputIds[0..., k, 0...]  // [B, T]
            let codebookEmb = embeddingsList[k](codebookIds)
            embeddings = embeddings + codebookEmb
        }

        return embeddings
    }

    /// Get embeddings for a specific codebook.
    public func getCodebookEmbedding(codebookIdx: Int, tokenIds: MLXArray) -> MLXArray {
        return embeddingsList[codebookIdx](tokenIds)
    }
}
