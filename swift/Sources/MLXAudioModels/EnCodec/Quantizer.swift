// Quantizer.swift
// Residual Vector Quantizer for EnCodec.

import Foundation
import MLX
import MLXNN

// MARK: - VectorQuantizer

/// Single codebook vector quantizer.
///
/// Maps continuous embeddings to nearest codebook entries using L2 distance.
class VectorQuantizer: Module, @unchecked Sendable {
    let codebookSize: Int
    let codebookDim: Int
    let embedding: Embedding

    init(codebookSize: Int = 1024, codebookDim: Int = 128) {
        self.codebookSize = codebookSize
        self.codebookDim = codebookDim
        self.embedding = Embedding(embeddingCount: codebookSize, dimensions: codebookDim)
        super.init()
    }

    /// Encode continuous vectors to discrete codes.
    ///
    /// Uses L2 distance to find nearest codebook entries.
    ///
    /// - Parameter x: Input embeddings [B, T, D]
    /// - Returns: Codebook indices [B, T]
    func encode(_ x: MLXArray) -> MLXArray {
        let shape = x.shape
        let batchSize = shape[0]
        let seqLen = shape[1]

        // Flatten batch and time dimensions
        let flatX = x.reshaped([batchSize * seqLen, codebookDim])  // [B*T, D]

        // Get codebook weights
        let codebook = embedding.weight  // [K, D]

        // Compute distances to all codebook entries
        // ||x - e||^2 = ||x||^2 + ||e||^2 - 2*x.e
        let xNormSq = sum(flatX * flatX, axis: -1, keepDims: true)  // [B*T, 1]
        let eNormSq = sum(codebook * codebook, axis: -1, keepDims: true).T  // [1, K]
        let dotProduct = matmul(flatX, codebook.T)  // [B*T, K]
        let distances = xNormSq + eNormSq - 2 * dotProduct  // [B*T, K]

        // Find nearest codebook entry
        let indices = argMin(distances, axis: -1)  // [B*T]

        // Reshape back to [B, T]
        return indices.reshaped([batchSize, seqLen])
    }

    /// Decode discrete codes to continuous vectors.
    ///
    /// - Parameter codes: Codebook indices [B, T] or [B, T, 1]
    /// - Returns: Quantized embeddings [B, T, D]
    func decode(_ codes: MLXArray) -> MLXArray {
        var c = codes
        if c.ndim == 3 {
            c = c.squeezed(axis: -1)
        }
        return embedding(c)
    }

    /// Quantize input embeddings.
    ///
    /// - Parameter x: Input embeddings [B, T, D]
    /// - Returns: Tuple of (quantized embeddings [B, T, D], codebook indices [B, T])
    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        let codes = encode(x)
        let quantized = decode(codes)
        return (quantized, codes)
    }
}

// MARK: - ResidualVectorQuantizer

/// Residual Vector Quantizer (RVQ) for multi-codebook quantization.
///
/// RVQ applies multiple codebooks sequentially, where each codebook
/// quantizes the residual from the previous codebook. This provides
/// a hierarchical representation where:
/// - First codebook captures coarse structure
/// - Subsequent codebooks refine the approximation
///
/// The final reconstruction is the sum of all quantized outputs.
class ResidualVectorQuantizer: Module, @unchecked Sendable {
    let numCodebooks: Int
    let codebookSize: Int
    let codebookDim: Int
    let layers: [VectorQuantizer]

    init(
        numCodebooks: Int = 4,
        codebookSize: Int = 1024,
        codebookDim: Int = 128
    ) {
        self.numCodebooks = numCodebooks
        self.codebookSize = codebookSize
        self.codebookDim = codebookDim

        // Create quantizer for each codebook level
        var layers: [VectorQuantizer] = []
        for _ in 0..<numCodebooks {
            layers.append(VectorQuantizer(codebookSize: codebookSize, codebookDim: codebookDim))
        }
        self.layers = layers

        super.init()
    }

    /// Encode continuous embeddings to multi-codebook codes.
    ///
    /// Each layer quantizes the residual from the previous layer.
    ///
    /// - Parameter x: Input embeddings [B, T, D]
    /// - Returns: Codebook indices [B, K, T] where K is numCodebooks
    func encode(_ x: MLXArray) -> MLXArray {
        var codes: [MLXArray] = []
        var residual = x

        for layer in layers {
            let indices = layer.encode(residual)
            codes.append(indices)
            let quantized = layer.decode(indices)
            residual = residual - quantized
        }

        // Stack codes: [K, B, T] -> [B, K, T]
        let stacked = stacked(codes, axis: 0)  // [K, B, T]
        return stacked.transposed(1, 0, 2)  // [B, K, T]
    }

    /// Decode multi-codebook codes to continuous embeddings.
    ///
    /// Sums the quantized outputs from all codebooks.
    ///
    /// - Parameter codes: Codebook indices [B, K, T] where K is numCodebooks
    /// - Returns: Reconstructed embeddings [B, T, D]
    func decode(_ codes: MLXArray) -> MLXArray {
        // Transpose to [K, B, T] for easier iteration
        let codesT = codes.transposed(1, 0, 2)

        let batchSize = codes.dim(0)
        let seqLen = codes.dim(2)

        // Sum quantized outputs from all codebooks
        var quantized = MLXArray.zeros([batchSize, seqLen, codebookDim])

        for i in 0..<layers.count {
            let layerCodes = codesT[i]  // [B, T]
            quantized = quantized + layers[i].decode(layerCodes)
        }

        return quantized
    }

    /// Quantize input embeddings using RVQ.
    ///
    /// - Parameter x: Input embeddings [B, T, D]
    /// - Returns: Tuple of (quantized embeddings [B, T, D], codebook indices [B, K, T])
    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        let codes = encode(x)
        let quantized = decode(codes)
        return (quantized, codes)
    }

    /// Get codebook weights for a specific layer.
    ///
    /// - Parameter layerIdx: Index of the codebook layer
    /// - Returns: Codebook weights [codebook_size, codebook_dim]
    func getCodebook(_ layerIdx: Int) -> MLXArray {
        return layers[layerIdx].embedding.weight
    }
}
