// Attention.swift
// Multi-head attention for MusicGen decoder with KV cache support.

import Foundation
import MLX
import MLXFast
import MLXNN

/// Multi-head attention with optional KV caching for efficient autoregressive decoding.
public class MusicGenAttention: Module {

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    let numHeads: Int
    let headDim: Int
    let hiddenSize: Int
    let scale: Float
    let isCrossAttention: Bool

    public init(config: MusicGenConfig, isCrossAttention: Bool = false) {
        self.numHeads = config.numAttentionHeads
        self.headDim = config.headDim
        self.hiddenSize = config.hiddenSize
        self.scale = 1.0 / sqrt(Float(headDim))
        self.isCrossAttention = isCrossAttention

        self._qProj.wrappedValue = Linear(hiddenSize, hiddenSize, bias: true)
        self._kProj.wrappedValue = Linear(hiddenSize, hiddenSize, bias: true)
        self._vProj.wrappedValue = Linear(hiddenSize, hiddenSize, bias: true)
        self._outProj.wrappedValue = Linear(hiddenSize, hiddenSize, bias: true)

        super.init()
    }

    /// Forward pass with optional KV cache.
    /// - Parameters:
    ///   - hiddenStates: Query input [B, T, D]
    ///   - keyValueStates: Key/value input for cross-attention [B, S, D] (nil for self-attention)
    ///   - mask: Attention mask [B, 1, T, S] or [1, 1, T, S]
    ///   - kvCache: Tuple of (cachedKeys, cachedValues) for incremental decoding
    /// - Returns: Tuple of (output [B, T, D], updatedKVCache)
    public func callAsFunction(
        _ hiddenStates: MLXArray,
        keyValueStates: MLXArray? = nil,
        mask: MLXArray? = nil,
        kvCache: (MLXArray, MLXArray)? = nil
    ) -> (MLXArray, (MLXArray, MLXArray)?) {
        let batchSize = hiddenStates.dim(0)
        let queryLength = hiddenStates.dim(1)

        // Compute queries
        var queries = qProj(hiddenStates)
        queries = queries.reshaped(batchSize, queryLength, numHeads, headDim)
        queries = queries.transposed(0, 2, 1, 3)  // [B, H, T, D]

        // Compute keys and values from appropriate source
        let kvSource = keyValueStates ?? hiddenStates
        var keys = kProj(kvSource)
        var values = vProj(kvSource)

        let kvLength = kvSource.dim(1)
        keys = keys.reshaped(batchSize, kvLength, numHeads, headDim)
        values = values.reshaped(batchSize, kvLength, numHeads, headDim)
        keys = keys.transposed(0, 2, 1, 3)  // [B, H, S, D]
        values = values.transposed(0, 2, 1, 3)  // [B, H, S, D]

        // Handle KV cache
        var updatedKVCache: (MLXArray, MLXArray)?

        if let cache = kvCache {
            // For cross-attention, we can reuse cached KV
            if isCrossAttention {
                keys = cache.0
                values = cache.1
                updatedKVCache = cache
            } else {
                // For self-attention, concatenate with cache
                keys = concatenated([cache.0, keys], axis: 2)
                values = concatenated([cache.1, values], axis: 2)
                updatedKVCache = (keys, values)
            }
        } else if isCrossAttention {
            // First call for cross-attention, cache the KV
            updatedKVCache = (keys, values)
        } else {
            // Self-attention without cache
            updatedKVCache = (keys, values)
        }

        // Scaled dot-product attention
        var attnWeights = matmul(queries * scale, keys.transposed(0, 1, 3, 2))

        // Apply mask
        if let mask = mask {
            attnWeights = attnWeights + mask
        }

        attnWeights = softmax(attnWeights, axis: -1)

        // Compute attention output
        var attnOutput = matmul(attnWeights, values)  // [B, H, T, D]
        attnOutput = attnOutput.transposed(0, 2, 1, 3)  // [B, T, H, D]
        attnOutput = attnOutput.reshaped(batchSize, queryLength, hiddenSize)

        return (outProj(attnOutput), updatedKVCache)
    }
}

/// KV Cache for MusicGen decoder.
/// Stores cached keys and values for each decoder layer.
public class MusicGenKVCache {

    var selfAttnCache: [(MLXArray, MLXArray)?]
    var crossAttnCache: [(MLXArray, MLXArray)?]
    let numLayers: Int

    public init(numLayers: Int) {
        self.numLayers = numLayers
        self.selfAttnCache = Array(repeating: nil, count: numLayers)
        self.crossAttnCache = Array(repeating: nil, count: numLayers)
    }

    public func getSelfAttnCache(layer: Int) -> (MLXArray, MLXArray)? {
        return selfAttnCache[layer]
    }

    public func getCrossAttnCache(layer: Int) -> (MLXArray, MLXArray)? {
        return crossAttnCache[layer]
    }

    public func updateSelfAttnCache(layer: Int, cache: (MLXArray, MLXArray)?) {
        selfAttnCache[layer] = cache
    }

    public func updateCrossAttnCache(layer: Int, cache: (MLXArray, MLXArray)?) {
        crossAttnCache[layer] = cache
    }

    public func reset() {
        selfAttnCache = Array(repeating: nil, count: numLayers)
        crossAttnCache = Array(repeating: nil, count: numLayers)
    }

    /// Get the current sequence length from cached keys.
    public var currentLength: Int {
        guard let cache = selfAttnCache.first, let kv = cache else {
            return 0
        }
        return kv.0.dim(2)  // [B, H, S, D]
    }
}

/// Creates a causal attention mask.
/// - Parameters:
///   - queryLength: Length of the query sequence
///   - keyLength: Length of the key sequence
///   - offset: Position offset for incremental decoding
/// - Returns: Causal mask of shape [1, 1, queryLength, keyLength]
public func createCausalMask(queryLength: Int, keyLength: Int, offset: Int = 0) -> MLXArray {
    // Create mask where position i can attend to positions <= i + offset
    let queryPositions = MLXArray(0 ..< queryLength).expandedDimensions(axis: 1)
    let keyPositions = MLXArray(0 ..< keyLength).expandedDimensions(axis: 0)

    // After offset adjustment: query at position i can attend to key positions [0, offset + i]
    let mask = keyPositions .> (queryPositions + offset)

    // Convert boolean mask to attention mask values
    let negInf = MLXArray(-Float.infinity)
    let zero = MLXArray(Float(0.0))
    let attnMask = MLX.where(mask, negInf, zero)

    // Add batch and head dimensions
    return attnMask.expandedDimensions(axes: [0, 1])  // [1, 1, T, S]
}
