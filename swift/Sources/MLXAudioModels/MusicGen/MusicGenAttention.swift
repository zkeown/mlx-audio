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

    /// Optimized forward pass using pre-computed K/V from cache.
    ///
    /// This version expects the cache to handle K/V updates externally,
    /// avoiding O(n) concatenation inside the attention layer.
    ///
    /// - Parameters:
    ///   - hiddenStates: Query input [B, T, D]
    ///   - keyValueStates: Key/value input for cross-attention [B, S, D] (nil for self-attention)
    ///   - mask: Attention mask [B, 1, T, S] or [1, 1, T, S]
    ///   - fullKV: Pre-computed (fullKeys, fullValues) from cache [B, H, L, D]
    /// - Returns: Tuple of (output [B, T, D], newK [B, H, T, D], newV [B, H, T, D])
    public func forwardOptimized(
        _ hiddenStates: MLXArray,
        keyValueStates: MLXArray? = nil,
        mask: MLXArray? = nil,
        fullKV: (MLXArray, MLXArray)? = nil
    ) -> (output: MLXArray, newK: MLXArray, newV: MLXArray) {
        let batchSize = hiddenStates.dim(0)
        let queryLength = hiddenStates.dim(1)

        // Compute queries
        var queries = qProj(hiddenStates)
        queries = queries.reshaped(batchSize, queryLength, numHeads, headDim)
        queries = queries.transposed(0, 2, 1, 3)  // [B, H, T, D]

        // Compute keys and values from appropriate source
        let kvSource = keyValueStates ?? hiddenStates
        var newKeys = kProj(kvSource)
        var newValues = vProj(kvSource)

        let kvLength = kvSource.dim(1)
        newKeys = newKeys.reshaped(batchSize, kvLength, numHeads, headDim)
        newValues = newValues.reshaped(batchSize, kvLength, numHeads, headDim)
        newKeys = newKeys.transposed(0, 2, 1, 3)  // [B, H, S, D]
        newValues = newValues.transposed(0, 2, 1, 3)  // [B, H, S, D]

        // Use pre-computed full K/V from cache (already includes current step)
        let keys: MLXArray
        let values: MLXArray
        if let (fullK, fullV) = fullKV {
            keys = fullK
            values = fullV
        } else {
            keys = newKeys
            values = newValues
        }

        // Scaled dot-product attention
        var attnWeights = matmul(queries * scale, keys.transposed(0, 1, 3, 2))

        if let mask = mask {
            attnWeights = attnWeights + mask
        }

        attnWeights = softmax(attnWeights, axis: -1)

        var attnOutput = matmul(attnWeights, values)
        attnOutput = attnOutput.transposed(0, 2, 1, 3)
        attnOutput = attnOutput.reshaped(batchSize, queryLength, hiddenSize)

        return (outProj(attnOutput), newKeys, newValues)
    }
}

/// KV Cache for MusicGen decoder with pre-allocated buffers.
///
/// Uses ring buffer pattern with at[].add() for O(1) append operations
/// instead of O(n) concatenation on each decode step.
public class MusicGenKVCache {

    /// Pre-allocated self-attention key caches per layer.
    /// Shape: [batch, numHeads, maxLength, headDim]
    private var selfAttnKeys: [MLXArray]

    /// Pre-allocated self-attention value caches per layer.
    private var selfAttnValues: [MLXArray]

    /// Cross-attention cache (fixed after first forward pass).
    var crossAttnCache: [(MLXArray, MLXArray)?]

    let numLayers: Int
    let maxLength: Int
    let batchSize: Int
    let numHeads: Int
    let headDim: Int

    /// Current sequence length for self-attention.
    private var _length: Int = 0

    /// Whether cache has been initialized with actual values.
    private var _initialized: Bool = false

    /// Initialize with pre-allocated buffers.
    ///
    /// - Parameters:
    ///   - numLayers: Number of decoder layers
    ///   - maxLength: Maximum sequence length (will be rounded to power of 2)
    ///   - batchSize: Batch size
    ///   - numHeads: Number of attention heads
    ///   - headDim: Dimension per head
    ///   - dtype: Data type for cache arrays
    public init(
        numLayers: Int,
        maxLength: Int = 4096,
        batchSize: Int = 1,
        numHeads: Int = 32,
        headDim: Int = 64,
        dtype: DType = .float16
    ) {
        self.numLayers = numLayers
        self.maxLength = Self.nextPowerOf2(maxLength)
        self.batchSize = batchSize
        self.numHeads = numHeads
        self.headDim = headDim

        // Pre-allocate self-attention caches
        var keys: [MLXArray] = []
        var values: [MLXArray] = []
        for _ in 0..<numLayers {
            keys.append(MLXArray.zeros([batchSize, numHeads, self.maxLength, headDim], dtype: dtype))
            values.append(MLXArray.zeros([batchSize, numHeads, self.maxLength, headDim], dtype: dtype))
        }
        self.selfAttnKeys = keys
        self.selfAttnValues = values

        // Cross-attention uses simple optional storage (computed once)
        self.crossAttnCache = Array(repeating: nil, count: numLayers)

        // Force allocation
        eval(selfAttnKeys + selfAttnValues)
    }

    /// Convenience initializer from config.
    public convenience init(numLayers: Int) {
        self.init(
            numLayers: numLayers,
            maxLength: 4096,
            batchSize: 1,
            numHeads: 32,
            headDim: 64
        )
    }

    private static func nextPowerOf2(_ n: Int) -> Int {
        guard n > 0 else { return 1 }
        var value = n - 1
        value |= value >> 1
        value |= value >> 2
        value |= value >> 4
        value |= value >> 8
        value |= value >> 16
        return value + 1
    }

    /// Get self-attention cache for a layer.
    ///
    /// Returns the valid portion up to current length, or nil if not initialized.
    public func getSelfAttnCache(layer: Int) -> (MLXArray, MLXArray)? {
        guard _initialized && _length > 0 else { return nil }
        return (
            selfAttnKeys[layer][0..., 0..., 0..<_length, 0...],
            selfAttnValues[layer][0..., 0..., 0..<_length, 0...]
        )
    }

    /// Update self-attention cache with new keys/values using at[].add() pattern.
    ///
    /// - Parameters:
    ///   - layer: Layer index
    ///   - k: New keys [B, H, T_new, D]
    ///   - v: New values [B, H, T_new, D]
    /// - Returns: Full cached K/V up to new position
    public func updateSelfAttnCache(layer: Int, k: MLXArray, v: MLXArray) -> (MLXArray, MLXArray) {
        let tNew = k.dim(2)
        let startPos = _length
        let endPos = startPos + tNew

        guard endPos <= maxLength else {
            fatalError("Sequence length \(endPos) exceeds maxLength \(maxLength)")
        }

        // Use at[].add() for O(1) update
        selfAttnKeys[layer] = selfAttnKeys[layer].at[0..., 0..., startPos..<endPos, 0...].add(
            k - selfAttnKeys[layer][0..., 0..., startPos..<endPos, 0...]
        )
        selfAttnValues[layer] = selfAttnValues[layer].at[0..., 0..., startPos..<endPos, 0...].add(
            v - selfAttnValues[layer][0..., 0..., startPos..<endPos, 0...]
        )

        _initialized = true

        return (
            selfAttnKeys[layer][0..., 0..., 0..<endPos, 0...],
            selfAttnValues[layer][0..., 0..., 0..<endPos, 0...]
        )
    }

    /// Legacy interface for compatibility.
    public func updateSelfAttnCache(layer: Int, cache: (MLXArray, MLXArray)?) {
        guard let (k, v) = cache else { return }
        // Extract only the new portion and update
        // This is used when cache comes from attention output
        let newLen = k.dim(2)
        if newLen > _length {
            let newK = k[0..., 0..., _length..<newLen, 0...]
            let newV = v[0..., 0..., _length..<newLen, 0...]
            _ = updateSelfAttnCache(layer: layer, k: newK, v: newV)
        }
    }

    public func getCrossAttnCache(layer: Int) -> (MLXArray, MLXArray)? {
        return crossAttnCache[layer]
    }

    public func updateCrossAttnCache(layer: Int, cache: (MLXArray, MLXArray)?) {
        crossAttnCache[layer] = cache
    }

    /// Advance position after all layers processed.
    public func step(nTokens: Int = 1) {
        _length += nTokens
    }

    public func reset() {
        _length = 0
        _initialized = false
        crossAttnCache = Array(repeating: nil, count: numLayers)
        // Note: We don't zero the pre-allocated buffers for efficiency
        // They will be overwritten on next use
    }

    /// Get the current sequence length from cached keys.
    public var currentLength: Int {
        return _length
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
