// Decoder.swift
// Transformer decoder for MusicGen audio generation.

import Foundation
import MLX
import MLXFast
import MLXNN

/// Feed-forward network for transformer decoder blocks.
public class MusicGenFeedForward: Module {

    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear

    let activationFunction: String

    public init(config: MusicGenConfig) {
        self._fc1.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: true)
        self._fc2.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: true)
        self.activationFunction = config.activationFunction

        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var hidden = fc1(x)

        // Apply activation function
        switch activationFunction.lowercased() {
        case "gelu":
            hidden = gelu(hidden)
        case "relu":
            hidden = relu(hidden)
        case "silu", "swish":
            hidden = silu(hidden)
        default:
            hidden = gelu(hidden)
        }

        return fc2(hidden)
    }
}

/// Single decoder block with self-attention, cross-attention, and FFN.
public class MusicGenDecoderBlock: Module {

    @ModuleInfo(key: "self_attn") var selfAttn: MusicGenAttention
    @ModuleInfo(key: "self_attn_layer_norm") var selfAttnLayerNorm: LayerNorm
    @ModuleInfo(key: "encoder_attn") var crossAttn: MusicGenAttention
    @ModuleInfo(key: "encoder_attn_layer_norm") var crossAttnLayerNorm: LayerNorm
    @ModuleInfo(key: "ffn") var ffn: MusicGenFeedForward
    @ModuleInfo(key: "final_layer_norm") var finalLayerNorm: LayerNorm

    public init(config: MusicGenConfig) {
        self._selfAttn.wrappedValue = MusicGenAttention(config: config, isCrossAttention: false)
        self._selfAttnLayerNorm.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize, eps: config.layerNormEps)

        self._crossAttn.wrappedValue = MusicGenAttention(config: config, isCrossAttention: true)
        self._crossAttnLayerNorm.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize, eps: config.layerNormEps)

        self._ffn.wrappedValue = MusicGenFeedForward(config: config)
        self._finalLayerNorm.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize, eps: config.layerNormEps)

        super.init()
    }

    /// Forward pass through the decoder block.
    /// - Parameters:
    ///   - hiddenStates: Input hidden states [B, T, D]
    ///   - encoderHiddenStates: Encoder output for cross-attention [B, S, D]
    ///   - attentionMask: Causal mask for self-attention [1, 1, T, T]
    ///   - encoderAttentionMask: Mask for cross-attention [B, 1, 1, S]
    ///   - selfAttnCache: KV cache for self-attention
    ///   - crossAttnCache: KV cache for cross-attention
    /// - Returns: Tuple of (output, selfAttnCache, crossAttnCache)
    public func callAsFunction(
        _ hiddenStates: MLXArray,
        encoderHiddenStates: MLXArray,
        attentionMask: MLXArray? = nil,
        encoderAttentionMask: MLXArray? = nil,
        selfAttnCache: (MLXArray, MLXArray)? = nil,
        crossAttnCache: (MLXArray, MLXArray)? = nil
    ) -> (MLXArray, (MLXArray, MLXArray)?, (MLXArray, MLXArray)?) {
        // Self-attention with pre-norm
        var residual = hiddenStates
        var hidden = selfAttnLayerNorm(hiddenStates)
        let (selfAttnOutput, newSelfAttnCache) = selfAttn(
            hidden,
            mask: attentionMask,
            kvCache: selfAttnCache
        )
        hidden = residual + selfAttnOutput

        // Cross-attention with pre-norm
        residual = hidden
        hidden = crossAttnLayerNorm(hidden)
        let (crossAttnOutput, newCrossAttnCache) = crossAttn(
            hidden,
            keyValueStates: encoderHiddenStates,
            mask: encoderAttentionMask,
            kvCache: crossAttnCache
        )
        hidden = residual + crossAttnOutput

        // FFN with pre-norm
        residual = hidden
        hidden = finalLayerNorm(hidden)
        hidden = residual + ffn(hidden)

        return (hidden, newSelfAttnCache, newCrossAttnCache)
    }

    /// Optimized forward pass using pre-computed K/V from cache.
    ///
    /// This version expects the cache to provide full K/V sequences,
    /// avoiding O(n) concatenation inside attention layers.
    ///
    /// - Parameters:
    ///   - hiddenStates: Input hidden states [B, T, D]
    ///   - encoderHiddenStates: Encoder output for cross-attention [B, S, D]
    ///   - attentionMask: Causal mask for self-attention
    ///   - encoderAttentionMask: Mask for cross-attention
    ///   - selfAttnKV: Pre-computed full self-attention K/V from cache
    ///   - crossAttnKV: Cached cross-attention K/V (fixed after first pass)
    /// - Returns: Tuple of (output, newSelfK, newSelfV, newCrossK, newCrossV)
    public func forwardOptimized(
        _ hiddenStates: MLXArray,
        encoderHiddenStates: MLXArray,
        attentionMask: MLXArray? = nil,
        encoderAttentionMask: MLXArray? = nil,
        selfAttnKV: (MLXArray, MLXArray)? = nil,
        crossAttnKV: (MLXArray, MLXArray)? = nil
    ) -> (output: MLXArray, selfK: MLXArray, selfV: MLXArray, crossK: MLXArray?, crossV: MLXArray?) {
        // Self-attention with pre-norm
        var residual = hiddenStates
        var hidden = selfAttnLayerNorm(hiddenStates)
        let (selfAttnOutput, selfK, selfV) = selfAttn.forwardOptimized(
            hidden,
            mask: attentionMask,
            fullKV: selfAttnKV
        )
        hidden = residual + selfAttnOutput

        // Cross-attention with pre-norm
        residual = hidden
        hidden = crossAttnLayerNorm(hidden)

        var crossK: MLXArray?
        var crossV: MLXArray?

        if let (cachedCrossK, cachedCrossV) = crossAttnKV {
            // Reuse cached cross-attention K/V
            let (crossAttnOutput, _, _) = crossAttn.forwardOptimized(
                hidden,
                keyValueStates: nil,
                mask: encoderAttentionMask,
                fullKV: (cachedCrossK, cachedCrossV)
            )
            hidden = residual + crossAttnOutput
            crossK = cachedCrossK
            crossV = cachedCrossV
        } else {
            // First pass: compute cross-attention K/V from encoder states
            let (crossAttnOutput, newCrossK, newCrossV) = crossAttn.forwardOptimized(
                hidden,
                keyValueStates: encoderHiddenStates,
                mask: encoderAttentionMask,
                fullKV: nil
            )
            hidden = residual + crossAttnOutput
            crossK = newCrossK
            crossV = newCrossV
        }

        // FFN with pre-norm
        residual = hidden
        hidden = finalLayerNorm(hidden)
        hidden = residual + ffn(hidden)

        return (hidden, selfK, selfV, crossK, crossV)
    }
}

/// Full MusicGen decoder with stacked transformer blocks.
public class MusicGenDecoder: Module {

    @ModuleInfo(key: "layers") var layers: [MusicGenDecoderBlock]
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm

    let config: MusicGenConfig

    public init(config: MusicGenConfig) {
        self.config = config

        var decoderLayers: [MusicGenDecoderBlock] = []
        for _ in 0 ..< config.numHiddenLayers {
            decoderLayers.append(MusicGenDecoderBlock(config: config))
        }
        self._layers.wrappedValue = decoderLayers

        self._layerNorm.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize, eps: config.layerNormEps)

        super.init()
    }

    /// Forward pass through the decoder.
    /// - Parameters:
    ///   - hiddenStates: Input embeddings [B, T, D]
    ///   - encoderHiddenStates: Encoder output for cross-attention [B, S, D]
    ///   - attentionMask: Causal mask for self-attention
    ///   - encoderAttentionMask: Mask for cross-attention
    ///   - kvCache: KV cache for incremental decoding
    /// - Returns: Tuple of (output [B, T, D], updatedKVCache)
    public func callAsFunction(
        _ hiddenStates: MLXArray,
        encoderHiddenStates: MLXArray,
        attentionMask: MLXArray? = nil,
        encoderAttentionMask: MLXArray? = nil,
        kvCache: MusicGenKVCache? = nil
    ) -> (MLXArray, MusicGenKVCache?) {
        var hidden = hiddenStates

        // Create or use existing KV cache
        let cache = kvCache ?? MusicGenKVCache(numLayers: config.numHiddenLayers)

        for (i, layer) in layers.enumerated() {
            let (output, newSelfCache, newCrossCache) = layer(
                hidden,
                encoderHiddenStates: encoderHiddenStates,
                attentionMask: attentionMask,
                encoderAttentionMask: encoderAttentionMask,
                selfAttnCache: cache.getSelfAttnCache(layer: i),
                crossAttnCache: cache.getCrossAttnCache(layer: i)
            )
            hidden = output
            cache.updateSelfAttnCache(layer: i, cache: newSelfCache)
            cache.updateCrossAttnCache(layer: i, cache: newCrossCache)
        }

        // Final layer norm
        hidden = layerNorm(hidden)

        return (hidden, cache)
    }

    /// Optimized forward pass using pre-allocated KV cache with O(1) updates.
    ///
    /// This version uses the `forwardOptimized` path through decoder blocks,
    /// properly using the pre-allocated cache for O(1) append operations.
    ///
    /// - Parameters:
    ///   - hiddenStates: Input embeddings [B, T, D]
    ///   - encoderHiddenStates: Encoder output for cross-attention [B, S, D]
    ///   - attentionMask: Causal mask for self-attention
    ///   - encoderAttentionMask: Mask for cross-attention
    ///   - kvCache: Pre-allocated KV cache
    /// - Returns: Tuple of (output [B, T, D], updatedKVCache)
    public func forwardOptimized(
        _ hiddenStates: MLXArray,
        encoderHiddenStates: MLXArray,
        attentionMask: MLXArray? = nil,
        encoderAttentionMask: MLXArray? = nil,
        kvCache: MusicGenKVCache
    ) -> (MLXArray, MusicGenKVCache) {
        var hidden = hiddenStates

        for (i, layer) in layers.enumerated() {
            // Get current cache state for this layer
            let selfAttnKV = kvCache.getSelfAttnCache(layer: i)
            let crossAttnKV = kvCache.getCrossAttnCache(layer: i)

            // Forward with optimized path
            let (output, selfK, selfV, crossK, crossV) = layer.forwardOptimized(
                hidden,
                encoderHiddenStates: encoderHiddenStates,
                attentionMask: attentionMask,
                encoderAttentionMask: encoderAttentionMask,
                selfAttnKV: selfAttnKV,
                crossAttnKV: crossAttnKV
            )
            hidden = output

            // Update cache with new K/V using O(1) pattern
            _ = kvCache.updateSelfAttnCache(layer: i, k: selfK, v: selfV)

            // Update cross-attention cache (only on first pass)
            if crossAttnKV == nil, let ck = crossK, let cv = crossV {
                kvCache.updateCrossAttnCache(layer: i, cache: (ck, cv))
            }
        }

        // Advance cache position
        kvCache.step(nTokens: hiddenStates.dim(1))

        // Final layer norm
        hidden = layerNorm(hidden)

        return (hidden, kvCache)
    }
}
