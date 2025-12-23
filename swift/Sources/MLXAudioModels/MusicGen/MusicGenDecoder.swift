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
}
