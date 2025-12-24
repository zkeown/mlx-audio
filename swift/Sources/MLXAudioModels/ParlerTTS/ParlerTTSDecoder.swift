// ParlerTTSDecoder.swift
// Transformer decoder for Parler-TTS.

import Foundation
@preconcurrency import MLX
import MLXNN

// MARK: - Multi-Head Attention

/// Multi-head attention with RoPE and optional KV caching.
///
/// Supports both self-attention and cross-attention modes.
/// Uses Grouped Query Attention (GQA) when numKVHeads < numHeads.
public class ParlerTTSAttention: Module, @unchecked Sendable {
    public let hiddenSize: Int
    public let numHeads: Int
    public let numKVHeads: Int
    public let headDim: Int
    public let scale: Float
    public let numHeadsPerKV: Int

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    var rotaryEmb: RotaryPositionalEmbedding?

    public init(
        hiddenSize: Int,
        numHeads: Int,
        numKVHeads: Int? = nil,
        dropout: Float = 0.0,
        ropeTheta: Float = 10000.0,
        maxPositionEmbeddings: Int = 4096,
        useRope: Bool = true
    ) {
        self.hiddenSize = hiddenSize
        self.numHeads = numHeads
        self.numKVHeads = numKVHeads ?? numHeads
        self.headDim = hiddenSize / numHeads
        self.scale = pow(Float(self.headDim), -0.5)
        self.numHeadsPerKV = numHeads / self.numKVHeads

        // Projections
        self._qProj.wrappedValue = Linear(hiddenSize, numHeads * headDim, bias: false)
        self._kProj.wrappedValue = Linear(hiddenSize, self.numKVHeads * headDim, bias: false)
        self._vProj.wrappedValue = Linear(hiddenSize, self.numKVHeads * headDim, bias: false)
        self._outProj.wrappedValue = Linear(numHeads * headDim, hiddenSize, bias: false)

        super.init()

        // Rotary embeddings (only for self-attention)
        if useRope {
            self.rotaryEmb = RotaryPositionalEmbedding(
                headDim: headDim,
                maxLength: maxPositionEmbeddings,
                base: ropeTheta
            )
        }
    }

    /// Forward pass.
    ///
    /// - Parameters:
    ///   - hiddenStates: Query input [B, T, D]
    ///   - keyValueStates: Key/value source for cross-attention [B, S, D]
    ///   - attentionMask: Attention mask [T, S] or [B, T, S]
    ///   - kvCache: Cached (key, value) from previous steps
    ///   - positionOffset: Position offset for RoPE
    /// - Returns: Tuple of output tensor [B, T, D] and updated KV cache
    public func callAsFunction(
        _ hiddenStates: MLXArray,
        keyValueStates: MLXArray? = nil,
        attentionMask: MLXArray? = nil,
        kvCache: (MLXArray, MLXArray)? = nil,
        positionOffset: Int = 0
    ) -> (MLXArray, (MLXArray, MLXArray)?) {
        let batchSize = hiddenStates.dim(0)
        let seqLength = hiddenStates.dim(1)

        // Compute query
        var query = qProj(hiddenStates)
        query = query.reshaped([batchSize, seqLength, numHeads, headDim])
        query = query.transposed(0, 2, 1, 3)  // [B, numHeads, T, headDim]

        // Determine if this is cross-attention
        let isCrossAttention = keyValueStates != nil
        var newKVCache: (MLXArray, MLXArray)? = nil

        var key: MLXArray
        var value: MLXArray

        if let kvStates = keyValueStates {
            // Cross-attention
            let kvSeqLength = kvStates.dim(1)
            key = kProj(kvStates)
            value = vProj(kvStates)

            key = key.reshaped([batchSize, kvSeqLength, numKVHeads, headDim])
            key = key.transposed(0, 2, 1, 3)

            value = value.reshaped([batchSize, kvSeqLength, numKVHeads, headDim])
            value = value.transposed(0, 2, 1, 3)
            // No RoPE for cross-attention keys
        } else {
            // Self-attention
            key = kProj(hiddenStates)
            value = vProj(hiddenStates)

            key = key.reshaped([batchSize, seqLength, numKVHeads, headDim])
            key = key.transposed(0, 2, 1, 3)

            value = value.reshaped([batchSize, seqLength, numKVHeads, headDim])
            value = value.transposed(0, 2, 1, 3)

            // Apply RoPE
            if let rope = rotaryEmb {
                (query, key) = rope(query: query, key: key, offset: positionOffset)
            }

            // Handle KV cache
            if let cache = kvCache {
                key = MLX.concatenated([cache.0, key], axis: 2)
                value = MLX.concatenated([cache.1, value], axis: 2)
            }

            newKVCache = (key, value)
        }

        // Expand KV heads for GQA
        if numHeadsPerKV > 1 {
            key = MLX.repeated(key, count: numHeadsPerKV, axis: 1)
            value = MLX.repeated(value, count: numHeadsPerKV, axis: 1)
        }

        // Compute attention scores
        var attnWeights = MLX.matmul(query, key.transposed(0, 1, 3, 2)) * scale

        // Apply attention mask
        if let mask = attentionMask {
            attnWeights = attnWeights + mask
        }

        // Softmax
        attnWeights = softmax(attnWeights, axis: -1)

        // Apply to values
        var attnOutput = MLX.matmul(attnWeights, value)

        // Reshape back
        attnOutput = attnOutput.transposed(0, 2, 1, 3)
        attnOutput = attnOutput.reshaped([batchSize, seqLength, hiddenSize])

        // Output projection
        attnOutput = outProj(attnOutput)

        return (attnOutput, newKVCache)
    }
}

// MARK: - Decoder Block

/// Single transformer decoder block for Parler-TTS.
public class ParlerTTSDecoderBlock: Module, @unchecked Sendable {
    public let hiddenSize: Int
    public let layerIdx: Int

    @ModuleInfo(key: "self_attn") var selfAttn: ParlerTTSAttention
    @ModuleInfo(key: "self_attn_layer_norm") var selfAttnLayerNorm: RMSNorm
    @ModuleInfo(key: "encoder_attn") var encoderAttn: ParlerTTSAttention
    @ModuleInfo(key: "encoder_attn_layer_norm") var encoderAttnLayerNorm: RMSNorm
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear
    @ModuleInfo(key: "fc3") var fc3: Linear
    @ModuleInfo(key: "final_layer_norm") var finalLayerNorm: RMSNorm

    public init(config: ParlerTTSConfig, layerIdx: Int = 0) {
        self.hiddenSize = config.hiddenSize
        self.layerIdx = layerIdx

        // Self-attention with RoPE
        self._selfAttn.wrappedValue = ParlerTTSAttention(
            hiddenSize: config.hiddenSize,
            numHeads: config.numAttentionHeads,
            numKVHeads: config.numKeyValueHeads,
            dropout: config.attentionDropout,
            ropeTheta: config.ropeTheta,
            maxPositionEmbeddings: config.maxPositionEmbeddings,
            useRope: true
        )
        self._selfAttnLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.layerNormEps
        )

        // Cross-attention (no RoPE)
        self._encoderAttn.wrappedValue = ParlerTTSAttention(
            hiddenSize: config.hiddenSize,
            numHeads: config.numAttentionHeads,
            numKVHeads: config.numKeyValueHeads,
            dropout: config.attentionDropout,
            useRope: false
        )
        self._encoderAttnLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.layerNormEps
        )

        // FFN (SwiGLU variant)
        self._fc1.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        self._fc2.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        self._fc3.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
        self._finalLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.layerNormEps
        )

        super.init()
    }

    /// Forward pass.
    public func callAsFunction(
        _ hiddenStates: MLXArray,
        encoderHiddenStates: MLXArray? = nil,
        attentionMask: MLXArray? = nil,
        encoderAttentionMask: MLXArray? = nil,
        kvCache: (MLXArray, MLXArray)? = nil,
        positionOffset: Int = 0
    ) -> (MLXArray, (MLXArray, MLXArray)?) {
        var hidden = hiddenStates
        var residual = hidden

        // Self-attention
        hidden = selfAttnLayerNorm(hidden)
        let (selfAttnOut, newKVCache) = selfAttn(
            hidden,
            attentionMask: attentionMask,
            kvCache: kvCache,
            positionOffset: positionOffset
        )
        hidden = residual + selfAttnOut

        // Cross-attention
        if let encHidden = encoderHiddenStates {
            residual = hidden
            hidden = encoderAttnLayerNorm(hidden)
            let (crossAttnOut, _) = encoderAttn(
                hidden,
                keyValueStates: encHidden,
                attentionMask: encoderAttentionMask
            )
            hidden = residual + crossAttnOut
        }

        // FFN (SwiGLU)
        residual = hidden
        hidden = finalLayerNorm(hidden)
        let gate = silu(fc1(hidden))
        let up = fc2(hidden)
        hidden = fc3(gate * up)
        hidden = residual + hidden

        return (hidden, newKVCache)
    }
}

// MARK: - Full Decoder

/// Full Parler-TTS transformer decoder.
public class ParlerTTSDecoder: Module, @unchecked Sendable {
    public let config: ParlerTTSConfig

    @ModuleInfo var layers: [ParlerTTSDecoderBlock]
    @ModuleInfo(key: "layer_norm") var layerNorm: RMSNorm

    public init(config: ParlerTTSConfig) {
        self.config = config

        var decoderLayers: [ParlerTTSDecoderBlock] = []
        for i in 0..<config.numHiddenLayers {
            decoderLayers.append(ParlerTTSDecoderBlock(config: config, layerIdx: i))
        }
        self._layers.wrappedValue = decoderLayers

        self._layerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.layerNormEps
        )

        super.init()
    }

    /// Forward pass.
    public func callAsFunction(
        _ hiddenStates: MLXArray,
        encoderHiddenStates: MLXArray? = nil,
        attentionMask: MLXArray? = nil,
        encoderAttentionMask: MLXArray? = nil,
        kvCache: [(MLXArray, MLXArray)]? = nil,
        positionOffset: Int = 0
    ) -> (MLXArray, [(MLXArray, MLXArray)]) {
        var hidden = hiddenStates
        var newKVCache: [(MLXArray, MLXArray)] = []

        for (i, layer) in layers.enumerated() {
            let layerCache = (kvCache != nil && i < kvCache!.count) ? kvCache![i] : nil

            let (layerOut, layerKVCache) = layer(
                hidden,
                encoderHiddenStates: encoderHiddenStates,
                attentionMask: attentionMask,
                encoderAttentionMask: encoderAttentionMask,
                kvCache: layerCache,
                positionOffset: positionOffset
            )
            hidden = layerOut

            if let cache = layerKVCache {
                newKVCache.append(cache)
            }
        }

        // Final layer norm
        hidden = layerNorm(hidden)

        return (hidden, newKVCache)
    }

    /// Create causal attention mask.
    public func createCausalMask(seqLength: Int, offset: Int = 0) -> MLXArray {
        let totalLength = seqLength + offset

        // Create mask where position i can attend to positions 0..i+offset
        let queryPos = MLXArray((0..<seqLength).map { Float($0) + Float(offset) })
            .expandedDimensions(axis: 1)
        let keyPos = MLXArray((0..<totalLength).map { Float($0) })
            .expandedDimensions(axis: 0)

        let mask = keyPos .<= queryPos
        return MLX.where(mask, MLXArray(0.0), MLXArray(Float.infinity * -1))
    }
}
