// T5Encoder.swift
// T5 transformer encoder for text conditioning in MusicGen.

import Foundation
import MLX
import MLXFast
import MLXNN

/// RMS Layer Normalization used in T5.
public class T5RMSNorm: Module, UnaryLayer {

    @ModuleInfo(key: "weight") var weight: MLXArray

    let eps: Float

    public init(dimensions: Int, eps: Float = 1e-6) {
        self.eps = eps
        self._weight.wrappedValue = MLXArray.ones([dimensions])
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // RMS normalization: x / sqrt(mean(x^2) + eps) * weight
        let variance = mean(x * x, axis: -1, keepDims: true)
        let normalized = x * rsqrt(variance + eps)
        return normalized * weight
    }
}

/// T5 relative position bias for attention.
/// Computes learned position biases based on relative distances.
public class T5RelativePositionBias: Module {

    @ModuleInfo(key: "relative_attention_bias") var relativeAttentionBias: Embedding

    let numBuckets: Int
    let maxDistance: Int
    let numHeads: Int
    let isDecoder: Bool

    public init(config: T5Config) {
        self.numBuckets = config.numBuckets
        self.maxDistance = config.maxDistance
        self.numHeads = config.numAttentionHeads
        self.isDecoder = config.isDecoder

        self._relativeAttentionBias.wrappedValue = Embedding(
            embeddingCount: numBuckets,
            dimensions: numHeads
        )

        super.init()
    }

    /// Compute relative position buckets.
    /// T5 uses a specific bucketing scheme for relative positions.
    private func computeRelativePositionBuckets(
        queryLength: Int,
        keyLength: Int
    ) -> MLXArray {
        // Create position indices
        let queryPositions = MLXArray(0 ..< queryLength).expandedDimensions(axis: 1)
        let keyPositions = MLXArray(0 ..< keyLength).expandedDimensions(axis: 0)

        // Relative positions: query_pos - key_pos
        var relativePosition = queryPositions - keyPositions

        // Convert to buckets
        var relativeBuckets = MLXArray.zeros([queryLength, keyLength]).asType(.int32)

        let numBucketsHalf = numBuckets / 2

        if !isDecoder {
            // Bidirectional: use negative positions for left context
            relativePosition = abs(relativePosition)
        } else {
            // Unidirectional: clamp negative to 0
            relativePosition = maximum(relativePosition, MLXArray(0))
        }

        // Bucket assignment:
        // - First half: exact positions 0..numBuckets/2-1
        // - Second half: log-spaced positions up to maxDistance
        let maxExact = numBucketsHalf
        let isSmall = relativePosition .< maxExact

        // Log-spaced buckets for larger distances
        let relativePositionFloat = relativePosition.asType(.float32)
        let logPosition =
            log(relativePositionFloat / Float(maxExact))
            / log(Float(maxDistance) / Float(maxExact))
            * Float(numBucketsHalf - 1)
        let logBuckets = minimum(logPosition.asType(.int32) + maxExact, MLXArray(numBuckets - 1))

        relativeBuckets = MLX.where(isSmall, relativePosition.asType(.int32), logBuckets)

        return relativeBuckets
    }

    public func callAsFunction(queryLength: Int, keyLength: Int) -> MLXArray {
        // Compute buckets
        let buckets = computeRelativePositionBuckets(
            queryLength: queryLength,
            keyLength: keyLength
        )

        // Look up biases: [Q, K] -> [Q, K, H]
        let biases = relativeAttentionBias(buckets)

        // Reshape to [1, H, Q, K] for broadcasting with attention weights
        return biases.transposed(2, 0, 1).expandedDimensions(axis: 0)
    }
}

/// T5 self-attention layer.
public class T5Attention: Module {

    @ModuleInfo(key: "q") var qProj: Linear
    @ModuleInfo(key: "k") var kProj: Linear
    @ModuleInfo(key: "v") var vProj: Linear
    @ModuleInfo(key: "o") var outProj: Linear

    let numHeads: Int
    let headDim: Int
    let scale: Float

    public init(config: T5Config) {
        self.numHeads = config.numAttentionHeads
        self.headDim = config.headDim
        self.scale = 1.0 / sqrt(Float(headDim))

        let hiddenSize = config.hiddenSize
        let innerDim = numHeads * headDim

        // T5 attention projections don't use bias
        self._qProj.wrappedValue = Linear(hiddenSize, innerDim, bias: false)
        self._kProj.wrappedValue = Linear(hiddenSize, innerDim, bias: false)
        self._vProj.wrappedValue = Linear(hiddenSize, innerDim, bias: false)
        self._outProj.wrappedValue = Linear(innerDim, hiddenSize, bias: false)

        super.init()
    }

    public func callAsFunction(
        _ hiddenStates: MLXArray,
        positionBias: MLXArray? = nil,
        attentionMask: MLXArray? = nil
    ) -> MLXArray {
        let batchSize = hiddenStates.dim(0)
        let seqLength = hiddenStates.dim(1)

        // Project Q, K, V
        var queries = qProj(hiddenStates)
        var keys = kProj(hiddenStates)
        var values = vProj(hiddenStates)

        // Reshape to multi-head format
        queries = queries.reshaped(batchSize, seqLength, numHeads, headDim).transposed(0, 2, 1, 3)
        keys = keys.reshaped(batchSize, seqLength, numHeads, headDim).transposed(0, 2, 1, 3)
        values = values.reshaped(batchSize, seqLength, numHeads, headDim).transposed(0, 2, 1, 3)

        // Compute attention scores
        var scores = matmul(queries * scale, keys.transposed(0, 1, 3, 2))

        // Add relative position bias
        if let bias = positionBias {
            scores = scores + bias
        }

        // Apply attention mask
        if let mask = attentionMask {
            scores = scores + mask
        }

        // Softmax and weighted sum
        let attnWeights = softmax(scores, axis: -1)
        var output = matmul(attnWeights, values)

        // Reshape back
        output = output.transposed(0, 2, 1, 3).reshaped(batchSize, seqLength, numHeads * headDim)

        return outProj(output)
    }
}

/// T5 feed-forward network.
public class T5FeedForward: Module {

    @ModuleInfo(key: "wi") var wi: Linear
    @ModuleInfo(key: "wo") var wo: Linear
    @ModuleInfo(key: "wi_0") var wi0: Linear?
    @ModuleInfo(key: "wi_1") var wi1: Linear?

    let isGatedAct: Bool
    let activationType: String

    public init(config: T5Config) {
        self.isGatedAct = config.isGatedAct || config.feedForwardProj.contains("gated")
        self.activationType = config.feedForwardProj

        let hiddenSize = config.hiddenSize
        let intermediateSize = config.intermediateSize

        if isGatedAct {
            // Gated activation: two input projections
            self._wi0.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
            self._wi1.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
            self._wi.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)  // Placeholder
        } else {
            // Standard: single input projection
            self._wi.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
            self._wi0.wrappedValue = nil
            self._wi1.wrappedValue = nil
        }

        self._wo.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)

        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        if isGatedAct, let wi0 = wi0, let wi1 = wi1 {
            // Gated GELU: gelu(wi0(x)) * wi1(x)
            let gated = gelu(wi0(x))
            let linear = wi1(x)
            return wo(gated * linear)
        } else {
            // Standard ReLU
            return wo(relu(wi(x)))
        }
    }
}

/// T5 encoder block.
public class T5EncoderBlock: Module {

    @ModuleInfo(key: "layer") var layers: [Module]

    let selfAttn: T5Attention
    let selfAttnLayerNorm: T5RMSNorm
    let ffn: T5FeedForward
    let ffnLayerNorm: T5RMSNorm

    public init(config: T5Config, hasRelativeAttentionBias: Bool = false) {
        self.selfAttn = T5Attention(config: config)
        self.selfAttnLayerNorm = T5RMSNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        self.ffn = T5FeedForward(config: config)
        self.ffnLayerNorm = T5RMSNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)

        // Store in layers array for weight loading compatibility
        self._layers.wrappedValue = []

        super.init()
    }

    public func callAsFunction(
        _ hiddenStates: MLXArray,
        positionBias: MLXArray? = nil,
        attentionMask: MLXArray? = nil
    ) -> MLXArray {
        // Self-attention with pre-norm
        var residual = hiddenStates
        var hidden = selfAttnLayerNorm(hiddenStates)
        hidden = selfAttn(hidden, positionBias: positionBias, attentionMask: attentionMask)
        hidden = residual + hidden

        // FFN with pre-norm
        residual = hidden
        hidden = ffnLayerNorm(hidden)
        hidden = residual + ffn(hidden)

        return hidden
    }
}

/// T5 encoder-only model.
public class T5Encoder: Module {

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "block") var blocks: [T5EncoderBlock]
    @ModuleInfo(key: "final_layer_norm") var finalLayerNorm: T5RMSNorm

    let positionBias: T5RelativePositionBias
    let config: T5Config

    public init(config: T5Config) {
        self.config = config

        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize,
            dimensions: config.hiddenSize
        )

        var encoderBlocks: [T5EncoderBlock] = []
        for i in 0 ..< config.numHiddenLayers {
            encoderBlocks.append(T5EncoderBlock(config: config, hasRelativeAttentionBias: i == 0))
        }
        self._blocks.wrappedValue = encoderBlocks

        self._finalLayerNorm.wrappedValue = T5RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.layerNormEps
        )

        // Position bias is computed once and shared
        self.positionBias = T5RelativePositionBias(config: config)

        super.init()
    }

    /// Forward pass through the T5 encoder.
    /// - Parameters:
    ///   - inputIds: Token IDs [B, S]
    ///   - attentionMask: Mask [B, S] (1 for real tokens, 0 for padding)
    /// - Returns: Encoder hidden states [B, S, D]
    public func callAsFunction(
        inputIds: MLXArray,
        attentionMask: MLXArray? = nil
    ) -> MLXArray {
        let seqLength = inputIds.dim(1)

        // Embed tokens
        var hidden = embedTokens(inputIds)

        // Compute position bias once
        let bias = positionBias(queryLength: seqLength, keyLength: seqLength)

        // Create attention mask
        var attnMask: MLXArray? = nil
        if let mask = attentionMask {
            // Convert [B, S] to [B, 1, 1, S]
            let maskExpanded = mask.expandedDimensions(axes: [1, 2])
            let negInf = MLXArray(-Float.infinity)
            let zero = MLXArray(Float(0.0))
            attnMask = MLX.where(maskExpanded .== 0, negInf, zero)
        }

        // Run through encoder blocks
        for block in blocks {
            hidden = block(hidden, positionBias: bias, attentionMask: attnMask)
        }

        // Final layer norm
        hidden = finalLayerNorm(hidden)

        return hidden
    }

    /// Load T5 encoder from pretrained weights.
    public static func fromPretrained(path: String, config: T5Config) throws -> T5Encoder {
        let encoder = T5Encoder(config: config)

        // Load weights
        let weightsPath = (path as NSString).appendingPathComponent("model.safetensors")
        let weights = try loadArrays(url: URL(fileURLWithPath: weightsPath))

        // Map weight keys and filter for encoder-only weights
        let mappedWeights = mapT5WeightKeys(weights)
        let parameters = ModuleParameters.unflattened(mappedWeights)
        try encoder.update(parameters: parameters, verify: .noUnusedKeys)

        return encoder
    }

    /// Map T5 weight keys from HuggingFace format to Swift naming.
    private static func mapT5WeightKeys(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var mapped: [String: MLXArray] = [:]

        for (key, value) in weights {
            // Skip decoder weights
            if key.hasPrefix("decoder.") {
                continue
            }

            var newKey = key

            // Remove "encoder." prefix
            if newKey.hasPrefix("encoder.") {
                newKey = String(newKey.dropFirst(8))
            }

            // Transform layer naming
            newKey = newKey.replacingOccurrences(
                of: "block.(\\d+).layer.0.SelfAttention",
                with: "blocks.$1.selfAttn",
                options: .regularExpression
            )
            newKey = newKey.replacingOccurrences(
                of: "block.(\\d+).layer.0.layer_norm",
                with: "blocks.$1.selfAttnLayerNorm",
                options: .regularExpression
            )
            newKey = newKey.replacingOccurrences(
                of: "block.(\\d+).layer.1.DenseReluDense",
                with: "blocks.$1.ffn",
                options: .regularExpression
            )
            newKey = newKey.replacingOccurrences(
                of: "block.(\\d+).layer.1.layer_norm",
                with: "blocks.$1.ffnLayerNorm",
                options: .regularExpression
            )

            // T5 projection naming
            newKey = newKey.replacingOccurrences(of: ".q.", with: ".qProj.")
            newKey = newKey.replacingOccurrences(of: ".k.", with: ".kProj.")
            newKey = newKey.replacingOccurrences(of: ".v.", with: ".vProj.")
            newKey = newKey.replacingOccurrences(of: ".o.", with: ".outProj.")

            // Embedding
            newKey = newKey.replacingOccurrences(of: "embed_tokens", with: "embedTokens")
            newKey = newKey.replacingOccurrences(of: "final_layer_norm", with: "finalLayerNorm")

            // Relative attention bias
            newKey = newKey.replacingOccurrences(
                of: "relative_attention_bias",
                with: "relativeAttentionBias"
            )

            mapped[newKey] = value
        }

        return mapped
    }
}
