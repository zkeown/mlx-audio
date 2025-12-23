// TextEncoder.swift
// RoBERTa-based text encoder for CLAP.
//
// Encodes text into fixed-size embeddings using a pre-trained
// RoBERTa model with a projection head.

import Foundation
@preconcurrency import MLX
import MLXNN

// MARK: - RoBERTa Embeddings

/// RoBERTa embeddings: token + position + token_type.
public class RobertaEmbeddings: Module, @unchecked Sendable {
    let config: CLAPTextConfig

    @ModuleInfo(key: "word_embeddings") var wordEmbeddings: Embedding
    @ModuleInfo(key: "position_embeddings") var positionEmbeddings: Embedding
    @ModuleInfo(key: "token_type_embeddings") var tokenTypeEmbeddings: Embedding
    @ModuleInfo(key: "LayerNorm") var layerNorm: LayerNorm
    let paddingIdx: Int

    public init(config: CLAPTextConfig) {
        self.config = config
        self.paddingIdx = config.padTokenId

        self._wordEmbeddings.wrappedValue = Embedding(
            embeddingCount: config.vocabSize,
            dimensions: config.hiddenSize
        )
        self._positionEmbeddings.wrappedValue = Embedding(
            embeddingCount: config.maxPositionEmbeddings,
            dimensions: config.hiddenSize
        )
        self._tokenTypeEmbeddings.wrappedValue = Embedding(
            embeddingCount: config.typeVocabSize,
            dimensions: config.hiddenSize
        )
        self._layerNorm.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize,
            eps: config.layerNormEps
        )

        super.init()
    }

    /// Forward pass.
    ///
    /// - Parameters:
    ///   - inputIds: Token IDs [B, L]
    ///   - tokenTypeIds: Token type IDs [B, L] (optional)
    ///   - positionIds: Position IDs [B, L] (optional)
    /// - Returns: Embeddings [B, L, hiddenSize]
    public func callAsFunction(
        _ inputIds: MLXArray,
        tokenTypeIds: MLXArray? = nil,
        positionIds: MLXArray? = nil
    ) -> MLXArray {
        let shape = inputIds.shape
        let B = shape[0]
        let L = shape[1]

        // Create position IDs starting from paddingIdx + 1
        let posIds: MLXArray
        if let ids = positionIds {
            posIds = ids
        } else {
            let positions = MLXArray(((paddingIdx + 1)..<(paddingIdx + 1 + L)).map { Int32($0) })
            posIds = MLX.broadcast(positions, to: [B, L])
        }

        // Create token type IDs (all zeros for RoBERTa)
        let typeIds: MLXArray
        if let ids = tokenTypeIds {
            typeIds = ids
        } else {
            typeIds = MLX.zeros([B, L], dtype: .int32)
        }

        let wordEmbeds = wordEmbeddings(inputIds)
        let posEmbeds = positionEmbeddings(posIds)
        let typeEmbeds = tokenTypeEmbeddings(typeIds)

        var embeddings = wordEmbeds + posEmbeds + typeEmbeds
        embeddings = layerNorm(embeddings)

        return embeddings
    }
}

// MARK: - RoBERTa Self Attention

/// RoBERTa self-attention layer.
public class RobertaSelfAttention: Module, @unchecked Sendable {
    let numHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo var query: Linear
    @ModuleInfo var key: Linear
    @ModuleInfo var value: Linear

    public init(config: CLAPTextConfig) {
        self.numHeads = config.numAttentionHeads
        self.headDim = config.hiddenSize / config.numAttentionHeads
        self.scale = pow(Float(headDim), -0.5)

        self._query.wrappedValue = Linear(config.hiddenSize, config.hiddenSize)
        self._key.wrappedValue = Linear(config.hiddenSize, config.hiddenSize)
        self._value.wrappedValue = Linear(config.hiddenSize, config.hiddenSize)

        super.init()
    }

    /// Forward pass.
    ///
    /// - Parameters:
    ///   - hiddenStates: Input [B, L, hiddenSize]
    ///   - attentionMask: Attention mask [B, 1, 1, L] or [B, L]
    /// - Returns: Output [B, L, hiddenSize]
    public func callAsFunction(
        _ hiddenStates: MLXArray,
        attentionMask: MLXArray? = nil
    ) -> MLXArray {
        let shape = hiddenStates.shape
        let B = shape[0]
        let L = shape[1]
        let C = shape[2]

        let q = query(hiddenStates)
        let k = key(hiddenStates)
        let v = value(hiddenStates)

        // Reshape for multi-head attention
        let qReshaped = q.reshaped([B, L, numHeads, headDim])
        let kReshaped = k.reshaped([B, L, numHeads, headDim])
        let vReshaped = v.reshaped([B, L, numHeads, headDim])

        // Transpose to [B, numHeads, L, headDim]
        let qT = qReshaped.transposed(axes: [0, 2, 1, 3])
        let kT = kReshaped.transposed(axes: [0, 2, 1, 3])
        let vT = vReshaped.transposed(axes: [0, 2, 1, 3])

        // Attention scores
        var attn = MLX.matmul(qT, kT.transposed(axes: [0, 1, 3, 2])) * scale

        // Apply attention mask
        if var mask = attentionMask {
            if mask.ndim == 2 {
                // Expand [B, L] to [B, 1, 1, L]
                mask = mask.expandedDimensions(axes: [1, 2])
            }
            // Convert mask: 1 for keep, 0 for mask -> additive mask
            let additiveMask = (1.0 - mask) * Float(-1e9)
            attn = attn + additiveMask
        }

        attn = softmax(attn, axis: -1)

        // Apply attention to values
        var out = MLX.matmul(attn, vT)  // [B, numHeads, L, headDim]
        out = out.transposed(axes: [0, 2, 1, 3])  // [B, L, numHeads, headDim]
        out = out.reshaped([B, L, C])  // [B, L, hiddenSize]

        return out
    }
}

// MARK: - RoBERTa Attention

/// RoBERTa attention layer with output projection.
public class RobertaAttention: Module, @unchecked Sendable {
    @ModuleInfo(key: "self_attn") var selfAttn: RobertaSelfAttention
    @ModuleInfo var output: Linear
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm

    public init(config: CLAPTextConfig) {
        self._selfAttn.wrappedValue = RobertaSelfAttention(config: config)
        self._output.wrappedValue = Linear(config.hiddenSize, config.hiddenSize)
        self._layerNorm.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize,
            eps: config.layerNormEps
        )

        super.init()
    }

    public func callAsFunction(
        _ hiddenStates: MLXArray,
        attentionMask: MLXArray? = nil
    ) -> MLXArray {
        var attnOutput = selfAttn(hiddenStates, attentionMask: attentionMask)
        attnOutput = output(attnOutput)
        let out = layerNorm(hiddenStates + attnOutput)
        return out
    }
}

// MARK: - RoBERTa Intermediate

/// RoBERTa intermediate (FFN first layer).
public class RobertaIntermediate: Module, @unchecked Sendable {
    @ModuleInfo var dense: Linear

    public init(config: CLAPTextConfig) {
        self._dense.wrappedValue = Linear(config.hiddenSize, config.intermediateSize)
        super.init()
    }

    public func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        return gelu(dense(hiddenStates))
    }
}

// MARK: - RoBERTa Output

/// RoBERTa output (FFN second layer with residual).
public class RobertaOutput: Module, @unchecked Sendable {
    @ModuleInfo var dense: Linear
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm

    public init(config: CLAPTextConfig) {
        self._dense.wrappedValue = Linear(config.intermediateSize, config.hiddenSize)
        self._layerNorm.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize,
            eps: config.layerNormEps
        )
        super.init()
    }

    public func callAsFunction(_ hiddenStates: MLXArray, inputTensor: MLXArray) -> MLXArray {
        var out = dense(hiddenStates)
        out = layerNorm(out + inputTensor)
        return out
    }
}

// MARK: - RoBERTa Layer

/// Single RoBERTa transformer layer.
public class RobertaLayer: Module, @unchecked Sendable {
    @ModuleInfo var attention: RobertaAttention
    @ModuleInfo var intermediate: RobertaIntermediate
    @ModuleInfo var output: RobertaOutput

    public init(config: CLAPTextConfig) {
        self._attention.wrappedValue = RobertaAttention(config: config)
        self._intermediate.wrappedValue = RobertaIntermediate(config: config)
        self._output.wrappedValue = RobertaOutput(config: config)
        super.init()
    }

    public func callAsFunction(
        _ hiddenStates: MLXArray,
        attentionMask: MLXArray? = nil
    ) -> MLXArray {
        let attentionOutput = attention(hiddenStates, attentionMask: attentionMask)
        let intermediateOutput = intermediate(attentionOutput)
        let layerOutput = output(intermediateOutput, inputTensor: attentionOutput)
        return layerOutput
    }
}

// MARK: - RoBERTa Encoder

/// RoBERTa encoder (stack of transformer layers).
public class RobertaEncoder: Module, @unchecked Sendable {
    @ModuleInfo(key: "layers") var layers: [RobertaLayer]

    public init(config: CLAPTextConfig) {
        var layerList: [RobertaLayer] = []
        for _ in 0..<config.numHiddenLayers {
            layerList.append(RobertaLayer(config: config))
        }
        self._layers.wrappedValue = layerList
        super.init()
    }

    public func callAsFunction(
        _ hiddenStates: MLXArray,
        attentionMask: MLXArray? = nil
    ) -> MLXArray {
        var out = hiddenStates
        for layer in layers {
            out = layer(out, attentionMask: attentionMask)
        }
        return out
    }
}

// MARK: - RoBERTa Pooler

/// Pooler for RoBERTa that extracts [CLS] token.
public class RobertaPooler: Module, @unchecked Sendable {
    @ModuleInfo var dense: Linear

    public init(hiddenSize: Int) {
        self._dense.wrappedValue = Linear(hiddenSize, hiddenSize)
        super.init()
    }

    public func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        // Take [CLS] token (first token)
        let firstToken = hiddenStates[0..., 0, 0...]
        var pooled = dense(firstToken)
        pooled = tanh(pooled)
        return pooled
    }
}

// MARK: - CLAP Text Projection

/// 2-layer MLP projection head for text encoder.
public class CLAPTextProjection: Module, @unchecked Sendable {
    @ModuleInfo var linear1: Linear
    @ModuleInfo var linear2: Linear

    public init(inDim: Int, outDim: Int) {
        self._linear1.wrappedValue = Linear(inDim, outDim)
        self._linear2.wrappedValue = Linear(outDim, outDim)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = linear1(x)
        out = relu(out)
        out = linear2(out)
        return out
    }
}

// MARK: - CLAP Text Encoder

/// CLAP text encoder based on RoBERTa.
///
/// Encodes text into fixed-size embeddings using a pre-trained
/// RoBERTa model with a projection head.
public class CLAPTextEncoder: Module, @unchecked Sendable {
    let config: CLAPTextConfig
    let projectionDim: Int

    @ModuleInfo var embeddings: RobertaEmbeddings
    @ModuleInfo var encoder: RobertaEncoder
    @ModuleInfo var pooler: RobertaPooler
    @ModuleInfo var projection: CLAPTextProjection

    public init(config: CLAPTextConfig, projectionDim: Int = 512) {
        self.config = config
        self.projectionDim = projectionDim

        self._embeddings.wrappedValue = RobertaEmbeddings(config: config)
        self._encoder.wrappedValue = RobertaEncoder(config: config)
        self._pooler.wrappedValue = RobertaPooler(hiddenSize: config.hiddenSize)
        self._projection.wrappedValue = CLAPTextProjection(
            inDim: config.hiddenSize,
            outDim: projectionDim
        )

        super.init()
    }

    /// Forward pass.
    ///
    /// - Parameters:
    ///   - inputIds: Token IDs [B, L]
    ///   - attentionMask: Attention mask [B, L] (1 for real, 0 for padding)
    ///   - tokenTypeIds: Token type IDs [B, L]
    ///   - normalize: Whether to L2-normalize output
    /// - Returns: Text embeddings [B, projectionDim]
    public func callAsFunction(
        _ inputIds: MLXArray,
        attentionMask: MLXArray? = nil,
        tokenTypeIds: MLXArray? = nil,
        normalize: Bool = true
    ) -> MLXArray {
        // Embeddings
        var hiddenStates = embeddings(
            inputIds,
            tokenTypeIds: tokenTypeIds
        )

        // Encoder
        hiddenStates = encoder(hiddenStates, attentionMask: attentionMask)

        // Pool using [CLS] token
        let pooled = pooler(hiddenStates)

        // Project to shared space
        var embeds = projection(pooled)

        if normalize {
            let normValue = MLX.sqrt(MLX.sum(embeds * embeds, axis: -1, keepDims: true))
            embeds = embeds / (normValue + 1e-8)
        }

        return embeds
    }

    /// Get raw hidden states without pooling.
    public func getEmbeddings(
        _ inputIds: MLXArray,
        attentionMask: MLXArray? = nil
    ) -> MLXArray {
        var hiddenStates = embeddings(inputIds)
        hiddenStates = encoder(hiddenStates, attentionMask: attentionMask)
        return hiddenStates
    }
}
