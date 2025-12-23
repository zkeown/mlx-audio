// T5Config.swift
// Configuration for T5 text encoder used in MusicGen.

import Foundation

/// Configuration for T5 encoder (t5-base defaults).
public struct T5Config: Codable, Sendable, Equatable {

    // MARK: - Model Architecture

    public var vocabSize: Int
    public var hiddenSize: Int
    public var numHiddenLayers: Int
    public var numAttentionHeads: Int
    public var intermediateSize: Int
    public var maxPositionEmbeddings: Int

    // MARK: - Attention Settings

    public var numBuckets: Int
    public var maxDistance: Int
    public var isDecoder: Bool

    // MARK: - Normalization and Dropout

    public var layerNormEps: Float
    public var dropout: Float
    public var attentionDropout: Float

    // MARK: - FFN Settings

    public var feedForwardProj: String  // "relu" or "gated-gelu"
    public var isGatedAct: Bool

    // MARK: - Special Tokens

    public var padTokenId: Int
    public var eosTokenId: Int
    public var decoderStartTokenId: Int

    // MARK: - Computed Properties

    public var headDim: Int { hiddenSize / numAttentionHeads }

    // MARK: - Coding Keys

    enum CodingKeys: String, CodingKey {
        case vocabSize = "vocab_size"
        case hiddenSize = "d_model"
        case numHiddenLayers = "num_layers"
        case numAttentionHeads = "num_heads"
        case intermediateSize = "d_ff"
        case maxPositionEmbeddings = "n_positions"
        case numBuckets = "relative_attention_num_buckets"
        case maxDistance = "relative_attention_max_distance"
        case isDecoder = "is_decoder"
        case layerNormEps = "layer_norm_epsilon"
        case dropout
        case attentionDropout = "attention_dropout"
        case feedForwardProj = "feed_forward_proj"
        case isGatedAct = "is_gated_act"
        case padTokenId = "pad_token_id"
        case eosTokenId = "eos_token_id"
        case decoderStartTokenId = "decoder_start_token_id"
    }

    // MARK: - Initialization

    public init(
        vocabSize: Int = 32128,
        hiddenSize: Int = 768,
        numHiddenLayers: Int = 12,
        numAttentionHeads: Int = 12,
        intermediateSize: Int = 3072,
        maxPositionEmbeddings: Int = 512,
        numBuckets: Int = 32,
        maxDistance: Int = 128,
        isDecoder: Bool = false,
        layerNormEps: Float = 1e-6,
        dropout: Float = 0.1,
        attentionDropout: Float = 0.1,
        feedForwardProj: String = "relu",
        isGatedAct: Bool = false,
        padTokenId: Int = 0,
        eosTokenId: Int = 1,
        decoderStartTokenId: Int = 0
    ) {
        self.vocabSize = vocabSize
        self.hiddenSize = hiddenSize
        self.numHiddenLayers = numHiddenLayers
        self.numAttentionHeads = numAttentionHeads
        self.intermediateSize = intermediateSize
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.numBuckets = numBuckets
        self.maxDistance = maxDistance
        self.isDecoder = isDecoder
        self.layerNormEps = layerNormEps
        self.dropout = dropout
        self.attentionDropout = attentionDropout
        self.feedForwardProj = feedForwardProj
        self.isGatedAct = isGatedAct
        self.padTokenId = padTokenId
        self.eosTokenId = eosTokenId
        self.decoderStartTokenId = decoderStartTokenId
    }

    // MARK: - Preset Configurations

    /// T5-small configuration (60M parameters)
    public static func small() -> T5Config {
        T5Config(
            hiddenSize: 512,
            numHiddenLayers: 6,
            numAttentionHeads: 8,
            intermediateSize: 2048
        )
    }

    /// T5-base configuration (220M parameters) - default for MusicGen
    public static func base() -> T5Config {
        T5Config()
    }

    /// T5-large configuration (770M parameters)
    public static func large() -> T5Config {
        T5Config(
            hiddenSize: 1024,
            numHiddenLayers: 24,
            numAttentionHeads: 16,
            intermediateSize: 4096
        )
    }
}
