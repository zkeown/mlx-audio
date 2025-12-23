// MusicGenConfig.swift
// Configuration for MusicGen text-to-music generation model.

import Foundation

/// Configuration for the MusicGen text-to-music generation model.
public struct MusicGenConfig: Codable, Sendable, Equatable {

    // MARK: - Audio Codec Settings

    public var numCodebooks: Int
    public var codebookSize: Int
    public var audioChannels: Int
    public var sampleRate: Int
    public var frameRate: Int

    // MARK: - Text Encoder Settings

    public var textEncoderName: String
    public var textHiddenSize: Int
    public var maxTextLength: Int

    // MARK: - Decoder Transformer Settings

    public var hiddenSize: Int
    public var numHiddenLayers: Int
    public var numAttentionHeads: Int
    public var intermediateSize: Int
    public var activationFunction: String
    public var dropout: Float
    public var attentionDropout: Float
    public var layerNormEps: Float

    // MARK: - Generation Settings

    public var maxDuration: Float
    public var useCache: Bool

    // MARK: - Special Tokens

    public var padTokenId: Int
    public var bosTokenId: Int
    public var eosTokenId: Int

    // MARK: - Computed Properties

    public var maxNewTokens: Int { Int(maxDuration * Float(frameRate)) }
    public var headDim: Int { hiddenSize / numAttentionHeads }
    public var vocabSize: Int { codebookSize + 1 }

    // MARK: - Coding Keys

    enum CodingKeys: String, CodingKey {
        case numCodebooks = "num_codebooks"
        case codebookSize = "codebook_size"
        case audioChannels = "audio_channels"
        case sampleRate = "sample_rate"
        case frameRate = "frame_rate"
        case textEncoderName = "text_encoder_name"
        case textHiddenSize = "text_hidden_size"
        case maxTextLength = "max_text_length"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case intermediateSize = "intermediate_size"
        case activationFunction = "activation_function"
        case dropout, attentionDropout = "attention_dropout"
        case layerNormEps = "layer_norm_eps"
        case maxDuration = "max_duration"
        case useCache = "use_cache"
        case padTokenId = "pad_token_id"
        case bosTokenId = "bos_token_id"
        case eosTokenId = "eos_token_id"
    }

    // MARK: - Initialization

    public init(
        numCodebooks: Int = 4, codebookSize: Int = 2048, audioChannels: Int = 1,
        sampleRate: Int = 32000, frameRate: Int = 50,
        textEncoderName: String = "t5-base", textHiddenSize: Int = 768, maxTextLength: Int = 256,
        hiddenSize: Int = 1024, numHiddenLayers: Int = 24, numAttentionHeads: Int = 16,
        intermediateSize: Int = 4096, activationFunction: String = "gelu",
        dropout: Float = 0.0, attentionDropout: Float = 0.0, layerNormEps: Float = 1e-5,
        maxDuration: Float = 30.0, useCache: Bool = true,
        padTokenId: Int? = nil, bosTokenId: Int? = nil, eosTokenId: Int? = nil
    ) {
        self.numCodebooks = numCodebooks
        self.codebookSize = codebookSize
        self.audioChannels = audioChannels
        self.sampleRate = sampleRate
        self.frameRate = frameRate
        self.textEncoderName = textEncoderName
        self.textHiddenSize = textHiddenSize
        self.maxTextLength = maxTextLength
        self.hiddenSize = hiddenSize
        self.numHiddenLayers = numHiddenLayers
        self.numAttentionHeads = numAttentionHeads
        self.intermediateSize = intermediateSize
        self.activationFunction = activationFunction
        self.dropout = dropout
        self.attentionDropout = attentionDropout
        self.layerNormEps = layerNormEps
        self.maxDuration = maxDuration
        self.useCache = useCache
        self.padTokenId = padTokenId ?? codebookSize
        self.bosTokenId = bosTokenId ?? codebookSize
        self.eosTokenId = eosTokenId ?? codebookSize
    }

    // MARK: - Preset Configurations

    public static func small() -> MusicGenConfig {
        MusicGenConfig(hiddenSize: 1024, numHiddenLayers: 24, numAttentionHeads: 16, intermediateSize: 4096)
    }

    public static func medium() -> MusicGenConfig {
        MusicGenConfig(hiddenSize: 1536, numHiddenLayers: 48, numAttentionHeads: 24, intermediateSize: 6144)
    }

    public static func large() -> MusicGenConfig {
        MusicGenConfig(hiddenSize: 2048, numHiddenLayers: 48, numAttentionHeads: 32, intermediateSize: 8192)
    }

    public static func melody() -> MusicGenConfig { medium() }

    public static func fromName(_ name: String) throws -> MusicGenConfig {
        switch name.lowercased().replacingOccurrences(of: "-", with: "_").replacingOccurrences(of: "musicgen_", with: "") {
        case "small": return .small()
        case "medium": return .medium()
        case "large": return .large()
        case "melody": return .melody()
        default: throw MusicGenError.unknownModel(name)
        }
    }
}

public enum MusicGenError: Error, LocalizedError {
    case unknownModel(String)
    case weightLoadingFailed(String)
    case invalidInput(String)

    public var errorDescription: String? {
        switch self {
        case .unknownModel(let name): return "Unknown MusicGen model: '\(name)'"
        case .weightLoadingFailed(let reason): return "Failed to load weights: \(reason)"
        case .invalidInput(let reason): return "Invalid input: \(reason)"
        }
    }
}
