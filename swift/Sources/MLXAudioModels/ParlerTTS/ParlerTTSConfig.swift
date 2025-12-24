// ParlerTTSConfig.swift
// Configuration for Parler-TTS text-to-speech model.

import Foundation

/// Configuration for Parler-TTS model.
///
/// Parler-TTS is a decoder-only transformer that generates speech audio tokens
/// conditioned on text input and optional voice description. It uses DAC/EnCodec
/// for audio tokenization/detokenization.
public struct ParlerTTSConfig: Codable, Sendable {
    // MARK: - Audio Codec Settings (DAC 24kHz)

    /// Number of audio codebooks (from DAC/EnCodec).
    public var numCodebooks: Int

    /// Vocabulary size per codebook.
    public var codebookSize: Int

    /// Number of audio channels (1=mono).
    public var audioChannels: Int

    /// Audio sample rate in Hz.
    public var sampleRate: Int

    /// Audio codec frame rate (tokens per second).
    public var frameRate: Int

    // MARK: - Text Encoder Settings (T5)

    /// Name of the text encoder model (T5).
    public var textEncoderName: String

    /// Dimension of text encoder hidden states.
    public var textHiddenSize: Int

    /// Maximum text sequence length.
    public var maxTextLength: Int

    // MARK: - Description Encoder Settings (T5)

    /// Name of the description encoder (T5).
    public var descriptionEncoderName: String

    /// Dimension of description encoder hidden states.
    public var descriptionHiddenSize: Int

    /// Maximum description sequence length.
    public var maxDescriptionLength: Int

    // MARK: - Decoder Transformer Settings

    /// Decoder hidden dimension.
    public var hiddenSize: Int

    /// Number of transformer layers.
    public var numHiddenLayers: Int

    /// Number of attention heads.
    public var numAttentionHeads: Int

    /// Number of key-value heads (for GQA).
    public var numKeyValueHeads: Int

    /// FFN intermediate dimension.
    public var intermediateSize: Int

    /// Activation function in FFN.
    public var activationFunction: String

    /// Dropout probability.
    public var dropout: Float

    /// Attention dropout probability.
    public var attentionDropout: Float

    /// Layer normalization epsilon.
    public var layerNormEps: Float

    // MARK: - Position Embedding Settings

    /// Maximum position embeddings.
    public var maxPositionEmbeddings: Int

    /// Base for RoPE frequencies.
    public var ropeTheta: Float

    // MARK: - Generation Settings

    /// Maximum generation duration in seconds.
    public var maxDuration: Float

    /// Whether to use KV caching during generation.
    public var useCache: Bool

    // MARK: - Special Tokens

    /// Padding token ID.
    public var padTokenId: Int

    /// Beginning of sequence token ID.
    public var bosTokenId: Int

    /// End of sequence token ID.
    public var eosTokenId: Int

    // MARK: - Computed Properties

    /// Maximum number of new tokens to generate.
    public var maxNewTokens: Int {
        Int(maxDuration * Float(frameRate))
    }

    /// Dimension of each attention head.
    public var headDim: Int {
        hiddenSize / numAttentionHeads
    }

    /// Total vocabulary size including special tokens.
    public var vocabSize: Int {
        codebookSize + 2  // +2 for pad and bos
    }

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
        case descriptionEncoderName = "description_encoder_name"
        case descriptionHiddenSize = "description_hidden_size"
        case maxDescriptionLength = "max_description_length"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case intermediateSize = "intermediate_size"
        case activationFunction = "activation_function"
        case dropout
        case attentionDropout = "attention_dropout"
        case layerNormEps = "layer_norm_eps"
        case maxPositionEmbeddings = "max_position_embeddings"
        case ropeTheta = "rope_theta"
        case maxDuration = "max_duration"
        case useCache = "use_cache"
        case padTokenId = "pad_token_id"
        case bosTokenId = "bos_token_id"
        case eosTokenId = "eos_token_id"
    }

    // MARK: - Initialization

    public init(
        numCodebooks: Int = 9,
        codebookSize: Int = 1024,
        audioChannels: Int = 1,
        sampleRate: Int = 24000,
        frameRate: Int = 75,
        textEncoderName: String = "google/flan-t5-large",
        textHiddenSize: Int = 1024,
        maxTextLength: Int = 600,
        descriptionEncoderName: String = "google/flan-t5-large",
        descriptionHiddenSize: Int = 1024,
        maxDescriptionLength: Int = 256,
        hiddenSize: Int = 1024,
        numHiddenLayers: Int = 24,
        numAttentionHeads: Int = 16,
        numKeyValueHeads: Int = 16,
        intermediateSize: Int = 4096,
        activationFunction: String = "gelu",
        dropout: Float = 0.0,
        attentionDropout: Float = 0.0,
        layerNormEps: Float = 1e-5,
        maxPositionEmbeddings: Int = 4096,
        ropeTheta: Float = 10000.0,
        maxDuration: Float = 30.0,
        useCache: Bool = true,
        padTokenId: Int = 1024,
        bosTokenId: Int = 1025,
        eosTokenId: Int = 1024
    ) {
        self.numCodebooks = numCodebooks
        self.codebookSize = codebookSize
        self.audioChannels = audioChannels
        self.sampleRate = sampleRate
        self.frameRate = frameRate
        self.textEncoderName = textEncoderName
        self.textHiddenSize = textHiddenSize
        self.maxTextLength = maxTextLength
        self.descriptionEncoderName = descriptionEncoderName
        self.descriptionHiddenSize = descriptionHiddenSize
        self.maxDescriptionLength = maxDescriptionLength
        self.hiddenSize = hiddenSize
        self.numHiddenLayers = numHiddenLayers
        self.numAttentionHeads = numAttentionHeads
        self.numKeyValueHeads = numKeyValueHeads
        self.intermediateSize = intermediateSize
        self.activationFunction = activationFunction
        self.dropout = dropout
        self.attentionDropout = attentionDropout
        self.layerNormEps = layerNormEps
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.ropeTheta = ropeTheta
        self.maxDuration = maxDuration
        self.useCache = useCache
        self.padTokenId = padTokenId
        self.bosTokenId = bosTokenId
        self.eosTokenId = eosTokenId
    }

    // MARK: - Preset Configurations

    /// Parler-TTS Mini configuration (~880M parameters).
    /// Fast inference, good for real-time applications.
    public static func mini() -> ParlerTTSConfig {
        var config = ParlerTTSConfig()
        config.numCodebooks = 9
        config.codebookSize = 1024
        config.hiddenSize = 1024
        config.numHiddenLayers = 24
        config.numAttentionHeads = 16
        config.numKeyValueHeads = 16
        config.intermediateSize = 4096
        config.textEncoderName = "google/flan-t5-large"
        config.textHiddenSize = 1024
        config.descriptionEncoderName = "google/flan-t5-large"
        config.descriptionHiddenSize = 1024
        return config
    }

    /// Parler-TTS Large configuration (~2.3B parameters).
    /// Best quality, slower inference.
    public static func large() -> ParlerTTSConfig {
        var config = ParlerTTSConfig()
        config.numCodebooks = 9
        config.codebookSize = 1024
        config.hiddenSize = 1536
        config.numHiddenLayers = 36
        config.numAttentionHeads = 24
        config.numKeyValueHeads = 24
        config.intermediateSize = 6144
        config.textEncoderName = "google/flan-t5-large"
        config.textHiddenSize = 1024
        config.descriptionEncoderName = "google/flan-t5-large"
        config.descriptionHiddenSize = 1024
        return config
    }

    /// Create config from model name.
    public static func fromName(_ name: String) throws -> ParlerTTSConfig {
        let normalized = name.lowercased()
            .replacingOccurrences(of: "-", with: "_")
            .replacingOccurrences(of: "parler_tts_", with: "")

        switch normalized {
        case "mini":
            return .mini()
        case "large":
            return .large()
        default:
            throw ParlerTTSError.unknownModel(name)
        }
    }

    /// Load config from JSON file.
    public static func load(from url: URL) throws -> ParlerTTSConfig {
        let data = try Data(contentsOf: url)
        let decoder = JSONDecoder()
        return try decoder.decode(ParlerTTSConfig.self, from: data)
    }
}

// MARK: - Errors

/// Errors that can occur during Parler-TTS operations.
public enum ParlerTTSError: Error, LocalizedError {
    case unknownModel(String)
    case codecNotSet
    case weightLoadingFailed(String)
    case generationFailed(String)
    case invalidInput(String)

    public var errorDescription: String? {
        switch self {
        case .unknownModel(let name):
            return "Unknown Parler-TTS model: \(name). Available: mini, large"
        case .codecNotSet:
            return "Audio codec not set. Call setAudioCodec() first."
        case .weightLoadingFailed(let message):
            return "Failed to load weights: \(message)"
        case .generationFailed(let message):
            return "Generation failed: \(message)"
        case .invalidInput(let message):
            return "Invalid input: \(message)"
        }
    }
}
