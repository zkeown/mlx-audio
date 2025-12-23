// CLAPConfig.swift
// Configuration for CLAP (Contrastive Language-Audio Pretraining) model.
//
// CLAP is a dual-encoder model that learns joint embeddings for audio and text,
// enabling zero-shot audio classification and audio-text retrieval.

import Foundation

// MARK: - Audio Encoder Configuration

/// Configuration for the HTSAT audio encoder.
///
/// HTSAT (Hierarchical Token-Semantic Audio Transformer) is a Swin Transformer-based
/// audio encoder that processes mel spectrograms into fixed-size embeddings.
public struct CLAPAudioConfig: Sendable, Codable {
    /// Audio sample rate in Hz.
    public var sampleRate: Int

    /// Number of mel filterbank bins.
    public var nMels: Int

    /// FFT window size.
    public var nFFT: Int

    /// Hop length for STFT (10ms at 48kHz).
    public var hopLength: Int

    /// Window length for STFT.
    public var windowLength: Int

    // MARK: - HTSAT Architecture

    /// Patch size for patch embedding.
    public var patchSize: Int

    /// Stride for patch extraction.
    public var patchStride: (Int, Int)

    /// Initial embedding dimension.
    public var embedDim: Int

    /// Number of blocks in each Swin stage.
    public var depths: [Int]

    /// Number of attention heads in each stage.
    public var numHeads: [Int]

    /// Window size for local attention.
    public var windowSize: Int

    /// MLP expansion ratio.
    public var mlpRatio: Float

    /// Whether to add bias to QKV projection.
    public var qkvBias: Bool

    /// Dropout rate.
    public var dropRate: Float

    /// Attention dropout rate.
    public var attnDropRate: Float

    /// Stochastic depth rate.
    public var dropPathRate: Float

    /// Output hidden size before projection.
    public var hiddenSize: Int

    // MARK: - Fusion

    /// Enable fusion for variable-length audio.
    public var enableFusion: Bool

    /// Fusion type (e.g., "aff_2d").
    public var fusionType: String

    /// Target spectrogram size (256x256 after reshaping).
    public var specSize: Int

    /// Frequency ratio for reshaping.
    public var freqRatio: Int

    /// Maximum audio length in seconds.
    public var maxLengthS: Float

    /// Creates an audio config with default values (HTSAT-tiny).
    public init(
        sampleRate: Int = 48000,
        nMels: Int = 64,
        nFFT: Int = 1024,
        hopLength: Int = 480,
        windowLength: Int = 1024,
        patchSize: Int = 4,
        patchStride: (Int, Int) = (4, 4),
        embedDim: Int = 96,
        depths: [Int] = [2, 2, 6, 2],
        numHeads: [Int] = [4, 8, 16, 32],
        windowSize: Int = 8,
        mlpRatio: Float = 4.0,
        qkvBias: Bool = true,
        dropRate: Float = 0.0,
        attnDropRate: Float = 0.0,
        dropPathRate: Float = 0.1,
        hiddenSize: Int = 768,
        enableFusion: Bool = true,
        fusionType: String = "aff_2d",
        specSize: Int = 256,
        freqRatio: Int = 4,
        maxLengthS: Float = 10.0
    ) {
        self.sampleRate = sampleRate
        self.nMels = nMels
        self.nFFT = nFFT
        self.hopLength = hopLength
        self.windowLength = windowLength
        self.patchSize = patchSize
        self.patchStride = patchStride
        self.embedDim = embedDim
        self.depths = depths
        self.numHeads = numHeads
        self.windowSize = windowSize
        self.mlpRatio = mlpRatio
        self.qkvBias = qkvBias
        self.dropRate = dropRate
        self.attnDropRate = attnDropRate
        self.dropPathRate = dropPathRate
        self.hiddenSize = hiddenSize
        self.enableFusion = enableFusion
        self.fusionType = fusionType
        self.specSize = specSize
        self.freqRatio = freqRatio
        self.maxLengthS = maxLengthS
    }

    // MARK: - Codable

    enum CodingKeys: String, CodingKey {
        case sampleRate = "sample_rate"
        case nMels = "n_mels"
        case nFFT = "n_fft"
        case hopLength = "hop_length"
        case windowLength = "window_length"
        case patchSize = "patch_size"
        case patchStride = "patch_stride"
        case embedDim = "embed_dim"
        case depths
        case numHeads = "num_heads"
        case windowSize = "window_size"
        case mlpRatio = "mlp_ratio"
        case qkvBias = "qkv_bias"
        case dropRate = "drop_rate"
        case attnDropRate = "attn_drop_rate"
        case dropPathRate = "drop_path_rate"
        case hiddenSize = "hidden_size"
        case enableFusion = "enable_fusion"
        case fusionType = "fusion_type"
        case specSize = "spec_size"
        case freqRatio = "freq_ratio"
        case maxLengthS = "max_length_s"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        sampleRate = try container.decodeIfPresent(Int.self, forKey: .sampleRate) ?? 48000
        nMels = try container.decodeIfPresent(Int.self, forKey: .nMels) ?? 64
        nFFT = try container.decodeIfPresent(Int.self, forKey: .nFFT) ?? 1024
        hopLength = try container.decodeIfPresent(Int.self, forKey: .hopLength) ?? 480
        windowLength = try container.decodeIfPresent(Int.self, forKey: .windowLength) ?? 1024
        patchSize = try container.decodeIfPresent(Int.self, forKey: .patchSize) ?? 4

        // Handle patch_stride as array or single int
        if let strideArray = try? container.decode([Int].self, forKey: .patchStride) {
            patchStride = (strideArray[0], strideArray.count > 1 ? strideArray[1] : strideArray[0])
        } else if let strideInt = try? container.decode(Int.self, forKey: .patchStride) {
            patchStride = (strideInt, strideInt)
        } else {
            patchStride = (4, 4)
        }

        embedDim = try container.decodeIfPresent(Int.self, forKey: .embedDim) ?? 96
        depths = try container.decodeIfPresent([Int].self, forKey: .depths) ?? [2, 2, 6, 2]
        numHeads = try container.decodeIfPresent([Int].self, forKey: .numHeads) ?? [4, 8, 16, 32]
        windowSize = try container.decodeIfPresent(Int.self, forKey: .windowSize) ?? 8
        mlpRatio = try container.decodeIfPresent(Float.self, forKey: .mlpRatio) ?? 4.0
        qkvBias = try container.decodeIfPresent(Bool.self, forKey: .qkvBias) ?? true
        dropRate = try container.decodeIfPresent(Float.self, forKey: .dropRate) ?? 0.0
        attnDropRate = try container.decodeIfPresent(Float.self, forKey: .attnDropRate) ?? 0.0
        dropPathRate = try container.decodeIfPresent(Float.self, forKey: .dropPathRate) ?? 0.1
        hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 768
        enableFusion = try container.decodeIfPresent(Bool.self, forKey: .enableFusion) ?? true
        fusionType = try container.decodeIfPresent(String.self, forKey: .fusionType) ?? "aff_2d"
        specSize = try container.decodeIfPresent(Int.self, forKey: .specSize) ?? 256
        freqRatio = try container.decodeIfPresent(Int.self, forKey: .freqRatio) ?? 4
        maxLengthS = try container.decodeIfPresent(Float.self, forKey: .maxLengthS) ?? 10.0
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)

        try container.encode(sampleRate, forKey: .sampleRate)
        try container.encode(nMels, forKey: .nMels)
        try container.encode(nFFT, forKey: .nFFT)
        try container.encode(hopLength, forKey: .hopLength)
        try container.encode(windowLength, forKey: .windowLength)
        try container.encode(patchSize, forKey: .patchSize)
        try container.encode([patchStride.0, patchStride.1], forKey: .patchStride)
        try container.encode(embedDim, forKey: .embedDim)
        try container.encode(depths, forKey: .depths)
        try container.encode(numHeads, forKey: .numHeads)
        try container.encode(windowSize, forKey: .windowSize)
        try container.encode(mlpRatio, forKey: .mlpRatio)
        try container.encode(qkvBias, forKey: .qkvBias)
        try container.encode(dropRate, forKey: .dropRate)
        try container.encode(attnDropRate, forKey: .attnDropRate)
        try container.encode(dropPathRate, forKey: .dropPathRate)
        try container.encode(hiddenSize, forKey: .hiddenSize)
        try container.encode(enableFusion, forKey: .enableFusion)
        try container.encode(fusionType, forKey: .fusionType)
        try container.encode(specSize, forKey: .specSize)
        try container.encode(freqRatio, forKey: .freqRatio)
        try container.encode(maxLengthS, forKey: .maxLengthS)
    }

    /// Computed final dimension after all stages.
    /// Each stage (except last) doubles the dimension via patch merging.
    public var finalDim: Int {
        var dim = embedDim
        for i in 0..<(depths.count - 1) {
            dim *= 2
        }
        return dim
    }

    /// Maximum number of samples for the configured duration.
    public var maxSamples: Int {
        Int(maxLengthS * Float(sampleRate))
    }
}

// MARK: - Text Encoder Configuration

/// Configuration for the RoBERTa text encoder.
public struct CLAPTextConfig: Sendable, Codable {
    /// Size of vocabulary.
    public var vocabSize: Int

    /// Hidden dimension.
    public var hiddenSize: Int

    /// Number of transformer layers.
    public var numHiddenLayers: Int

    /// Number of attention heads.
    public var numAttentionHeads: Int

    /// FFN intermediate dimension.
    public var intermediateSize: Int

    /// Activation function.
    public var hiddenAct: String

    /// Dropout probability.
    public var hiddenDropoutProb: Float

    /// Attention dropout probability.
    public var attentionProbsDropoutProb: Float

    /// Maximum sequence length.
    public var maxPositionEmbeddings: Int

    /// Number of token type IDs.
    public var typeVocabSize: Int

    /// Layer normalization epsilon.
    public var layerNormEps: Float

    /// Padding token ID.
    public var padTokenId: Int

    /// Creates a text config with default values (RoBERTa-base).
    public init(
        vocabSize: Int = 50265,
        hiddenSize: Int = 768,
        numHiddenLayers: Int = 12,
        numAttentionHeads: Int = 12,
        intermediateSize: Int = 3072,
        hiddenAct: String = "gelu",
        hiddenDropoutProb: Float = 0.1,
        attentionProbsDropoutProb: Float = 0.1,
        maxPositionEmbeddings: Int = 514,
        typeVocabSize: Int = 1,
        layerNormEps: Float = 1e-5,
        padTokenId: Int = 1
    ) {
        self.vocabSize = vocabSize
        self.hiddenSize = hiddenSize
        self.numHiddenLayers = numHiddenLayers
        self.numAttentionHeads = numAttentionHeads
        self.intermediateSize = intermediateSize
        self.hiddenAct = hiddenAct
        self.hiddenDropoutProb = hiddenDropoutProb
        self.attentionProbsDropoutProb = attentionProbsDropoutProb
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.typeVocabSize = typeVocabSize
        self.layerNormEps = layerNormEps
        self.padTokenId = padTokenId
    }

    enum CodingKeys: String, CodingKey {
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case intermediateSize = "intermediate_size"
        case hiddenAct = "hidden_act"
        case hiddenDropoutProb = "hidden_dropout_prob"
        case attentionProbsDropoutProb = "attention_probs_dropout_prob"
        case maxPositionEmbeddings = "max_position_embeddings"
        case typeVocabSize = "type_vocab_size"
        case layerNormEps = "layer_norm_eps"
        case padTokenId = "pad_token_id"
    }

    /// Head dimension (hidden_size / num_attention_heads).
    public var headDim: Int {
        hiddenSize / numAttentionHeads
    }
}

// MARK: - Full CLAP Configuration

/// Full CLAP model configuration.
public struct CLAPConfig: Sendable, Codable {
    /// Audio encoder configuration.
    public var audio: CLAPAudioConfig

    /// Text encoder configuration.
    public var text: CLAPTextConfig

    /// Shared projection dimension.
    public var projectionDim: Int

    /// Initial value for logit scale (log(14.2857) ≈ 2.6592).
    public var logitScaleInit: Float

    /// Creates a CLAP config with default values.
    public init(
        audio: CLAPAudioConfig = CLAPAudioConfig(),
        text: CLAPTextConfig = CLAPTextConfig(),
        projectionDim: Int = 512,
        logitScaleInit: Float = 2.6592
    ) {
        self.audio = audio
        self.text = text
        self.projectionDim = projectionDim
        self.logitScaleInit = logitScaleInit
    }

    enum CodingKeys: String, CodingKey {
        case audio
        case text
        case projectionDim = "projection_dim"
        case logitScaleInit = "logit_scale_init"
    }

    // MARK: - Preset Configurations

    /// HTSAT-tiny configuration (laion/clap-htsat-fused).
    public static func htsatTiny() -> CLAPConfig {
        CLAPConfig(
            audio: CLAPAudioConfig(embedDim: 96, hiddenSize: 768),
            text: CLAPTextConfig()
        )
    }

    /// HTSAT-base configuration (larger_clap_*).
    public static func htsatBase() -> CLAPConfig {
        CLAPConfig(
            audio: CLAPAudioConfig(embedDim: 128, hiddenSize: 1024),
            text: CLAPTextConfig()
        )
    }

    /// Load configuration from JSON file.
    public static func load(from url: URL) throws -> CLAPConfig {
        let data = try Data(contentsOf: url)
        let decoder = JSONDecoder()
        return try decoder.decode(CLAPConfig.self, from: data)
    }

    /// Save configuration to JSON file.
    public func save(to url: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(self)
        try data.write(to: url)
    }
}
