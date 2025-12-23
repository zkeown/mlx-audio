// BanquetConfig.swift
// Configuration for Banquet query-based source separation model.

import Foundation

/// Configuration for the Banquet query-based source separation model.
///
/// Banquet uses a reference audio query to extract matching sounds from a mixture.
/// This mirrors the Python implementation for weight compatibility.
public struct BanquetConfig: Codable, Sendable, Equatable {

    // MARK: - Audio Settings

    /// Expected sample rate in Hz.
    public var sampleRate: Int

    /// FFT size for STFT.
    public var nFFT: Int

    /// Hop length for STFT.
    public var hopLength: Int

    /// Window length for STFT (defaults to nFFT if nil).
    public var winLength: Int?

    /// Number of input audio channels (1=mono, 2=stereo).
    public var inChannel: Int

    // MARK: - Band Split Settings

    /// Number of frequency bands for band splitting.
    public var nBands: Int

    /// Band type specification.
    public var bandType: String

    // MARK: - Model Architecture

    /// Embedding dimension for band features.
    public var embDim: Int

    /// Hidden dimension for RNN layers.
    public var rnnDim: Int

    /// Number of sequential band modelling modules.
    public var nSQMModules: Int

    /// MLP hidden dimension for mask estimation.
    public var mlpDim: Int

    /// Whether to use bidirectional RNN.
    public var bidirectional: Bool

    /// RNN type (LSTM or GRU).
    public var rnnType: String

    /// Whether to use complex-valued masks.
    public var complexMask: Bool

    /// Whether to use frequency weights in mask estimation.
    public var useFreqWeights: Bool

    // MARK: - FiLM Settings

    /// Conditioning embedding dimension (PaSST output: 768).
    public var condEmbDim: Int

    /// Whether to use additive modulation (beta).
    public var filmAdditive: Bool

    /// Whether to use multiplicative modulation (gamma).
    public var filmMultiplicative: Bool

    /// Depth of FiLM modulation networks.
    public var filmDepth: Int

    /// Channels per group for GroupNorm.
    public var channelsPerGroup: Int

    // MARK: - Mask Estimation

    /// Hidden activation function.
    public var hiddenActivation: String

    // MARK: - Computed Properties

    /// Number of frequency bins in STFT.
    public var freqBins: Int {
        nFFT / 2 + 1
    }

    /// Effective window length.
    public var effectiveWinLength: Int {
        winLength ?? nFFT
    }

    // MARK: - Initialization

    /// Creates a Banquet configuration with default values matching ev-pre-aug checkpoint.
    public init(
        sampleRate: Int = 44100,
        nFFT: Int = 2048,
        hopLength: Int = 512,
        winLength: Int? = nil,
        inChannel: Int = 2,
        nBands: Int = 64,
        bandType: String = "musical",
        embDim: Int = 128,
        rnnDim: Int = 256,
        nSQMModules: Int = 12,
        mlpDim: Int = 512,
        bidirectional: Bool = true,
        rnnType: String = "LSTM",
        complexMask: Bool = true,
        useFreqWeights: Bool = true,
        condEmbDim: Int = 768,
        filmAdditive: Bool = true,
        filmMultiplicative: Bool = true,
        filmDepth: Int = 2,
        channelsPerGroup: Int = 16,
        hiddenActivation: String = "Tanh"
    ) {
        self.sampleRate = sampleRate
        self.nFFT = nFFT
        self.hopLength = hopLength
        self.winLength = winLength
        self.inChannel = inChannel
        self.nBands = nBands
        self.bandType = bandType
        self.embDim = embDim
        self.rnnDim = rnnDim
        self.nSQMModules = nSQMModules
        self.mlpDim = mlpDim
        self.bidirectional = bidirectional
        self.rnnType = rnnType
        self.complexMask = complexMask
        self.useFreqWeights = useFreqWeights
        self.condEmbDim = condEmbDim
        self.filmAdditive = filmAdditive
        self.filmMultiplicative = filmMultiplicative
        self.filmDepth = filmDepth
        self.channelsPerGroup = channelsPerGroup
        self.hiddenActivation = hiddenActivation
    }

    // MARK: - Coding Keys

    enum CodingKeys: String, CodingKey {
        case sampleRate = "sample_rate"
        case nFFT = "n_fft"
        case hopLength = "hop_length"
        case winLength = "win_length"
        case inChannel = "in_channel"
        case nBands = "n_bands"
        case bandType = "band_type"
        case embDim = "emb_dim"
        case rnnDim = "rnn_dim"
        case nSQMModules = "n_sqm_modules"
        case mlpDim = "mlp_dim"
        case bidirectional
        case rnnType = "rnn_type"
        case complexMask = "complex_mask"
        case useFreqWeights = "use_freq_weights"
        case condEmbDim = "cond_emb_dim"
        case filmAdditive = "film_additive"
        case filmMultiplicative = "film_multiplicative"
        case filmDepth = "film_depth"
        case channelsPerGroup = "channels_per_group"
        case hiddenActivation = "hidden_activation"
    }

    // MARK: - Preset Configurations

    /// Default Banquet configuration (ev-pre-aug checkpoint).
    public static func banquet() -> BanquetConfig {
        BanquetConfig()
    }
}

/// Configuration for the PaSST query encoder.
///
/// PaSST (Patchout Spectrogram Transformer) encodes reference audio
/// into a 768-dimensional embedding for query-based separation.
public struct PaSSTConfig: Codable, Sendable {

    // MARK: - Audio Settings

    /// PaSST expected sample rate (32kHz).
    public var sampleRate: Int

    /// Number of mel filterbank bins.
    public var nMels: Int

    /// FFT size for mel spectrogram.
    public var nFFT: Int

    /// Hop length for mel spectrogram.
    public var hopLength: Int

    /// Window length for mel spectrogram.
    public var winLength: Int

    /// Number of expected time frames.
    public var nTimeFrames: Int

    // MARK: - Model Architecture

    /// Patch size for patch embedding.
    public var patchSize: (Int, Int)

    /// Embedding dimension.
    public var embedDim: Int

    /// Number of attention heads.
    public var numHeads: Int

    /// Number of transformer layers.
    public var numLayers: Int

    /// MLP ratio (hidden_dim = embed_dim * mlp_ratio).
    public var mlpRatio: Float

    /// Dropout rate.
    public var dropout: Float

    /// Attention dropout rate.
    public var attentionDropout: Float

    // MARK: - Initialization

    /// Creates a PaSST configuration with default values.
    public init(
        sampleRate: Int = 32000,
        nMels: Int = 128,
        nFFT: Int = 1024,
        hopLength: Int = 320,
        winLength: Int = 800,
        nTimeFrames: Int = 998,
        patchSize: (Int, Int) = (16, 16),
        embedDim: Int = 768,
        numHeads: Int = 12,
        numLayers: Int = 12,
        mlpRatio: Float = 4.0,
        dropout: Float = 0.0,
        attentionDropout: Float = 0.0
    ) {
        self.sampleRate = sampleRate
        self.nMels = nMels
        self.nFFT = nFFT
        self.hopLength = hopLength
        self.winLength = winLength
        self.nTimeFrames = nTimeFrames
        self.patchSize = patchSize
        self.embedDim = embedDim
        self.numHeads = numHeads
        self.numLayers = numLayers
        self.mlpRatio = mlpRatio
        self.dropout = dropout
        self.attentionDropout = attentionDropout
    }

    // MARK: - Computed Properties

    /// Number of frequency patches.
    public var numFreqPatches: Int {
        nMels / patchSize.0
    }

    /// Number of time patches.
    public var numTimePatches: Int {
        nTimeFrames / patchSize.1
    }

    /// Total number of patches.
    public var numPatches: Int {
        numFreqPatches * numTimePatches
    }

    // MARK: - Coding Keys

    enum CodingKeys: String, CodingKey {
        case sampleRate = "sample_rate"
        case nMels = "n_mels"
        case nFFT = "n_fft"
        case hopLength = "hop_length"
        case winLength = "win_length"
        case nTimeFrames = "n_time_frames"
        case patchSize = "patch_size"
        case embedDim = "embed_dim"
        case numHeads = "num_heads"
        case numLayers = "num_layers"
        case mlpRatio = "mlp_ratio"
        case dropout
        case attentionDropout = "attention_dropout"
    }

    // MARK: - Custom Coding for Tuple

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        sampleRate = try container.decode(Int.self, forKey: .sampleRate)
        nMels = try container.decode(Int.self, forKey: .nMels)
        nFFT = try container.decode(Int.self, forKey: .nFFT)
        hopLength = try container.decode(Int.self, forKey: .hopLength)
        winLength = try container.decode(Int.self, forKey: .winLength)
        nTimeFrames = try container.decode(Int.self, forKey: .nTimeFrames)
        let patchSizeArray = try container.decode([Int].self, forKey: .patchSize)
        patchSize = (patchSizeArray[0], patchSizeArray[1])
        embedDim = try container.decode(Int.self, forKey: .embedDim)
        numHeads = try container.decode(Int.self, forKey: .numHeads)
        numLayers = try container.decode(Int.self, forKey: .numLayers)
        mlpRatio = try container.decode(Float.self, forKey: .mlpRatio)
        dropout = try container.decode(Float.self, forKey: .dropout)
        attentionDropout = try container.decode(Float.self, forKey: .attentionDropout)
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(sampleRate, forKey: .sampleRate)
        try container.encode(nMels, forKey: .nMels)
        try container.encode(nFFT, forKey: .nFFT)
        try container.encode(hopLength, forKey: .hopLength)
        try container.encode(winLength, forKey: .winLength)
        try container.encode(nTimeFrames, forKey: .nTimeFrames)
        try container.encode([patchSize.0, patchSize.1], forKey: .patchSize)
        try container.encode(embedDim, forKey: .embedDim)
        try container.encode(numHeads, forKey: .numHeads)
        try container.encode(numLayers, forKey: .numLayers)
        try container.encode(mlpRatio, forKey: .mlpRatio)
        try container.encode(dropout, forKey: .dropout)
        try container.encode(attentionDropout, forKey: .attentionDropout)
    }

    // MARK: - Preset Configurations

    /// Default PaSST configuration (openmic architecture).
    public static func passt() -> PaSSTConfig {
        PaSSTConfig()
    }
}

// MARK: - Equatable

extension PaSSTConfig: Equatable {
    public static func == (lhs: PaSSTConfig, rhs: PaSSTConfig) -> Bool {
        lhs.sampleRate == rhs.sampleRate &&
        lhs.nMels == rhs.nMels &&
        lhs.nFFT == rhs.nFFT &&
        lhs.hopLength == rhs.hopLength &&
        lhs.winLength == rhs.winLength &&
        lhs.nTimeFrames == rhs.nTimeFrames &&
        lhs.patchSize.0 == rhs.patchSize.0 &&
        lhs.patchSize.1 == rhs.patchSize.1 &&
        lhs.embedDim == rhs.embedDim &&
        lhs.numHeads == rhs.numHeads &&
        lhs.numLayers == rhs.numLayers &&
        lhs.mlpRatio == rhs.mlpRatio &&
        lhs.dropout == rhs.dropout &&
        lhs.attentionDropout == rhs.attentionDropout
    }
}

/// Band specification for frequency band splitting.
public struct BandSpec: Codable, Sendable, Equatable {
    /// Start frequency bin (inclusive).
    public var startBin: Int

    /// End frequency bin (exclusive).
    public var endBin: Int

    /// Bandwidth in bins.
    public var bandwidth: Int {
        endBin - startBin
    }

    public init(startBin: Int, endBin: Int) {
        self.startBin = startBin
        self.endBin = endBin
    }

    enum CodingKeys: String, CodingKey {
        case startBin = "start_bin"
        case endBin = "end_bin"
    }
}
