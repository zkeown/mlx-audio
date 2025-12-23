// WhisperConfig.swift
// Configuration for Whisper speech recognition model.

import Foundation

/// Configuration for Whisper speech recognition model.
///
/// Whisper is an encoder-decoder transformer model trained on multilingual
/// speech recognition, translation, language identification, and voice activity
/// detection tasks.
public struct WhisperConfig: Codable, Sendable {

    // MARK: - Audio Processing

    /// Number of mel filterbank bins (80 for v1/v2, 128 for v3).
    public var nMels: Int

    /// Expected audio sample rate in Hz.
    public var sampleRate: Int

    /// FFT window size (400 = 25ms at 16kHz).
    public var nFft: Int

    /// STFT hop length (160 = 10ms at 16kHz).
    public var hopLength: Int

    /// Audio chunk length in seconds.
    public var chunkLength: Int

    // MARK: - Encoder Architecture

    /// Audio context length (encoder output frames).
    public var nAudioCtx: Int

    /// Encoder hidden dimension.
    public var nAudioState: Int

    /// Number of encoder attention heads.
    public var nAudioHead: Int

    /// Number of encoder transformer layers.
    public var nAudioLayer: Int

    // MARK: - Decoder Architecture

    /// Text context length (max tokens).
    public var nTextCtx: Int

    /// Decoder hidden dimension.
    public var nTextState: Int

    /// Number of decoder attention heads.
    public var nTextHead: Int

    /// Number of decoder transformer layers.
    public var nTextLayer: Int

    // MARK: - Vocabulary

    /// Vocabulary size.
    public var nVocab: Int

    // MARK: - Computed Properties

    /// Number of audio samples per chunk.
    public var nSamples: Int {
        chunkLength * sampleRate
    }

    /// Number of mel frames per chunk (before conv downsampling).
    public var nFrames: Int {
        nSamples / hopLength
    }

    /// Whether this is a multilingual model.
    public var isMultilingual: Bool {
        nVocab >= 51865
    }

    /// Whether this uses v3 architecture (128 mel bins).
    public var isV3: Bool {
        nMels == 128
    }

    // MARK: - Constants

    /// Standard sample rate for Whisper models.
    public static let whisperSampleRate = 16000

    /// Standard FFT window size.
    public static let whisperNFft = 400

    /// Standard hop length.
    public static let whisperHopLength = 160

    /// Standard chunk length in seconds.
    public static let whisperChunkLength = 30

    /// Standard mel bins for v1/v2.
    public static let whisperNMels = 80

    /// Mel bins for v3.
    public static let whisperV3NMels = 128

    /// Audio context length.
    public static let whisperNAudioCtx = 1500

    /// Text context length.
    public static let whisperNTextCtx = 448

    /// Multilingual vocabulary size.
    public static let whisperNVocab = 51865

    /// English-only vocabulary size.
    public static let whisperNVocabEn = 51864

    // MARK: - Initialization

    public init(
        nMels: Int = whisperNMels,
        sampleRate: Int = whisperSampleRate,
        nFft: Int = whisperNFft,
        hopLength: Int = whisperHopLength,
        chunkLength: Int = whisperChunkLength,
        nAudioCtx: Int = whisperNAudioCtx,
        nAudioState: Int = 384,
        nAudioHead: Int = 6,
        nAudioLayer: Int = 4,
        nTextCtx: Int = whisperNTextCtx,
        nTextState: Int = 384,
        nTextHead: Int = 6,
        nTextLayer: Int = 4,
        nVocab: Int = whisperNVocab
    ) {
        self.nMels = nMels
        self.sampleRate = sampleRate
        self.nFft = nFft
        self.hopLength = hopLength
        self.chunkLength = chunkLength
        self.nAudioCtx = nAudioCtx
        self.nAudioState = nAudioState
        self.nAudioHead = nAudioHead
        self.nAudioLayer = nAudioLayer
        self.nTextCtx = nTextCtx
        self.nTextState = nTextState
        self.nTextHead = nTextHead
        self.nTextLayer = nTextLayer
        self.nVocab = nVocab
    }

    // MARK: - Factory Methods

    /// Whisper tiny configuration (39M parameters).
    public static func tiny() -> WhisperConfig {
        WhisperConfig(
            nMels: whisperNMels,
            nAudioState: 384,
            nAudioHead: 6,
            nAudioLayer: 4,
            nTextState: 384,
            nTextHead: 6,
            nTextLayer: 4
        )
    }

    /// Whisper tiny.en configuration (English-only, 39M parameters).
    public static func tinyEn() -> WhisperConfig {
        var config = tiny()
        config.nVocab = whisperNVocabEn
        return config
    }

    /// Whisper base configuration (74M parameters).
    public static func base() -> WhisperConfig {
        WhisperConfig(
            nMels: whisperNMels,
            nAudioState: 512,
            nAudioHead: 8,
            nAudioLayer: 6,
            nTextState: 512,
            nTextHead: 8,
            nTextLayer: 6
        )
    }

    /// Whisper base.en configuration (English-only, 74M parameters).
    public static func baseEn() -> WhisperConfig {
        var config = base()
        config.nVocab = whisperNVocabEn
        return config
    }

    /// Whisper small configuration (244M parameters).
    public static func small() -> WhisperConfig {
        WhisperConfig(
            nMels: whisperNMels,
            nAudioState: 768,
            nAudioHead: 12,
            nAudioLayer: 12,
            nTextState: 768,
            nTextHead: 12,
            nTextLayer: 12
        )
    }

    /// Whisper small.en configuration (English-only, 244M parameters).
    public static func smallEn() -> WhisperConfig {
        var config = small()
        config.nVocab = whisperNVocabEn
        return config
    }

    /// Whisper medium configuration (769M parameters).
    public static func medium() -> WhisperConfig {
        WhisperConfig(
            nMels: whisperNMels,
            nAudioState: 1024,
            nAudioHead: 16,
            nAudioLayer: 24,
            nTextState: 1024,
            nTextHead: 16,
            nTextLayer: 24
        )
    }

    /// Whisper medium.en configuration (English-only, 769M parameters).
    public static func mediumEn() -> WhisperConfig {
        var config = medium()
        config.nVocab = whisperNVocabEn
        return config
    }

    /// Whisper large configuration (1.5B parameters).
    public static func large() -> WhisperConfig {
        WhisperConfig(
            nMels: whisperNMels,
            nAudioState: 1280,
            nAudioHead: 20,
            nAudioLayer: 32,
            nTextState: 1280,
            nTextHead: 20,
            nTextLayer: 32
        )
    }

    /// Whisper large-v2 configuration (1.5B parameters, improved training).
    public static func largeV2() -> WhisperConfig {
        large()
    }

    /// Whisper large-v3 configuration (1.5B parameters, 128 mel bins).
    public static func largeV3() -> WhisperConfig {
        var config = large()
        config.nMels = whisperV3NMels
        return config
    }

    /// Whisper large-v3-turbo configuration (809M parameters).
    ///
    /// Uses large-v3 encoder (32 layers) with only 4 decoder layers,
    /// providing ~8x faster inference with minimal quality loss.
    public static func largeV3Turbo() -> WhisperConfig {
        WhisperConfig(
            nMels: whisperV3NMels,
            nAudioState: 1280,
            nAudioHead: 20,
            nAudioLayer: 32,
            nTextState: 1280,
            nTextHead: 20,
            nTextLayer: 4  // Reduced from 32 to 4
        )
    }

    /// Alias for largeV3Turbo.
    public static func turbo() -> WhisperConfig {
        largeV3Turbo()
    }

    /// Create config from model name.
    ///
    /// - Parameter name: Model name (e.g., "tiny", "base.en", "large-v3-turbo")
    /// - Returns: WhisperConfig for the specified model
    /// - Throws: WhisperError if model name is not recognized
    public static func fromName(_ name: String) throws -> WhisperConfig {
        // Normalize name
        var normalized = name.lowercased()
            .replacingOccurrences(of: "-", with: "_")
            .replacingOccurrences(of: ".", with: "_")

        // Remove common prefixes
        let prefixes = ["whisper_", "openai_whisper_", "openai/whisper_"]
        for prefix in prefixes {
            if normalized.hasPrefix(prefix) {
                normalized = String(normalized.dropFirst(prefix.count))
            }
        }

        switch normalized {
        case "tiny":
            return tiny()
        case "tiny_en":
            return tinyEn()
        case "base":
            return base()
        case "base_en":
            return baseEn()
        case "small":
            return small()
        case "small_en":
            return smallEn()
        case "medium":
            return medium()
        case "medium_en":
            return mediumEn()
        case "large", "large_v1":
            return large()
        case "large_v2":
            return largeV2()
        case "large_v3":
            return largeV3()
        case "large_v3_turbo", "turbo":
            return largeV3Turbo()
        default:
            throw WhisperError.unknownModel(name)
        }
    }

    /// Load config from a JSON file.
    public static func fromFile(_ url: URL) throws -> WhisperConfig {
        let data = try Data(contentsOf: url)
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode(WhisperConfig.self, from: data)
    }

    /// Save config to a JSON file.
    public func save(to url: URL) throws {
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(self)
        try data.write(to: url)
    }

    // MARK: - Codable

    enum CodingKeys: String, CodingKey {
        case nMels = "n_mels"
        case sampleRate = "sample_rate"
        case nFft = "n_fft"
        case hopLength = "hop_length"
        case chunkLength = "chunk_length"
        case nAudioCtx = "n_audio_ctx"
        case nAudioState = "n_audio_state"
        case nAudioHead = "n_audio_head"
        case nAudioLayer = "n_audio_layer"
        case nTextCtx = "n_text_ctx"
        case nTextState = "n_text_state"
        case nTextHead = "n_text_head"
        case nTextLayer = "n_text_layer"
        case nVocab = "n_vocab"
    }
}

// MARK: - Errors

/// Errors that can occur when working with Whisper.
public enum WhisperError: Error, LocalizedError {
    case unknownModel(String)
    case invalidConfig(String)
    case weightLoadingFailed(String)
    case tokenizerError(String)
    case audioProcessingError(String)
    case decodingError(String)

    public var errorDescription: String? {
        switch self {
        case .unknownModel(let name):
            let available = [
                "tiny", "tiny.en", "base", "base.en", "small", "small.en",
                "medium", "medium.en", "large", "large-v2", "large-v3", "large-v3-turbo"
            ].joined(separator: ", ")
            return "Unknown Whisper model: '\(name)'. Available: \(available)"
        case .invalidConfig(let message):
            return "Invalid configuration: \(message)"
        case .weightLoadingFailed(let message):
            return "Failed to load weights: \(message)"
        case .tokenizerError(let message):
            return "Tokenizer error: \(message)"
        case .audioProcessingError(let message):
            return "Audio processing error: \(message)"
        case .decodingError(let message):
            return "Decoding error: \(message)"
        }
    }
}
