// WhisperModel.swift
// Whisper speech recognition model.

import Foundation
import MLX
import MLXNN

/// Whisper speech recognition model.
///
/// Whisper is an encoder-decoder transformer trained on multilingual
/// speech recognition, translation, language identification, and
/// voice activity detection tasks.
///
/// Architecture:
/// - Encoder: Log-mel spectrogram -> Conv1d -> Transformer -> Audio features
/// - Decoder: Tokens -> Transformer (with cross-attention) -> Logits
///
/// The model supports:
/// - Speech transcription (multilingual)
/// - Speech translation (to English)
/// - Language identification
/// - Timestamp prediction
public class WhisperModel: Module {

    /// Model configuration.
    public let config: WhisperConfig

    /// Audio encoder.
    @ModuleInfo var encoder: AudioEncoder

    /// Text decoder.
    @ModuleInfo var decoder: TextDecoder

    /// Initialize Whisper model.
    ///
    /// - Parameter config: Model configuration. If nil, uses tiny config.
    public init(config: WhisperConfig? = nil) {
        self.config = config ?? WhisperConfig.tiny()

        self._encoder.wrappedValue = AudioEncoder(config: self.config)
        self._decoder.wrappedValue = TextDecoder(config: self.config)
    }

    /// Encode audio to features.
    ///
    /// - Parameter mel: Log-mel spectrogram [B, nMels, T] or [nMels, T]
    /// - Returns: Audio features [B, T//2, nState]
    public func encode(_ mel: MLXArray) -> MLXArray {
        encoder(mel)
    }

    /// Decode tokens to logits.
    ///
    /// - Parameters:
    ///   - tokens: Input token IDs [B, T]
    ///   - audioFeatures: Encoder output [B, S, D]
    ///   - kvCache: Cached key/value pairs for incremental decoding
    /// - Returns: Tuple of logits [B, T, nVocab] and updated KV cache
    public func decode(
        tokens: MLXArray,
        audioFeatures: MLXArray,
        kvCache: [(MLXArray, MLXArray)]? = nil
    ) -> (MLXArray, [(MLXArray, MLXArray)]) {
        decoder(tokens: tokens, audioFeatures: audioFeatures, kvCache: kvCache)
    }

    /// Optimized decode using pre-allocated KV cache.
    ///
    /// This version uses O(1) cache updates instead of O(n) concatenation.
    ///
    /// - Parameters:
    ///   - tokens: Input token IDs [B, T]
    ///   - audioFeatures: Encoder output [B, S, D]
    ///   - selfAttnKVs: Pre-computed self-attention K/V per layer
    ///   - crossAttnKVs: Cached cross-attention K/V per layer (fixed for sequence)
    ///   - offset: Position offset for positional embeddings
    /// - Returns: Tuple of (logits, newKs, newVs, crossKs, crossVs)
    public func decodeOptimized(
        tokens: MLXArray,
        audioFeatures: MLXArray,
        selfAttnKVs: [(MLXArray, MLXArray)]?,
        crossAttnKVs: [(MLXArray, MLXArray)]?,
        offset: Int
    ) -> (logits: MLXArray, newKs: [MLXArray], newVs: [MLXArray], crossKs: [MLXArray], crossVs: [MLXArray]) {
        decoder.forwardOptimized(
            tokens: tokens,
            audioFeatures: audioFeatures,
            selfAttnKVs: selfAttnKVs,
            crossAttnKVs: crossAttnKVs,
            offset: offset
        )
    }

    /// Full forward pass.
    ///
    /// - Parameters:
    ///   - mel: Log-mel spectrogram [B, nMels, T]
    ///   - tokens: Input token IDs [B, L]
    /// - Returns: Logits [B, L, nVocab]
    public func callAsFunction(mel: MLXArray, tokens: MLXArray) -> MLXArray {
        let audioFeatures = encode(mel)
        let (logits, _) = decode(tokens: tokens, audioFeatures: audioFeatures)
        return logits
    }

    /// Detect the language of the audio.
    ///
    /// - Parameters:
    ///   - mel: Log-mel spectrogram [B, nMels, T] or [nMels, T]
    ///   - tokenizer: Whisper tokenizer
    /// - Returns: Tuple of (languageCode, probability)
    public func detectLanguage(
        mel: MLXArray,
        tokenizer: WhisperTokenizer
    ) -> (String, Float) {
        // Encode audio
        let audioFeatures = encode(mel)

        // Get initial tokens (just SOT)
        let sotToken = MLXArray([Int32(tokenizer.sot)]).reshaped([1, 1])

        // Get logits for next token
        let (logits, _) = decode(tokens: sotToken, audioFeatures: audioFeatures)

        // Get probabilities over language tokens
        let langTokens = tokenizer.allLanguageTokens
        let langTokenIndices = MLXArray(langTokens.map { Int32($0) })
        let langLogits = logits[0, 0, langTokenIndices]
        let langProbs = softmax(langLogits, axis: -1)

        // Find best language
        let bestIdx = Int(argMax(langProbs).item(Int.self))
        let bestProb = langProbs[bestIdx].item(Float.self)

        // Get language code from token
        let langCodes = Array(WhisperTokenizer.languages.keys)
        let bestLang = langCodes[bestIdx]

        return (bestLang, bestProb)
    }

    /// Get model dimensions.
    public var dims: [String: Int] {
        [
            "n_mels": config.nMels,
            "n_audio_ctx": config.nAudioCtx,
            "n_audio_state": config.nAudioState,
            "n_audio_head": config.nAudioHead,
            "n_audio_layer": config.nAudioLayer,
            "n_text_ctx": config.nTextCtx,
            "n_text_state": config.nTextState,
            "n_text_head": config.nTextHead,
            "n_text_layer": config.nTextLayer,
            "n_vocab": config.nVocab,
        ]
    }

    /// Load pretrained Whisper model from a directory.
    ///
    /// - Parameters:
    ///   - path: Path to model directory containing config.json and model.safetensors
    /// - Returns: Loaded Whisper model
    public static func fromPretrained(path: URL) throws -> WhisperModel {
        // Load config
        let configPath = path.appendingPathComponent("config.json")
        let config: WhisperConfig
        if FileManager.default.fileExists(atPath: configPath.path) {
            config = try WhisperConfig.fromFile(configPath)
        } else {
            // Default to tiny
            config = WhisperConfig.tiny()
        }

        // Create model
        let model = WhisperModel(config: config)

        // Load weights
        let weightsPath = path.appendingPathComponent("model.safetensors")
        if FileManager.default.fileExists(atPath: weightsPath.path) {
            try model.loadWeights(from: weightsPath)
        } else {
            // Try .npz format
            let npzPath = path.appendingPathComponent("weights.npz")
            if FileManager.default.fileExists(atPath: npzPath.path) {
                try model.loadWeights(from: npzPath)
            }
        }

        return model
    }

    /// Load model weights from a file.
    ///
    /// - Parameter url: URL to weights file (.safetensors or .npz)
    public func loadWeights(from url: URL) throws {
        // Load arrays from file
        let arrays = try MLX.loadArrays(url: url)

        // Convert flat dictionary to nested module parameters
        let params = ModuleParameters.unflattened(arrays)

        // Update model with loaded parameters
        try update(parameters: params, verify: .noUnusedKeys)

        // Also handle positional embeddings which are stored as plain arrays
        if let decoderPosEmb = arrays["decoder.positional_embedding"] {
            decoder.positionalEmbedding = decoderPosEmb
        }
        if let encoderPosEmb = arrays["encoder.positional_embedding"] {
            encoder.positionalEmbedding = encoderPosEmb
        }
    }

    /// Save model to directory.
    ///
    /// - Parameter path: Output directory
    public func save(to path: URL) throws {
        try FileManager.default.createDirectory(at: path, withIntermediateDirectories: true)

        // Save config
        let configPath = path.appendingPathComponent("config.json")
        try config.save(to: configPath)

        // Get all parameters as flat dictionary
        var arrays = Dictionary(uniqueKeysWithValues: parameters().flattened())

        // Add positional embeddings
        arrays["encoder.positional_embedding"] = encoder.positionalEmbedding
        arrays["decoder.positional_embedding"] = decoder.positionalEmbedding

        // Save weights
        let weightsPath = path.appendingPathComponent("model.safetensors")
        try MLX.save(arrays: arrays, url: weightsPath)
    }
}

// MARK: - Convenience Type Alias

/// Type alias for backward compatibility.
public typealias Whisper = WhisperModel
