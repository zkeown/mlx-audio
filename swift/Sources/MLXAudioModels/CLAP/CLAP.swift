// CLAP.swift
// CLAP (Contrastive Language-Audio Pretraining) model.
//
// Dual-encoder model that learns joint embeddings for audio and text,
// enabling zero-shot audio classification, audio-text retrieval,
// and audio similarity search.

import Foundation
@preconcurrency import MLX
import MLXNN
import MLXAudioPrimitives

// MARK: - CLAP Projection

/// 2-layer MLP projection head used by CLAP.
public class CLAPProjection: Module, @unchecked Sendable {
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

// MARK: - CLAP Model

/// CLAP model for audio-text embeddings.
///
/// CLAP (Contrastive Language-Audio Pretraining) learns a joint embedding
/// space for audio and text, enabling:
/// - Audio similarity search
/// - Zero-shot audio classification
/// - Audio-text retrieval
///
/// Architecture:
/// - Audio encoder: HTSAT (Hierarchical Token-Semantic Audio Transformer)
/// - Text encoder: RoBERTa-base
/// - Shared projection space: 512 dimensions
/// - Learnable temperature for similarity scaling
public class CLAPModel: Module, @unchecked Sendable {
    /// Model configuration.
    public let config: CLAPConfig

    /// Audio encoder (HTSAT).
    @ModuleInfo(key: "audio_encoder") public var audioEncoder: HTSAT

    /// Audio projection head.
    @ModuleInfo(key: "audio_projection") public var audioProjection: CLAPProjection

    /// Text encoder (RoBERTa + projection).
    @ModuleInfo(key: "text_encoder") public var textEncoder: CLAPTextEncoder

    /// Audio fusion for variable-length audio.
    public var audioFusion: AudioFusion?

    /// Learnable temperature (logit scale).
    public var logitScale: MLXArray

    /// Feature extractor for audio preprocessing.
    public let featureExtractor: CLAPFeatureExtractor

    /// Tokenizer for text preprocessing.
    public var tokenizer: CLAPTokenizer?

    /// Creates a CLAP model with the given configuration.
    public init(config: CLAPConfig = CLAPConfig()) {
        self.config = config

        // Audio encoder
        self._audioEncoder.wrappedValue = HTSAT(config: config.audio)

        // Audio projection
        let audioEncoderDim = config.audio.finalDim
        self._audioProjection.wrappedValue = CLAPProjection(
            inDim: audioEncoderDim,
            outDim: config.projectionDim
        )

        // Text encoder (includes projection)
        self._textEncoder.wrappedValue = CLAPTextEncoder(
            config: config.text,
            projectionDim: config.projectionDim
        )

        // Audio fusion for variable-length
        if config.audio.enableFusion {
            self.audioFusion = AudioFusion(config: config.audio)
        }

        // Learnable temperature
        self.logitScale = MLXArray([config.logitScaleInit])

        // Feature extractor
        self.featureExtractor = CLAPFeatureExtractor.fromConfig(config.audio)

        // Default tokenizer
        self.tokenizer = CLAPTokenizer.createBasic()

        super.init()
    }

    /// Encode audio to embeddings.
    ///
    /// - Parameters:
    ///   - audio: Mel spectrogram [B, C, F, T] where C=1 or 4 for fusion,
    ///            or raw waveform [B, T]
    ///   - normalize: L2 normalize embeddings
    ///   - isLonger: Boolean tensor [B] indicating which samples need fusion
    /// - Returns: Audio embeddings [B, projectionDim]
    public func encodeAudio(
        _ audio: MLXArray,
        normalize: Bool = true,
        isLonger: MLXArray? = nil
    ) throws -> MLXArray {
        var processedAudio = audio
        var longer = isLonger

        // Handle waveform input
        if audio.ndim == 2 {
            // Assume waveform [B, T] - needs mel conversion
            let result = try featureExtractor(audio)
            processedAudio = result.inputFeatures
            longer = result.isLonger
        } else if audio.ndim == 3 {
            // Add channel dim: [B, F, T] -> [B, 1, F, T]
            processedAudio = audio.expandedDimensions(axis: 1)
        }

        // Transpose from [B, C, F, T] to [B, C, T, F] for HTSAT
        // Note: This depends on the expected input format
        // The Python code expects [B, C, F, T], HTSAT internally handles it

        // Use fusion for long audio
        var embeddings: MLXArray
        if let fusion = audioFusion, processedAudio.shape[3] > 1024 {
            embeddings = fusion(processedAudio, encoder: audioEncoder)
        } else {
            embeddings = audioEncoder(processedAudio, isLonger: longer)
        }

        // Project to shared space
        embeddings = audioProjection(embeddings)

        if normalize {
            let normValue = MLX.sqrt(MLX.sum(embeddings * embeddings, axis: -1, keepDims: true))
            embeddings = embeddings / (normValue + 1e-8)
        }

        return embeddings
    }

    /// Encode text to embeddings.
    ///
    /// - Parameters:
    ///   - text: Text string to encode
    ///   - normalize: L2 normalize embeddings
    /// - Returns: Text embeddings [1, projectionDim]
    public func encodeText(_ text: String, normalize: Bool = true) throws -> MLXArray {
        guard let tokenizer = tokenizer else {
            throw CLAPError.tokenizerNotConfigured
        }

        let (inputIds, attentionMask) = tokenizer.encodeWithMask(text)

        let inputIdsTensor = MLXArray(inputIds.map { Int32($0) }).reshaped([1, -1])
        let maskTensor = MLXArray(attentionMask.map { Float($0) }).reshaped([1, -1])

        return try encodeText(inputIdsTensor, attentionMask: maskTensor, normalize: normalize)
    }

    /// Encode text from token IDs.
    ///
    /// - Parameters:
    ///   - inputIds: Token IDs [B, L]
    ///   - attentionMask: Attention mask [B, L] (1=keep, 0=mask)
    ///   - normalize: L2 normalize embeddings
    /// - Returns: Text embeddings [B, projectionDim]
    public func encodeText(
        _ inputIds: MLXArray,
        attentionMask: MLXArray? = nil,
        normalize: Bool = true
    ) throws -> MLXArray {
        let embeddings = textEncoder(
            inputIds,
            attentionMask: attentionMask,
            normalize: normalize
        )
        return embeddings
    }

    /// Compute audio-text similarity matrix.
    ///
    /// - Parameters:
    ///   - audioEmbeds: Audio embeddings [B_a, dim]
    ///   - textEmbeds: Text embeddings [B_t, dim]
    /// - Returns: Similarity matrix [B_a, B_t]
    public func similarity(audioEmbeds: MLXArray, textEmbeds: MLXArray) -> MLXArray {
        let scale = MLX.exp(logitScale)
        return scale * MLX.matmul(audioEmbeds, textEmbeds.transposed())
    }

    /// Full forward pass.
    ///
    /// - Parameters:
    ///   - audio: Mel spectrogram [B, 1, F, T] or waveform [B, T]
    ///   - inputIds: Token IDs [B, L]
    ///   - attentionMask: Attention mask [B, L]
    /// - Returns: Dictionary with embeddings and logits
    public func callAsFunction(
        audio: MLXArray? = nil,
        inputIds: MLXArray? = nil,
        attentionMask: MLXArray? = nil
    ) throws -> CLAPOutput {
        var audioEmbeds: MLXArray? = nil
        var textEmbeds: MLXArray? = nil

        if let audio = audio {
            audioEmbeds = try encodeAudio(audio, normalize: true)
        }

        if let ids = inputIds {
            textEmbeds = try encodeText(ids, attentionMask: attentionMask, normalize: true)
        }

        var logitsPerAudio: MLXArray? = nil
        var logitsPerText: MLXArray? = nil

        if let aEmb = audioEmbeds, let tEmb = textEmbeds {
            logitsPerAudio = similarity(audioEmbeds: aEmb, textEmbeds: tEmb)
            logitsPerText = logitsPerAudio?.transposed()
        }

        return CLAPOutput(
            audioEmbeds: audioEmbeds,
            textEmbeds: textEmbeds,
            logitsPerAudio: logitsPerAudio,
            logitsPerText: logitsPerText
        )
    }

    /// Set tokenizer.
    public func setTokenizer(_ tokenizer: CLAPTokenizer) {
        self.tokenizer = tokenizer
    }
}

// MARK: - CLAP Output

/// Output from CLAP forward pass.
public struct CLAPOutput: Sendable {
    /// Audio embeddings [B, projectionDim]
    public let audioEmbeds: MLXArray?

    /// Text embeddings [B, projectionDim]
    public let textEmbeds: MLXArray?

    /// Audio-to-text similarity logits [B_audio, B_text]
    public let logitsPerAudio: MLXArray?

    /// Text-to-audio similarity logits [B_text, B_audio]
    public let logitsPerText: MLXArray?
}

// MARK: - CLAP Errors

/// Errors that can occur during CLAP operations.
public enum CLAPError: Error, LocalizedError {
    case tokenizerNotConfigured
    case weightLoadingFailed(String)
    case invalidInputShape(String)
    case featureExtractionFailed(String)

    public var errorDescription: String? {
        switch self {
        case .tokenizerNotConfigured:
            return "Tokenizer not configured. Call setTokenizer() first."
        case .weightLoadingFailed(let message):
            return "Failed to load weights: \(message)"
        case .invalidInputShape(let message):
            return "Invalid input shape: \(message)"
        case .featureExtractionFailed(let message):
            return "Feature extraction failed: \(message)"
        }
    }
}

// MARK: - Weight Loading

extension CLAPModel {
    /// Load weights from safetensors file.
    public func loadWeights(from path: URL) throws {
        // Load safetensors file
        let weights = try loadArrays(url: path)

        // Map HuggingFace keys to our model keys
        var mappedWeights: [String: MLXArray] = [:]
        for (key, value) in weights {
            let mappedKey = mapWeightKey(key)
            mappedWeights[mappedKey] = value
        }

        // Apply weights to model
        try update(parameters: ModuleParameters.unflattened(mappedWeights))
    }

    /// Map HuggingFace weight key to our model key.
    private func mapWeightKey(_ key: String) -> String {
        var mapped = key

        // HuggingFace uses audio_model, we use audio_encoder
        mapped = mapped.replacingOccurrences(of: "audio_model.", with: "audio_encoder.")

        // HuggingFace uses text_model, we use text_encoder
        mapped = mapped.replacingOccurrences(of: "text_model.", with: "text_encoder.")

        // Handle projection key differences
        mapped = mapped.replacingOccurrences(of: "audio_projection.0.", with: "audio_projection.linear1.")
        mapped = mapped.replacingOccurrences(of: "audio_projection.2.", with: "audio_projection.linear2.")
        mapped = mapped.replacingOccurrences(of: "text_projection.0.", with: "text_encoder.projection.linear1.")
        mapped = mapped.replacingOccurrences(of: "text_projection.2.", with: "text_encoder.projection.linear2.")

        return mapped
    }

    /// Load pretrained CLAP model.
    public static func fromPretrained(path: URL) throws -> CLAPModel {
        // Load config
        let configPath = path.appendingPathComponent("config.json")
        let config: CLAPConfig
        if FileManager.default.fileExists(atPath: configPath.path) {
            config = try CLAPConfig.load(from: configPath)
        } else {
            config = CLAPConfig()
        }

        // Create model
        let model = CLAPModel(config: config)

        // Load weights
        let weightsPath = path.appendingPathComponent("model.safetensors")
        if FileManager.default.fileExists(atPath: weightsPath.path) {
            try model.loadWeights(from: weightsPath)
        }

        // Load tokenizer
        let vocabPath = path.appendingPathComponent("vocab.json")
        let mergesPath = path.appendingPathComponent("merges.txt")
        if FileManager.default.fileExists(atPath: vocabPath.path),
           FileManager.default.fileExists(atPath: mergesPath.path) {
            let tokenizer = try CLAPTokenizer.load(vocabPath: vocabPath, mergesPath: mergesPath)
            model.setTokenizer(tokenizer)
        }

        return model
    }
}

// MARK: - Convenience Methods

extension CLAPModel {
    /// Zero-shot audio classification.
    ///
    /// - Parameters:
    ///   - audio: Audio waveform or mel spectrogram
    ///   - labels: Text labels to classify against
    /// - Returns: Classification probabilities [numLabels]
    public func classify(_ audio: MLXArray, labels: [String]) throws -> MLXArray {
        // Encode audio
        let audioEmbed = try encodeAudio(audio, normalize: true)

        // Encode all labels
        var textEmbeds: [MLXArray] = []
        for label in labels {
            let embed = try encodeText(label, normalize: true)
            textEmbeds.append(embed)
        }
        let textEmbedsStacked = MLX.concatenated(textEmbeds, axis: 0)  // [numLabels, dim]

        // Compute similarities
        let similarities = similarity(audioEmbeds: audioEmbed, textEmbeds: textEmbedsStacked)

        // Softmax to get probabilities
        let probs = softmax(similarities, axis: -1)

        return probs.squeezed(axis: 0)  // [numLabels]
    }

    /// Compute cosine similarity between audio and text.
    ///
    /// - Parameters:
    ///   - audio: Audio waveform or mel spectrogram
    ///   - text: Text string
    /// - Returns: Similarity score (scalar)
    public func cosineSimilarity(audio: MLXArray, text: String) throws -> Float {
        let audioEmbed = try encodeAudio(audio, normalize: true)
        let textEmbed = try encodeText(text, normalize: true)

        let similarity = MLX.sum(audioEmbed * textEmbed)
        return similarity.item()
    }
}
