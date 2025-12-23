// WhisperAdapter.swift
// Whisper model with LoRA adapters for fine-tuning.
//
// Adds LoRA to decoder attention layers for domain-specific
// adaptation (accents, vocabulary, etc.).

import Foundation
import MLX
import MLXNN

// MARK: - Whisper Adapter Configuration

/// Configuration for Whisper adapter.
public struct WhisperAdapterConfig: Codable, Sendable {
    /// LoRA rank.
    public let loraRank: Int

    /// LoRA alpha (scaling factor).
    public let loraAlpha: Float?

    /// LoRA dropout.
    public let loraDropout: Float

    /// Which attention modules to adapt.
    public let targetModules: [String]

    /// Whether to freeze the encoder.
    public let freezeEncoder: Bool

    /// Whether to freeze embedding layers.
    public let freezeEmbeddings: Bool

    /// Creates a Whisper adapter configuration.
    public init(
        loraRank: Int = 8,
        loraAlpha: Float? = nil,
        loraDropout: Float = 0.0,
        targetModules: [String] = ["query", "value"],
        freezeEncoder: Bool = true,
        freezeEmbeddings: Bool = true
    ) {
        self.loraRank = loraRank
        self.loraAlpha = loraAlpha
        self.loraDropout = loraDropout
        self.targetModules = targetModules
        self.freezeEncoder = freezeEncoder
        self.freezeEmbeddings = freezeEmbeddings
    }

    /// Default configuration for accent adaptation.
    public static let accent = WhisperAdapterConfig(
        loraRank: 8,
        targetModules: ["query", "value"],
        freezeEncoder: true
    )

    /// Configuration for vocabulary adaptation.
    public static let vocabulary = WhisperAdapterConfig(
        loraRank: 16,
        targetModules: ["query", "key", "value", "out_proj"],
        freezeEncoder: true
    )

    /// Full fine-tuning configuration.
    public static let full = WhisperAdapterConfig(
        loraRank: 32,
        targetModules: ["query", "key", "value", "out_proj"],
        freezeEncoder: false,
        freezeEmbeddings: false
    )
}

// MARK: - Whisper Adapter

/// Whisper model with LoRA adapters for efficient fine-tuning.
///
/// This wrapper adds LoRA to Whisper's decoder attention layers,
/// enabling adaptation to specific accents, vocabularies, or domains
/// with minimal trainable parameters.
///
/// Example:
/// ```swift
/// let whisper = try WhisperModel.fromPretrained(path: whisperPath)
/// let adapter = WhisperAdapter(
///     whisper: whisper,
///     config: .accent
/// )
///
/// // Train on domain-specific data
/// let trainer = Trainer()
/// trainer.fit(module: adapter, trainData: transcriptionData)
///
/// // Merge for deployment
/// adapter.merge()
/// ```
public class WhisperAdapter: Module, TrainModule, @unchecked Sendable {
    /// Configuration.
    public let config: WhisperAdapterConfig

    /// Base Whisper model.
    public let whisper: Module

    /// Encoder module (for reference).
    private let encoder: Module?

    /// Decoder module (for LoRA).
    private let decoder: Module?

    /// LoRA application result.
    public private(set) var loraResult: LoRAApplicationResult?

    /// Learning rate.
    private let learningRate: Float

    /// Weight decay.
    private let weightDecay: Float

    /// Total training steps (for scheduler).
    private let totalSteps: Int

    /// Creates a Whisper adapter.
    ///
    /// - Parameters:
    ///   - whisper: Pre-trained Whisper model
    ///   - config: Adapter configuration
    ///   - learningRate: Learning rate (default: 1e-4)
    ///   - weightDecay: Weight decay (default: 0.01)
    ///   - totalSteps: Total training steps (default: 10000)
    public init(
        whisper: Module,
        config: WhisperAdapterConfig = WhisperAdapterConfig(),
        learningRate: Float = 1e-4,
        weightDecay: Float = 0.01,
        totalSteps: Int = 10000
    ) {
        self.config = config
        self.whisper = whisper
        self.learningRate = learningRate
        self.weightDecay = weightDecay
        self.totalSteps = totalSteps

        // Try to get encoder and decoder
        // This assumes Whisper has encoder and decoder properties
        self.encoder = nil  // Would be whisper.encoder
        self.decoder = nil  // Would be whisper.decoder

        super.init()

        // Apply LoRA to decoder attention
        let loraConfig = LoRAConfig(
            rank: config.loraRank,
            alpha: config.loraAlpha,
            dropout: config.loraDropout,
            targetModules: Set(config.targetModules)
        )

        // Apply LoRA with predicate to match decoder attention
        loraResult = applyLoRA(to: whisper, rank: config.loraRank, alpha: config.loraAlpha) { path, _ in
            // Only apply to decoder if encoder is frozen
            let isDecoder = path.contains("decoder")
            let matchesTarget = config.targetModules.contains { path.contains($0) }
            let isAttention = path.contains("attn") || path.contains("attention")

            if config.freezeEncoder {
                return isDecoder && matchesTarget && isAttention
            } else {
                return matchesTarget && isAttention
            }
        }

        // Freeze components as configured
        if config.freezeEncoder, let enc = encoder {
            enc.freeze()
        }

        if config.freezeEmbeddings {
            // Freeze embedding layers
            for (path, module) in whisper.leafModules().flattened() {
                if path.contains("embed") || path.contains("embedding") {
                    module.freeze()
                }
            }
        }
    }

    /// Get trainable parameter count.
    public var trainableParameterCount: Int {
        loraResult?.trainableParameters ?? 0
    }

    /// Get compression ratio.
    public var compressionRatio: Float {
        let stats = getLoRAStats(for: whisper)
        return 100.0 / max(1, stats.trainablePercentage)
    }

    // MARK: - Forward Pass

    /// Encode audio to features.
    /// Note: Placeholder - actual implementation should call whisper.encode()
    public func encode(_ mel: MLXArray) -> MLXArray {
        // Placeholder: return mel as-is
        // In actual use, call whisper's encode method
        return mel
    }

    /// Decode tokens with audio features.
    /// Note: Placeholder - actual implementation should call whisper.decode()
    public func decode(
        tokens: MLXArray,
        audioFeatures: MLXArray,
        kvCache: Any? = nil
    ) -> MLXArray {
        // Placeholder: return tokens as-is
        // In actual use, call whisper's decode method
        return tokens
    }

    /// Full forward pass for training.
    public func callAsFunction(mel: MLXArray, tokens: MLXArray) -> MLXArray {
        // Encode audio
        let audioFeatures = encode(mel)

        // Decode tokens
        let logits = decode(tokens: tokens, audioFeatures: audioFeatures)

        return logits
    }

    // MARK: - TrainModule

    public func computeLoss(batch: [MLXArray]) -> (MLXArray, [String: MLXArray]) {
        guard batch.count >= 3 else {
            return (MLXArray(Float.infinity), [:])
        }

        let mel = batch[0]      // [B, C, T]
        let tokens = batch[1]   // [B, S] - input tokens
        let labels = batch[2]   // [B, S] - target tokens

        // Forward pass
        let audioFeatures = encode(mel)
        let logits = decode(tokens: tokens, audioFeatures: audioFeatures)

        // Flatten for cross-entropy
        let B = logits.dim(0)
        let S = logits.dim(1)
        let V = logits.dim(2)

        let logitsFlat = logits.reshaped([B * S, V])
        let labelsFlat = labels.reshaped([B * S])

        // Compute loss (ignore padding token = -100)
        // For simplicity, we use all tokens here
        let loss = crossEntropyLoss(
            logits: logitsFlat,
            targets: labelsFlat,
            reduction: .mean
        )

        // Compute perplexity
        let perplexity = MLX.exp(loss)

        // Token accuracy
        let predictions = argMax(logitsFlat, axis: -1)
        let correct = (predictions .== labelsFlat).asType(.float32)
        let accuracy = MLX.mean(correct)

        return (loss, [
            "perplexity": perplexity,
            "accuracy": accuracy
        ])
    }

    public func configureOptimizers() -> OptimizerConfig {
        let optimizer = AdamW(
            schedule: WarmupCosineSchedule(
                peakLR: learningRate,
                warmupSteps: Int(Float(totalSteps) * 0.05),
                totalSteps: totalSteps,
                minLR: learningRate / 100
            ),
            weightDecay: weightDecay
        )

        return OptimizerConfig(
            optimizer: optimizer,
            schedulerName: "warmup_cosine"
        )
    }

    // MARK: - LoRA Operations

    /// Merge LoRA weights into base model.
    ///
    /// After merging, the model can be used without LoRA overhead.
    /// This is useful for deployment.
    public func merge() {
        let mergedCount = mergeLoRA(in: whisper)
        loraResult = nil
        print("Merged \(mergedCount) LoRA layers")
    }

    /// Reset LoRA weights.
    public func resetLoRA() {
        for (_, module) in whisper.leafModules().flattened() {
            if let lora = module as? LoRALinear {
                lora.resetLoRA()
            }
        }
    }

    /// Get LoRA statistics.
    public func getStats() -> LoRAStats {
        getLoRAStats(for: whisper)
    }

    // MARK: - Save/Load

    /// Save only LoRA weights.
    public func saveLoRAWeights(to path: URL) throws {
        try MLXAudioTraining.saveLoRAWeights(from: whisper, to: path)
    }

    /// Load LoRA weights.
    public func loadLoRAWeights(from path: URL) throws {
        try MLXAudioTraining.loadLoRAWeights(from: path, into: whisper)
    }
}

// MARK: - Whisper Transcription Adapter

/// Simplified interface for Whisper transcription fine-tuning.
public class WhisperTranscriptionAdapter: @unchecked Sendable {
    private let adapter: WhisperAdapter

    /// Tokenizer for text processing.
    public var tokenizer: Any?

    /// Creates a transcription adapter.
    public init(
        whisper: Module,
        config: WhisperAdapterConfig = .accent
    ) {
        self.adapter = WhisperAdapter(whisper: whisper, config: config)
    }

    /// Fine-tune on transcription pairs.
    ///
    /// - Parameters:
    ///   - audioFiles: Paths to audio files
    ///   - transcriptions: Corresponding transcriptions
    ///   - epochs: Number of training epochs
    public func fineTune(
        audioFiles: [URL],
        transcriptions: [String],
        epochs: Int = 3
    ) throws {
        // This would:
        // 1. Load and preprocess audio
        // 2. Tokenize transcriptions
        // 3. Create training batches
        // 4. Train with the adapter
        fatalError("Implementation requires full Whisper API integration")
    }

    /// Transcribe audio.
    public func transcribe(_ audio: MLXArray) -> String {
        // Use adapter for transcription
        fatalError("Implementation requires full Whisper API integration")
    }

    /// Export merged model.
    public func exportMerged(to path: URL) throws {
        adapter.merge()
        // Save full model
        let params = adapter.whisper.parameters().flattened()
        var paramDict: [String: MLXArray] = [:]
        for (key, value) in params {
            paramDict[key] = value
        }
        try save(arrays: paramDict, url: path)
    }
}
