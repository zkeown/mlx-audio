// HTDemucsAdapter.swift
// HTDemucs model with LoRA adapters for fine-tuning.
//
// Adds LoRA to cross-transformer attention for adapting
// source separation to specific instrument types.

import Foundation
import MLX
import MLXNN

// MARK: - HTDemucs Adapter Configuration

/// Configuration for HTDemucs adapter.
public struct HTDemucsAdapterConfig: Codable, Sendable {
    /// LoRA rank.
    public let loraRank: Int

    /// LoRA alpha (scaling factor).
    public let loraAlpha: Float?

    /// LoRA dropout.
    public let loraDropout: Float

    /// Which modules to adapt.
    public let targetModules: [String]

    /// Whether to freeze encoder/decoder convolutions.
    public let freezeConvolutions: Bool

    /// Source names (for logging).
    public let sourceNames: [String]

    /// Creates an HTDemucs adapter configuration.
    public init(
        loraRank: Int = 16,
        loraAlpha: Float? = nil,
        loraDropout: Float = 0.0,
        targetModules: [String] = ["in_proj", "out_proj", "query", "key", "value"],
        freezeConvolutions: Bool = true,
        sourceNames: [String] = ["drums", "bass", "other", "vocals"]
    ) {
        self.loraRank = loraRank
        self.loraAlpha = loraAlpha
        self.loraDropout = loraDropout
        self.targetModules = targetModules
        self.freezeConvolutions = freezeConvolutions
        self.sourceNames = sourceNames
    }

    /// Default configuration for instrument adaptation.
    public static let instrument = HTDemucsAdapterConfig(
        loraRank: 16,
        targetModules: ["in_proj", "out_proj"],
        freezeConvolutions: true
    )

    /// Configuration for full transformer adaptation.
    public static let transformer = HTDemucsAdapterConfig(
        loraRank: 32,
        targetModules: ["in_proj", "out_proj", "query", "key", "value"],
        freezeConvolutions: true
    )

    /// Configuration for aggressive adaptation (more parameters).
    public static let aggressive = HTDemucsAdapterConfig(
        loraRank: 64,
        targetModules: ["in_proj", "out_proj", "query", "key", "value", "linear"],
        freezeConvolutions: false
    )
}

// MARK: - HTDemucs Adapter

/// HTDemucs model with LoRA adapters for source separation fine-tuning.
///
/// This wrapper adds LoRA to HTDemucs's cross-transformer attention,
/// enabling adaptation to specific instrument types or audio domains.
///
/// Example:
/// ```swift
/// let demucs = try HTDemucs.fromPretrained(path: demucsPath)
/// let adapter = HTDemucsAdapter(
///     demucs: demucs,
///     config: .instrument
/// )
///
/// // Train on domain-specific data
/// let trainer = Trainer()
/// trainer.fit(module: adapter, trainData: separationData)
/// ```
public class HTDemucsAdapter: Module, TrainModule, @unchecked Sendable {
    /// Configuration.
    public let config: HTDemucsAdapterConfig

    /// Base HTDemucs model.
    public let demucs: Module

    /// LoRA application result.
    public private(set) var loraResult: LoRAApplicationResult?

    /// Learning rate.
    private let learningRate: Float

    /// Weight decay.
    private let weightDecay: Float

    /// Total training steps.
    private let totalSteps: Int

    /// Creates an HTDemucs adapter.
    ///
    /// - Parameters:
    ///   - demucs: Pre-trained HTDemucs model
    ///   - config: Adapter configuration
    ///   - learningRate: Learning rate (default: 3e-5)
    ///   - weightDecay: Weight decay (default: 0.01)
    ///   - totalSteps: Total training steps (default: 50000)
    public init(
        demucs: Module,
        config: HTDemucsAdapterConfig = HTDemucsAdapterConfig(),
        learningRate: Float = 3e-5,
        weightDecay: Float = 0.01,
        totalSteps: Int = 50000
    ) {
        self.config = config
        self.demucs = demucs
        self.learningRate = learningRate
        self.weightDecay = weightDecay
        self.totalSteps = totalSteps

        super.init()

        // Apply LoRA to transformer layers
        loraResult = applyLoRA(
            to: demucs,
            rank: config.loraRank,
            alpha: config.loraAlpha,
            dropout: config.loraDropout
        ) { path, _ in
            // Match transformer attention layers
            let isTransformer = path.contains("transformer") ||
                               path.contains("crosstransformer")
            let matchesTarget = config.targetModules.contains { path.contains($0) }

            return isTransformer && matchesTarget
        }

        // Freeze convolutions if configured
        if config.freezeConvolutions {
            freezeConvolutionalLayers()
        }
    }

    /// Freeze all convolutional layers.
    private func freezeConvolutionalLayers() {
        for (path, module) in demucs.leafModules().flattened() {
            // Freeze encoder and decoder conv layers
            let isConv = path.contains("encoder") || path.contains("decoder") ||
                        path.contains("tencoder") || path.contains("tdecoder")
            let notTransformer = !path.contains("transformer")

            if isConv && notTransformer {
                module.freeze()
            }
        }
    }

    /// Get trainable parameter count.
    public var trainableParameterCount: Int {
        loraResult?.trainableParameters ?? 0
    }

    // MARK: - Forward Pass

    /// Separate audio into stems.
    ///
    /// - Parameter mix: Mixed audio [B, C, T] or [B, S, C, T]
    /// - Returns: Separated stems [B, S, C, T]
    /// Note: Placeholder - actual implementation should call demucs forward pass
    public func callAsFunction(_ mix: MLXArray) -> MLXArray {
        // Placeholder: return mix expanded for stems
        // In actual use, call the demucs model's forward method
        return mix.expandedDimensions(axis: 1)
    }

    /// Separate and return individual stems.
    public func separate(_ mix: MLXArray) -> [String: MLXArray] {
        let stems = self(mix)

        var result: [String: MLXArray] = [:]
        for (i, name) in config.sourceNames.enumerated() {
            if i < stems.dim(1) {
                result[name] = stems[0..., i...(i+1), 0..., 0...]
            }
        }

        return result
    }

    // MARK: - TrainModule

    public func computeLoss(batch: [MLXArray]) -> (MLXArray, [String: MLXArray]) {
        guard batch.count >= 2 else {
            return (MLXArray(Float.infinity), [:])
        }

        let mix = batch[0]      // [B, C, T] mixed audio
        let stems = batch[1]    // [B, S, C, T] target stems

        // Forward pass
        let predictions = self(mix)

        // SDR loss
        let loss = sdrLoss(
            predictions: predictions,
            targets: stems,
            reduction: .mean
        )

        // Compute per-stem SDR for logging
        var metrics: [String: MLXArray] = [:]

        let numStems = min(predictions.dim(1), config.sourceNames.count)
        for i in 0..<numStems {
            let predStem = predictions[0..., i...(i+1), 0..., 0...]
            let targetStem = stems[0..., i...(i+1), 0..., 0...]

            let stemLoss = sdrLoss(
                predictions: predStem,
                targets: targetStem,
                reduction: .mean
            )

            // Store positive SDR (loss is negative SDR)
            metrics["\(config.sourceNames[i])_sdr"] = -stemLoss
        }

        // Average SDR across stems
        metrics["avg_sdr"] = -loss

        return (loss, metrics)
    }

    public func configureOptimizers() -> OptimizerConfig {
        let optimizer = AdamW(
            schedule: WarmupCosineSchedule(
                peakLR: learningRate,
                warmupSteps: Int(Float(totalSteps) * 0.02),
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
    public func merge() {
        let mergedCount = mergeLoRA(in: demucs)
        loraResult = nil
        print("Merged \(mergedCount) LoRA layers")
    }

    /// Reset LoRA weights.
    public func resetLoRA() {
        for (_, module) in demucs.leafModules().flattened() {
            if let lora = module as? LoRALinear {
                lora.resetLoRA()
            }
        }
    }

    /// Get LoRA statistics.
    public func getStats() -> LoRAStats {
        getLoRAStats(for: demucs)
    }

    // MARK: - Save/Load

    /// Save only LoRA weights.
    public func saveLoRAWeights(to path: URL) throws {
        try MLXAudioTraining.saveLoRAWeights(from: demucs, to: path)
    }

    /// Load LoRA weights.
    public func loadLoRAWeights(from path: URL) throws {
        try MLXAudioTraining.loadLoRAWeights(from: path, into: demucs)
    }

    /// Export merged model.
    public func exportMerged(to path: URL) throws {
        merge()
        let params = demucs.parameters().flattened()
        var paramDict: [String: MLXArray] = [:]
        for (key, value) in params {
            paramDict[key] = value
        }
        try save(arrays: paramDict, url: path)
    }
}

// MARK: - Separation Metrics

/// Compute separation quality metrics.
public struct SeparationMetrics {
    /// Signal-to-Distortion Ratio for each stem.
    public let sdr: [String: Float]

    /// Signal-to-Interference Ratio for each stem.
    public let sir: [String: Float]

    /// Signal-to-Artifacts Ratio for each stem.
    public let sar: [String: Float]

    /// Average SDR across stems.
    public var avgSDR: Float {
        guard !sdr.isEmpty else { return 0 }
        return sdr.values.reduce(0, +) / Float(sdr.count)
    }
}

/// Compute separation metrics for predictions.
///
/// - Parameters:
///   - predictions: Predicted stems [B, S, C, T]
///   - targets: Target stems [B, S, C, T]
///   - sourceNames: Names of sources
/// - Returns: Separation metrics
public func computeSeparationMetrics(
    predictions: MLXArray,
    targets: MLXArray,
    sourceNames: [String]
) -> SeparationMetrics {
    var sdrDict: [String: Float] = [:]
    var sirDict: [String: Float] = [:]
    var sarDict: [String: Float] = [:]

    let numStems = min(predictions.dim(1), sourceNames.count)

    for i in 0..<numStems {
        let pred = predictions[0..., i...(i+1), 0..., 0...]
        let target = targets[0..., i...(i+1), 0..., 0...]

        // Compute SDR
        let sdr = -sdrLoss(predictions: pred, targets: target, reduction: .mean)
        eval(sdr)
        sdrDict[sourceNames[i]] = sdr.item(Float.self)

        // SIR and SAR would require more complex computation
        // (separating interference from artifacts)
        sirDict[sourceNames[i]] = 0  // Placeholder
        sarDict[sourceNames[i]] = 0  // Placeholder
    }

    return SeparationMetrics(sdr: sdrDict, sir: sirDict, sar: sarDict)
}
