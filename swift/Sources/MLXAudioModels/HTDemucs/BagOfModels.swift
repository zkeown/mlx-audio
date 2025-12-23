// BagOfModels.swift
// Ensemble of HTDemucs models for improved source separation.

import Foundation
@preconcurrency import MLX

// MARK: - BagOfModels

/// Ensemble of HTDemucs models for improved source separation.
///
/// htdemucs_ft uses 4 specialized models, each trained to excel at one
/// source type. The weight matrix selects which model's output to use
/// for each stem:
/// - Model 0 → drums
/// - Model 1 → bass
/// - Model 2 → other
/// - Model 3 → vocals
///
/// This provides ~3dB SI-SDR improvement over a single general-purpose model.
///
/// Example:
/// ```swift
/// let bag = try BagOfModels.fromPretrained(path: modelURL)
/// let stems = bag(mixture)  // [B, 4, C, T]
/// ```
public class BagOfModels: @unchecked Sendable {

    /// The HTDemucs models in this ensemble.
    public let models: [HTDemucs]

    /// Number of models in the ensemble.
    public let numModels: Int

    /// Weight matrix `[numModels, numSources]` for combining outputs.
    public let weights: MLXArray

    /// Configuration from the first model.
    public var config: HTDemucsConfig {
        models[0].config
    }

    /// Creates a BagOfModels ensemble.
    /// - Parameters:
    ///   - models: List of HTDemucs models (typically 4).
    ///   - weights: Weight matrix for combining outputs. Default is identity matrix.
    public init(models: [HTDemucs], weights: MLXArray? = nil) {
        self.models = models
        self.numModels = models.count
        self.weights = weights ?? MLXArray.eye(numModels)
    }

    /// Run all models and combine outputs.
    ///
    /// Uses sequential processing to minimize peak memory usage.
    /// Instead of running all models and stacking outputs (~4GB peak),
    /// processes models one at a time with weighted accumulation (~1GB peak).
    ///
    /// - Parameter mix: Input mixture `[B, C, T]` or `[C, T]`.
    /// - Returns: Separated stems `[B, S, C, T]` or `[S, C, T]`.
    public func callAsFunction(_ mix: MLXArray) -> MLXArray {
        var input = mix
        var squeezeBatch = false

        // Handle unbatched input
        if input.ndim == 2 {
            input = input.expandedDimensions(axis: 0)
            squeezeBatch = true
        }

        let B = input.shape[0]
        let C = input.shape[1]
        let T = input.shape[2]
        let S = config.num_sources

        // Pre-allocate result buffer
        var result = MLXArray.zeros([B, S, C, T])

        // Process models sequentially with weighted accumulation.
        // This avoids holding all model outputs simultaneously (~3GB savings).
        for (modelIdx, model) in models.enumerated() {
            let modelOutput = model(input)  // [B, S, C, T]

            // Accumulate weighted output for each stem
            for stemIdx in 0..<S {
                let w = weights[modelIdx, stemIdx]
                // Use einsum for efficient weighted accumulation: w * output[:, stemIdx]
                // This adds w * modelOutput[:, stemIdx, :, :] to result[:, stemIdx, :, :]
                let stemOutput = modelOutput[0..., stemIdx, 0..., 0...]  // [B, C, T]
                let weighted = w * stemOutput
                result = result.at[0..., stemIdx, 0..., 0...].add(weighted)
            }

            // Evaluate after each model to release model output from memory
            eval(result)
        }

        if squeezeBatch {
            result = result.squeezed(axis: 0)
        }

        return result
    }

    /// Load a bag of pretrained HTDemucs models.
    ///
    /// Expected directory structure:
    /// ```
    /// path/
    ///     model_0/
    ///         config.json
    ///         model.safetensors
    ///     model_1/
    ///         ...
    ///     model_2/
    ///         ...
    ///     model_3/
    ///         ...
    ///     weights.npy  (optional, defaults to identity)
    /// ```
    ///
    /// - Parameters:
    ///   - path: Path to bag directory.
    ///   - numModels: Expected number of models (default: 4).
    /// - Returns: BagOfModels instance.
    public static func fromPretrained(path: URL, numModels: Int = 4) throws -> BagOfModels {
        var models: [HTDemucs] = []

        // Load individual models
        for i in 0..<numModels {
            let modelPath = path.appendingPathComponent("model_\(i)")
            guard FileManager.default.fileExists(atPath: modelPath.path) else {
                throw BagOfModelsError.modelNotFound(index: i, path: modelPath)
            }
            let model = try HTDemucs.fromPretrained(path: modelPath)
            models.append(model)
        }

        // Load weights if available, otherwise use identity
        let weightsPath = path.appendingPathComponent("weights.npy")
        let weights: MLXArray?
        if FileManager.default.fileExists(atPath: weightsPath.path) {
            // Load numpy weights
            weights = try MLX.loadArray(url: weightsPath)
        } else {
            weights = nil  // Will default to identity
        }

        return BagOfModels(models: models, weights: weights)
    }
}

// MARK: - Errors

/// Errors for BagOfModels operations.
public enum BagOfModelsError: Error, LocalizedError {
    case modelNotFound(index: Int, path: URL)

    public var errorDescription: String? {
        switch self {
        case .modelNotFound(let index, let path):
            return "Model \(index) not found at \(path.path)"
        }
    }
}
