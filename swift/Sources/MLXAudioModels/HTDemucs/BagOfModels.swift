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

        // Run all models
        var outputs: [MLXArray] = []
        for model in models {
            let out = model(input)  // [B, S, C, T]
            outputs.append(out)
        }

        // Stack outputs: [numModels, B, S, C, T]
        let stacked = MLX.stacked(outputs, axis: 0)

        // Apply weight matrix to combine model outputs
        // weights[m, s] indicates how much model m contributes to stem s
        // Using einsum: 'mbsct,ms->bsct'
        var result = MLX.einsum("mbsct,ms->bsct", stacked, weights)

        // Ensure evaluation to prevent memory buildup
        eval(result)

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
