// LoRAUtils.swift
// Utility functions for applying LoRA to models.

import Foundation
import MLX
import MLXNN

// MARK: - LoRA Configuration

/// Configuration for LoRA adaptation.
public struct LoRAConfig: Sendable {
    /// Rank of the LoRA matrices.
    public let rank: Int

    /// Alpha scaling factor.
    public let alpha: Float?

    /// Dropout probability.
    public let dropout: Float

    /// Module name patterns to apply LoRA to.
    /// Uses simple string matching (contains).
    public let targetModules: Set<String>

    /// Module name patterns to exclude.
    public let excludeModules: Set<String>

    /// Creates a LoRA configuration.
    ///
    /// - Parameters:
    ///   - rank: LoRA rank (default: 8)
    ///   - alpha: Scaling factor (default: nil, uses rank)
    ///   - dropout: Dropout probability (default: 0.0)
    ///   - targetModules: Modules to adapt (default: ["query", "value"])
    ///   - excludeModules: Modules to skip (default: empty)
    public init(
        rank: Int = 8,
        alpha: Float? = nil,
        dropout: Float = 0.0,
        targetModules: Set<String> = ["query", "value"],
        excludeModules: Set<String> = []
    ) {
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.targetModules = targetModules
        self.excludeModules = excludeModules
    }

    /// Default configuration for attention layers.
    public static let attention = LoRAConfig(
        rank: 8,
        targetModules: ["query", "key", "value", "out_proj"]
    )

    /// Conservative configuration (query and value only).
    public static let conservative = LoRAConfig(
        rank: 4,
        targetModules: ["query", "value"]
    )

    /// Aggressive configuration (all linear layers).
    public static let aggressive = LoRAConfig(
        rank: 16,
        targetModules: ["linear", "proj", "fc", "dense"]
    )
}

// MARK: - LoRA Application

/// Result of applying LoRA to a model.
public struct LoRAApplicationResult {
    /// Number of layers adapted.
    public let layersAdapted: Int

    /// Total trainable parameters added.
    public let trainableParameters: Int

    /// Paths of adapted layers.
    public let adaptedPaths: [String]

    /// Compression ratio.
    public var compressionRatio: Float {
        // Approximate: typical Linear is ~100x more params than LoRA
        Float(layersAdapted) * 100.0 / Float(max(1, trainableParameters))
    }
}

/// Apply LoRA to all matching Linear layers in a model.
///
/// This function traverses the module hierarchy and replaces matching
/// Linear layers with LoRALinear wrappers. The original weights are
/// frozen and only the LoRA matrices are trainable.
///
/// - Parameters:
///   - model: The model to adapt
///   - config: LoRA configuration
/// - Returns: Result describing the adaptation
@discardableResult
public func applyLoRA(
    to model: Module,
    config: LoRAConfig
) -> LoRAApplicationResult {
    var adaptedPaths: [String] = []
    var trainableParams = 0

    // Get all leaf modules
    let modules = model.leafModules().flattened()

    for (path, module) in modules {
        // Check if this is a Linear layer
        guard let linear = module as? Linear else { continue }

        // Check if path matches target patterns
        let shouldAdapt = config.targetModules.contains { pattern in
            path.contains(pattern)
        }

        // Check if path matches exclude patterns
        let shouldExclude = config.excludeModules.contains { pattern in
            path.contains(pattern)
        }

        if shouldAdapt && !shouldExclude {
            // Create LoRA wrapper
            let loraLinear = LoRALinear(
                base: linear,
                rank: config.rank,
                alpha: config.alpha,
                dropout: config.dropout
            )

            // Replace in model
            replaceModule(in: model, path: path, with: loraLinear)

            adaptedPaths.append(path)
            trainableParams += loraLinear.trainableParameterCount
        }
    }

    return LoRAApplicationResult(
        layersAdapted: adaptedPaths.count,
        trainableParameters: trainableParams,
        adaptedPaths: adaptedPaths
    )
}

/// Apply LoRA with a custom predicate function.
///
/// - Parameters:
///   - model: The model to adapt
///   - rank: LoRA rank
///   - alpha: Scaling factor
///   - dropout: Dropout probability
///   - predicate: Function that returns true for layers to adapt
/// - Returns: Result describing the adaptation
@discardableResult
public func applyLoRA(
    to model: Module,
    rank: Int = 8,
    alpha: Float? = nil,
    dropout: Float = 0.0,
    where predicate: (String, Linear) -> Bool
) -> LoRAApplicationResult {
    var adaptedPaths: [String] = []
    var trainableParams = 0

    let modules = model.leafModules().flattened()

    for (path, module) in modules {
        guard let linear = module as? Linear else { continue }

        if predicate(path, linear) {
            let loraLinear = LoRALinear(
                base: linear,
                rank: rank,
                alpha: alpha,
                dropout: dropout
            )

            replaceModule(in: model, path: path, with: loraLinear)

            adaptedPaths.append(path)
            trainableParams += loraLinear.trainableParameterCount
        }
    }

    return LoRAApplicationResult(
        layersAdapted: adaptedPaths.count,
        trainableParameters: trainableParams,
        adaptedPaths: adaptedPaths
    )
}

/// Merge all LoRA layers back into regular Linear layers.
///
/// This combines the LoRA adaptations with the base weights,
/// creating standard Linear layers suitable for inference.
/// After merging, the model has no LoRA overhead.
///
/// - Parameter model: The model to merge
/// - Returns: Number of layers merged
@discardableResult
public func mergeLoRA(in model: Module) -> Int {
    var mergedCount = 0

    let modules = model.leafModules().flattened()

    for (path, module) in modules {
        guard let loraLinear = module as? LoRALinear else { continue }

        // Merge LoRA into base weights
        let mergedLinear = loraLinear.merge()

        // Replace LoRA layer with merged Linear
        replaceModule(in: model, path: path, with: mergedLinear)

        mergedCount += 1
    }

    return mergedCount
}

/// Unmerge LoRA by converting Linear layers back to LoRA layers.
///
/// This is the reverse of merge - it wraps Linear layers in LoRA
/// and resets the LoRA weights. Useful for continuing training
/// after a merge.
///
/// - Parameters:
///   - model: The model to unmerge
///   - config: LoRA configuration
///   - paths: Specific paths to unmerge (nil = use config patterns)
/// - Returns: Number of layers converted
@discardableResult
public func unmergeLoRA(
    in model: Module,
    config: LoRAConfig,
    paths: [String]? = nil
) -> Int {
    var unmergedCount = 0

    let modules = model.leafModules().flattened()

    for (path, module) in modules {
        guard let linear = module as? Linear else { continue }
        guard !(module is LoRALinear) else { continue }  // Already LoRA

        let shouldConvert: Bool
        if let specificPaths = paths {
            shouldConvert = specificPaths.contains(path)
        } else {
            shouldConvert = config.targetModules.contains { path.contains($0) }
                && !config.excludeModules.contains { path.contains($0) }
        }

        if shouldConvert {
            let loraLinear = LoRALinear(
                base: linear,
                rank: config.rank,
                alpha: config.alpha,
                dropout: config.dropout
            )

            replaceModule(in: model, path: path, with: loraLinear)
            unmergedCount += 1
        }
    }

    return unmergedCount
}

/// Get statistics about LoRA layers in a model.
public struct LoRAStats {
    /// Number of LoRA layers.
    public let loraLayerCount: Int

    /// Number of regular Linear layers.
    public let linearLayerCount: Int

    /// Total trainable LoRA parameters.
    public let trainableLoRAParams: Int

    /// Total frozen parameters.
    public let frozenParams: Int

    /// Paths of LoRA layers.
    public let loraPaths: [String]

    /// Percentage of parameters that are trainable.
    public var trainablePercentage: Float {
        let total = trainableLoRAParams + frozenParams
        guard total > 0 else { return 0 }
        return Float(trainableLoRAParams) / Float(total) * 100
    }
}

/// Get LoRA statistics for a model.
///
/// - Parameter model: The model to analyze
/// - Returns: Statistics about LoRA layers
public func getLoRAStats(for model: Module) -> LoRAStats {
    var loraCount = 0
    var linearCount = 0
    var trainableParams = 0
    var frozenParams = 0
    var loraPaths: [String] = []

    let modules = model.leafModules().flattened()

    for (path, module) in modules {
        if let loraLinear = module as? LoRALinear {
            loraCount += 1
            trainableParams += loraLinear.trainableParameterCount
            frozenParams += loraLinear.totalParameterCount - loraLinear.trainableParameterCount
            loraPaths.append(path)
        } else if module is Linear {
            linearCount += 1
            // Count parameters in regular linear
            let params = module.parameters().flattened()
            for (_, p) in params {
                frozenParams += p.size
            }
        }
    }

    return LoRAStats(
        loraLayerCount: loraCount,
        linearLayerCount: linearCount,
        trainableLoRAParams: trainableParams,
        frozenParams: frozenParams,
        loraPaths: loraPaths
    )
}

// MARK: - Helper Functions

/// Replace a module at the given path.
private func replaceModule(in model: Module, path: String, with newModule: Module) {
    // Parse the path to navigate to parent and get the key
    let components = path.split(separator: ".").map(String.init)
    guard !components.isEmpty else { return }

    // For now, we use the module update mechanism
    // This works with @ModuleInfo property wrappers
    let updates = [(path, newModule)]
    model.update(modules: ModuleChildren.unflattened(updates))
}

// MARK: - Save/Load LoRA Weights

/// Save only the LoRA weights from a model.
///
/// This saves a smaller checkpoint containing just the LoRA adaptations,
/// not the full model weights.
///
/// - Parameters:
///   - model: The model with LoRA layers
///   - path: Path to save weights
public func saveLoRAWeights(from model: Module, to path: URL) throws {
    var loraWeights: [String: MLXArray] = [:]

    let params = model.trainableParameters().flattened()
    for (key, value) in params {
        // Only save LoRA parameters
        if key.contains("lora_A") || key.contains("lora_B") {
            loraWeights[key] = value
        }
    }

    try save(arrays: loraWeights, url: path)
}

/// Load LoRA weights into a model.
///
/// The model must already have LoRA layers applied. This loads
/// just the LoRA A and B matrices.
///
/// - Parameters:
///   - path: Path to LoRA weights
///   - model: The model to load into
public func loadLoRAWeights(from path: URL, into model: Module) throws {
    let loraWeights = try loadArrays(url: path)

    var updates: [(String, MLXArray)] = []
    for (key, value) in loraWeights {
        updates.append((key, value))
    }

    if !updates.isEmpty {
        try model.update(parameters: ModuleParameters.unflattened(updates))
    }
}
