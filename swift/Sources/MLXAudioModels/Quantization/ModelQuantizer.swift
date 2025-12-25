// ModelQuantizer.swift
// Utilities for quantizing entire models.
//
// Provides functions to convert full-precision models to quantized versions
// and to load pre-quantized models from disk.

import Foundation
import MLX
import MLXNN

/// Utility for quantizing models.
public enum ModelQuantizer {

    // MARK: - Model Quantization

    /// Quantize all Linear layers in a module.
    ///
    /// Recursively traverses the module tree and replaces Linear layers
    /// with QuantizedLinear layers.
    ///
    /// - Parameters:
    ///   - module: The module to quantize
    ///   - config: Quantization configuration
    ///   - keyPrefix: Prefix for layer key matching (for skipLayers)
    /// - Returns: Dictionary mapping layer paths to their quantized versions
    public static func quantizeLinearLayers(
        in module: Module,
        config: QuantizationConfig,
        keyPrefix: String = ""
    ) throws -> [String: QuantizedLinear] {
        var quantizedLayers: [String: QuantizedLinear] = [:]

        // Get all parameters to find Linear layers
        let params = module.parameters()
        let flatParams = Dictionary(uniqueKeysWithValues: params.flattened())

        // Traverse and find Linear layers by their weight shape
        for (key, weight) in flatParams {
            // Linear layers have weights with key ending in ".weight"
            if key.hasSuffix(".weight") {
                let layerKey = String(key.dropLast(7))  // Remove ".weight"
                let fullKey = keyPrefix.isEmpty ? layerKey : "\(keyPrefix).\(layerKey)"

                // Check if this layer should be skipped
                if config.skipLayers.contains(where: { fullKey.contains($0) }) {
                    continue
                }

                // Only quantize 2D weight matrices (Linear layers)
                if weight.ndim == 2 {
                    let outputDim = weight.dim(0)
                    let inputDim = weight.dim(1)

                    // Check for corresponding bias
                    let biasKey = "\(layerKey).bias"
                    let hasBias = flatParams[biasKey] != nil

                    let quantized = QuantizedLinear(
                        inputDim: inputDim,
                        outputDim: outputDim,
                        bias: hasBias,
                        config: config
                    )

                    try quantized.quantize(from: weight)

                    if hasBias, let bias = flatParams[biasKey] {
                        quantized.bias = bias.asType(.float16)
                    }

                    quantizedLayers[fullKey] = quantized
                }
            }
        }

        return quantizedLayers
    }

    // MARK: - Quantized Weight Saving/Loading

    /// Save quantized weights to a file.
    ///
    /// - Parameters:
    ///   - layers: Dictionary of quantized layers
    ///   - url: File URL to save to
    ///   - config: Quantization config to embed
    public static func saveQuantized(
        layers: [String: QuantizedLinear],
        to url: URL,
        config: QuantizationConfig
    ) throws {
        var arrays: [String: MLXArray] = [:]

        for (key, layer) in layers {
            arrays["\(key).quantized_weight"] = layer.quantizedWeight
            arrays["\(key).scales"] = layer.scales
            if let zeros = layer.zeros {
                arrays["\(key).zeros"] = zeros
            }
            if let bias = layer.bias {
                arrays["\(key).bias"] = bias
            }
        }

        // Save using MLX's save function
        try MLX.save(arrays: arrays, url: url)

        // Save config alongside
        let configUrl = url.deletingPathExtension().appendingPathExtension("quant_config.json")
        let encoder = JSONEncoder()
        encoder.outputFormatting = .prettyPrinted
        let configData = try encoder.encode(config)
        try configData.write(to: configUrl)
    }

    /// Load quantized weights from a file.
    ///
    /// - Parameter url: File URL to load from
    /// - Returns: Tuple of (quantized layers, config)
    public static func loadQuantized(
        from url: URL
    ) throws -> ([String: QuantizedLinear], QuantizationConfig) {
        // Load config
        let configUrl = url.deletingPathExtension().appendingPathExtension("quant_config.json")
        let configData = try Data(contentsOf: configUrl)
        let config = try JSONDecoder().decode(QuantizationConfig.self, from: configData)

        // Load arrays
        let arrays = try MLX.loadArrays(url: url)

        // Group by layer name
        var layerArrays: [String: [String: MLXArray]] = [:]
        for (key, array) in arrays {
            let parts = key.split(separator: ".")
            if parts.count >= 2 {
                let layerKey = parts.dropLast().joined(separator: ".")
                let arrayType = String(parts.last!)
                layerArrays[layerKey, default: [:]][arrayType] = array
            }
        }

        // Reconstruct layers
        var layers: [String: QuantizedLinear] = [:]
        for (layerKey, layerData) in layerArrays {
            guard let quantizedWeight = layerData["quantized_weight"],
                  let scales = layerData["scales"] else {
                continue
            }

            // Infer dimensions from shapes
            let outputDim = scales.dim(0)
            let numGroups = scales.dim(1)
            let groupSize = config.groupSize
            let inputDim = numGroups * groupSize  // Approximate

            let layer = QuantizedLinear(
                inputDim: inputDim,
                outputDim: outputDim,
                bias: layerData["bias"] != nil,
                config: config
            )

            layer.quantizedWeight = quantizedWeight
            layer.scales = scales
            layer.zeros = layerData["zeros"]
            layer.bias = layerData["bias"]

            layers[layerKey] = layer
        }

        return (layers, config)
    }

    // MARK: - Memory Estimation

    /// Estimate memory savings from quantizing a model.
    ///
    /// - Parameters:
    ///   - module: The module to analyze
    ///   - config: Quantization configuration
    /// - Returns: Tuple of (original bytes, quantized bytes, savings ratio)
    public static func estimateMemorySavings(
        for module: Module,
        config: QuantizationConfig
    ) -> (originalBytes: UInt64, quantizedBytes: UInt64, savingsRatio: Double) {
        var originalBytes: UInt64 = 0
        var quantizedBytes: UInt64 = 0

        let params = module.parameters()

        for (key, weight) in params.flattened() {
            if key.hasSuffix(".weight") {
                let layerKey = String(key.dropLast(7))

                // Skip layers that won't be quantized
                if config.skipLayers.contains(where: { layerKey.contains($0) }) {
                    // Not quantized - count as float32
                    let bytes = UInt64(weight.size) * 4
                    originalBytes += bytes
                    quantizedBytes += bytes
                    continue
                }

                if weight.ndim == 2 {
                    let paramCount = weight.size
                    originalBytes += UInt64(paramCount) * 4  // float32
                    quantizedBytes += config.estimateMemory(parameterCount: paramCount)
                } else {
                    // Non-linear weights (conv, etc.) - not quantized
                    let bytes = UInt64(weight.size) * 4
                    originalBytes += bytes
                    quantizedBytes += bytes
                }
            }
        }

        let ratio = originalBytes > 0 ? Double(originalBytes) / Double(quantizedBytes) : 1.0
        return (originalBytes, quantizedBytes, ratio)
    }

    // MARK: - Quality Validation

    /// Compute max absolute difference between original and dequantized weights.
    ///
    /// - Parameters:
    ///   - original: Original weight matrix
    ///   - quantized: Quantized linear layer
    /// - Returns: Maximum absolute difference
    public static func maxQuantizationError(
        original: MLXArray,
        quantized: QuantizedLinear
    ) throws -> Float {
        let dequantized = try quantized.dequantize()
        let diff = MLX.abs(original.asType(.float16) - dequantized)
        return MLX.max(diff).item(Float.self)
    }

    /// Compute mean absolute difference between original and dequantized weights.
    public static func meanQuantizationError(
        original: MLXArray,
        quantized: QuantizedLinear
    ) throws -> Float {
        let dequantized = try quantized.dequantize()
        let diff = MLX.abs(original.asType(.float16) - dequantized)
        return MLX.mean(diff).item(Float.self)
    }
}

// MARK: - Quantization Report

/// Report on quantization results.
public struct QuantizationReport: Sendable {
    /// Model identifier.
    public let modelId: String

    /// Quantization configuration used.
    public let config: QuantizationConfig

    /// Number of layers quantized.
    public let layersQuantized: Int

    /// Number of layers skipped.
    public let layersSkipped: Int

    /// Original model size in bytes.
    public let originalBytes: UInt64

    /// Quantized model size in bytes.
    public let quantizedBytes: UInt64

    /// Memory savings ratio.
    public var savingsRatio: Double {
        originalBytes > 0 ? Double(originalBytes) / Double(quantizedBytes) : 1.0
    }

    /// Memory saved in bytes.
    public var savedBytes: UInt64 {
        originalBytes > quantizedBytes ? originalBytes - quantizedBytes : 0
    }

    /// Memory saved in MB.
    public var savedMB: Double {
        Double(savedBytes) / (1024 * 1024)
    }

    /// Per-layer quantization errors (if measured).
    public var layerErrors: [String: Float]?

    /// Maximum quantization error across all layers.
    public var maxError: Float? {
        layerErrors?.values.max()
    }

    /// Print formatted report.
    public func printReport() {
        print("\n=== Quantization Report: \(modelId) ===")
        print("Config: \(config.bits)-bit, group size \(config.groupSize), \(config.mode)")
        print("Layers quantized: \(layersQuantized)")
        print("Layers skipped: \(layersSkipped)")
        print("Original size: \(formatBytes(originalBytes))")
        print("Quantized size: \(formatBytes(quantizedBytes))")
        print("Savings: \(formatBytes(savedBytes)) (\(String(format: "%.1fx", savingsRatio)) compression)")
        if let maxErr = maxError {
            print("Max quantization error: \(String(format: "%.6f", maxErr))")
        }
        print("==========================================\n")
    }
}
