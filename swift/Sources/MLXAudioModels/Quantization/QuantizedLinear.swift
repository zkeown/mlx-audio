// QuantizedLinear.swift
// Quantized linear layer for memory-efficient inference.
//
// Stores weights in compressed format (int4/int8) and dequantizes
// on-the-fly during forward pass. This trades compute for memory,
// which is beneficial on memory-constrained devices.

import Foundation
import MLX
import MLXNN

/// Errors that can occur during quantization operations.
public enum QuantizationError: Error, LocalizedError {
    /// The specified bit width is not supported.
    case unsupportedBitWidth(Int)

    public var errorDescription: String? {
        switch self {
        case .unsupportedBitWidth(let bits):
            return "Unsupported quantization bit width: \(bits). Supported values are 4 and 8."
        }
    }
}

/// Linear layer with quantized weights.
///
/// Weights are stored in a packed format with per-group scales and zero-points.
/// During forward pass, weights are dequantized to float16/32 for computation.
///
/// Memory savings depend on quantization config:
/// - 8-bit: ~2x reduction
/// - 4-bit: ~4x reduction (with scale overhead)
///
/// Example:
/// ```swift
/// let linear = QuantizedLinear(
///     inputDim: 512,
///     outputDim: 2048,
///     config: .int4
/// )
/// let output = linear(input)  // Dequantizes weights internally
/// ```
public class QuantizedLinear: Module, @unchecked Sendable {

    /// Quantization configuration.
    public let config: QuantizationConfig

    /// Packed quantized weights.
    /// Shape depends on packing: [outputDim, packedInputDim] for 4-bit.
    var quantizedWeight: MLXArray

    /// Per-group scales for dequantization.
    /// Shape: [outputDim, numGroups]
    var scales: MLXArray

    /// Per-group zero-points (only for affine mode).
    /// Shape: [outputDim, numGroups]
    var zeros: MLXArray?

    /// Optional bias (not quantized).
    var bias: MLXArray?

    /// Original input dimension.
    public let inputDim: Int

    /// Original output dimension.
    public let outputDim: Int

    /// Number of quantization groups per output row.
    public let numGroups: Int

    /// Initialize a quantized linear layer.
    ///
    /// - Parameters:
    ///   - inputDim: Input feature dimension
    ///   - outputDim: Output feature dimension
    ///   - bias: Whether to include bias
    ///   - config: Quantization configuration
    public init(
        inputDim: Int,
        outputDim: Int,
        bias: Bool = true,
        config: QuantizationConfig = .int4
    ) {
        self.inputDim = inputDim
        self.outputDim = outputDim
        self.config = config
        self.numGroups = (inputDim + config.groupSize - 1) / config.groupSize

        // Calculate packed dimension based on bit width
        let packedDim = Self.packedDimension(inputDim: inputDim, bits: config.bits)

        // Initialize with zeros (will be set by quantization or loading)
        self.quantizedWeight = MLXArray.zeros([outputDim, packedDim], dtype: .uint8)
        self.scales = MLXArray.zeros([outputDim, numGroups], dtype: .float16)

        if config.mode == .affine {
            self.zeros = MLXArray.zeros([outputDim, numGroups], dtype: .float16)
        }

        if bias {
            self.bias = MLXArray.zeros([outputDim], dtype: .float16)
        }

        super.init()
    }

    /// Forward pass with on-the-fly dequantization.
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Dequantize weights
        // Using try! is safe here because config.bits is validated at init time
        // and only supports 4 and 8, which are handled by dequantize()
        let weight = try! dequantize()

        // Standard linear: output = x @ weight.T + bias
        var output = MLX.matmul(x, weight.T)

        if let b = bias {
            output = output + b
        }

        return output
    }

    /// Dequantize weights to full precision.
    ///
    /// - Throws: `QuantizationError.unsupportedBitWidth` if bit width is not 4 or 8.
    public func dequantize() throws -> MLXArray {
        switch config.bits {
        case 4:
            return dequantize4Bit()
        case 8:
            return dequantize8Bit()
        default:
            throw QuantizationError.unsupportedBitWidth(config.bits)
        }
    }

    /// Dequantize 4-bit packed weights.
    private func dequantize4Bit() -> MLXArray {
        // Unpack 4-bit values from uint8
        // Each uint8 contains 2 weights: low nibble and high nibble
        let low = quantizedWeight & 0x0F
        let high = (quantizedWeight >> 4) & 0x0F

        // Interleave to get original order
        // low: [outputDim, packedDim], high: [outputDim, packedDim]
        // Expand to [outputDim, packedDim, 1], stack on axis=-1 -> [outputDim, packedDim, 2]
        let lowExpanded = low.expandedDimensions(axis: -1)
        let highExpanded = high.expandedDimensions(axis: -1)
        var unpacked = concatenated([lowExpanded, highExpanded], axis: -1)
        unpacked = unpacked.reshaped([outputDim, -1])

        // Trim to actual input dimension
        if unpacked.dim(1) > inputDim {
            unpacked = unpacked[0..., 0..<inputDim]
        }

        // Apply dequantization per group
        return applyDequantization(unpacked.asType(DType.float16))
    }

    /// Dequantize 8-bit weights.
    private func dequantize8Bit() -> MLXArray {
        // 8-bit weights are stored directly as uint8
        let unpacked = quantizedWeight.asType(.float16)

        // Trim to actual input dimension if padded
        let trimmed = unpacked.dim(1) > inputDim
            ? unpacked[0..., 0..<inputDim]
            : unpacked

        return applyDequantization(trimmed)
    }

    /// Apply scale and zero-point dequantization.
    private func applyDequantization(_ quantized: MLXArray) -> MLXArray {
        // Reshape for group-wise operations
        // From: [outputDim, inputDim]
        // To: [outputDim, numGroups, groupSize]
        let groupedShape = [outputDim, numGroups, config.groupSize]

        // Pad if input dimension is not divisible by group size
        var padded = quantized
        let remainder = inputDim % config.groupSize
        if remainder != 0 {
            let padAmount = config.groupSize - remainder
            let padding = MLXArray.zeros([outputDim, padAmount], dtype: quantized.dtype)
            padded = concatenated([quantized, padding], axis: 1)
        }

        let grouped = padded.reshaped(groupedShape)

        // Expand scales and zeros for broadcasting
        // From: [outputDim, numGroups] to [outputDim, numGroups, 1]
        let expandedScales = scales.expandedDimensions(axis: -1)

        // Dequantize: value = scale * (quantized - zero)
        var dequantized: MLXArray
        if let z = zeros {
            let expandedZeros = z.expandedDimensions(axis: -1)
            dequantized = expandedScales * (grouped - expandedZeros)
        } else {
            // Symmetric mode: value = scale * quantized
            dequantized = expandedScales * grouped
        }

        // Reshape back to [outputDim, paddedInputDim]
        dequantized = dequantized.reshaped([outputDim, -1])

        // Trim padding if added
        if dequantized.dim(1) > inputDim {
            dequantized = dequantized[0..., 0..<inputDim]
        }

        return dequantized
    }

    // MARK: - Quantization from Full Precision

    /// Quantize a full-precision weight matrix.
    ///
    /// - Parameter weight: Full precision weights [outputDim, inputDim]
    /// - Returns: Self with quantized weights set
    /// - Throws: `QuantizationError.unsupportedBitWidth` if bit width is not 4 or 8.
    @discardableResult
    public func quantize(from weight: MLXArray) throws -> Self {
        precondition(weight.shape == [outputDim, inputDim],
                     "Weight shape mismatch: expected [\(outputDim), \(inputDim)], got \(weight.shape)")

        switch config.bits {
        case 4:
            quantize4Bit(weight)
        case 8:
            quantize8Bit(weight)
        default:
            throw QuantizationError.unsupportedBitWidth(config.bits)
        }

        return self
    }

    /// Quantize weights to 4-bit.
    private func quantize4Bit(_ weight: MLXArray) {
        // Pad input dimension to multiple of group size
        var padded = weight
        let remainder = inputDim % config.groupSize
        if remainder != 0 {
            let padAmount = config.groupSize - remainder
            let padding = MLXArray.zeros([outputDim, padAmount], dtype: weight.dtype)
            padded = concatenated([weight, padding], axis: 1)
        }

        // Reshape to groups: [outputDim, numGroups, groupSize]
        let grouped = padded.reshaped([outputDim, numGroups, config.groupSize])

        // Compute per-group statistics
        let groupMin = MLX.min(grouped, axis: -1)  // [outputDim, numGroups]
        let groupMax = MLX.max(grouped, axis: -1)

        // Compute scale and zero-point for 4-bit range [0, 15]
        let range = groupMax - groupMin
        let scale = range / 15.0  // 4-bit has 16 levels (0-15)

        // Avoid division by zero
        let safeScale = MLX.maximum(scale, MLXArray(1e-8))

        // Zero point maps min value to 0
        let zeroPoint = groupMin / safeScale

        // Quantize: q = round((value - min) / scale)
        let expandedScale = safeScale.expandedDimensions(axis: -1)
        let expandedMin = groupMin.expandedDimensions(axis: -1)

        var quantized = MLX.round((grouped - expandedMin) / expandedScale)
        quantized = MLX.clip(quantized, min: 0, max: 15)

        // Pack pairs of 4-bit values into uint8
        let flat = quantized.reshaped([outputDim, -1]).asType(.uint8)
        let even = flat[0..., stride(from: 0, to: flat.dim(1), by: 2)]
        let odd = flat[0..., stride(from: 1, to: flat.dim(1), by: 2)]
        let packed = even | (odd << 4)

        self.quantizedWeight = packed
        self.scales = safeScale.asType(.float16)
        self.zeros = zeroPoint.asType(.float16)

        eval(quantizedWeight, scales, zeros!)
    }

    /// Quantize weights to 8-bit.
    private func quantize8Bit(_ weight: MLXArray) {
        // Pad input dimension to multiple of group size
        var padded = weight
        let remainder = inputDim % config.groupSize
        if remainder != 0 {
            let padAmount = config.groupSize - remainder
            let padding = MLXArray.zeros([outputDim, padAmount], dtype: weight.dtype)
            padded = concatenated([weight, padding], axis: 1)
        }

        // Reshape to groups: [outputDim, numGroups, groupSize]
        let grouped = padded.reshaped([outputDim, numGroups, config.groupSize])

        // Compute per-group statistics
        let groupMin = MLX.min(grouped, axis: -1)
        let groupMax = MLX.max(grouped, axis: -1)

        // Compute scale for 8-bit range [0, 255]
        let range = groupMax - groupMin
        let scale = range / 255.0

        let safeScale = MLX.maximum(scale, MLXArray(1e-8))
        let zeroPoint = groupMin / safeScale

        // Quantize
        let expandedScale = safeScale.expandedDimensions(axis: -1)
        let expandedMin = groupMin.expandedDimensions(axis: -1)

        var quantized = MLX.round((grouped - expandedMin) / expandedScale)
        quantized = MLX.clip(quantized, min: 0, max: 255)

        // Reshape and store
        self.quantizedWeight = quantized.reshaped([outputDim, -1]).asType(.uint8)
        self.scales = safeScale.asType(.float16)
        self.zeros = zeroPoint.asType(.float16)

        eval(quantizedWeight, scales, zeros!)
    }

    // MARK: - Utility

    /// Calculate packed dimension for quantized weights.
    ///
    /// - Precondition: `bits` must be 4 or 8.
    static func packedDimension(inputDim: Int, bits: Int) -> Int {
        switch bits {
        case 4:
            // Two 4-bit values per uint8
            return (inputDim + 1) / 2
        case 8:
            // One 8-bit value per uint8
            return inputDim
        default:
            preconditionFailure("Unsupported bit width: \(bits). Use 4 or 8.")
        }
    }

    /// Memory usage of quantized weights in bytes.
    public var memoryBytes: UInt64 {
        var total = UInt64(quantizedWeight.size) * 1  // uint8
        total += UInt64(scales.size) * 2  // float16
        if let z = zeros {
            total += UInt64(z.size) * 2  // float16
        }
        if let b = bias {
            total += UInt64(b.size) * 2  // float16
        }
        return total
    }

    /// Memory savings compared to float32.
    public var memorySavingsRatio: Double {
        let fullPrecisionBytes = UInt64(inputDim * outputDim * 4)  // float32
        return Double(fullPrecisionBytes) / Double(memoryBytes)
    }
}

// MARK: - Factory Methods

extension QuantizedLinear {

    /// Create a quantized linear layer from a full-precision Linear layer.
    public static func from(
        _ linear: Linear,
        config: QuantizationConfig = .int4
    ) throws -> QuantizedLinear {
        let inputDim = linear.weight.dim(1)
        let outputDim = linear.weight.dim(0)

        let quantized = QuantizedLinear(
            inputDim: inputDim,
            outputDim: outputDim,
            bias: linear.bias != nil,
            config: config
        )

        try quantized.quantize(from: linear.weight)

        if let b = linear.bias {
            quantized.bias = b.asType(.float16)
        }

        return quantized
    }
}
