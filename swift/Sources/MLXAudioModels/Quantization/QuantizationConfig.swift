// QuantizationConfig.swift
// Configuration for weight quantization to reduce model memory footprint.
//
// Supports int4 and int8 quantization with configurable group sizes.
// Quantization reduces memory by 2-8x with minimal quality loss for most models.

import Foundation

/// Quantization mode determining how weights are compressed.
public enum QuantizationMode: String, Codable, Sendable {
    /// Affine quantization with scale and zero-point per group.
    /// Formula: dequantized = scale * (quantized - zero_point)
    case affine

    /// Symmetric quantization with only scale per group.
    /// Formula: dequantized = scale * quantized
    /// Slightly faster but may have worse quality for asymmetric weight distributions.
    case symmetric
}

/// Configuration for model weight quantization.
public struct QuantizationConfig: Codable, Sendable, Equatable {

    /// Number of bits per weight (2, 3, 4, 5, 6, or 8).
    public let bits: Int

    /// Number of weights per quantization group.
    /// Smaller groups = better quality but more overhead.
    /// Common values: 32, 64, 128.
    public let groupSize: Int

    /// Quantization mode (affine or symmetric).
    public let mode: QuantizationMode

    /// Layer names to skip quantization (keep in full precision).
    /// Embeddings and final layers are often sensitive.
    public let skipLayers: [String]

    public init(
        bits: Int = 4,
        groupSize: Int = 64,
        mode: QuantizationMode = .affine,
        skipLayers: [String] = []
    ) {
        precondition([2, 3, 4, 5, 6, 8].contains(bits), "bits must be 2, 3, 4, 5, 6, or 8")
        precondition(groupSize > 0 && groupSize.isPowerOfTwo, "groupSize must be a positive power of 2")

        self.bits = bits
        self.groupSize = groupSize
        self.mode = mode
        self.skipLayers = skipLayers
    }

    // MARK: - Presets

    /// 4-bit quantization (2-4x memory reduction, minimal quality loss for text models).
    public static let int4 = QuantizationConfig(
        bits: 4,
        groupSize: 64,
        mode: .affine,
        skipLayers: []
    )

    /// 8-bit quantization (2x memory reduction, negligible quality loss).
    public static let int8 = QuantizationConfig(
        bits: 8,
        groupSize: 64,
        mode: .affine,
        skipLayers: []
    )

    /// Conservative 8-bit for audio-sensitive models.
    /// Skips embeddings and output projections.
    public static let audioConservative = QuantizationConfig(
        bits: 8,
        groupSize: 64,
        mode: .affine,
        skipLayers: ["embed", "embedding", "proj_out", "output"]
    )

    /// Aggressive 4-bit for maximum memory savings.
    /// Use with caution - may affect audio quality.
    public static let aggressive = QuantizationConfig(
        bits: 4,
        groupSize: 32,
        mode: .affine,
        skipLayers: []
    )

    // MARK: - Memory Estimation

    /// Estimate memory savings ratio compared to float32.
    public var compressionRatio: Double {
        // Float32 = 32 bits per weight
        // Quantized = bits + overhead for scales/zeros
        // Overhead: ~16 bits per group (scale) + optional 16 bits (zero)
        let overheadBitsPerWeight = (mode == .affine ? 32.0 : 16.0) / Double(groupSize)
        let effectiveBits = Double(bits) + overheadBitsPerWeight
        return 32.0 / effectiveBits
    }

    /// Estimate memory for quantized weights.
    public func estimateMemory(parameterCount: Int) -> UInt64 {
        let numGroups = (parameterCount + groupSize - 1) / groupSize

        // Packed weight bits
        let weightBits = parameterCount * bits

        // Scale storage (float16 per group)
        let scaleBits = numGroups * 16

        // Zero-point storage (only for affine mode)
        let zeroBits = mode == .affine ? numGroups * 16 : 0

        return UInt64((weightBits + scaleBits + zeroBits + 7) / 8)
    }
}

// MARK: - Device-Specific Recommendations

extension QuantizationConfig {

    /// Recommended quantization config for a device profile.
    public static func recommended(for device: DeviceProfile) -> QuantizationConfig? {
        switch device {
        case .phone:
            // iPhone: aggressive quantization for memory savings
            return .int4
        case .tablet:
            // iPad: balanced quantization
            return .int8
        case .mac, .macPro:
            // Mac: no quantization needed (plenty of memory)
            return nil
        }
    }

    /// Recommended quantization for a specific model type.
    public static func recommended(
        for modelType: QuantizableModelType,
        device: DeviceProfile
    ) -> QuantizationConfig? {
        switch modelType {
        case .whisper:
            // Whisper: text output tolerant, 4-bit works well
            switch device {
            case .phone: return .int4
            case .tablet: return .int8
            case .mac, .macPro: return nil
            }

        case .htdemucs:
            // HTDemucs: audio quality sensitive, prefer 8-bit
            switch device {
            case .phone: return .audioConservative
            case .tablet: return .int8
            case .mac, .macPro: return nil
            }

        case .musicgen:
            // MusicGen: large model, 4-bit acceptable
            switch device {
            case .phone: return .int4
            case .tablet: return .int4
            case .mac, .macPro: return nil
            }

        case .clap:
            // CLAP: embedding precision matters
            switch device {
            case .phone: return .int8
            case .tablet: return .int8
            case .mac, .macPro: return nil
            }

        case .encodec:
            // EnCodec: codec quality sensitive
            switch device {
            case .phone: return .audioConservative
            case .tablet: return nil
            case .mac, .macPro: return nil
            }
        }
    }
}

/// Model types that support quantization.
public enum QuantizableModelType: String, Codable, Sendable {
    case whisper
    case htdemucs
    case musicgen
    case clap
    case encodec
}

// MARK: - Helpers

extension Int {
    var isPowerOfTwo: Bool {
        self > 0 && (self & (self - 1)) == 0
    }
}
