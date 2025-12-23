// QuantizationTests.swift
// Tests for weight quantization infrastructure.

import XCTest
@testable import MLXAudioModels
import MLX
import MLXNN
import Foundation

final class QuantizationTests: XCTestCase {

    // MARK: - QuantizationConfig Tests

    func testQuantizationConfigPresets() {
        // Test int4 preset
        let int4 = QuantizationConfig.int4
        XCTAssertEqual(int4.bits, 4)
        XCTAssertEqual(int4.groupSize, 64)
        XCTAssertEqual(int4.mode, .affine)

        // Test int8 preset
        let int8 = QuantizationConfig.int8
        XCTAssertEqual(int8.bits, 8)
        XCTAssertEqual(int8.groupSize, 64)
        XCTAssertEqual(int8.mode, .affine)

        // Test audio conservative preset
        let conservative = QuantizationConfig.audioConservative
        XCTAssertEqual(conservative.bits, 8)
        XCTAssertFalse(conservative.skipLayers.isEmpty)
    }

    func testCompressionRatio() {
        let int4 = QuantizationConfig.int4
        let int8 = QuantizationConfig.int8

        // 4-bit should have ~4-8x compression
        XCTAssertGreaterThan(int4.compressionRatio, 4.0)
        XCTAssertLessThan(int4.compressionRatio, 8.0)

        // 8-bit should have ~2-4x compression
        XCTAssertGreaterThan(int8.compressionRatio, 2.0)
        XCTAssertLessThan(int8.compressionRatio, 4.0)

        // 4-bit should compress better than 8-bit
        XCTAssertGreaterThan(int4.compressionRatio, int8.compressionRatio)
    }

    func testMemoryEstimation() {
        let config = QuantizationConfig.int4
        let paramCount = 1_000_000  // 1M parameters

        let estimated = config.estimateMemory(parameterCount: paramCount)

        // Should be significantly less than float32 (4MB)
        XCTAssertLessThan(estimated, 4_000_000)

        // Should be more than pure 4-bit (500KB) due to scale overhead
        XCTAssertGreaterThan(estimated, 500_000)
    }

    func testDeviceRecommendations() {
        // Phone should get quantization
        let phoneConfig = QuantizationConfig.recommended(for: .phone)
        XCTAssertNotNil(phoneConfig)
        XCTAssertEqual(phoneConfig?.bits, 4)

        // Tablet should get 8-bit
        let tabletConfig = QuantizationConfig.recommended(for: .tablet)
        XCTAssertNotNil(tabletConfig)
        XCTAssertEqual(tabletConfig?.bits, 8)

        // Mac should not get quantization
        let macConfig = QuantizationConfig.recommended(for: .mac)
        XCTAssertNil(macConfig)
    }

    func testModelSpecificRecommendations() {
        // Whisper on phone should be 4-bit
        let whisperPhone = QuantizationConfig.recommended(for: .whisper, device: .phone)
        XCTAssertEqual(whisperPhone?.bits, 4)

        // HTDemucs (audio-sensitive) should be 8-bit on phone
        let htdemucsPhone = QuantizationConfig.recommended(for: .htdemucs, device: .phone)
        XCTAssertEqual(htdemucsPhone?.bits, 8)

        // EnCodec on tablet should be nil (small enough)
        let encodecTablet = QuantizationConfig.recommended(for: .encodec, device: .tablet)
        XCTAssertNil(encodecTablet)
    }

    // MARK: - QuantizedLinear Tests

    func testQuantizedLinearCreation() {
        let layer = QuantizedLinear(
            inputDim: 512,
            outputDim: 1024,
            bias: true,
            config: .int4
        )

        XCTAssertEqual(layer.inputDim, 512)
        XCTAssertEqual(layer.outputDim, 1024)
        XCTAssertNotNil(layer.bias)
        XCTAssertEqual(layer.config.bits, 4)
    }

    func testQuantizedLinearNoBias() {
        let layer = QuantizedLinear(
            inputDim: 256,
            outputDim: 512,
            bias: false,
            config: .int8
        )

        XCTAssertNil(layer.bias)
    }

    func testQuantize4Bit() {
        let inputDim = 128
        let outputDim = 256

        // Create random weights
        let weights = MLXRandom.normal([outputDim, inputDim])
        eval(weights)

        // Create and quantize
        let layer = QuantizedLinear(
            inputDim: inputDim,
            outputDim: outputDim,
            bias: false,
            config: .int4
        )
        layer.quantize(from: weights)

        // Dequantize and check shape
        let dequantized = layer.dequantize()
        XCTAssertEqual(dequantized.shape, [outputDim, inputDim])

        // Check error is reasonable (4-bit should have some error)
        let error = MLX.max(MLX.abs(weights.asType(.float16) - dequantized)).item(Float.self)
        XCTAssertLessThan(error, 1.0, "4-bit quantization error should be bounded")
    }

    func testQuantize8Bit() {
        let inputDim = 128
        let outputDim = 256

        let weights = MLXRandom.normal([outputDim, inputDim])
        eval(weights)

        let layer = QuantizedLinear(
            inputDim: inputDim,
            outputDim: outputDim,
            bias: false,
            config: .int8
        )
        layer.quantize(from: weights)

        let dequantized = layer.dequantize()
        XCTAssertEqual(dequantized.shape, [outputDim, inputDim])

        // 8-bit should have lower error than 4-bit
        let error = MLX.max(MLX.abs(weights.asType(.float16) - dequantized)).item(Float.self)
        XCTAssertLessThan(error, 0.1, "8-bit quantization error should be small")
    }

    func testQuantizedLinearForward() {
        let inputDim = 64
        let outputDim = 128
        let batchSize = 2
        let seqLen = 10

        // Create and quantize layer
        let weights = MLXRandom.normal([outputDim, inputDim])
        let bias = MLXRandom.normal([outputDim])
        eval(weights, bias)

        let layer = QuantizedLinear(
            inputDim: inputDim,
            outputDim: outputDim,
            bias: true,
            config: .int8
        )
        layer.quantize(from: weights)
        layer.bias = bias.asType(.float16)

        // Forward pass
        let input = MLXRandom.normal([batchSize, seqLen, inputDim])
        eval(input)

        let output = layer(input)

        // Check output shape
        XCTAssertEqual(output.shape, [batchSize, seqLen, outputDim])
    }

    func testMemorySavings() {
        let inputDim = 1024
        let outputDim = 4096

        let layer = QuantizedLinear(
            inputDim: inputDim,
            outputDim: outputDim,
            bias: false,
            config: .int4
        )

        // Memory savings should be significant
        XCTAssertGreaterThan(layer.memorySavingsRatio, 3.0)

        // Memory bytes should be less than float32
        let fullPrecisionBytes = UInt64(inputDim * outputDim * 4)
        XCTAssertLessThan(layer.memoryBytes, fullPrecisionBytes)
    }

    func testQuantizedFromLinear() {
        let inputDim = 256
        let outputDim = 512

        // Create a regular Linear layer
        let linear = Linear(inputDim, outputDim)

        // Convert to quantized
        let quantized = QuantizedLinear.from(linear, config: .int8)

        XCTAssertEqual(quantized.inputDim, inputDim)
        XCTAssertEqual(quantized.outputDim, outputDim)
        XCTAssertNotNil(quantized.bias)

        // Forward should work
        let input = MLXRandom.normal([1, 10, inputDim])
        let output = quantized(input)
        XCTAssertEqual(output.shape, [1, 10, outputDim])
    }

    // MARK: - ModelQuantizer Tests

    func testEstimateMemorySavings() {
        // Create a simple model with Linear layers
        let model = Sequential([
            Linear(512, 1024),
            ReLU(),
            Linear(1024, 512),
        ])

        let (original, quantized, ratio) = ModelQuantizer.estimateMemorySavings(
            for: model,
            config: .int4
        )

        XCTAssertGreaterThan(original, 0)
        XCTAssertGreaterThan(quantized, 0)
        XCTAssertLessThan(quantized, original)
        XCTAssertGreaterThan(ratio, 1.0)
    }

    func testQuantizationError() {
        let inputDim = 128
        let outputDim = 256

        let weights = MLXRandom.normal([outputDim, inputDim])
        eval(weights)

        let layer = QuantizedLinear(
            inputDim: inputDim,
            outputDim: outputDim,
            bias: false,
            config: .int8
        )
        layer.quantize(from: weights)

        let maxError = ModelQuantizer.maxQuantizationError(original: weights, quantized: layer)
        let meanError = ModelQuantizer.meanQuantizationError(original: weights, quantized: layer)

        XCTAssertGreaterThan(maxError, 0)
        XCTAssertGreaterThan(meanError, 0)
        XCTAssertGreaterThan(maxError, meanError)  // Max should be >= mean
    }

    // MARK: - Edge Cases

    func testQuantizationWithPadding() {
        // Input dimension not divisible by group size (64)
        let inputDim = 100  // Not divisible by 64
        let outputDim = 128

        let weights = MLXRandom.normal([outputDim, inputDim])
        eval(weights)

        let layer = QuantizedLinear(
            inputDim: inputDim,
            outputDim: outputDim,
            bias: false,
            config: .int4
        )
        layer.quantize(from: weights)

        let dequantized = layer.dequantize()

        // Should still match original dimensions
        XCTAssertEqual(dequantized.shape, [outputDim, inputDim])
    }

    func testLargeLayerQuantization() {
        // Test with a larger layer (more realistic)
        let inputDim = 4096
        let outputDim = 4096

        let weights = MLXRandom.normal([outputDim, inputDim])
        eval(weights)

        let layer = QuantizedLinear(
            inputDim: inputDim,
            outputDim: outputDim,
            bias: false,
            config: .int4
        )
        layer.quantize(from: weights)

        // Memory should be much less than float32
        let fullPrecisionMB = Double(inputDim * outputDim * 4) / (1024 * 1024)
        let quantizedMB = Double(layer.memoryBytes) / (1024 * 1024)

        XCTAssertLessThan(quantizedMB, fullPrecisionMB / 3)
    }

    // MARK: - QuantizationReport Tests

    func testQuantizationReport() {
        let report = QuantizationReport(
            modelId: "test-model",
            config: .int4,
            layersQuantized: 10,
            layersSkipped: 2,
            originalBytes: 100_000_000,
            quantizedBytes: 30_000_000,
            layerErrors: ["layer1": 0.01, "layer2": 0.02]
        )

        XCTAssertEqual(report.layersQuantized, 10)
        XCTAssertEqual(report.savedBytes, 70_000_000)
        XCTAssertGreaterThan(report.savingsRatio, 3.0)
        XCTAssertEqual(report.maxError, 0.02)
    }
}

// MARK: - Helper Classes for Testing

/// Simple ReLU for testing.
private class ReLU: Module, @unchecked Sendable {
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLX.maximum(x, MLXArray(0))
    }
}

/// Simple Sequential for testing.
private class Sequential: Module, @unchecked Sendable {
    let layers: [Module]

    init(_ layers: [Module]) {
        self.layers = layers
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var result = x
        for layer in layers {
            if let linear = layer as? Linear {
                result = linear(result)
            } else if let relu = layer as? ReLU {
                result = relu(result)
            }
        }
        return result
    }
}
