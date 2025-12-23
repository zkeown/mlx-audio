// LoRATests.swift
// Tests for LoRA implementation.

import XCTest
import MLX
import MLXNN
@testable import MLXAudioTraining

final class LoRATests: XCTestCase {

    // MARK: - LoRALinear Tests

    func testLoRALinearCreation() {
        let base = Linear(64, 32)
        let lora = LoRALinear(base: base, rank: 8)

        XCTAssertEqual(lora.rank, 8)
        XCTAssertEqual(lora.inFeatures, 64)
        XCTAssertEqual(lora.outFeatures, 32)
        XCTAssertEqual(lora.alpha, 8)  // Default: alpha = rank
    }

    func testLoRALinearTrainableParams() {
        let base = Linear(64, 32)
        let lora = LoRALinear(base: base, rank: 8)

        // Trainable: A (8x64) + B (32x8) = 512 + 256 = 768
        XCTAssertEqual(lora.trainableParameterCount, 768)
    }

    func testLoRALinearCompressionRatio() {
        let base = Linear(1024, 512, bias: true)
        let lora = LoRALinear(base: base, rank: 8)

        // Total: 1024*512 + 512 + 8*1024 + 512*8 = 524288 + 512 + 8192 + 4096
        // Trainable: 8*1024 + 512*8 = 8192 + 4096 = 12288
        XCTAssertGreaterThan(lora.compressionRatio, 40)  // ~42x compression
    }

    func testLoRALinearForwardShape() {
        let base = Linear(64, 32)
        let lora = LoRALinear(base: base, rank: 8)

        let input = MLXArray.zeros([4, 64])
        let output = lora(input)

        XCTAssertEqual(output.shape, [4, 32])
    }

    func testLoRALinearInitialOutputEqualsBase() {
        let base = Linear(64, 32)
        let lora = LoRALinear(base: base, rank: 8)

        let input = MLXRandom.uniform(low: 0.0, high: 1.0, [4, 64])

        // Since B is initialized to zeros, LoRA output should equal base output
        let baseOutput = base(input)
        let loraOutput = lora(input)

        eval(baseOutput, loraOutput)

        // Check outputs are approximately equal
        let diff = MLX.max(MLX.abs(MLX.subtract(baseOutput, loraOutput)))
        eval(diff)
        XCTAssertLessThan(diff.item(Float.self), 1e-5)
    }

    func testLoRALinearMerge() {
        let base = Linear(64, 32)
        let lora = LoRALinear(base: base, rank: 8)

        // Train the LoRA by calling forward a few times (to trigger any lazy init)
        let input = MLXRandom.uniform(low: 0.0, high: 1.0, [4, 64])
        let loraOutput = lora(input)

        // Merge LoRA into base
        let merged = lora.merge()

        let mergedOutput = merged(input)

        eval(loraOutput, mergedOutput)

        // Merged output should equal LoRA output (when B is zeros, they're equal)
        let diff = MLX.max(MLX.abs(MLX.subtract(loraOutput, mergedOutput)))
        eval(diff)
        XCTAssertLessThan(diff.item(Float.self), 1e-4)
    }

    func testLoRALinearReset() {
        let base = Linear(64, 32)
        let lora = LoRALinear(base: base, rank: 8)

        // Reset
        lora.resetLoRA()

        // B should be zeros
        let bSum = MLX.sum(MLX.abs(lora.loraB))
        eval(bSum)
        XCTAssertEqual(bSum.item(Float.self), 0)
    }

    // MARK: - LoRA Config Tests

    func testLoRAConfigDefault() {
        let config = LoRAConfig()
        XCTAssertEqual(config.rank, 8)
        XCTAssertEqual(config.dropout, 0)
        XCTAssertTrue(config.targetModules.contains("query"))
        XCTAssertTrue(config.targetModules.contains("value"))
    }

    func testLoRAConfigAttention() {
        let config = LoRAConfig.attention
        XCTAssertEqual(config.rank, 8)
        XCTAssertTrue(config.targetModules.contains("out_proj"))
    }

    func testLoRAConfigAggressive() {
        let config = LoRAConfig.aggressive
        XCTAssertEqual(config.rank, 16)
    }

    // MARK: - LoRA Stats Tests

    func testLoRAStatsEmpty() {
        // Create a simple module without LoRA
        let linear = Linear(64, 32)

        let stats = getLoRAStats(for: linear)

        XCTAssertEqual(stats.loraLayerCount, 0)
        XCTAssertEqual(stats.linearLayerCount, 1)
        XCTAssertEqual(stats.trainableLoRAParams, 0)
    }

    // MARK: - Linear Extension Tests

    func testLinearWithLoRA() {
        let linear = Linear(64, 32)
        let lora = linear.withLoRA(rank: 4, alpha: 8)

        XCTAssertEqual(lora.rank, 4)
        XCTAssertEqual(lora.alpha, 8)
    }
}
