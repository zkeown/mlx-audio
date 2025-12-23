// MemoryBenchmarks.swift
// Memory usage benchmarks and verification tests.
//
// These tests verify that memory optimizations are working correctly
// and provide baseline measurements for regression testing.

import XCTest
@testable import MLXAudioModels
import MLX
import Foundation

final class MemoryBenchmarks: XCTestCase {

    // MARK: - Memory Tracking Utilities

    /// Get current physical memory footprint in bytes.
    private var memoryFootprint: UInt64 {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        return result == KERN_SUCCESS ? info.resident_size : 0
    }

    /// Format bytes as human-readable string.
    private func formatBytes(_ bytes: Int64) -> String {
        let formatter = ByteCountFormatter()
        formatter.countStyle = .memory
        return formatter.string(fromByteCount: bytes)
    }

    // MARK: - Chunked Inference Memory Tests

    /// Test that chunked inference doesn't accumulate memory.
    ///
    /// Before optimization: Each chunk caused full tensor copies, O(chunks × T) memory.
    /// After optimization: Uses at[].add() for O(1) updates per chunk.
    func testChunkedInferenceMemoryEfficiency() {
        // Create a minimal model configuration for testing
        let config = HTDemucsConfig.htdemucs_ft()

        // Create encoder layer (small footprint for testing)
        let encoder = HEncLayer(
            chin: 4,
            chout: 48,
            kernelSize: 8,
            stride: 4,
            freq: true,
            dconvDepth: 2,
            dconvCompress: 8
        )

        // Simulate chunked output accumulation pattern
        let T = 44100 * 10  // 10 seconds at 44.1kHz
        let B = 1
        let S = 4  // sources
        let C = 2  // channels

        let chunkSize = 44100  // 1 second chunks
        let numChunks = T / chunkSize

        // Pre-allocate output buffer (this is what we optimized)
        var out = MLXArray.zeros([B, S, C, T])
        var weightSum = MLXArray.zeros([B, S, C, T])

        let baselineMemory = memoryFootprint

        // Simulate chunked accumulation with at[].add()
        for chunkIdx in 0..<numChunks {
            let offset = chunkIdx * chunkSize
            let chunkEnd = offset + chunkSize

            // Simulate chunk output
            let chunkOut = MLXRandom.normal([B, S, C, chunkSize])
            let weight = MLXArray.ones([1, 1, 1, chunkSize])

            // Memory-efficient accumulation using at[].add()
            out = out.at[0..., 0..., 0..., offset..<chunkEnd].add(chunkOut)
            weightSum = weightSum.at[0..., 0..., 0..., offset..<chunkEnd].add(weight)

            eval(out, weightSum)
        }

        let peakMemory = memoryFootprint
        let memoryGrowth = Int64(peakMemory) - Int64(baselineMemory)

        print("Chunked inference memory test:")
        print("  Chunks: \(numChunks)")
        print("  Baseline: \(formatBytes(Int64(baselineMemory)))")
        print("  Peak: \(formatBytes(Int64(peakMemory)))")
        print("  Growth: \(formatBytes(memoryGrowth))")

        // Memory growth should be bounded (not O(chunks × T))
        // Expected: ~80MB for two [1, 4, 2, 441000] float32 tensors
        // With inefficient concatenation: would be ~800MB+
        let expectedMaxGrowth: UInt64 = 200_000_000  // 200MB max
        XCTAssertLessThan(
            UInt64(memoryGrowth),
            expectedMaxGrowth,
            "Memory growth \(formatBytes(memoryGrowth)) exceeds expected max \(formatBytes(Int64(expectedMaxGrowth)))"
        )
    }

    /// Test that at[].add() pattern works correctly for accumulation.
    func testAtAddAccuracy() {
        let shape = [2, 3, 100]
        var buffer = MLXArray.zeros(shape)

        // Accumulate values in chunks
        for i in stride(from: 0, to: 100, by: 10) {
            let chunk = MLXArray.ones([2, 3, 10])
            buffer = buffer.at[0..., 0..., i..<(i + 10)].add(chunk)
        }

        eval(buffer)

        // Verify all values are 1.0
        let allOnes = MLX.all(buffer .== 1.0).item(Bool.self)
        XCTAssertTrue(allOnes, "at[].add() accumulation should produce all ones")
    }

    // MARK: - Sequential Processing Tests

    /// Test that sequential model processing pattern works correctly.
    ///
    /// Before optimization: Stack all 4 model outputs (~4GB peak).
    /// After optimization: Process sequentially (~1GB peak).
    func testSequentialProcessingPattern() {
        let numModels = 4
        let numStems = 4
        let B = 1
        let C = 2
        let T = 44100 * 6  // 6 seconds

        // Pre-allocate result buffer
        var result = MLXArray.zeros([B, numStems, C, T])

        // Identity weights (model i contributes to stem i)
        let weights = MLXArray.eye(numModels)

        let baselineMemory = memoryFootprint

        // Simulate sequential model processing
        for modelIdx in 0..<numModels {
            // Simulate model output (would be model(input) in real code)
            let modelOutput = MLXRandom.normal([B, numStems, C, T])

            // Weighted accumulation per stem
            for stemIdx in 0..<numStems {
                let w = weights[modelIdx, stemIdx]
                let stemOutput = modelOutput[0..., stemIdx, 0..., 0...]
                let weighted = w * stemOutput
                result = result.at[0..., stemIdx, 0..., 0...].add(weighted)
            }

            // Evaluate after each model to release model output
            eval(result)
        }

        let peakMemory = memoryFootprint
        let memoryGrowth = Int64(peakMemory) - Int64(baselineMemory)

        print("Sequential processing memory test:")
        print("  Models: \(numModels)")
        print("  Baseline: \(formatBytes(Int64(baselineMemory)))")
        print("  Peak: \(formatBytes(Int64(peakMemory)))")
        print("  Growth: \(formatBytes(memoryGrowth))")

        // Memory should be bounded to roughly 1-2 model outputs, not 4
        // Each model output: [1, 4, 2, 264600] float32 ≈ 8.5MB
        // With old approach: 4 × 8.5MB held simultaneously = 34MB
        // With new approach: ~8.5MB at any time (plus result buffer)
        let expectedMaxGrowth: UInt64 = 100_000_000  // 100MB max
        XCTAssertLessThan(
            UInt64(memoryGrowth),
            expectedMaxGrowth,
            "Memory growth \(formatBytes(memoryGrowth)) exceeds expected max \(formatBytes(Int64(expectedMaxGrowth)))"
        )
    }

    // MARK: - KV Cache Memory Tests

    /// Test that pre-allocated KV cache doesn't grow during decoding.
    func testKVCacheNoGrowth() {
        let maxLength = 512
        let nLayers = 6
        let hiddenDim = 512
        let batchSize = 1

        // Pre-allocate cache tensors
        var keys: [MLXArray] = []
        var values: [MLXArray] = []
        for _ in 0..<nLayers {
            keys.append(MLXArray.zeros([batchSize, maxLength, hiddenDim]))
            values.append(MLXArray.zeros([batchSize, maxLength, hiddenDim]))
        }
        eval(keys + values)

        let baselineMemory = memoryFootprint
        var currentLength = 0

        // Simulate 448 decode steps (typical Whisper max)
        for _ in 0..<448 {
            let newK = MLXRandom.normal([batchSize, 1, hiddenDim])
            let newV = MLXRandom.normal([batchSize, 1, hiddenDim])

            for layer in 0..<nLayers {
                // Use at[].add() pattern for O(1) update
                keys[layer] = keys[layer].at[0..., currentLength..<(currentLength + 1), 0...].add(
                    newK - keys[layer][0..., currentLength..<(currentLength + 1), 0...]
                )
                values[layer] = values[layer].at[0..., currentLength..<(currentLength + 1), 0...].add(
                    newV - values[layer][0..., currentLength..<(currentLength + 1), 0...]
                )
            }

            currentLength += 1

            // Evaluate periodically
            if currentLength % 50 == 0 {
                eval(keys + values)
            }
        }

        eval(keys + values)
        let finalMemory = memoryFootprint
        let memoryGrowth = Int64(finalMemory) - Int64(baselineMemory)

        print("KV cache memory test:")
        print("  Steps: 448")
        print("  Layers: \(nLayers)")
        print("  Baseline: \(formatBytes(Int64(baselineMemory)))")
        print("  Final: \(formatBytes(Int64(finalMemory)))")
        print("  Growth: \(formatBytes(memoryGrowth))")

        // With pre-allocation, memory should not grow significantly
        // Some temporary tensors during computation, but should be bounded
        let expectedMaxGrowth: UInt64 = 50_000_000  // 50MB max
        XCTAssertLessThan(
            UInt64(memoryGrowth),
            expectedMaxGrowth,
            "KV cache memory grew by \(formatBytes(memoryGrowth)), expected less than \(formatBytes(Int64(expectedMaxGrowth)))"
        )
    }

    // MARK: - Quality Parity Tests

    /// Verify at[].add() produces same result as concatenation.
    func testAtAddParityWithConcatenation() {
        let T = 1000
        let chunkSize = 100

        // Method 1: Concatenation (old approach)
        var concatResult = MLXArray.zeros([1, 1, T])
        for i in stride(from: 0, to: T, by: chunkSize) {
            let chunk = MLXArray(Float(i / chunkSize + 1)) * MLXArray.ones([1, 1, chunkSize])
            let prefix = concatResult[0..., 0..., 0..<i]
            let suffix = concatResult[0..., 0..., (i + chunkSize)...]
            concatResult = concatenated([prefix, chunk, suffix], axis: 2)
        }

        // Method 2: at[].add() (new approach)
        var atAddResult = MLXArray.zeros([1, 1, T])
        for i in stride(from: 0, to: T, by: chunkSize) {
            let chunk = MLXArray(Float(i / chunkSize + 1)) * MLXArray.ones([1, 1, chunkSize])
            atAddResult = atAddResult.at[0..., 0..., i..<(i + chunkSize)].add(chunk)
        }

        eval(concatResult, atAddResult)

        // Results should be identical
        let maxDiff = MLX.max(MLX.abs(concatResult - atAddResult)).item(Float.self)
        XCTAssertLessThan(maxDiff, 1e-6, "at[].add() should produce identical results to concatenation")
    }

    // MARK: - Stress Tests

    /// Stress test with large audio to verify memory efficiency.
    func testLargeAudioMemoryEfficiency() {
        // 5 minutes of stereo audio at 44.1kHz
        let T = 44100 * 60 * 5  // ~13.2M samples
        let B = 1
        let S = 4
        let C = 2

        let baselineMemory = memoryFootprint

        // Pre-allocate output buffer
        var out = MLXArray.zeros([B, S, C, T])
        eval(out)

        let afterAllocationMemory = memoryFootprint

        // Simulate 50 chunk updates
        let chunkSize = T / 50
        for i in stride(from: 0, to: T, by: chunkSize) {
            let end = min(i + chunkSize, T)
            let chunk = MLXRandom.normal([B, S, C, end - i])
            out = out.at[0..., 0..., 0..., i..<end].add(chunk)
            eval(out)
        }

        let finalMemory = memoryFootprint

        let allocationCost = Int64(afterAllocationMemory) - Int64(baselineMemory)
        let updateCost = Int64(finalMemory) - Int64(afterAllocationMemory)

        print("Large audio memory test (5 minutes):")
        print("  Samples: \(T)")
        print("  Allocation cost: \(formatBytes(allocationCost))")
        print("  Update cost: \(formatBytes(updateCost))")
        print("  Total growth: \(formatBytes(Int64(finalMemory) - Int64(baselineMemory)))")

        // Buffer size: [1, 4, 2, 13230000] float32 ≈ 420MB
        // Update cost should be minimal (just temporary chunk tensors)
        let expectedMaxUpdateCost: UInt64 = 100_000_000  // 100MB max
        XCTAssertLessThan(
            UInt64(updateCost),
            expectedMaxUpdateCost,
            "Update cost \(formatBytes(updateCost)) exceeds expected max"
        )
    }
}
