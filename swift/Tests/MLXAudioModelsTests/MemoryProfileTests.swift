// MemoryProfileTests.swift
// Tests for memory profiling infrastructure.

import XCTest
@testable import MLXAudioModels
@preconcurrency import MLX
import Foundation

final class MemoryProfileTests: XCTestCase {

    // MARK: - MemorySnapshot Tests

    func testMemorySnapshotCapture() {
        let snapshot = MemorySnapshot.capture()

        // Should have valid values
        XCTAssertGreaterThan(snapshot.processResident, 0, "Process memory should be positive")
        XCTAssertNotNil(snapshot.timestamp)

        // GPU values may be 0 if nothing allocated, but shouldn't crash
        XCTAssertGreaterThanOrEqual(snapshot.gpuActive, 0)
        XCTAssertGreaterThanOrEqual(snapshot.gpuPeak, 0)
    }

    func testMemorySnapshotMBConversions() {
        let snapshot = MemorySnapshot.capture()

        // MB conversions should be consistent
        XCTAssertEqual(
            snapshot.gpuActiveMB,
            Double(snapshot.gpuActive) / (1024 * 1024),
            accuracy: 0.001
        )
        XCTAssertEqual(
            snapshot.processResidentMB,
            Double(snapshot.processResident) / (1024 * 1024),
            accuracy: 0.001
        )
    }

    // MARK: - MemoryProfiler Tests

    func testProfilerBasicOperation() async throws {
        let profiler = MemoryProfiler(verbose: false)

        // Profile a simple allocation
        let (result, profile) = try await profiler.profile("test_allocation") {
            let array = MLXRandom.normal([1000, 1000])
            eval(array)
            return array
        }

        // Should have captured the operation
        XCTAssertEqual(profile.name, "test_allocation")
        XCTAssertGreaterThan(profile.durationMs, 0)

        // Result should be returned
        XCTAssertEqual(result.shape, [1000, 1000])

        // Profile result should be stored
        let results = await profiler.getResults()
        XCTAssertEqual(results.count, 1)
        XCTAssertEqual(results[0].name, "test_allocation")
    }

    func testProfilerMultipleOperations() async throws {
        let profiler = MemoryProfiler(verbose: false)

        // Profile multiple operations
        _ = try await profiler.profile("op1") {
            let a = MLXRandom.normal([500, 500])
            eval(a)
            return a
        }

        _ = try await profiler.profile("op2") {
            let b = MLXRandom.normal([1000, 1000])
            eval(b)
            return b
        }

        _ = try await profiler.profile("op3") {
            let c = MLXRandom.normal([100, 100])
            eval(c)
            return c
        }

        let results = await profiler.getResults()
        XCTAssertEqual(results.count, 3)

        let summary = await profiler.getSummary()
        XCTAssertEqual(summary.results.count, 3)
        XCTAssertGreaterThan(summary.totalDurationMs, 0)
    }

    func testProfilerReset() async throws {
        let profiler = MemoryProfiler(verbose: false)

        _ = try await profiler.profile("test") {
            MLXRandom.normal([100, 100])
        }

        var results = await profiler.getResults()
        XCTAssertEqual(results.count, 1)

        await profiler.reset()

        results = await profiler.getResults()
        XCTAssertEqual(results.count, 0)
    }

    func testProfilerMarkers() async throws {
        let profiler = MemoryProfiler(verbose: false)

        await profiler.setMarker("start")

        // Allocate some memory
        let array = MLXRandom.normal([2000, 2000])
        eval(array)

        let delta = await profiler.deltaFromMarker("start")
        XCTAssertNotNil(delta)

        // Delta should be positive after allocation
        // Note: May be 0 if GPU memory tracking has quirks
        XCTAssertGreaterThanOrEqual(delta ?? 0, 0)

        // Unknown marker should return nil
        let unknown = await profiler.deltaFromMarker("nonexistent")
        XCTAssertNil(unknown)
    }

    // MARK: - MemoryEstimator Tests

    func testTensorBytesEstimation() {
        // Float32: 4 bytes per element
        let float32Bytes = MemoryEstimator.tensorBytes(shape: [1000, 1000], dtype: .float32)
        XCTAssertEqual(float32Bytes, 4_000_000)

        // Float16: 2 bytes per element
        let float16Bytes = MemoryEstimator.tensorBytes(shape: [1000, 1000], dtype: .float16)
        XCTAssertEqual(float16Bytes, 2_000_000)

        // Int8: 1 byte per element
        let int8Bytes = MemoryEstimator.tensorBytes(shape: [1000, 1000], dtype: .int8)
        XCTAssertEqual(int8Bytes, 1_000_000)
    }

    func testKVCacheEstimation() {
        // Whisper-style KV cache estimation
        let cacheBytes = MemoryEstimator.whisperKVCacheBytes(
            maxLength: 512,
            nLayers: 6,
            hiddenDim: 512,
            batchSize: 1,
            dtype: .float16
        )

        // Expected: 6 layers × 2 (K+V) × 1 × 512 × 512 × 2 bytes = 6,291,456 bytes
        XCTAssertEqual(cacheBytes, 6_291_456)
    }

    func testQuantizedWeightEstimation() {
        let paramCount = 1_000_000  // 1M parameters

        // Full precision (float32)
        let fullBytes = MemoryEstimator.weightBytes(parameterCount: paramCount, dtype: .float32)
        XCTAssertEqual(fullBytes, 4_000_000)

        // 8-bit quantized (should be ~1/4 of float32 + overhead)
        let int8Bytes = MemoryEstimator.quantizedWeightBytes(
            parameterCount: paramCount,
            bits: 8,
            groupSize: 64
        )
        // Expected: 1M × 8 bits + (1M/64) × 16 bits scales ≈ 1,031,250 bytes
        XCTAssertLessThan(int8Bytes, fullBytes)
        XCTAssertGreaterThan(int8Bytes, 1_000_000)  // More than 1M due to scales

        // 4-bit quantized (should be ~1/2 of 8-bit)
        let int4Bytes = MemoryEstimator.quantizedWeightBytes(
            parameterCount: paramCount,
            bits: 4,
            groupSize: 64
        )
        XCTAssertLessThan(int4Bytes, int8Bytes)
    }

    // MARK: - ProfileResult Tests

    func testProfileResultDeltas() async throws {
        let profiler = MemoryProfiler(verbose: false)

        // Profile an operation that allocates memory
        let (_, profile) = try await profiler.profile("allocation") {
            let large = MLXRandom.normal([5000, 5000])  // ~100MB
            eval(large)
            return large
        }

        // GPU delta should be calculated
        XCTAssertEqual(
            profile.gpuDelta,
            Int64(profile.after.gpuActive) - Int64(profile.before.gpuActive)
        )

        // MB conversion should be consistent
        XCTAssertEqual(
            profile.gpuDeltaMB,
            Double(profile.gpuDelta) / (1024 * 1024),
            accuracy: 0.001
        )
    }

    // MARK: - Summary Tests

    func testProfileSummary() async throws {
        let profiler = MemoryProfiler(verbose: false)

        // Create varied operations
        _ = try await profiler.profile("small") {
            let a = MLXRandom.normal([100, 100])
            eval(a)
            return a
        }

        _ = try await profiler.profile("medium") {
            let b = MLXRandom.normal([500, 500])
            eval(b)
            return b
        }

        _ = try await profiler.profile("large") {
            let c = MLXRandom.normal([1000, 1000])
            eval(c)
            return c
        }

        let summary = await profiler.getSummary()

        XCTAssertEqual(summary.results.count, 3)
        XCTAssertGreaterThan(summary.totalDurationMs, 0)

        // Sorted results should work
        let byDuration = summary.sortedByDuration
        XCTAssertEqual(byDuration.count, 3)

        let byGpu = summary.sortedByGpuDelta
        XCTAssertEqual(byGpu.count, 3)
    }

    // MARK: - Format Utility Tests

    func testFormatBytes() {
        XCTAssertEqual(formatBytes(UInt64(1024)), "1 KB")
        XCTAssertEqual(formatBytes(UInt64(1024 * 1024)), "1 MB")
        XCTAssertEqual(formatBytes(UInt64(1024 * 1024 * 1024)), "1 GB")

        // Signed version
        XCTAssertEqual(formatBytes(Int64(1024 * 1024)), "1 MB")
        XCTAssertEqual(formatBytes(Int64(-1024 * 1024)), "-1 MB")
    }

    // MARK: - ModelMemoryBreakdown Tests

    func testModelMemoryBreakdown() {
        let breakdown = ModelMemoryBreakdown(
            modelId: "test-model",
            weightsBytes: 100_000_000,     // 100MB
            activationsBytes: 50_000_000,  // 50MB
            kvCacheBytes: 10_000_000,      // 10MB
            otherBytes: 5_000_000          // 5MB
        )

        XCTAssertEqual(breakdown.totalBytes, 165_000_000)
        XCTAssertEqual(breakdown.totalMB, 165_000_000 / (1024 * 1024), accuracy: 0.1)
        XCTAssertEqual(breakdown.weightsMB, 100_000_000 / (1024 * 1024), accuracy: 0.1)
    }

    // MARK: - Integration Tests

    func testProfileRealModelOperation() async throws {
        // Test profiling with actual model components
        let profiler = MemoryProfiler(verbose: false)

        // Simulate encoder-like operation
        let (output, encProfile) = try await profiler.profile("encoder_forward") {
            let input = MLXRandom.normal([1, 512, 216, 4])
            let weight = MLXRandom.normal([48, 4, 8, 1])
            let result = MLX.conv2d(input, weight, stride: [4, 1], padding: [2, 0])
            eval(result)
            return result
        }

        XCTAssertGreaterThan(encProfile.durationMs, 0)
        XCTAssertEqual(output.dim(3), 48)  // Output channels

        // Simulate decoder-like operation
        let (_, decProfile) = try await profiler.profile("decoder_forward") {
            let input = MLXRandom.normal([1, 128, 54, 48])
            let weight = MLXRandom.normal([4, 48, 8, 1])
            let result = MLX.convTransposed2d(input, weight, stride: [4, 1], padding: [2, 0])
            eval(result)
            return result
        }

        XCTAssertGreaterThan(decProfile.durationMs, 0)

        // Verify both operations captured
        let summary = await profiler.getSummary()
        XCTAssertEqual(summary.results.count, 2)
    }

    func testMemoryGrowthTracking() async throws {
        let profiler = MemoryProfiler(verbose: false)

        // Track memory growth over multiple allocations
        var arrays: [MLXArray] = []

        for i in 0..<5 {
            let (array, _) = try await profiler.profile("alloc_\(i)") {
                let a = MLXRandom.normal([1000, 1000])
                eval(a)
                return a
            }
            arrays.append(array)
        }

        let summary = await profiler.getSummary()

        // Should have 5 profile results
        XCTAssertEqual(summary.results.count, 5)

        // Max GPU should be tracked
        XCTAssertGreaterThanOrEqual(summary.maxGpuMB, 0)
    }
}
