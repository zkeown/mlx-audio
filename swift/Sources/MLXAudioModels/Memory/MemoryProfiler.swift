// MemoryProfiler.swift
// Memory profiling infrastructure for MLX audio models.
//
// Provides detailed memory tracking for optimizing model performance
// on memory-constrained devices like iPhone and iPad.

import Foundation
import MLX

/// Memory snapshot capturing GPU and process memory state.
public struct MemorySnapshot: Sendable {
    /// GPU active memory in bytes (from GPU.activeMemory).
    public let gpuActive: UInt64

    /// GPU peak memory in bytes (from GPU.peakMemory).
    public let gpuPeak: UInt64

    /// GPU cache memory in bytes (from GPU.cacheMemory).
    public let gpuCache: UInt64

    /// Process resident memory in bytes (from mach_task_basic_info).
    public let processResident: UInt64

    /// Timestamp when snapshot was taken.
    public let timestamp: Date

    /// Total GPU memory usage (active + cache).
    public var gpuTotal: UInt64 {
        gpuActive + gpuCache
    }

    /// GPU active memory in megabytes.
    public var gpuActiveMB: Double {
        Double(gpuActive) / (1024 * 1024)
    }

    /// GPU peak memory in megabytes.
    public var gpuPeakMB: Double {
        Double(gpuPeak) / (1024 * 1024)
    }

    /// Process resident memory in megabytes.
    public var processResidentMB: Double {
        Double(processResident) / (1024 * 1024)
    }

    /// Create a snapshot of current memory state.
    public static func capture() -> MemorySnapshot {
        MemorySnapshot(
            gpuActive: UInt64(GPU.activeMemory),
            gpuPeak: UInt64(GPU.peakMemory),
            gpuCache: UInt64(GPU.cacheMemory),
            processResident: Self.getProcessMemory(),
            timestamp: Date()
        )
    }

    /// Get process resident memory using mach_task_basic_info.
    private static func getProcessMemory() -> UInt64 {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        return result == KERN_SUCCESS ? info.resident_size : 0
    }
}

/// Profiling result for a single operation.
public struct ProfileResult: Sendable {
    /// Name of the profiled operation.
    public let name: String

    /// Snapshot before operation.
    public let before: MemorySnapshot

    /// Snapshot after operation.
    public let after: MemorySnapshot

    /// Duration of operation in milliseconds.
    public let durationMs: Double

    /// GPU memory delta (after - before) in bytes.
    public var gpuDelta: Int64 {
        Int64(after.gpuActive) - Int64(before.gpuActive)
    }

    /// GPU memory delta in megabytes.
    public var gpuDeltaMB: Double {
        Double(gpuDelta) / (1024 * 1024)
    }

    /// Process memory delta in bytes.
    public var processDelta: Int64 {
        Int64(after.processResident) - Int64(before.processResident)
    }

    /// Peak GPU memory during operation (from after.gpuPeak).
    public var peakGpuMB: Double {
        after.gpuPeakMB
    }
}

/// Actor for thread-safe memory profiling.
public actor MemoryProfiler {

    /// Collected profile results.
    private var results: [ProfileResult] = []

    /// Named memory markers for tracking allocations.
    private var markers: [String: UInt64] = [:]

    /// Whether verbose logging is enabled.
    public var verbose: Bool

    public init(verbose: Bool = false) {
        self.verbose = verbose
    }

    /// Profile an operation and return its result along with the operation's return value.
    ///
    /// - Parameters:
    ///   - name: Name for this profiled operation
    ///   - body: The operation to profile
    /// - Returns: Tuple of (operation result, profile result)
    public func profile<T>(
        _ name: String,
        body: () throws -> T
    ) throws -> (T, ProfileResult) {
        // Clear cache and reset peak to get accurate measurements
        GPU.clearCache()
        GPU.resetPeakMemory()

        let before = MemorySnapshot.capture()
        let start = CFAbsoluteTimeGetCurrent()

        let result = try body()

        // Force GPU sync for accurate timing
        eval()

        let end = CFAbsoluteTimeGetCurrent()
        let after = MemorySnapshot.capture()

        let profileResult = ProfileResult(
            name: name,
            before: before,
            after: after,
            durationMs: (end - start) * 1000
        )

        results.append(profileResult)

        if verbose {
            printResult(profileResult)
        }

        return (result, profileResult)
    }

    /// Profile an async operation.
    public func profileAsync<T: Sendable>(
        _ name: String,
        body: @Sendable () async throws -> T
    ) async throws -> (T, ProfileResult) {
        GPU.clearCache()
        GPU.resetPeakMemory()

        let before = MemorySnapshot.capture()
        let start = CFAbsoluteTimeGetCurrent()

        let result = try await body()

        eval()

        let end = CFAbsoluteTimeGetCurrent()
        let after = MemorySnapshot.capture()

        let profileResult = ProfileResult(
            name: name,
            before: before,
            after: after,
            durationMs: (end - start) * 1000
        )

        results.append(profileResult)

        if verbose {
            printResult(profileResult)
        }

        return (result, profileResult)
    }

    /// Set a named memory marker at current GPU memory level.
    public func setMarker(_ name: String) {
        markers[name] = UInt64(GPU.activeMemory)
    }

    /// Get memory delta since a named marker.
    public func deltaFromMarker(_ name: String) -> Int64? {
        guard let markerMemory = markers[name] else { return nil }
        return Int64(GPU.activeMemory) - Int64(markerMemory)
    }

    /// Get all collected profile results.
    public func getResults() -> [ProfileResult] {
        results
    }

    /// Clear all collected results and markers.
    public func reset() {
        results.removeAll()
        markers.removeAll()
    }

    /// Get a summary of all profiled operations.
    public func getSummary() -> MemoryProfileSummary {
        MemoryProfileSummary(results: results)
    }

    /// Print a formatted result to console.
    private func printResult(_ result: ProfileResult) {
        print("""
            [\(result.name)]
              Duration: \(String(format: "%.2f", result.durationMs))ms
              GPU: \(String(format: "%.2f", result.before.gpuActiveMB))MB -> \
            \(String(format: "%.2f", result.after.gpuActiveMB))MB \
            (delta: \(String(format: "%+.2f", result.gpuDeltaMB))MB)
              Peak GPU: \(String(format: "%.2f", result.peakGpuMB))MB
              Process: \(String(format: "%.2f", result.after.processResidentMB))MB
            """)
    }
}

/// Summary of profiled operations.
public struct MemoryProfileSummary: Sendable {
    public let results: [ProfileResult]

    /// Total duration of all operations.
    public var totalDurationMs: Double {
        results.reduce(0) { $0 + $1.durationMs }
    }

    /// Maximum GPU memory observed.
    public var maxGpuMB: Double {
        results.map { $0.after.gpuActiveMB }.max() ?? 0
    }

    /// Maximum peak GPU memory observed.
    public var maxPeakGpuMB: Double {
        results.map { $0.peakGpuMB }.max() ?? 0
    }

    /// Get results sorted by GPU delta (largest first).
    public var sortedByGpuDelta: [ProfileResult] {
        results.sorted { $0.gpuDelta > $1.gpuDelta }
    }

    /// Get results sorted by duration (longest first).
    public var sortedByDuration: [ProfileResult] {
        results.sorted { $0.durationMs > $1.durationMs }
    }

    /// Print formatted summary to console.
    public func printSummary() {
        print("\n=== Memory Profile Summary ===")
        print("Total operations: \(results.count)")
        print("Total duration: \(String(format: "%.2f", totalDurationMs))ms")
        print("Max GPU memory: \(String(format: "%.2f", maxGpuMB))MB")
        print("Max peak GPU: \(String(format: "%.2f", maxPeakGpuMB))MB")

        if !results.isEmpty {
            print("\nTop 5 by GPU usage:")
            for result in sortedByGpuDelta.prefix(5) {
                print("  \(result.name): \(String(format: "%+.2f", result.gpuDeltaMB))MB")
            }

            print("\nTop 5 by duration:")
            for result in sortedByDuration.prefix(5) {
                print("  \(result.name): \(String(format: "%.2f", result.durationMs))ms")
            }
        }
        print("==============================\n")
    }
}

/// Model-specific memory breakdown.
public struct ModelMemoryBreakdown: Sendable {
    /// Model identifier.
    public let modelId: String

    /// Estimated weight memory in bytes.
    public let weightsBytes: UInt64

    /// Estimated activation memory in bytes.
    public let activationsBytes: UInt64

    /// KV cache memory in bytes (if applicable).
    public let kvCacheBytes: UInt64

    /// Other memory (buffers, etc.) in bytes.
    public let otherBytes: UInt64

    /// Total memory in bytes.
    public var totalBytes: UInt64 {
        weightsBytes + activationsBytes + kvCacheBytes + otherBytes
    }

    /// Total memory in megabytes.
    public var totalMB: Double {
        Double(totalBytes) / (1024 * 1024)
    }

    /// Weights memory in megabytes.
    public var weightsMB: Double {
        Double(weightsBytes) / (1024 * 1024)
    }

    /// Activations memory in megabytes.
    public var activationsMB: Double {
        Double(activationsBytes) / (1024 * 1024)
    }

    /// KV cache memory in megabytes.
    public var kvCacheMB: Double {
        Double(kvCacheBytes) / (1024 * 1024)
    }
}

/// Utility for estimating model memory requirements.
public enum MemoryEstimator {

    /// Estimate memory for a tensor shape with given dtype.
    public static func tensorBytes(shape: [Int], dtype: DType = .float32) -> UInt64 {
        let elements = shape.reduce(1, *)
        let bytesPerElement: Int
        switch dtype {
        case .float32:
            bytesPerElement = 4
        case .float16, .bfloat16:
            bytesPerElement = 2
        case .int32:
            bytesPerElement = 4
        case .int16:
            bytesPerElement = 2
        case .int8, .uint8:
            bytesPerElement = 1
        default:
            bytesPerElement = 4  // Default assumption
        }
        return UInt64(elements * bytesPerElement)
    }

    /// Estimate KV cache memory for Whisper-style decoder.
    public static func whisperKVCacheBytes(
        maxLength: Int,
        nLayers: Int,
        hiddenDim: Int,
        batchSize: Int = 1,
        dtype: DType = .float16
    ) -> UInt64 {
        // Each layer has keys and values: [batch, maxLength, hiddenDim]
        let perTensor = tensorBytes(shape: [batchSize, maxLength, hiddenDim], dtype: dtype)
        return perTensor * UInt64(nLayers) * 2  // keys + values
    }

    /// Estimate weight memory for a model with given parameter count.
    public static func weightBytes(parameterCount: Int, dtype: DType = .float32) -> UInt64 {
        tensorBytes(shape: [parameterCount], dtype: dtype)
    }

    /// Estimate quantized weight memory.
    public static func quantizedWeightBytes(
        parameterCount: Int,
        bits: Int,
        groupSize: Int = 64
    ) -> UInt64 {
        // Quantized weights: bits per weight + scales (float16 per group)
        let weightBits = UInt64(parameterCount * bits)
        let numGroups = (parameterCount + groupSize - 1) / groupSize
        let scaleBits = UInt64(numGroups * 16)  // float16 scales
        return (weightBits + scaleBits + 7) / 8  // Convert to bytes
    }
}

/// Format bytes as human-readable string.
public func formatBytes(_ bytes: UInt64) -> String {
    let formatter = ByteCountFormatter()
    formatter.countStyle = .memory
    return formatter.string(fromByteCount: Int64(bytes))
}

/// Format bytes as human-readable string (Int64 version).
public func formatBytes(_ bytes: Int64) -> String {
    let formatter = ByteCountFormatter()
    formatter.countStyle = .memory
    return formatter.string(fromByteCount: bytes)
}
