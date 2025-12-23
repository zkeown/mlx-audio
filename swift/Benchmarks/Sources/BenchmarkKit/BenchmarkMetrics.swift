// BenchmarkMetrics.swift
// Measurement utilities for benchmarks.

import Foundation
import MLX

/// Configuration for benchmark runs.
public struct BenchmarkConfig: Sendable {
    public let warmup: Int
    public let iterations: Int
    public let verbose: Bool

    public init(warmup: Int = 3, iterations: Int = 10, verbose: Bool = false) {
        self.warmup = warmup
        self.iterations = iterations
        self.verbose = verbose
    }

    public static let quick = BenchmarkConfig(warmup: 1, iterations: 3)
    public static let standard = BenchmarkConfig(warmup: 3, iterations: 10)
    public static let thorough = BenchmarkConfig(warmup: 5, iterations: 20)
}

/// Timing result with statistics.
public struct TimingResult: Sendable {
    public let times: [Double]
    public let peakMemoryMB: Double

    public var mean: Double {
        times.reduce(0, +) / Double(times.count)
    }

    public var std: Double {
        let m = mean
        let variance = times.map { ($0 - m) * ($0 - m) }.reduce(0, +) / Double(times.count)
        return variance.squareRoot()
    }

    public var min: Double {
        times.min() ?? 0
    }

    public var max: Double {
        times.max() ?? 0
    }
}

/// Benchmark measurement utilities.
public enum BenchmarkMetrics {

    /// Measure execution time with warmup and GPU synchronization.
    public static func measureTime(
        warmup: Int = 3,
        iterations: Int = 10,
        operation: () throws -> Void
    ) throws -> TimingResult {
        // Warmup runs
        for _ in 0..<warmup {
            try operation()
            eval()
        }

        // Reset peak memory tracking
        GPU.set(cacheLimit: 0)
        GPU.clearCache()

        var times: [Double] = []
        times.reserveCapacity(iterations)

        for _ in 0..<iterations {
            let start = CFAbsoluteTimeGetCurrent()
            try operation()
            eval()  // Force GPU sync for accurate timing
            let end = CFAbsoluteTimeGetCurrent()
            times.append((end - start) * 1000)  // Convert to ms
        }

        let peakMemory = Double(GPU.activeMemory) / (1024 * 1024)
        return TimingResult(times: times, peakMemoryMB: peakMemory)
    }

    /// Measure with real-time factor calculation.
    public static func measureRealtimeFactor(
        audioDurationSec: Double,
        sampleRate: Int = 44100,
        warmup: Int = 3,
        iterations: Int = 10,
        operation: () throws -> Void
    ) throws -> (timing: TimingResult, realtimeFactor: Double) {
        let timing = try measureTime(
            warmup: warmup,
            iterations: iterations,
            operation: operation
        )

        let processingTimeSec = timing.mean / 1000
        let rtf = audioDurationSec / processingTimeSec

        return (timing, rtf)
    }

    /// Create a BenchmarkResult from timing data.
    public static func createResult(
        name: String,
        timing: TimingResult,
        audioDurationSec: Double? = nil,
        sampleRate: Int = 44100,
        iterations: Int,
        params: [String: AnyCodableValue] = [:]
    ) -> BenchmarkResult {
        let throughput: Double
        let rtf: Double?

        if let duration = audioDurationSec {
            let processingTimeSec = timing.mean / 1000
            throughput = duration * Double(sampleRate) / processingTimeSec
            rtf = duration / processingTimeSec
        } else {
            throughput = 0
            rtf = nil
        }

        return BenchmarkResult(
            name: name,
            meanTimeMs: timing.mean,
            stdTimeMs: timing.std,
            minTimeMs: timing.min,
            maxTimeMs: timing.max,
            throughput: throughput,
            peakMemoryMB: timing.peakMemoryMB,
            realtimeFactor: rtf,
            iterations: iterations,
            params: params
        )
    }
}

/// Simple progress reporter for long-running benchmarks.
public final class BenchmarkProgress: @unchecked Sendable {
    private let total: Int
    private var current: Int = 0
    private let startTime: CFAbsoluteTime
    private let name: String

    public init(name: String, total: Int) {
        self.name = name
        self.total = total
        self.startTime = CFAbsoluteTimeGetCurrent()
    }

    public func update(_ message: String = "") {
        current += 1
        let elapsed = CFAbsoluteTimeGetCurrent() - startTime
        let percent = Double(current) / Double(total) * 100
        let eta = elapsed / Double(current) * Double(total - current)

        print(
            String(
                format: "\r[%@] %.1f%% (%d/%d) - Elapsed: %.1fs, ETA: %.1fs %@",
                name, percent, current, total, elapsed, eta, message
            ),
            terminator: ""
        )
        fflush(stdout)

        if current == total {
            print()  // New line at end
        }
    }
}
