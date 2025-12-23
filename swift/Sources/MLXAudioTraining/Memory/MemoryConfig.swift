// MemoryConfig.swift
// Memory-aware training configuration.

import Foundation
import MLX

// MARK: - Device Tier

/// Device capability tier based on memory.
public enum DeviceTier: String, Sendable, CaseIterable {
    /// iPhone, low-memory iPad (6-8GB)
    case low

    /// iPad Pro, base Mac (8-16GB)
    case medium

    /// Mac with 16-64GB
    case high

    /// Mac Pro, Studio with 64GB+
    case extreme

    /// Detect current device tier.
    public static var current: DeviceTier {
        let totalMemory = ProcessInfo.processInfo.physicalMemory
        let memoryGB = totalMemory / (1024 * 1024 * 1024)

        switch memoryGB {
        case 0..<8:
            return .low
        case 8..<16:
            return .medium
        case 16..<64:
            return .high
        default:
            return .extreme
        }
    }
}

// MARK: - Memory Configuration

/// Memory-aware training configuration.
public struct MemoryConfig: Sendable {
    /// Maximum batch size.
    public let maxBatchSize: Int

    /// Gradient accumulation steps.
    public let gradientAccumulationSteps: Int

    /// Whether to clear cache between batches.
    public let clearCacheBetweenBatches: Bool

    /// Whether to clear cache between epochs.
    public let clearCacheBetweenEpochs: Bool

    /// Memory budget in MB.
    public let memoryBudgetMB: Int

    /// Eval frequency (eval() every N steps).
    public let evalFrequency: Int

    /// Creates a memory configuration.
    public init(
        maxBatchSize: Int = 4,
        gradientAccumulationSteps: Int = 1,
        clearCacheBetweenBatches: Bool = false,
        clearCacheBetweenEpochs: Bool = true,
        memoryBudgetMB: Int = 4096,
        evalFrequency: Int = 1
    ) {
        self.maxBatchSize = maxBatchSize
        self.gradientAccumulationSteps = gradientAccumulationSteps
        self.clearCacheBetweenBatches = clearCacheBetweenBatches
        self.clearCacheBetweenEpochs = clearCacheBetweenEpochs
        self.memoryBudgetMB = memoryBudgetMB
        self.evalFrequency = evalFrequency
    }

    /// Create configuration for current device.
    public static var forCurrentDevice: MemoryConfig {
        forDevice(DeviceTier.current)
    }

    /// Create configuration for specific device tier.
    public static func forDevice(_ tier: DeviceTier) -> MemoryConfig {
        switch tier {
        case .low:
            return MemoryConfig(
                maxBatchSize: 1,
                gradientAccumulationSteps: 8,
                clearCacheBetweenBatches: true,
                clearCacheBetweenEpochs: true,
                memoryBudgetMB: 3000,
                evalFrequency: 1
            )
        case .medium:
            return MemoryConfig(
                maxBatchSize: 2,
                gradientAccumulationSteps: 4,
                clearCacheBetweenBatches: true,
                clearCacheBetweenEpochs: true,
                memoryBudgetMB: 5000,
                evalFrequency: 2
            )
        case .high:
            return MemoryConfig(
                maxBatchSize: 8,
                gradientAccumulationSteps: 1,
                clearCacheBetweenBatches: false,
                clearCacheBetweenEpochs: true,
                memoryBudgetMB: 10000,
                evalFrequency: 4
            )
        case .extreme:
            return MemoryConfig(
                maxBatchSize: 16,
                gradientAccumulationSteps: 1,
                clearCacheBetweenBatches: false,
                clearCacheBetweenEpochs: false,
                memoryBudgetMB: 30000,
                evalFrequency: 8
            )
        }
    }

    /// Effective batch size (accounting for accumulation).
    public var effectiveBatchSize: Int {
        maxBatchSize * gradientAccumulationSteps
    }
}

// MARK: - Memory Monitor

/// Monitor memory usage during training.
public final class MemoryMonitor: @unchecked Sendable {
    /// Peak GPU memory seen.
    public private(set) var peakGPUMemory: UInt64 = 0

    /// Peak process memory seen.
    public private(set) var peakProcessMemory: UInt64 = 0

    /// Memory samples.
    private var samples: [(timestamp: Date, gpu: UInt64, process: UInt64)] = []

    /// Lock for thread safety.
    private let lock = NSLock()

    /// Sampling interval.
    private var timer: Timer?

    public init() {}

    /// Start monitoring.
    public func start(interval: TimeInterval = 1.0) {
        timer = Timer.scheduledTimer(withTimeInterval: interval, repeats: true) { [weak self] _ in
            self?.sample()
        }
    }

    /// Stop monitoring.
    public func stop() {
        timer?.invalidate()
        timer = nil
    }

    /// Take a memory sample.
    public func sample() {
        lock.lock()
        defer { lock.unlock() }

        let gpuMemory = UInt64(GPU.activeMemory)
        let processMemory = getProcessMemory()

        samples.append((Date(), gpuMemory, processMemory))

        peakGPUMemory = max(peakGPUMemory, gpuMemory)
        peakProcessMemory = max(peakProcessMemory, processMemory)
    }

    /// Get current GPU memory in MB.
    public var currentGPUMemoryMB: Float {
        Float(GPU.activeMemory) / (1024 * 1024)
    }

    /// Get peak GPU memory in MB.
    public var peakGPUMemoryMB: Float {
        Float(peakGPUMemory) / (1024 * 1024)
    }

    /// Get memory statistics.
    public func getStats() -> MemoryStats {
        lock.lock()
        defer { lock.unlock() }

        let gpuValues = samples.map { Float($0.gpu) / (1024 * 1024) }
        let processValues = samples.map { Float($0.process) / (1024 * 1024) }

        return MemoryStats(
            peakGPUMB: Float(peakGPUMemory) / (1024 * 1024),
            peakProcessMB: Float(peakProcessMemory) / (1024 * 1024),
            avgGPUMB: gpuValues.isEmpty ? 0 : gpuValues.reduce(0, +) / Float(gpuValues.count),
            avgProcessMB: processValues.isEmpty ? 0 : processValues.reduce(0, +) / Float(processValues.count),
            sampleCount: samples.count
        )
    }

    /// Reset statistics.
    public func reset() {
        lock.lock()
        defer { lock.unlock() }

        peakGPUMemory = 0
        peakProcessMemory = 0
        samples.removeAll()
        GPU.resetPeakMemory()
    }

    /// Clear GPU cache.
    public func clearCache() {
        GPU.clearCache()
    }

    /// Get process memory using mach_task_basic_info.
    private func getProcessMemory() -> UInt64 {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        return result == KERN_SUCCESS ? info.resident_size : 0
    }
}

/// Memory statistics.
public struct MemoryStats: Sendable {
    public let peakGPUMB: Float
    public let peakProcessMB: Float
    public let avgGPUMB: Float
    public let avgProcessMB: Float
    public let sampleCount: Int
}

// MARK: - Batch Size Tuner

/// Automatically find optimal batch size for available memory.
public actor BatchSizeTuner {
    private let memoryBudgetMB: Int
    private let safetyMargin: Float

    /// Creates a batch size tuner.
    ///
    /// - Parameters:
    ///   - memoryBudgetMB: Memory budget in MB
    ///   - safetyMargin: Safety margin (0.8 = use 80% of budget)
    public init(memoryBudgetMB: Int, safetyMargin: Float = 0.8) {
        self.memoryBudgetMB = memoryBudgetMB
        self.safetyMargin = safetyMargin
    }

    /// Find optimal batch size by binary search.
    ///
    /// - Parameters:
    ///   - minBatch: Minimum batch size to try
    ///   - maxBatch: Maximum batch size to try
    ///   - testFn: Function that runs one forward pass with given batch size
    /// - Returns: Optimal batch size
    public func findOptimalBatchSize(
        minBatch: Int = 1,
        maxBatch: Int = 64,
        testFn: (Int) async throws -> Void
    ) async -> Int {
        var low = minBatch
        var high = maxBatch
        var optimal = minBatch

        while low <= high {
            let mid = (low + high) / 2

            // Clear memory before test
            GPU.clearCache()
            GPU.resetPeakMemory()

            do {
                try await testFn(mid)
                eval()

                let peakMemory = GPU.peakMemory
                let effectiveBudget = Float(memoryBudgetMB) * safetyMargin * 1024 * 1024

                if Float(peakMemory) <= effectiveBudget {
                    optimal = mid
                    low = mid + 1
                } else {
                    high = mid - 1
                }
            } catch {
                // OOM or other error, reduce batch size
                high = mid - 1
            }

            GPU.clearCache()
        }

        return optimal
    }

    /// Estimate batch size based on model size.
    ///
    /// - Parameters:
    ///   - modelSizeMB: Model size in MB
    ///   - sampleSizeMB: Size of one sample in MB
    /// - Returns: Estimated optimal batch size
    public func estimateBatchSize(
        modelSizeMB: Float,
        sampleSizeMB: Float
    ) -> Int {
        let effectiveBudget = Float(memoryBudgetMB) * safetyMargin

        // Reserve space for model weights and gradients (2x)
        let modelMemory = modelSizeMB * 2

        // Available for batches
        let batchBudget = effectiveBudget - modelMemory

        // Each sample needs space for activations (estimate 4x sample size)
        let perSampleMemory = sampleSizeMB * 4

        let estimated = Int(batchBudget / perSampleMemory)
        return max(1, estimated)
    }
}

// MARK: - Gradient Accumulation Helper

/// Helper for gradient accumulation.
public struct GradientAccumulator {
    /// Accumulated gradients.
    private var accumulatedGrads: [String: MLXArray] = [:]

    /// Number of accumulated steps.
    private var accumulatedSteps: Int = 0

    /// Target accumulation steps.
    public let accumulationSteps: Int

    public init(accumulationSteps: Int) {
        self.accumulationSteps = accumulationSteps
    }

    /// Add gradients to accumulator.
    ///
    /// - Parameter gradients: Gradients to add
    /// - Returns: True if ready to update (accumulated enough)
    public mutating func accumulate(_ gradients: [String: MLXArray]) -> Bool {
        for (key, grad) in gradients {
            if let existing = accumulatedGrads[key] {
                accumulatedGrads[key] = existing + grad
            } else {
                accumulatedGrads[key] = grad
            }
        }

        accumulatedSteps += 1
        return accumulatedSteps >= accumulationSteps
    }

    /// Get averaged gradients and reset.
    public mutating func getAndReset() -> [String: MLXArray] {
        let scale = Float(1) / Float(accumulatedSteps)
        var result: [String: MLXArray] = [:]

        for (key, grad) in accumulatedGrads {
            result[key] = grad * scale
        }

        accumulatedGrads.removeAll()
        accumulatedSteps = 0

        return result
    }

    /// Whether accumulator has gradients.
    public var hasGradients: Bool {
        accumulatedSteps > 0
    }
}
