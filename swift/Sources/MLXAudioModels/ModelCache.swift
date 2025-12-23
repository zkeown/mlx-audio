// ModelCache.swift
// LRU cache for loaded models with automatic eviction.
//
// For memory-constrained devices like iPhone (6-8GB), we can't keep
// all models loaded simultaneously. This cache automatically evicts
// least-recently-used models when memory limits are reached.

import Foundation
import MLX
import MLXNN

/// Errors related to memory management.
public enum MemoryError: Error, LocalizedError {
    case insufficientMemory(required: UInt64, available: UInt64)
    case modelTooLarge(modelId: String, size: UInt64, budget: UInt64)

    public var errorDescription: String? {
        switch self {
        case .insufficientMemory(let required, let available):
            return "Insufficient memory: requires \(formatBytes(required)), only \(formatBytes(available)) available"
        case .modelTooLarge(let modelId, let size, let budget):
            return "Model '\(modelId)' (\(formatBytes(size))) exceeds budget (\(formatBytes(budget)))"
        }
    }
}

/// LRU cache for loaded models with automatic eviction and memory budgeting.
///
/// Designed for memory-constrained devices where keeping all models
/// loaded simultaneously isn't feasible. Uses LRU eviction to keep
/// the most recently used models in memory, with optional memory budget
/// enforcement.
///
/// Example:
/// ```swift
/// let cache = ModelCache.forCurrentDevice()
///
/// // Load models on demand with memory tracking
/// let htdemucs = try await cache.get(
///     id: "htdemucs_ft",
///     estimatedMemoryMB: 2000
/// ) {
///     try HTDemucs.fromPretrained(path: htdemucsURL)
/// }
///
/// // Later, different model (may evict htdemucs if needed)
/// let whisper = try await cache.get(
///     id: "whisper-large-v3",
///     estimatedMemoryMB: 3000
/// ) {
///     try WhisperModel.fromPretrained(path: whisperURL)
/// }
/// ```
public actor ModelCache {

    /// Shared instance for global model caching.
    public static let shared = ModelCache.forCurrentDevice()

    /// Create cache configured for current device profile.
    public static func forCurrentDevice() -> ModelCache {
        let profile = DeviceProfile.current
        return ModelCache(
            maxModels: profile.recommendedMaxModels,
            memoryBudgetMB: profile.memoryBudgetMB
        )
    }

    /// Cache entry with metadata.
    private struct CacheEntry {
        let model: any Sendable
        let loadTime: Date
        var lastAccessTime: Date
        let estimatedMemoryMB: UInt64
    }

    /// Ordered cache storage (insertion order = LRU order).
    private var cache: [String: CacheEntry] = [:]

    /// Order of keys for LRU tracking (most recently used at end).
    private var accessOrder: [String] = []

    /// Maximum number of models to keep in cache.
    public let maxModels: Int

    /// Total memory budget for all cached models in MB.
    public let memoryBudgetMB: UInt64

    /// Memory pressure handler for automatic eviction.
    private var pressureHandler: MemoryPressureHandler?

    /// Current ID being accessed (to protect from eviction).
    private var currentModelId: String?

    /// Initialize cache with maximum model count and memory budget.
    ///
    /// - Parameters:
    ///   - maxModels: Maximum models to keep loaded
    ///   - memoryBudgetMB: Total memory budget in MB (0 = unlimited)
    public init(maxModels: Int = 2, memoryBudgetMB: UInt64 = 0) {
        self.maxModels = maxModels
        self.memoryBudgetMB = memoryBudgetMB
    }

    // MARK: - Model Access

    /// Get or load a model by ID with memory tracking.
    ///
    /// If the model is already cached, returns it immediately and updates
    /// LRU order. If not cached, loads it using the provided loader,
    /// evicting least-recently-used models if necessary to stay within
    /// memory budget.
    ///
    /// - Parameters:
    ///   - id: Unique identifier for the model
    ///   - estimatedMemoryMB: Estimated memory usage in MB
    ///   - loader: Closure to load the model if not cached
    /// - Returns: The loaded model
    /// - Throws: MemoryError if model cannot fit within budget
    public func get<T: Sendable>(
        id: String,
        estimatedMemoryMB: UInt64 = 0,
        loader: () throws -> T
    ) throws -> T {
        currentModelId = id
        defer { currentModelId = nil }

        // Check if already cached
        if let entry = cache[id], let model = entry.model as? T {
            // Update access order (move to end)
            updateAccessOrder(id)
            // Update last access time
            var updatedEntry = entry
            updatedEntry.lastAccessTime = Date()
            cache[id] = updatedEntry
            return model
        }

        // Check memory budget
        if memoryBudgetMB > 0 && estimatedMemoryMB > 0 {
            // Evict until we have room
            while currentTotalMemoryMB + estimatedMemoryMB > memoryBudgetMB {
                if !evictLRU() {
                    // Nothing left to evict
                    throw MemoryError.insufficientMemory(
                        required: estimatedMemoryMB * 1024 * 1024,
                        available: (memoryBudgetMB - currentTotalMemoryMB) * 1024 * 1024
                    )
                }
            }
        }

        // Evict if at model count capacity
        while cache.count >= maxModels {
            if !evictLRU() {
                break
            }
        }

        // Load the model
        let model = try loader()

        // Store in cache
        let entry = CacheEntry(
            model: model,
            loadTime: Date(),
            lastAccessTime: Date(),
            estimatedMemoryMB: estimatedMemoryMB
        )
        cache[id] = entry
        accessOrder.append(id)

        return model
    }

    /// Get or load a model by ID (legacy API without memory tracking).
    public func get<T: Sendable>(
        id: String,
        loader: () throws -> T
    ) throws -> T {
        try get(id: id, estimatedMemoryMB: 0, loader: loader)
    }

    // MARK: - Eviction

    /// Evict a specific model from cache.
    ///
    /// - Parameter id: Model identifier to evict
    public func evict(_ id: String) {
        cache.removeValue(forKey: id)
        accessOrder.removeAll { $0 == id }
    }

    /// Clear all cached models.
    public func clearAll() {
        cache.removeAll()
        accessOrder.removeAll()
        GPU.clearCache()
    }

    /// Evict all models except the one currently being used.
    public func evictAllExceptCurrent() {
        let toEvict = accessOrder.filter { $0 != currentModelId }
        for id in toEvict {
            cache.removeValue(forKey: id)
        }
        accessOrder.removeAll { $0 != currentModelId }
        GPU.clearCache()
    }

    // MARK: - Query

    /// Check if a model is currently cached.
    ///
    /// - Parameter id: Model identifier to check
    /// - Returns: True if model is in cache
    public func contains(_ id: String) -> Bool {
        return cache[id] != nil
    }

    /// List of currently cached model IDs (in LRU order, oldest first).
    public var loadedModelIds: [String] {
        return accessOrder
    }

    /// Number of currently cached models.
    public var count: Int {
        return cache.count
    }

    /// Current total estimated memory usage in MB.
    public var currentTotalMemoryMB: UInt64 {
        cache.values.reduce(0) { $0 + $1.estimatedMemoryMB }
    }

    /// Remaining memory budget in MB.
    public var remainingBudgetMB: UInt64 {
        guard memoryBudgetMB > 0 else { return UInt64.max }
        let used = currentTotalMemoryMB
        return used < memoryBudgetMB ? memoryBudgetMB - used : 0
    }

    /// Get memory usage per model.
    public var memoryPerModel: [String: UInt64] {
        var result: [String: UInt64] = [:]
        for (id, entry) in cache {
            result[id] = entry.estimatedMemoryMB
        }
        return result
    }

    // MARK: - Memory Pressure Integration

    /// Start monitoring memory pressure.
    public func startMemoryPressureMonitoring() async {
        pressureHandler = MemoryPressureHandler()

        await pressureHandler?.setOnPressureChange { [weak self] level in
            Task {
                await self?.handleMemoryPressure(level)
            }
        }

        await pressureHandler?.startMonitoring()
    }

    /// Stop monitoring memory pressure.
    public func stopMemoryPressureMonitoring() async {
        await pressureHandler?.stopMonitoring()
        pressureHandler = nil
    }

    /// Handle memory pressure event.
    private func handleMemoryPressure(_ level: MemoryPressureLevel) {
        switch level {
        case .warning:
            _ = evictLRU()
            GPU.clearCache()
        case .critical:
            evictAllExceptCurrent()
        case .terminal:
            clearAll()
        case .normal:
            break
        }
    }

    // MARK: - Private Methods

    /// Update access order for a key (move to end).
    private func updateAccessOrder(_ id: String) {
        accessOrder.removeAll { $0 == id }
        accessOrder.append(id)
    }

    /// Evict the least recently used model.
    ///
    /// - Returns: True if a model was evicted
    @discardableResult
    private func evictLRU() -> Bool {
        // Find LRU that isn't the current model
        guard let lruId = accessOrder.first(where: { $0 != currentModelId }) else {
            return false
        }
        cache.removeValue(forKey: lruId)
        accessOrder.removeAll { $0 == lruId }
        return true
    }
}

// MARK: - MemoryPressureHandler Extension

extension MemoryPressureHandler {
    /// Set callback (actor-safe setter).
    func setOnPressureChange(_ callback: @escaping MemoryPressureCallback) {
        self.onPressureChange = callback
    }
}

/// Device profile for model selection and memory management.
///
/// Helps choose appropriate model sizes and memory budgets based on device capabilities.
public enum DeviceProfile: String, Codable, Sendable {
    /// iPhone with 6-8GB RAM
    case phone
    /// iPad with 8-16GB RAM
    case tablet
    /// Mac with 16-64GB RAM
    case mac
    /// Mac Studio/Pro with 64GB+ RAM
    case macPro

    /// Detect current device profile based on available memory.
    public static var current: DeviceProfile {
        let totalMemory = ProcessInfo.processInfo.physicalMemory
        if totalMemory <= 8 * 1024 * 1024 * 1024 {
            return .phone
        } else if totalMemory <= 16 * 1024 * 1024 * 1024 {
            return .tablet
        } else if totalMemory <= 64 * 1024 * 1024 * 1024 {
            return .mac
        } else {
            return .macPro
        }
    }

    /// Total physical memory in bytes.
    public static var totalMemory: UInt64 {
        ProcessInfo.processInfo.physicalMemory
    }

    /// Currently available memory in bytes (iOS/macOS).
    public static var availableMemory: UInt64 {
        #if canImport(Darwin)
        var available: UInt64 = 0
        var size = MemoryLayout<UInt64>.size
        if sysctlbyname("hw.memsize", &available, &size, nil, 0) == 0 {
            // Estimate available as ~40% of total on iOS, ~60% on macOS
            #if os(iOS) || os(tvOS)
            return available * 40 / 100
            #else
            return available * 60 / 100
            #endif
        }
        #endif
        return totalMemory / 2  // Fallback
    }

    /// Recommended maximum models in cache for this profile.
    public var recommendedMaxModels: Int {
        switch self {
        case .phone: return 1
        case .tablet: return 2
        case .mac: return 4
        case .macPro: return 8
        }
    }

    /// Memory budget for models in MB.
    public var memoryBudgetMB: UInt64 {
        switch self {
        case .phone: return 2048      // 2GB max for models
        case .tablet: return 4096     // 4GB max
        case .mac: return 8192        // 8GB max
        case .macPro: return 16384    // 16GB max
        }
    }

    /// Memory budget per single model in MB.
    public var perModelBudgetMB: UInt64 {
        switch self {
        case .phone: return 1500      // ~1.5GB max per model
        case .tablet: return 3000     // ~3GB max per model
        case .mac: return 6000        // ~6GB max per model
        case .macPro: return 12000    // ~12GB max per model
        }
    }

    // MARK: - Model Recommendations

    /// Recommended Whisper model variant.
    public var whisperModel: String {
        switch self {
        case .phone: return "whisper-small"
        case .tablet: return "whisper-medium"
        case .mac: return "whisper-large-v3-turbo"
        case .macPro: return "whisper-large-v3"
        }
    }

    /// Recommended MusicGen model variant.
    public var musicgenModel: String {
        switch self {
        case .phone: return "musicgen-small"
        case .tablet: return "musicgen-medium"
        case .mac: return "musicgen-medium"
        case .macPro: return "musicgen-large"
        }
    }

    /// Recommended HTDemucs model variant.
    public var htdemucsModel: String {
        switch self {
        case .phone: return "htdemucs"
        case .tablet: return "htdemucs_ft"
        case .mac: return "htdemucs_ft"
        case .macPro: return "htdemucs_6s"
        }
    }

    /// Recommended CLAP model variant.
    public var clapModel: String {
        switch self {
        case .phone: return "clap-htsat-tiny"
        case .tablet: return "clap-htsat-fused"
        case .mac: return "clap-htsat-fused"
        case .macPro: return "clap-htsat-fused"
        }
    }

    /// Recommended EnCodec model variant.
    public var encodecModel: String {
        // EnCodec is small enough for all devices
        return "encodec-24khz"
    }

    // MARK: - Quantization Recommendations

    /// Default quantization config for this device (nil = full precision).
    public var defaultQuantization: QuantizationConfig? {
        switch self {
        case .phone: return .int4
        case .tablet: return .int8
        case .mac: return nil
        case .macPro: return nil
        }
    }

    /// Whether to prefer quantized models on this device.
    public var preferQuantized: Bool {
        switch self {
        case .phone, .tablet: return true
        case .mac, .macPro: return false
        }
    }
}
