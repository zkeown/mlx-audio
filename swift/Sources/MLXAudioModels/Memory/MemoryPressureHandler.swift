// MemoryPressureHandler.swift
// iOS/macOS memory pressure monitoring and response.
//
// Monitors system memory pressure and provides callbacks for graceful
// memory reduction when the system is under pressure.

import Foundation
import MLX

#if canImport(Dispatch)
import Dispatch
#endif

/// Memory pressure levels.
public enum MemoryPressureLevel: Int, Comparable, Sendable {
    /// Normal memory conditions.
    case normal = 0

    /// Memory is getting tight, consider releasing caches.
    case warning = 1

    /// Memory is critically low, must release memory.
    case critical = 2

    /// System will terminate the app if memory isn't freed.
    case terminal = 3

    public static func < (lhs: MemoryPressureLevel, rhs: MemoryPressureLevel) -> Bool {
        lhs.rawValue < rhs.rawValue
    }

    /// Human-readable description.
    public var description: String {
        switch self {
        case .normal: return "Normal"
        case .warning: return "Warning"
        case .critical: return "Critical"
        case .terminal: return "Terminal"
        }
    }
}

/// Callback type for memory pressure changes.
public typealias MemoryPressureCallback = @Sendable (MemoryPressureLevel) -> Void

/// Handler for system memory pressure events.
///
/// On iOS, this monitors `DispatchSource.MemoryPressure` events.
/// On macOS, it monitors the same but also provides manual checks.
///
/// Example:
/// ```swift
/// let handler = MemoryPressureHandler()
///
/// handler.onPressureChange = { level in
///     switch level {
///     case .warning:
///         cache.evictLRU()
///     case .critical:
///         cache.clearAll()
///     default:
///         break
///     }
/// }
///
/// handler.startMonitoring()
/// ```
public actor MemoryPressureHandler {

    /// Dispatch source for memory pressure events.
    private var memoryPressureSource: DispatchSourceMemoryPressure?

    /// Current pressure level.
    private var _currentLevel: MemoryPressureLevel = .normal

    /// Whether monitoring is active.
    private var isMonitoring: Bool = false

    /// Callback for pressure changes.
    public var onPressureChange: MemoryPressureCallback?

    /// Queue for memory pressure monitoring.
    private let monitorQueue = DispatchQueue(
        label: "com.mlxaudio.memorypressure",
        qos: .utility
    )

    public init() {}

    // MARK: - Public Interface

    /// Current memory pressure level.
    public var currentLevel: MemoryPressureLevel {
        _currentLevel
    }

    /// Start monitoring memory pressure.
    public func startMonitoring() {
        guard !isMonitoring else { return }
        isMonitoring = true

        // Create dispatch source for memory pressure
        memoryPressureSource = DispatchSource.makeMemoryPressureSource(
            eventMask: [.warning, .critical],
            queue: monitorQueue
        )

        let source = memoryPressureSource
        memoryPressureSource?.setEventHandler { [weak self] in
            guard let self else { return }
            let handler = self
            Task { @Sendable in
                await handler.handleMemoryPressureEvent()
            }
        }

        memoryPressureSource?.setCancelHandler { [weak self] in
            guard let self else { return }
            let handler = self
            Task { @Sendable in
                await handler.handleCancellation()
            }
        }
        _ = source // silence unused warning

        memoryPressureSource?.resume()
    }

    /// Stop monitoring memory pressure.
    public func stopMonitoring() {
        guard isMonitoring else { return }
        isMonitoring = false

        memoryPressureSource?.cancel()
        memoryPressureSource = nil
    }

    /// Manually trigger a pressure check.
    ///
    /// Useful for proactive memory management based on GPU memory usage.
    public func checkPressure() -> MemoryPressureLevel {
        let gpuMemory = GPU.activeMemory
        let profile = DeviceProfile.current
        let budgetBytes = profile.memoryBudgetMB * 1024 * 1024

        // Calculate pressure based on GPU memory usage vs budget
        let usageRatio = Double(gpuMemory) / Double(budgetBytes)

        if usageRatio > 0.95 {
            updateLevel(.terminal)
        } else if usageRatio > 0.85 {
            updateLevel(.critical)
        } else if usageRatio > 0.70 {
            updateLevel(.warning)
        } else {
            updateLevel(.normal)
        }

        return _currentLevel
    }

    /// Get detailed memory statistics.
    public func getMemoryStats() -> MemoryStats {
        MemoryStats(
            gpuActiveBytes: UInt64(GPU.activeMemory),
            gpuPeakBytes: UInt64(GPU.peakMemory),
            gpuCacheBytes: UInt64(GPU.cacheMemory),
            processResidentBytes: getProcessMemory(),
            pressureLevel: _currentLevel,
            deviceProfile: DeviceProfile.current
        )
    }

    // MARK: - Private Methods

    /// Handle memory pressure event from dispatch source.
    private func handleMemoryPressureEvent() {
        guard let source = memoryPressureSource else { return }

        let event = source.data
        let newLevel: MemoryPressureLevel

        if event.contains(.critical) {
            newLevel = .critical
        } else if event.contains(.warning) {
            newLevel = .warning
        } else {
            newLevel = .normal
        }

        updateLevel(newLevel)
    }

    /// Handle dispatch source cancellation.
    private func handleCancellation() {
        isMonitoring = false
        memoryPressureSource = nil
    }

    /// Update pressure level and notify callback.
    private func updateLevel(_ newLevel: MemoryPressureLevel) {
        guard newLevel != _currentLevel else { return }

        let oldLevel = _currentLevel
        _currentLevel = newLevel

        // Log level change
        #if DEBUG
        print("[MemoryPressure] Level changed: \(oldLevel.description) -> \(newLevel.description)")
        #endif

        // Notify callback
        if let callback = onPressureChange {
            callback(newLevel)
        }
    }

    /// Get process resident memory.
    private func getProcessMemory() -> UInt64 {
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

// MARK: - Memory Statistics

/// Detailed memory statistics.
public struct MemoryStats: Sendable {
    /// GPU active memory in bytes.
    public let gpuActiveBytes: UInt64

    /// GPU peak memory in bytes.
    public let gpuPeakBytes: UInt64

    /// GPU cache memory in bytes.
    public let gpuCacheBytes: UInt64

    /// Process resident memory in bytes.
    public let processResidentBytes: UInt64

    /// Current pressure level.
    public let pressureLevel: MemoryPressureLevel

    /// Current device profile.
    public let deviceProfile: DeviceProfile

    /// GPU active memory in MB.
    public var gpuActiveMB: Double {
        Double(gpuActiveBytes) / (1024 * 1024)
    }

    /// Total GPU memory (active + cache) in MB.
    public var gpuTotalMB: Double {
        Double(gpuActiveBytes + gpuCacheBytes) / (1024 * 1024)
    }

    /// Process memory in MB.
    public var processMB: Double {
        Double(processResidentBytes) / (1024 * 1024)
    }

    /// Memory budget utilization (0.0 to 1.0+).
    public var budgetUtilization: Double {
        let budgetBytes = deviceProfile.memoryBudgetMB * 1024 * 1024
        return Double(gpuActiveBytes) / Double(budgetBytes)
    }

    /// Print formatted summary.
    public func printSummary() {
        print("\n=== Memory Stats ===")
        print("Device: \(deviceProfile)")
        print("Pressure: \(pressureLevel.description)")
        print("GPU Active: \(String(format: "%.1f", gpuActiveMB))MB")
        print("GPU Peak: \(String(format: "%.1f", Double(gpuPeakBytes) / (1024 * 1024)))MB")
        print("GPU Cache: \(String(format: "%.1f", Double(gpuCacheBytes) / (1024 * 1024)))MB")
        print("Process: \(String(format: "%.1f", processMB))MB")
        print("Budget Utilization: \(String(format: "%.1f", budgetUtilization * 100))%")
        print("====================\n")
    }
}

// MARK: - iOS Integration

#if canImport(UIKit)
import UIKit

/// Extension for iOS app memory warning handling.
public extension MemoryPressureHandler {

    /// Handle UIKit memory warning (call from AppDelegate).
    ///
    /// Example usage in AppDelegate:
    /// ```swift
    /// func applicationDidReceiveMemoryWarning(_ application: UIApplication) {
    ///     Task {
    ///         await memoryHandler.handleAppMemoryWarning()
    ///     }
    /// }
    /// ```
    func handleAppMemoryWarning() async {
        updateLevel(.critical)

        // Clear GPU cache
        GPU.clearCache()

        // Log the event
        #if DEBUG
        let stats = getMemoryStats()
        print("[MemoryPressure] App received memory warning")
        stats.printSummary()
        #endif
    }

    /// Handle entering background (reduce memory footprint).
    func handleEnteringBackground() async {
        // Force evaluation to complete pending work
        eval()

        // Clear GPU cache
        GPU.clearCache()

        updateLevel(.warning)
    }

    /// Handle entering foreground.
    func handleEnteringForeground() async {
        // Check current pressure
        _ = checkPressure()
    }
}
#endif

// MARK: - macOS Integration

#if os(macOS)
import AppKit

/// Extension for macOS memory pressure handling.
public extension MemoryPressureHandler {

    /// Handle NSApplication memory pressure notification.
    func handleMacOSMemoryPressure(_ notification: Notification) async {
        // Check current pressure based on GPU usage
        _ = checkPressure()
    }
}
#endif

// MARK: - Convenience Methods

extension MemoryPressureHandler {

    /// Create handler with automatic ModelCache integration.
    public static func withModelCache(_ cache: ModelCache) -> MemoryPressureHandler {
        let handler = MemoryPressureHandler()

        Task {
            await handler.setCallback { level in
                Task {
                    switch level {
                    case .warning:
                        // Evict least recently used model
                        if let lruId = await cache.loadedModelIds.first {
                            await cache.evict(lruId)
                        }
                        GPU.clearCache()

                    case .critical, .terminal:
                        // Evict all but current model
                        let ids = await cache.loadedModelIds
                        for id in ids.dropLast() {
                            await cache.evict(id)
                        }
                        GPU.clearCache()

                    case .normal:
                        break
                    }
                }
            }
        }

        return handler
    }

    /// Set callback (actor-safe wrapper).
    private func setCallback(_ callback: @escaping MemoryPressureCallback) {
        self.onPressureChange = callback
    }
}
