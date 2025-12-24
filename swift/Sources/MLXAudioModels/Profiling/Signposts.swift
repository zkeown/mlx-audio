// Signposts.swift
// Integration with Apple Instruments for performance profiling.
//
// Provides os_signpost integration for detailed performance analysis
// of model inference, audio processing, and streaming operations.

import Foundation
import os.signpost

// MARK: - Signpost Categories

/// Signpost subsystem for mlx-audio operations.
private let signpostSubsystem = "com.mlx-audio"

/// Signpost categories for different operation types.
public enum SignpostCategory: String, Sendable {
    /// Model loading and initialization
    case modelLoading = "Model Loading"
    /// Model inference operations
    case inference = "Inference"
    /// Audio DSP operations (STFT, mel-spectrogram, etc.)
    case dsp = "DSP"
    /// Streaming audio I/O
    case streaming = "Streaming"
    /// Memory operations (allocation, cache, eviction)
    case memory = "Memory"
    /// Weight loading from disk
    case weightLoading = "Weight Loading"

    /// OSLog for this category.
    public var log: OSLog {
        OSLog(subsystem: signpostSubsystem, category: self.rawValue)
    }
}

// MARK: - Signpost Manager

/// Manager for os_signpost integration.
///
/// Provides a unified interface for performance instrumentation that integrates
/// with Apple Instruments. When profiling with Instruments, signposts appear
/// in the "Points of Interest" and "os_signpost" instruments.
///
/// ## Usage
///
/// ```swift
/// // Simple interval measurement
/// Signposts.begin(.inference, name: "HTDemucs Forward Pass")
/// let result = model(input)
/// Signposts.end(.inference, name: "HTDemucs Forward Pass")
///
/// // Using the interval helper
/// let result = Signposts.measure(.inference, name: "Attention") {
///     attention(query, key, value)
/// }
///
/// // Async operations
/// let result = await Signposts.measureAsync(.modelLoading, name: "Load Whisper") {
///     try await loadWhisperModel()
/// }
///
/// // With metadata
/// Signposts.begin(.dsp, name: "STFT", metadata: ["n_fft": "2048", "hop": "512"])
/// ```
///
/// ## Instruments Integration
///
/// To view signposts in Instruments:
/// 1. Open Instruments and select "Blank" template
/// 2. Add "os_signpost" instrument
/// 3. Run your app with profiling
/// 4. Filter by subsystem "com.mlx-audio"
///
/// Signposts provide:
/// - Start/end times for intervals
/// - Duration measurements
/// - Custom metadata for each operation
/// - Hierarchical nesting for complex operations
public enum Signposts {

    /// Whether signpost logging is enabled.
    ///
    /// Set to `false` to disable all signpost overhead in release builds.
    /// Default is `true` when running in DEBUG mode.
    #if DEBUG
    public static var isEnabled: Bool = true
    #else
    public static var isEnabled: Bool = false
    #endif

    /// Unique ID generator for signpost intervals.
    private static let idGenerator = OSSignpostID.makeExclusive

    // MARK: - Interval Signposts

    /// Begin a signpost interval.
    ///
    /// Call `end(_:name:)` with the same category and name to complete the interval.
    ///
    /// - Parameters:
    ///   - category: The signpost category
    ///   - name: A descriptive name for the operation
    ///   - metadata: Optional key-value metadata to include
    /// - Returns: A signpost ID to pass to `end()`
    @discardableResult
    public static func begin(
        _ category: SignpostCategory,
        name: StaticString,
        metadata: [String: String]? = nil
    ) -> OSSignpostID {
        guard isEnabled else { return .exclusive }

        let log = category.log
        let id = OSSignpostID(log: log)

        if let metadata = metadata {
            let metaString = metadata.map { "\($0.key)=\($0.value)" }.joined(separator: ", ")
            os_signpost(.begin, log: log, name: name, signpostID: id, "%{public}s", metaString)
        } else {
            os_signpost(.begin, log: log, name: name, signpostID: id)
        }

        return id
    }

    /// End a signpost interval.
    ///
    /// - Parameters:
    ///   - category: The signpost category (must match `begin()`)
    ///   - name: The operation name (must match `begin()`)
    ///   - id: The signpost ID returned from `begin()`
    ///   - metadata: Optional metadata to include at end
    public static func end(
        _ category: SignpostCategory,
        name: StaticString,
        id: OSSignpostID = .exclusive,
        metadata: [String: String]? = nil
    ) {
        guard isEnabled else { return }

        let log = category.log

        if let metadata = metadata {
            let metaString = metadata.map { "\($0.key)=\($0.value)" }.joined(separator: ", ")
            os_signpost(.end, log: log, name: name, signpostID: id, "%{public}s", metaString)
        } else {
            os_signpost(.end, log: log, name: name, signpostID: id)
        }
    }

    /// Emit a single-point signpost event.
    ///
    /// Use for instantaneous events rather than intervals.
    ///
    /// - Parameters:
    ///   - category: The signpost category
    ///   - name: A descriptive name for the event
    ///   - message: Optional message to include
    public static func event(
        _ category: SignpostCategory,
        name: StaticString,
        message: String? = nil
    ) {
        guard isEnabled else { return }

        let log = category.log

        if let message = message {
            os_signpost(.event, log: log, name: name, "%{public}s", message)
        } else {
            os_signpost(.event, log: log, name: name)
        }
    }

    // MARK: - Measurement Helpers

    /// Measure the duration of a synchronous operation.
    ///
    /// Automatically begins and ends a signpost interval around the operation.
    ///
    /// - Parameters:
    ///   - category: The signpost category
    ///   - name: A descriptive name for the operation
    ///   - metadata: Optional metadata to include
    ///   - operation: The operation to measure
    /// - Returns: The result of the operation
    @inlinable
    public static func measure<T>(
        _ category: SignpostCategory,
        name: StaticString,
        metadata: [String: String]? = nil,
        operation: () throws -> T
    ) rethrows -> T {
        let id = begin(category, name: name, metadata: metadata)
        defer { end(category, name: name, id: id) }
        return try operation()
    }

    /// Measure the duration of an async operation.
    ///
    /// - Parameters:
    ///   - category: The signpost category
    ///   - name: A descriptive name for the operation
    ///   - metadata: Optional metadata to include
    ///   - operation: The async operation to measure
    /// - Returns: The result of the operation
    @inlinable
    public static func measureAsync<T>(
        _ category: SignpostCategory,
        name: StaticString,
        metadata: [String: String]? = nil,
        operation: () async throws -> T
    ) async rethrows -> T {
        let id = begin(category, name: name, metadata: metadata)
        defer { end(category, name: name, id: id) }
        return try await operation()
    }
}

// MARK: - Convenience Extensions

extension Signposts {

    /// Begin a model inference interval with common metadata.
    ///
    /// - Parameters:
    ///   - modelName: Name of the model being run
    ///   - batchSize: Batch size for inference
    ///   - inputShape: Shape of input tensor
    /// - Returns: Signpost ID for the interval
    @discardableResult
    public static func beginInference(
        model modelName: String,
        batchSize: Int = 1,
        inputShape: [Int]? = nil
    ) -> OSSignpostID {
        var metadata = [
            "model": modelName,
            "batch": String(batchSize)
        ]
        if let shape = inputShape {
            metadata["shape"] = shape.map(String.init).joined(separator: "x")
        }
        return begin(.inference, name: "Model Inference", metadata: metadata)
    }

    /// End a model inference interval with output metadata.
    ///
    /// - Parameters:
    ///   - id: Signpost ID from `beginInference()`
    ///   - outputShape: Shape of output tensor
    public static func endInference(
        id: OSSignpostID,
        outputShape: [Int]? = nil
    ) {
        var metadata: [String: String]? = nil
        if let shape = outputShape {
            metadata = ["output_shape": shape.map(String.init).joined(separator: "x")]
        }
        end(.inference, name: "Model Inference", id: id, metadata: metadata)
    }

    /// Begin a DSP operation interval.
    ///
    /// - Parameters:
    ///   - operation: Name of the DSP operation (e.g., "STFT", "Mel Spectrogram")
    ///   - sampleRate: Audio sample rate
    ///   - duration: Audio duration in seconds
    /// - Returns: Signpost ID for the interval
    @discardableResult
    public static func beginDSP(
        operation: String,
        sampleRate: Int? = nil,
        duration: Double? = nil
    ) -> OSSignpostID {
        var metadata = ["operation": operation]
        if let sr = sampleRate {
            metadata["sample_rate"] = String(sr)
        }
        if let dur = duration {
            metadata["duration_sec"] = String(format: "%.2f", dur)
        }
        return begin(.dsp, name: "DSP Operation", metadata: metadata)
    }

    /// End a DSP operation interval.
    public static func endDSP(id: OSSignpostID) {
        end(.dsp, name: "DSP Operation", id: id)
    }

    /// Begin a weight loading interval.
    ///
    /// - Parameters:
    ///   - file: Weights file name
    ///   - keys: Number of weight keys being loaded
    /// - Returns: Signpost ID for the interval
    @discardableResult
    public static func beginWeightLoading(
        file: String,
        keys: Int? = nil
    ) -> OSSignpostID {
        var metadata = ["file": file]
        if let k = keys {
            metadata["keys"] = String(k)
        }
        return begin(.weightLoading, name: "Load Weights", metadata: metadata)
    }

    /// End a weight loading interval.
    ///
    /// - Parameters:
    ///   - id: Signpost ID from `beginWeightLoading()`
    ///   - bytesLoaded: Total bytes loaded
    public static func endWeightLoading(
        id: OSSignpostID,
        bytesLoaded: Int64? = nil
    ) {
        var metadata: [String: String]? = nil
        if let bytes = bytesLoaded {
            metadata = ["bytes": formatBytes(UInt64(bytes))]
        }
        end(.weightLoading, name: "Load Weights", id: id, metadata: metadata)
    }

    /// Log a memory event.
    ///
    /// - Parameters:
    ///   - event: Type of memory event
    ///   - details: Additional details
    public static func memoryEvent(
        _ event: String,
        details: String? = nil
    ) {
        let message = details.map { "\(event): \($0)" } ?? event
        Signposts.event(.memory, name: "Memory Event", message: message)
    }
}

// MARK: - Scoped Signpost

/// A scoped signpost that automatically ends when deallocated.
///
/// Useful for measuring operations with complex control flow or early returns.
///
/// ```swift
/// func processAudio() {
///     let signpost = ScopedSignpost(.inference, name: "Process Audio")
///
///     guard let audio = loadAudio() else { return }  // Signpost ends here
///
///     // ... processing ...
///
///     // Signpost ends when function returns
/// }
/// ```
public final class ScopedSignpost: @unchecked Sendable {
    private let category: SignpostCategory
    private let name: StaticString
    private let id: OSSignpostID
    private var ended: Bool = false

    /// Create a scoped signpost that begins immediately.
    ///
    /// - Parameters:
    ///   - category: The signpost category
    ///   - name: A descriptive name for the operation
    ///   - metadata: Optional metadata to include
    public init(
        _ category: SignpostCategory,
        name: StaticString,
        metadata: [String: String]? = nil
    ) {
        self.category = category
        self.name = name
        self.id = Signposts.begin(category, name: name, metadata: metadata)
    }

    deinit {
        end()
    }

    /// Manually end the signpost early.
    ///
    /// - Parameter metadata: Optional metadata to include at end
    public func end(metadata: [String: String]? = nil) {
        guard !ended else { return }
        ended = true
        Signposts.end(category, name: name, id: id, metadata: metadata)
    }
}
