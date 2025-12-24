// RingBuffer.swift
// Lock-free circular buffer for real-time audio streaming.
//
// Uses atomic operations for thread-safe producer-consumer pattern.
// Designed for audio callbacks (producer) and async contexts (consumer).

import Atomics
import Foundation

// MARK: - Audio Ring Buffer

/// Lock-free circular buffer for real-time audio streaming.
///
/// This buffer is designed for the producer-consumer pattern common in audio:
/// - Producer (audio callback): Writes samples in real-time thread
/// - Consumer (async context): Reads samples for processing
///
/// Thread Safety:
/// - Uses atomic operations with proper memory ordering
/// - No locks or blocking operations
/// - Safe for single producer, single consumer
///
/// Example:
/// ```swift
/// let buffer = AudioRingBuffer(capacity: 8192, channels: 2)
///
/// // Producer (in audio callback)
/// buffer.write(samples)
///
/// // Consumer (in async context)
/// if let audio = buffer.read(count: 1024) {
///     // Process audio
/// }
/// ```
public final class AudioRingBuffer: @unchecked Sendable {
    // MARK: - Properties

    /// Pre-allocated buffer storage (interleaved samples)
    private let storage: UnsafeMutableBufferPointer<Float>

    /// Buffer capacity in frames (samples per channel)
    public let capacity: Int

    /// Number of audio channels
    public let channels: Int

    /// Total storage size (capacity * channels)
    private let storageSize: Int

    /// Mask for fast modulo (capacity must be power of 2)
    private let mask: Int

    /// Atomic write position (producer)
    private let writePosition: ManagedAtomic<Int>

    /// Atomic read position (consumer)
    private let readPosition: ManagedAtomic<Int>

    /// Statistics: total samples written
    private let totalWritten: ManagedAtomic<Int>

    /// Statistics: total samples read
    private let totalRead: ManagedAtomic<Int>

    /// Statistics: samples dropped due to full buffer
    private let droppedSamples: ManagedAtomic<Int>

    // MARK: - Initialization

    /// Maximum buffer capacity (1 million frames = ~23 seconds at 44.1kHz stereo).
    /// This prevents accidental OOM from unreasonable capacity requests.
    public static let maxCapacity: Int = 1_000_000

    /// Creates a new audio ring buffer.
    ///
    /// - Parameters:
    ///   - capacity: Buffer capacity in frames. Will be rounded up to power of 2.
    ///              Maximum capacity is 1 million frames.
    ///   - channels: Number of audio channels (default: 2 for stereo).
    /// - Precondition: channels must be positive, capacity must not exceed maxCapacity.
    public init(capacity: Int, channels: Int = 2) {
        precondition(channels > 0, "Channel count must be positive")
        precondition(capacity > 0, "Capacity must be positive")
        precondition(capacity <= Self.maxCapacity,
                     "Capacity \(capacity) exceeds maximum \(Self.maxCapacity)")

        // Round up to power of 2 for fast modulo
        let powerOf2Capacity = Self.nextPowerOf2(min(capacity, Self.maxCapacity))

        self.capacity = powerOf2Capacity
        self.channels = channels
        self.storageSize = powerOf2Capacity * channels
        self.mask = powerOf2Capacity - 1

        // Allocate storage
        let pointer = UnsafeMutablePointer<Float>.allocate(capacity: storageSize)
        pointer.initialize(repeating: 0, count: storageSize)
        self.storage = UnsafeMutableBufferPointer(start: pointer, count: storageSize)

        // Initialize atomics
        self.writePosition = ManagedAtomic(0)
        self.readPosition = ManagedAtomic(0)
        self.totalWritten = ManagedAtomic(0)
        self.totalRead = ManagedAtomic(0)
        self.droppedSamples = ManagedAtomic(0)
    }

    deinit {
        storage.baseAddress?.deinitialize(count: storageSize)
        storage.baseAddress?.deallocate()
    }

    // MARK: - Producer API (Real-Time Safe)

    /// Write samples to the buffer.
    ///
    /// This method is designed to be called from real-time audio callbacks.
    /// It performs no memory allocation and uses atomic operations.
    ///
    /// - Parameter samples: Interleaved audio samples [L0, R0, L1, R1, ...]
    /// - Returns: Number of frames actually written (may be less if buffer full)
    @inline(__always)
    public func write(_ samples: UnsafeBufferPointer<Float>) -> Int {
        let frameCount = samples.count / channels
        guard frameCount > 0 else { return 0 }

        let currentWrite = writePosition.load(ordering: .relaxed)
        let currentRead = readPosition.load(ordering: .acquiring)

        // Calculate available space
        let used = (currentWrite - currentRead + capacity) & mask
        let available = capacity - used - 1  // Keep one slot empty to distinguish full from empty

        let framesToWrite = min(frameCount, available)
        guard framesToWrite > 0 else {
            droppedSamples.wrappingIncrement(by: frameCount, ordering: .relaxed)
            return 0
        }

        // Write samples with wrap-around
        let samplesToWrite = framesToWrite * channels
        let writeStart = (currentWrite * channels) % storageSize

        if writeStart + samplesToWrite <= storageSize {
            // Contiguous write
            for i in 0..<samplesToWrite {
                storage[writeStart + i] = samples[i]
            }
        } else {
            // Wrap-around write
            let firstPart = storageSize - writeStart
            for i in 0..<firstPart {
                storage[writeStart + i] = samples[i]
            }
            let secondPart = samplesToWrite - firstPart
            for i in 0..<secondPart {
                storage[i] = samples[firstPart + i]
            }
        }

        // Update write position with release ordering (makes writes visible to consumer)
        let newWrite = (currentWrite + framesToWrite) & mask
        writePosition.store(newWrite, ordering: .releasing)

        totalWritten.wrappingIncrement(by: framesToWrite, ordering: .relaxed)

        if framesToWrite < frameCount {
            droppedSamples.wrappingIncrement(by: frameCount - framesToWrite, ordering: .relaxed)
        }

        return framesToWrite
    }

    /// Write samples from a contiguous array.
    ///
    /// Convenience wrapper for write(_:UnsafeBufferPointer).
    ///
    /// - Parameter samples: Interleaved audio samples
    /// - Returns: Number of frames actually written
    @inline(__always)
    public func write(_ samples: [Float]) -> Int {
        samples.withUnsafeBufferPointer { buffer in
            write(buffer)
        }
    }

    // MARK: - Consumer API

    /// Read a specific number of frames from the buffer.
    ///
    /// - Parameter count: Number of frames to read
    /// - Returns: Interleaved samples, or nil if insufficient data available
    public func read(count: Int) -> [Float]? {
        guard count > 0 else { return [] }

        let currentRead = readPosition.load(ordering: .relaxed)
        let currentWrite = writePosition.load(ordering: .acquiring)

        // Calculate available frames
        let available = (currentWrite - currentRead + capacity) & mask
        guard available >= count else { return nil }

        // Allocate output buffer
        var output = [Float](repeating: 0, count: count * channels)

        // Read samples with wrap-around
        let readStart = (currentRead * channels) % storageSize
        let samplesToRead = count * channels

        if readStart + samplesToRead <= storageSize {
            // Contiguous read
            for i in 0..<samplesToRead {
                output[i] = storage[readStart + i]
            }
        } else {
            // Wrap-around read
            let firstPart = storageSize - readStart
            for i in 0..<firstPart {
                output[i] = storage[readStart + i]
            }
            let secondPart = samplesToRead - firstPart
            for i in 0..<secondPart {
                output[firstPart + i] = storage[i]
            }
        }

        // Update read position with release ordering
        let newRead = (currentRead + count) & mask
        readPosition.store(newRead, ordering: .releasing)

        totalRead.wrappingIncrement(by: count, ordering: .relaxed)

        return output
    }

    /// Read all available frames from the buffer.
    ///
    /// - Returns: All available interleaved samples, or empty array if none available
    public func readAvailable() -> [Float] {
        let currentRead = readPosition.load(ordering: .relaxed)
        let currentWrite = writePosition.load(ordering: .acquiring)

        let available = (currentWrite - currentRead + capacity) & mask
        guard available > 0 else { return [] }

        return read(count: available) ?? []
    }

    /// Peek at samples without consuming them.
    ///
    /// - Parameter count: Number of frames to peek
    /// - Returns: Interleaved samples, or nil if insufficient data available
    public func peek(count: Int) -> [Float]? {
        guard count > 0 else { return [] }

        let currentRead = readPosition.load(ordering: .relaxed)
        let currentWrite = writePosition.load(ordering: .acquiring)

        let available = (currentWrite - currentRead + capacity) & mask
        guard available >= count else { return nil }

        var output = [Float](repeating: 0, count: count * channels)

        let readStart = (currentRead * channels) % storageSize
        let samplesToRead = count * channels

        if readStart + samplesToRead <= storageSize {
            for i in 0..<samplesToRead {
                output[i] = storage[readStart + i]
            }
        } else {
            let firstPart = storageSize - readStart
            for i in 0..<firstPart {
                output[i] = storage[readStart + i]
            }
            let secondPart = samplesToRead - firstPart
            for i in 0..<secondPart {
                output[firstPart + i] = storage[i]
            }
        }

        // Note: read position NOT updated

        return output
    }

    // MARK: - State Queries

    /// Number of frames available to read.
    public var available: Int {
        let currentRead = readPosition.load(ordering: .relaxed)
        let currentWrite = writePosition.load(ordering: .acquiring)
        return (currentWrite - currentRead + capacity) & mask
    }

    /// Space available for writing (in frames).
    public var space: Int {
        let currentRead = readPosition.load(ordering: .acquiring)
        let currentWrite = writePosition.load(ordering: .relaxed)
        let used = (currentWrite - currentRead + capacity) & mask
        return capacity - used - 1
    }

    /// Whether the buffer is empty.
    public var isEmpty: Bool {
        available == 0
    }

    /// Whether the buffer is full.
    public var isFull: Bool {
        space == 0
    }

    /// Fill level as a percentage (0.0 to 1.0).
    public var fillLevel: Double {
        Double(available) / Double(capacity)
    }

    // MARK: - Statistics

    /// Total frames written since creation.
    public var statisticsTotalWritten: Int {
        totalWritten.load(ordering: .relaxed)
    }

    /// Total frames read since creation.
    public var statisticsTotalRead: Int {
        totalRead.load(ordering: .relaxed)
    }

    /// Frames dropped due to buffer overflow.
    public var statisticsDropped: Int {
        droppedSamples.load(ordering: .relaxed)
    }

    // MARK: - Control

    /// Reset the buffer, clearing all data.
    ///
    /// Warning: Only call when not actively streaming.
    public func reset() {
        readPosition.store(0, ordering: .releasing)
        writePosition.store(0, ordering: .releasing)
    }

    /// Reset statistics counters.
    public func resetStatistics() {
        totalWritten.store(0, ordering: .relaxed)
        totalRead.store(0, ordering: .relaxed)
        droppedSamples.store(0, ordering: .relaxed)
    }

    // MARK: - Utilities

    /// Round up to the next power of 2.
    private static func nextPowerOf2(_ n: Int) -> Int {
        guard n > 0 else { return 1 }
        var value = n - 1
        value |= value >> 1
        value |= value >> 2
        value |= value >> 4
        value |= value >> 8
        value |= value >> 16
        value |= value >> 32
        return value + 1
    }
}

// MARK: - Buffer Statistics

extension AudioRingBuffer {
    /// Snapshot of buffer statistics.
    public struct Statistics: Sendable {
        /// Frames currently available to read
        public let available: Int

        /// Space available for writing (frames)
        public let space: Int

        /// Fill level (0.0 to 1.0)
        public let fillLevel: Double

        /// Total frames written
        public let totalWritten: Int

        /// Total frames read
        public let totalRead: Int

        /// Frames dropped due to overflow
        public let dropped: Int
    }

    /// Get a snapshot of current statistics.
    public var statistics: Statistics {
        Statistics(
            available: available,
            space: space,
            fillLevel: fillLevel,
            totalWritten: statisticsTotalWritten,
            totalRead: statisticsTotalRead,
            dropped: statisticsDropped
        )
    }
}
