// MicrophoneSource.swift
// Real-time microphone capture using AVAudioEngine.
//
// Captures audio from the device microphone and provides it as MLXArrays.

import AVFoundation
import Foundation
@preconcurrency import MLX

// MARK: - Safety Limits

/// Maximum samples per audio buffer allocation to prevent OOM.
/// This allows up to ~10 seconds of 8-channel 192kHz audio (15M samples).
private let kMaxSamplesPerAllocation: Int = 15_000_000

// MARK: - Microphone Source Configuration

/// Configuration for microphone capture.
public struct MicrophoneConfiguration: Sendable {
    /// Sample rate in Hz
    public var sampleRate: Int

    /// Number of channels (1 = mono, 2 = stereo)
    public var channels: Int

    /// Buffer size in samples per callback
    public var bufferSize: Int

    /// Ring buffer duration in seconds
    public var bufferDurationSeconds: Double

    /// Default mono configuration at 44.1kHz
    public static let mono = MicrophoneConfiguration(
        sampleRate: 44100,
        channels: 1,
        bufferSize: 1024,
        bufferDurationSeconds: 2.0
    )

    /// Default stereo configuration at 44.1kHz
    public static let stereo = MicrophoneConfiguration(
        sampleRate: 44100,
        channels: 2,
        bufferSize: 1024,
        bufferDurationSeconds: 2.0
    )

    /// Configuration for Whisper (16kHz mono)
    public static let whisper = MicrophoneConfiguration(
        sampleRate: 16000,
        channels: 1,
        bufferSize: 512,
        bufferDurationSeconds: 30.0
    )

    public init(
        sampleRate: Int = 44100,
        channels: Int = 1,
        bufferSize: Int = 1024,
        bufferDurationSeconds: Double = 2.0
    ) {
        self.sampleRate = sampleRate
        self.channels = channels
        self.bufferSize = bufferSize
        self.bufferDurationSeconds = bufferDurationSeconds
    }
}

// MARK: - Microphone Source Errors

/// Errors that can occur during microphone capture.
public enum MicrophoneError: Error, LocalizedError {
    /// Audio engine failed to start
    case engineStartFailed(String)
    /// Microphone permission not granted
    case permissionDenied
    /// No input device available
    case noInputDevice
    /// Format conversion failed
    case formatConversionFailed
    /// Already running
    case alreadyRunning
    /// Not running
    case notRunning

    public var errorDescription: String? {
        switch self {
        case .engineStartFailed(let message):
            return "Audio engine failed to start: \(message)"
        case .permissionDenied:
            return "Microphone permission not granted"
        case .noInputDevice:
            return "No audio input device available"
        case .formatConversionFailed:
            return "Audio format conversion failed"
        case .alreadyRunning:
            return "Microphone is already running"
        case .notRunning:
            return "Microphone is not running"
        }
    }
}

// MARK: - Microphone Source

/// Real-time microphone capture using AVAudioEngine.
///
/// This actor captures audio from the device microphone and provides it as MLXArrays.
/// It uses a lock-free ring buffer for thread-safe transfer from the audio callback.
///
/// Example:
/// ```swift
/// let mic = MicrophoneSource(configuration: .mono)
/// try await mic.start()
///
/// while let audio = try await mic.read(count: 1024) {
///     // Process audio (shape: [channels, samples])
/// }
///
/// try await mic.stop()
/// ```
public actor MicrophoneSource: @preconcurrency AudioSource {
    // MARK: - AudioStream Protocol

    public nonisolated let sampleRate: Int
    public nonisolated let channels: Int

    public var isActive: Bool {
        _isActive
    }

    // MARK: - Properties

    private let configuration: MicrophoneConfiguration
    private var engine: AVAudioEngine?
    private var ringBuffer: AudioRingBuffer?
    private var _isActive = false

    // Statistics
    private var samplesCaptured: Int = 0
    private var overruns: Int = 0

    // MARK: - Initialization

    /// Creates a new microphone source with the specified configuration.
    ///
    /// - Parameter configuration: Capture configuration
    public init(configuration: MicrophoneConfiguration = .mono) {
        self.configuration = configuration
        self.sampleRate = configuration.sampleRate
        self.channels = configuration.channels
    }

    // MARK: - AudioSource Protocol

    /// Start capturing from the microphone.
    ///
    /// This will request microphone permission if not already granted.
    ///
    /// - Throws: MicrophoneError if capture cannot be started
    public func start() async throws {
        guard !_isActive else {
            throw MicrophoneError.alreadyRunning
        }

        // Check/request permission
        let hasPermission = await AudioSessionManager.shared.requestMicrophonePermission()
        guard hasPermission else {
            throw MicrophoneError.permissionDenied
        }

        // Configure audio session
        try await AudioSessionManager.shared.configure(.playAndRecord)
        try await AudioSessionManager.shared.activate()

        // Create ring buffer
        let bufferCapacity = Int(Double(configuration.sampleRate) * configuration.bufferDurationSeconds)
        ringBuffer = AudioRingBuffer(capacity: bufferCapacity, channels: configuration.channels)

        // Create and configure engine
        let engine = AVAudioEngine()
        self.engine = engine

        let inputNode = engine.inputNode

        // Check for input
        guard inputNode.inputFormat(forBus: 0).channelCount > 0 else {
            throw MicrophoneError.noInputDevice
        }

        // Get hardware format
        let hardwareFormat = inputNode.inputFormat(forBus: 0)

        // Create our desired format
        guard let desiredFormat = AVAudioFormat(
            standardFormatWithSampleRate: Double(configuration.sampleRate),
            channels: AVAudioChannelCount(configuration.channels)
        ) else {
            throw MicrophoneError.formatConversionFailed
        }

        // Install tap on input node
        // Note: AVAudioEngine handles sample rate conversion automatically
        let bufferSize = AVAudioFrameCount(configuration.bufferSize)

        // Capture ring buffer reference for callback
        let buffer = ringBuffer!

        inputNode.installTap(onBus: 0, bufferSize: bufferSize, format: desiredFormat) {
            [channels = configuration.channels] pcmBuffer, _ in
            // This runs on real-time audio thread
            // No allocation, no blocking, no Swift async

            guard let floatData = pcmBuffer.floatChannelData else { return }

            let frameCount = Int(pcmBuffer.frameLength)
            let channelCount = Int(pcmBuffer.format.channelCount)

            // Convert to interleaved format and write to ring buffer
            // Pre-allocate a stack buffer if possible, otherwise use small array
            if channelCount == 1 {
                // Mono: direct copy
                floatData[0].withMemoryRebound(to: Float.self, capacity: frameCount) { ptr in
                    let bufferPtr = UnsafeBufferPointer(start: ptr, count: frameCount)
                    _ = buffer.write(bufferPtr)
                }
            } else {
                // Multi-channel: interleave
                // This allocates, but it's small and infrequent enough to be acceptable
                let allocationSize = frameCount * channelCount
                guard allocationSize <= kMaxSamplesPerAllocation else {
                    // Skip this buffer if it's unreasonably large (indicates corruption)
                    return
                }
                var interleaved = [Float](repeating: 0, count: allocationSize)
                for frame in 0..<frameCount {
                    for channel in 0..<min(channelCount, channels) {
                        interleaved[frame * channels + channel] = floatData[channel][frame]
                    }
                }
                _ = buffer.write(interleaved)
            }
        }

        // Start the engine
        engine.prepare()
        do {
            try engine.start()
        } catch {
            inputNode.removeTap(onBus: 0)
            throw MicrophoneError.engineStartFailed(error.localizedDescription)
        }

        _isActive = true
    }

    /// Stop capturing from the microphone.
    public func stop() async throws {
        guard _isActive else {
            throw MicrophoneError.notRunning
        }

        if let engine = engine {
            engine.inputNode.removeTap(onBus: 0)
            engine.stop()
        }

        engine = nil
        _isActive = false
    }

    /// Read samples from the microphone buffer.
    ///
    /// This method supports cooperative cancellation - if the current Task is
    /// cancelled, it will throw `CancellationError` on the next call.
    ///
    /// - Parameter count: Number of frames to read
    /// - Returns: Audio samples as MLXArray [channels, samples], or nil if not enough data
    /// - Throws: `MicrophoneError.notRunning` if microphone is not active,
    ///           `CancellationError` if the task was cancelled
    public func read(count: Int) async throws -> MLXArray? {
        // Check for task cancellation before processing
        try Task.checkCancellation()

        guard _isActive else {
            throw MicrophoneError.notRunning
        }

        guard let buffer = ringBuffer else { return nil }

        guard let samples = buffer.read(count: count) else {
            return nil
        }

        samplesCaptured += count

        // Convert to MLXArray
        // Input is interleaved [L0, R0, L1, R1, ...] or mono [S0, S1, ...]
        // Output shape: [channels, samples]
        if channels == 1 {
            return MLXArray(samples).reshaped([1, count])
        } else {
            // Deinterleave: [L0, R0, L1, R1, ...] -> [[L0, L1, ...], [R0, R1, ...]]
            var deinterleaved = [[Float]](repeating: [Float](repeating: 0, count: count), count: channels)
            for i in 0..<count {
                for c in 0..<channels {
                    deinterleaved[c][i] = samples[i * channels + c]
                }
            }
            // Flatten and create array with proper shape
            let flat = deinterleaved.flatMap { $0 }
            return MLXArray(flat).reshaped([channels, count])
        }
    }

    // MARK: - Additional Methods

    /// Read all available samples from the buffer.
    ///
    /// This method supports cooperative cancellation - if the current Task is
    /// cancelled, it will throw `CancellationError` on the next call.
    ///
    /// - Returns: Available audio samples as MLXArray, or empty array if none
    /// - Throws: `MicrophoneError.notRunning` if microphone is not active,
    ///           `CancellationError` if the task was cancelled
    public func readAvailable() async throws -> MLXArray {
        // Check for task cancellation before processing
        try Task.checkCancellation()

        guard _isActive else {
            throw MicrophoneError.notRunning
        }

        guard let buffer = ringBuffer else {
            return MLXArray.zeros([channels, 0])
        }

        let samples = buffer.readAvailable()
        guard !samples.isEmpty else {
            return MLXArray.zeros([channels, 0])
        }

        let frameCount = samples.count / channels
        samplesCaptured += frameCount

        if channels == 1 {
            return MLXArray(samples).reshaped([1, frameCount])
        } else {
            var deinterleaved = [[Float]](repeating: [Float](repeating: 0, count: frameCount), count: channels)
            for i in 0..<frameCount {
                for c in 0..<channels {
                    deinterleaved[c][i] = samples[i * channels + c]
                }
            }
            let flat = deinterleaved.flatMap { $0 }
            return MLXArray(flat).reshaped([channels, frameCount])
        }
    }

    // MARK: - Statistics

    /// Number of frames available in the buffer.
    public var available: Int {
        ringBuffer?.available ?? 0
    }

    /// Total frames captured since start.
    public var totalCaptured: Int {
        samplesCaptured
    }

    /// Buffer statistics.
    public var bufferStatistics: AudioRingBuffer.Statistics? {
        ringBuffer?.statistics
    }
}
