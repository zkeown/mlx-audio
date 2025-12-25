// SpeakerSink.swift
// Real-time speaker output using AVAudioEngine.
//
// Plays audio through the device speakers with low latency.

import AVFoundation
import Foundation
@preconcurrency import MLX

// MARK: - Speaker Sink Configuration

/// Configuration for speaker output.
public struct SpeakerConfiguration: Sendable {
    /// Sample rate in Hz
    public var sampleRate: Int

    /// Number of channels (1 = mono, 2 = stereo)
    public var channels: Int

    /// Number of buffers to pre-schedule for smooth playback
    public var scheduledBufferCount: Int

    /// Size of each scheduled buffer in frames
    public var bufferFrameCount: Int

    /// Ring buffer duration in seconds
    public var bufferDurationSeconds: Double

    /// Default stereo configuration at 44.1kHz
    public static let stereo = SpeakerConfiguration(
        sampleRate: 44100,
        channels: 2,
        scheduledBufferCount: 3,
        bufferFrameCount: 1024,
        bufferDurationSeconds: 2.0
    )

    /// Default mono configuration at 44.1kHz
    public static let mono = SpeakerConfiguration(
        sampleRate: 44100,
        channels: 1,
        scheduledBufferCount: 3,
        bufferFrameCount: 1024,
        bufferDurationSeconds: 2.0
    )

    /// Low latency configuration
    public static let lowLatency = SpeakerConfiguration(
        sampleRate: 44100,
        channels: 2,
        scheduledBufferCount: 2,
        bufferFrameCount: 512,
        bufferDurationSeconds: 1.0
    )

    public init(
        sampleRate: Int = 44100,
        channels: Int = 2,
        scheduledBufferCount: Int = 3,
        bufferFrameCount: Int = 1024,
        bufferDurationSeconds: Double = 2.0
    ) {
        self.sampleRate = sampleRate
        self.channels = channels
        self.scheduledBufferCount = scheduledBufferCount
        self.bufferFrameCount = bufferFrameCount
        self.bufferDurationSeconds = bufferDurationSeconds
    }
}

// MARK: - Speaker Sink Errors

/// Errors that can occur during speaker output.
public enum SpeakerError: Error, LocalizedError {
    /// Audio engine failed to start
    case engineStartFailed(String)
    /// Buffer allocation failed
    case bufferAllocationFailed
    /// Format creation failed
    case formatCreationFailed
    /// Already running
    case alreadyRunning
    /// Not running
    case notRunning

    public var errorDescription: String? {
        switch self {
        case .engineStartFailed(let message):
            return "Audio engine failed to start: \(message)"
        case .bufferAllocationFailed:
            return "Failed to allocate audio buffer"
        case .formatCreationFailed:
            return "Failed to create audio format"
        case .alreadyRunning:
            return "Speaker is already running"
        case .notRunning:
            return "Speaker is not running"
        }
    }
}

// MARK: - Speaker Sink

/// Real-time speaker output using AVAudioEngine.
///
/// This actor plays audio through the device speakers with low latency.
/// It uses a ring buffer and pre-scheduled AVAudioPlayerNode buffers.
///
/// Example:
/// ```swift
/// let speaker = SpeakerSink(configuration: .stereo)
/// try await speaker.start()
///
/// // Write audio samples (shape: [channels, samples])
/// try await speaker.write(audioData)
///
/// try await speaker.stop()
/// ```
public actor SpeakerSink: @preconcurrency AudioSink {
    // MARK: - AudioStream Protocol

    public nonisolated let sampleRate: Int
    public nonisolated let channels: Int

    public var isActive: Bool {
        _isActive
    }

    // MARK: - Properties

    private let configuration: SpeakerConfiguration
    private var engine: AVAudioEngine?
    private var playerNode: AVAudioPlayerNode?
    private var ringBuffer: AudioRingBuffer?
    private var format: AVAudioFormat?
    private var _isActive = false

    // Pre-allocated buffer pool
    private var bufferPool: [AVAudioPCMBuffer] = []
    private var bufferPoolIndex = 0

    // Scheduling state
    private var isScheduling = false
    private var schedulingTask: Task<Void, Never>?

    // Statistics
    private var samplesPlayed: Int = 0
    private var underruns: Int = 0

    // MARK: - Initialization

    /// Creates a new speaker sink with the specified configuration.
    ///
    /// - Parameter configuration: Output configuration
    public init(configuration: SpeakerConfiguration = .stereo) {
        self.configuration = configuration
        self.sampleRate = configuration.sampleRate
        self.channels = configuration.channels
    }

    // MARK: - AudioSink Protocol

    /// Start speaker output.
    ///
    /// - Throws: SpeakerError if output cannot be started
    public func start() async throws {
        guard !_isActive else {
            throw SpeakerError.alreadyRunning
        }

        // Configure audio session
        try await AudioSessionManager.shared.configure(.playback)
        try await AudioSessionManager.shared.activate()

        // Create ring buffer
        let bufferCapacity = Int(Double(configuration.sampleRate) * configuration.bufferDurationSeconds)
        ringBuffer = AudioRingBuffer(capacity: bufferCapacity, channels: configuration.channels)

        // Create format
        guard let audioFormat = AVAudioFormat(
            standardFormatWithSampleRate: Double(configuration.sampleRate),
            channels: AVAudioChannelCount(configuration.channels)
        ) else {
            throw SpeakerError.formatCreationFailed
        }
        format = audioFormat

        // Pre-allocate buffer pool
        bufferPool = []
        for _ in 0..<configuration.scheduledBufferCount {
            guard let buffer = AVAudioPCMBuffer(
                pcmFormat: audioFormat,
                frameCapacity: AVAudioFrameCount(configuration.bufferFrameCount)
            ) else {
                throw SpeakerError.bufferAllocationFailed
            }
            bufferPool.append(buffer)
        }

        // Create and configure engine
        let engine = AVAudioEngine()
        let playerNode = AVAudioPlayerNode()

        engine.attach(playerNode)
        engine.connect(playerNode, to: engine.mainMixerNode, format: audioFormat)

        self.engine = engine
        self.playerNode = playerNode

        // Start engine
        engine.prepare()
        do {
            try engine.start()
        } catch {
            throw SpeakerError.engineStartFailed(error.localizedDescription)
        }

        // Start player
        playerNode.play()

        _isActive = true

        // Start scheduling loop
        startSchedulingLoop()
    }

    /// Stop speaker output.
    public func stop() async throws {
        guard _isActive else {
            throw SpeakerError.notRunning
        }

        // Stop scheduling
        schedulingTask?.cancel()
        schedulingTask = nil
        isScheduling = false

        // Stop player and engine
        playerNode?.stop()
        engine?.stop()

        playerNode = nil
        engine = nil
        format = nil
        bufferPool = []

        _isActive = false
    }

    /// Write samples to the speaker.
    ///
    /// This method supports cooperative cancellation - if the current Task is
    /// cancelled, it will throw `CancellationError` on the next call.
    ///
    /// - Parameter samples: Audio samples as MLXArray [channels, samples]
    /// - Throws: `SpeakerError.notRunning` if speaker is not active,
    ///           `CancellationError` if the task was cancelled
    public func write(_ samples: MLXArray) async throws {
        // Check for task cancellation before processing
        try Task.checkCancellation()

        guard _isActive else {
            throw SpeakerError.notRunning
        }

        guard let buffer = ringBuffer else { return }

        // Evaluate MLXArray and convert to interleaved format
        let shape = samples.shape
        let sampleCount: Int
        let channelCount: Int

        if shape.count == 1 {
            // Mono: [samples]
            channelCount = 1
            sampleCount = shape[0]
        } else {
            // Multi-channel: [channels, samples]
            channelCount = shape[0]
            sampleCount = shape[1]
        }

        // Get float data
        let floatData = samples.asArray(Float.self)

        // Convert to interleaved if multi-channel
        if channelCount == 1 || channels == 1 {
            // Already interleaved or mono
            _ = buffer.write(floatData)
        } else {
            // Deinterleave: [channels, samples] -> [L0, R0, L1, R1, ...]
            var interleaved = [Float](repeating: 0, count: sampleCount * min(channelCount, channels))
            for i in 0..<sampleCount {
                for c in 0..<min(channelCount, channels) {
                    interleaved[i * channels + c] = floatData[c * sampleCount + i]
                }
            }
            _ = buffer.write(interleaved)
        }
    }

    // MARK: - Scheduling

    private func startSchedulingLoop() {
        guard !isScheduling else { return }
        isScheduling = true

        schedulingTask = Task { [weak self] in
            while !Task.isCancelled {
                await self?.scheduleNextBuffer()

                // Small sleep to prevent busy waiting
                try? await Task.sleep(nanoseconds: 1_000_000)  // 1ms
            }
        }
    }

    private func scheduleNextBuffer() {
        guard _isActive,
              let buffer = ringBuffer,
              let playerNode = playerNode,
              let format = format
        else { return }

        // Check if we have enough data
        let frameCount = configuration.bufferFrameCount
        guard buffer.available >= frameCount else {
            // Not enough data - this is an underrun if we're playing
            return
        }

        // Get a buffer from the pool
        let pcmBuffer = bufferPool[bufferPoolIndex]
        bufferPoolIndex = (bufferPoolIndex + 1) % bufferPool.count

        // Read from ring buffer
        guard let samples = buffer.read(count: frameCount) else { return }

        // Fill PCM buffer
        pcmBuffer.frameLength = AVAudioFrameCount(frameCount)

        if let channelData = pcmBuffer.floatChannelData {
            let channelCount = Int(format.channelCount)

            if channelCount == 1 {
                // Mono
                for i in 0..<frameCount {
                    channelData[0][i] = samples[i]
                }
            } else {
                // Deinterleave for playback
                for frame in 0..<frameCount {
                    for channel in 0..<channelCount {
                        let sampleIndex = frame * channelCount + channel
                        if sampleIndex < samples.count {
                            channelData[channel][frame] = samples[sampleIndex]
                        }
                    }
                }
            }
        }

        // Schedule buffer
        playerNode.scheduleBuffer(pcmBuffer) { [self] in
            Task {
                await self.onBufferCompleted()
            }
        }

        samplesPlayed += frameCount
    }

    private func onBufferCompleted() {
        // Buffer finished playing, schedule next if data available
        scheduleNextBuffer()
    }

    // MARK: - Statistics

    /// Number of frames available in the buffer.
    public var available: Int {
        ringBuffer?.available ?? 0
    }

    /// Total frames played since start.
    public var totalPlayed: Int {
        samplesPlayed
    }

    /// Number of underruns (buffer empty when playback needed).
    public var underrunCount: Int {
        underruns
    }

    /// Buffer statistics.
    public var bufferStatistics: AudioRingBuffer.Statistics? {
        ringBuffer?.statistics
    }

    /// Estimated latency in seconds.
    public var estimatedLatency: TimeInterval {
        let bufferLatency = Double(configuration.scheduledBufferCount * configuration.bufferFrameCount) / Double(sampleRate)
        let sessionLatency = Task {
            await AudioSessionManager.shared.outputLatency
        }
        return bufferLatency
    }
}
