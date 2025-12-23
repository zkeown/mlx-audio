// Streaming.swift
// Real-time audio streaming support.
//
// This module provides production-ready audio streaming for iOS/macOS:
// - Audio sources: MicrophoneSource, FileSource
// - Audio sinks: SpeakerSink, FileSink
// - Streaming processors: StreamingSTFT, StreamingMelSpectrogram, StreamingTranscriber
// - Pipeline orchestration: StreamingPipeline
// - Audio session management: AudioSessionManager
// - Lock-free ring buffer: AudioRingBuffer

import Foundation
@preconcurrency import MLX
import MLXAudioPrimitives

// Re-export core streaming components
@_exported import struct Foundation.URL
@_exported import struct Foundation.TimeInterval

// MARK: - Audio Stream Protocol

/// Protocol for audio streams.
public protocol AudioStream: Sendable {
    /// Sample rate of the stream.
    var sampleRate: Int { get }

    /// Number of channels.
    var channels: Int { get }

    /// Whether the stream is currently active.
    var isActive: Bool { get }
}

// MARK: - Audio Source

/// Protocol for audio input sources.
public protocol AudioSource: AudioStream {
    /// Read samples from the source.
    ///
    /// - Parameter count: Number of samples to read.
    /// - Returns: Audio samples as MLXArray, or nil if end of stream.
    func read(count: Int) async throws -> MLXArray?

    /// Start the audio source.
    func start() async throws

    /// Stop the audio source.
    func stop() async throws
}

// MARK: - Audio Sink

/// Protocol for audio output sinks.
public protocol AudioSink: AudioStream {
    /// Write samples to the sink.
    ///
    /// - Parameter samples: Audio samples to write.
    func write(_ samples: MLXArray) async throws

    /// Start the audio sink.
    func start() async throws

    /// Stop the audio sink.
    func stop() async throws
}

// MARK: - Streaming Processor

/// Protocol for real-time audio processors.
public protocol StreamingProcessor: Sendable {
    /// Process a chunk of audio.
    ///
    /// - Parameter input: Input audio chunk.
    /// - Returns: Processed audio chunk.
    func process(_ input: MLXArray) async throws -> MLXArray

    /// Reset processor state.
    func reset() async
}

// MARK: - Streaming Errors

/// Errors that can occur during audio streaming.
public enum AudioStreamingError: Error, LocalizedError, Sendable {
    /// Audio session setup failed
    case audioSessionSetupFailed(String)
    /// Audio engine failed to start
    case engineStartFailed(String)
    /// Audio format mismatch
    case formatMismatch(String)
    /// Buffer overflow (input too fast)
    case bufferOverrun(dropped: Int)
    /// Buffer underflow (output too fast)
    case bufferUnderrun(duration: TimeInterval)
    /// Device not available
    case deviceNotAvailable(String)
    /// Permission denied
    case permissionDenied
    /// Interruption could not be recovered
    case interruptionNotRecoverable

    public var errorDescription: String? {
        switch self {
        case .audioSessionSetupFailed(let message):
            return "Audio session setup failed: \(message)"
        case .engineStartFailed(let message):
            return "Audio engine failed to start: \(message)"
        case .formatMismatch(let message):
            return "Audio format mismatch: \(message)"
        case .bufferOverrun(let dropped):
            return "Buffer overrun: \(dropped) samples dropped"
        case .bufferUnderrun(let duration):
            return "Buffer underrun: \(String(format: "%.3f", duration))s of silence"
        case .deviceNotAvailable(let device):
            return "Audio device not available: \(device)"
        case .permissionDenied:
            return "Audio permission denied"
        case .interruptionNotRecoverable:
            return "Audio interruption could not be recovered"
        }
    }
}

// MARK: - Audio Pipeline

/// A pipeline of streaming processors.
public struct AudioPipeline: Sendable {
    private let processors: [any StreamingProcessor]

    public init(processors: [any StreamingProcessor]) {
        self.processors = processors
    }

    /// Process audio through all processors in sequence.
    public func process(_ input: MLXArray) async throws -> MLXArray {
        var result = input
        for processor in processors {
            result = try await processor.process(result)
        }
        return result
    }

    /// Reset all processors.
    public func reset() async {
        for processor in processors {
            await processor.reset()
        }
    }
}
