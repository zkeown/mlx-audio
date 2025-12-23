// StreamingTranscriber.swift
// Streaming Whisper transcription processor.
//
// Accumulates audio and performs periodic transcription using Whisper.

import Foundation
@preconcurrency import MLX
import MLXAudioPrimitives

// MARK: - Transcription Result

/// Result from streaming transcription.
public struct StreamingTranscriptionResult: Sendable {
    /// Transcribed text
    public let text: String

    /// Start time in seconds
    public let startTime: TimeInterval

    /// End time in seconds
    public let endTime: TimeInterval

    /// Whether this is a final result (vs partial/interim)
    public let isFinal: Bool

    /// Language detected (if available)
    public let language: String?

    /// Confidence score (0-1, if available)
    public let confidence: Float?
}

// MARK: - Transcriber Configuration

/// Configuration for streaming transcription.
public struct StreamingTranscriberConfiguration: Sendable {
    /// Sample rate expected by the model (Whisper uses 16kHz)
    public var sampleRate: Int

    /// Chunk duration in seconds for transcription
    public var chunkDuration: TimeInterval

    /// Overlap between chunks (seconds) for context
    public var overlapDuration: TimeInterval

    /// Language hint (nil for auto-detect)
    public var language: String?

    /// Whether to include timestamps in output
    public var includeTimestamps: Bool

    /// Task type
    public var task: TranscriptionTask

    /// Default configuration
    public static let `default` = StreamingTranscriberConfiguration(
        sampleRate: 16000,
        chunkDuration: 5.0,
        overlapDuration: 0.5,
        language: nil,
        includeTimestamps: false,
        task: .transcribe
    )

    /// Real-time configuration with shorter chunks
    public static let realtime = StreamingTranscriberConfiguration(
        sampleRate: 16000,
        chunkDuration: 3.0,
        overlapDuration: 0.3,
        language: nil,
        includeTimestamps: false,
        task: .transcribe
    )

    public init(
        sampleRate: Int = 16000,
        chunkDuration: TimeInterval = 5.0,
        overlapDuration: TimeInterval = 0.5,
        language: String? = nil,
        includeTimestamps: Bool = false,
        task: TranscriptionTask = .transcribe
    ) {
        self.sampleRate = sampleRate
        self.chunkDuration = chunkDuration
        self.overlapDuration = overlapDuration
        self.language = language
        self.includeTimestamps = includeTimestamps
        self.task = task
    }
}

/// Transcription task type.
public enum TranscriptionTask: String, Sendable {
    /// Transcribe audio to text
    case transcribe
    /// Translate audio to English
    case translate
}

// MARK: - Transcriber Delegate

/// Delegate for receiving transcription results.
public protocol StreamingTranscriberDelegate: AnyObject, Sendable {
    /// Called when a transcription result is available.
    func transcriber(_ transcriber: StreamingTranscriber, didReceive result: StreamingTranscriptionResult) async

    /// Called when an error occurs during transcription.
    func transcriber(_ transcriber: StreamingTranscriber, didEncounterError error: Error) async
}

// MARK: - Streaming Transcriber

/// Streaming transcription processor using Whisper.
///
/// This actor accumulates audio samples and periodically performs transcription
/// using the Whisper model. It's designed for real-time streaming scenarios.
///
/// Note: This is a framework for streaming transcription. The actual Whisper
/// model inference is performed by MLXAudioModels, which should be loaded
/// separately and passed to this processor.
///
/// Example:
/// ```swift
/// let transcriber = StreamingTranscriber(configuration: .realtime)
/// transcriber.setDelegate(self)
///
/// // In processing loop
/// let _ = try await transcriber.process(audioChunk)
/// // Results delivered via delegate
/// ```
public actor StreamingTranscriber: StreamingProcessor {
    // MARK: - Properties

    private let configuration: StreamingTranscriberConfiguration
    private let melProcessor: StreamingMelSpectrogram

    /// Accumulated audio samples (at 16kHz)
    private var audioBuffer: [Float] = []

    /// Current position in the audio stream (samples)
    private var streamPosition: Int = 0

    /// Transcription results queue
    private var pendingResults: [StreamingTranscriptionResult] = []

    /// Delegate for results
    private weak var delegate: (any StreamingTranscriberDelegate)?

    /// Model inference function (to be set externally)
    private var inferenceFunction: ((MLXArray) async throws -> String)?

    // MARK: - Initialization

    /// Creates a new streaming transcriber.
    ///
    /// - Parameter configuration: Transcriber configuration
    public init(configuration: StreamingTranscriberConfiguration = .default) {
        self.configuration = configuration

        // Create mel spectrogram processor for Whisper
        // Whisper uses n_fft=400, hop_length=160, n_mels=80 at 16kHz
        self.melProcessor = StreamingMelSpectrogram(
            sampleRate: configuration.sampleRate,
            nFFT: 400,
            hopLength: 160,
            nMels: 80
        )
    }

    // MARK: - Configuration

    /// Set the transcription delegate.
    public func setDelegate(_ delegate: (any StreamingTranscriberDelegate)?) {
        self.delegate = delegate
    }

    /// Set the model inference function.
    ///
    /// This function takes mel spectrogram features and returns transcribed text.
    /// It should be connected to a loaded Whisper model.
    public func setInferenceFunction(_ function: @escaping (MLXArray) async throws -> String) {
        self.inferenceFunction = function
    }

    // MARK: - StreamingProcessor Protocol

    /// Process audio and perform transcription when enough data is available.
    ///
    /// Returns the input audio unchanged (passthrough) while triggering
    /// transcription callbacks via the delegate.
    ///
    /// - Parameter input: Audio samples [channels, samples] at model sample rate
    /// - Returns: Passthrough audio (same as input)
    public func process(_ input: MLXArray) async throws -> MLXArray {
        // Convert to mono if needed
        let monoAudio: [Float]
        if input.ndim == 1 {
            monoAudio = input.asArray(Float.self)
        } else {
            let mono = MLX.mean(input, axis: 0)
            monoAudio = mono.asArray(Float.self)
        }

        // Accumulate audio
        audioBuffer.append(contentsOf: monoAudio)

        // Check if we have enough audio for transcription
        let chunkSamples = Int(configuration.chunkDuration * Double(configuration.sampleRate))

        while audioBuffer.count >= chunkSamples {
            // Extract chunk for transcription
            let chunk = Array(audioBuffer.prefix(chunkSamples))

            // Perform transcription (async)
            Task {
                await transcribeChunk(chunk)
            }

            // Remove processed samples (keeping overlap)
            let overlapSamples = Int(configuration.overlapDuration * Double(configuration.sampleRate))
            let removeCount = chunkSamples - overlapSamples
            audioBuffer.removeFirst(removeCount)
            streamPosition += removeCount
        }

        // Return audio unchanged (passthrough)
        return input
    }

    /// Reset transcriber state.
    public func reset() async {
        audioBuffer.removeAll(keepingCapacity: true)
        streamPosition = 0
        pendingResults.removeAll()
        await melProcessor.reset()
    }

    // MARK: - Transcription

    private func transcribeChunk(_ samples: [Float]) async {
        // Calculate timestamps
        let startTime = Double(streamPosition) / Double(configuration.sampleRate)
        let endTime = startTime + configuration.chunkDuration

        // Convert to mel spectrogram
        let audio = MLXArray(samples)
        let melSpec: MLXArray
        do {
            melSpec = try await melProcessor.process(audio)
        } catch {
            await delegate?.transcriber(self, didEncounterError: error)
            return
        }

        // Run inference if function is set
        let text: String
        if let inference = inferenceFunction {
            do {
                text = try await inference(melSpec)
            } catch {
                await delegate?.transcriber(self, didEncounterError: error)
                return
            }
        } else {
            // No inference function - return placeholder
            text = "[Whisper model not loaded]"
        }

        // Create result
        let result = StreamingTranscriptionResult(
            text: text,
            startTime: startTime,
            endTime: endTime,
            isFinal: true,
            language: configuration.language,
            confidence: nil
        )

        // Deliver result
        await delegate?.transcriber(self, didReceive: result)
    }

    // MARK: - State Queries

    /// Amount of audio buffered (seconds).
    public var bufferedDuration: TimeInterval {
        Double(audioBuffer.count) / Double(configuration.sampleRate)
    }

    /// Current stream position (seconds).
    public var currentPosition: TimeInterval {
        Double(streamPosition) / Double(configuration.sampleRate)
    }

    /// Whether transcription is ready (enough audio buffered).
    public var isReady: Bool {
        let chunkSamples = Int(configuration.chunkDuration * Double(configuration.sampleRate))
        return audioBuffer.count >= chunkSamples
    }
}

// MARK: - Voice Activity Detection

/// Simple voice activity detector for streaming.
///
/// This can be used to filter audio before transcription,
/// only transcribing segments with detected speech.
public actor VoiceActivityDetector: StreamingProcessor {
    // MARK: - Properties

    /// Energy threshold for speech detection
    private let energyThreshold: Float

    /// Minimum speech duration (samples)
    private let minSpeechSamples: Int

    /// Current speech state
    private var isSpeaking = false

    /// Samples since last state change
    private var samplesSinceChange = 0

    // MARK: - Initialization

    /// Creates a voice activity detector.
    ///
    /// - Parameters:
    ///   - energyThreshold: RMS energy threshold for speech (default: 0.01)
    ///   - minSpeechDuration: Minimum speech duration in seconds (default: 0.1)
    ///   - sampleRate: Audio sample rate (default: 16000)
    public init(
        energyThreshold: Float = 0.01,
        minSpeechDuration: TimeInterval = 0.1,
        sampleRate: Int = 16000
    ) {
        self.energyThreshold = energyThreshold
        self.minSpeechSamples = Int(minSpeechDuration * Double(sampleRate))
    }

    /// Process audio and detect voice activity.
    ///
    /// Returns audio unchanged, but updates internal speech state.
    /// Use `isSpeechDetected` to check current state.
    public func process(_ input: MLXArray) async throws -> MLXArray {
        // Calculate RMS energy
        let squared = input * input
        let mean = MLX.mean(squared)
        let rms = MLX.sqrt(mean)

        let energy = rms.item(Float.self)
        let wasSpeeking = isSpeaking

        if energy > energyThreshold {
            if !isSpeaking {
                samplesSinceChange += input.shape.last ?? 0
                if samplesSinceChange >= minSpeechSamples {
                    isSpeaking = true
                    samplesSinceChange = 0
                }
            } else {
                samplesSinceChange = 0
            }
        } else {
            if isSpeaking {
                samplesSinceChange += input.shape.last ?? 0
                if samplesSinceChange >= minSpeechSamples {
                    isSpeaking = false
                    samplesSinceChange = 0
                }
            } else {
                samplesSinceChange = 0
            }
        }

        return input
    }

    /// Reset detector state.
    public func reset() async {
        isSpeaking = false
        samplesSinceChange = 0
    }

    /// Whether speech is currently detected.
    public var isSpeechDetected: Bool {
        isSpeaking
    }
}
