// StreamingPipeline.swift
// Orchestrates source -> processor -> sink audio pipeline.
//
// Manages the streaming lifecycle and coordinates audio flow.

import Foundation
@preconcurrency import MLX

// MARK: - Pipeline State

/// State of the streaming pipeline.
public enum StreamingPipelineState: Sendable, Equatable {
    /// Pipeline is idle, not started
    case idle
    /// Pipeline is running and processing audio
    case running
    /// Pipeline is paused (source stopped, sink buffering)
    case paused
    /// Pipeline has stopped (can be restarted)
    case stopped
    /// Pipeline encountered an error
    case error(String)

    public static func == (lhs: StreamingPipelineState, rhs: StreamingPipelineState) -> Bool {
        switch (lhs, rhs) {
        case (.idle, .idle), (.running, .running), (.paused, .paused), (.stopped, .stopped):
            return true
        case (.error(let a), .error(let b)):
            return a == b
        default:
            return false
        }
    }
}

// MARK: - Pipeline Statistics

/// Statistics for the streaming pipeline.
public struct StreamingPipelineStatistics: Sendable {
    /// Number of chunks processed
    public var chunksProcessed: Int = 0

    /// Total samples processed
    public var samplesProcessed: Int = 0

    /// Total audio duration processed (seconds)
    public var totalDuration: TimeInterval = 0

    /// Total CPU time spent processing (seconds)
    public var processingTime: TimeInterval = 0

    /// Number of input buffer underruns
    public var bufferUnderruns: Int = 0

    /// Number of output buffer overruns
    public var bufferOverruns: Int = 0

    /// Real-time factor (> 1 means faster than real-time)
    public var realtimeFactor: Double {
        guard processingTime > 0 else { return .infinity }
        return totalDuration / processingTime
    }
}

// MARK: - Pipeline Configuration

/// Configuration for the streaming pipeline.
public struct StreamingPipelineConfiguration: Sendable {
    /// Number of samples to read per chunk
    public var chunkSize: Int

    /// Maximum time to wait for source data (seconds)
    public var readTimeout: TimeInterval

    /// Enable statistics collection
    public var collectStatistics: Bool

    /// Callback interval for progress updates
    public var progressInterval: TimeInterval?

    /// Default configuration
    public static let `default` = StreamingPipelineConfiguration(
        chunkSize: 1024,
        readTimeout: 1.0,
        collectStatistics: true,
        progressInterval: 0.1
    )

    /// Low-latency configuration
    public static let lowLatency = StreamingPipelineConfiguration(
        chunkSize: 256,
        readTimeout: 0.5,
        collectStatistics: false,
        progressInterval: nil
    )

    public init(
        chunkSize: Int = 1024,
        readTimeout: TimeInterval = 1.0,
        collectStatistics: Bool = true,
        progressInterval: TimeInterval? = 0.1
    ) {
        self.chunkSize = chunkSize
        self.readTimeout = readTimeout
        self.collectStatistics = collectStatistics
        self.progressInterval = progressInterval
    }
}

// MARK: - Pipeline Errors

/// Errors that can occur in the streaming pipeline.
public enum StreamingPipelineError: Error, LocalizedError {
    /// Pipeline is in wrong state for operation
    case invalidState(current: StreamingPipelineState, expected: String)
    /// Source error
    case sourceError(Error)
    /// Processor error
    case processorError(Error)
    /// Sink error
    case sinkError(Error)
    /// Pipeline was cancelled
    case cancelled

    public var errorDescription: String? {
        switch self {
        case .invalidState(let current, let expected):
            return "Pipeline in invalid state: \(current), expected: \(expected)"
        case .sourceError(let error):
            return "Source error: \(error.localizedDescription)"
        case .processorError(let error):
            return "Processor error: \(error.localizedDescription)"
        case .sinkError(let error):
            return "Sink error: \(error.localizedDescription)"
        case .cancelled:
            return "Pipeline was cancelled"
        }
    }
}

// MARK: - Progress Delegate

/// Delegate for receiving pipeline progress updates.
public protocol StreamingPipelineDelegate: AnyObject, Sendable {
    /// Called when pipeline state changes.
    func pipelineDidChangeState(_ pipeline: StreamingPipeline, to state: StreamingPipelineState) async

    /// Called periodically with progress updates.
    func pipelineDidProgress(_ pipeline: StreamingPipeline, statistics: StreamingPipelineStatistics) async

    /// Called when an error occurs.
    func pipelineDidEncounterError(_ pipeline: StreamingPipeline, error: Error) async
}

// MARK: - Streaming Pipeline

/// Orchestrates a streaming audio pipeline from source through processor to sink.
///
/// The pipeline manages the lifecycle of audio streaming and coordinates
/// the flow of audio data from a source, through optional processing,
/// to a sink.
///
/// Example:
/// ```swift
/// let pipeline = StreamingPipeline(
///     source: MicrophoneSource(),
///     processor: StreamingSTFT(),
///     sink: SpeakerSink()
/// )
///
/// try await pipeline.start()
/// // Pipeline runs until stopped or source ends
/// try await pipeline.wait()
/// ```
public actor StreamingPipeline {
    // MARK: - Properties

    private let source: any AudioSource
    private let processor: (any StreamingProcessor)?
    private let sink: (any AudioSink)?
    private let configuration: StreamingPipelineConfiguration

    private var state: StreamingPipelineState = .idle
    private var statistics = StreamingPipelineStatistics()

    private var pipelineTask: Task<Void, Error>?
    private weak var delegate: (any StreamingPipelineDelegate)?

    // MARK: - Initialization

    /// Creates a new streaming pipeline.
    ///
    /// - Parameters:
    ///   - source: Audio source (e.g., MicrophoneSource, FileSource)
    ///   - processor: Optional audio processor (e.g., StreamingSTFT)
    ///   - sink: Optional audio sink (e.g., SpeakerSink, FileSink)
    ///   - configuration: Pipeline configuration
    public init(
        source: any AudioSource,
        processor: (any StreamingProcessor)? = nil,
        sink: (any AudioSink)? = nil,
        configuration: StreamingPipelineConfiguration = .default
    ) {
        self.source = source
        self.processor = processor
        self.sink = sink
        self.configuration = configuration
    }

    /// Creates a passthrough pipeline (source directly to sink).
    ///
    /// - Parameters:
    ///   - source: Audio source
    ///   - sink: Audio sink
    public init(
        source: any AudioSource,
        sink: any AudioSink
    ) {
        self.source = source
        self.processor = nil
        self.sink = sink
        self.configuration = .default
    }

    // MARK: - Delegate

    /// Set the progress delegate.
    public func setDelegate(_ delegate: (any StreamingPipelineDelegate)?) {
        self.delegate = delegate
    }

    // MARK: - Control

    /// Start the pipeline.
    ///
    /// Starts the source, processor, and sink, then begins the processing loop.
    ///
    /// - Throws: StreamingPipelineError if pipeline cannot be started
    public func start() async throws {
        guard state == .idle || state == .stopped else {
            throw StreamingPipelineError.invalidState(current: state, expected: "idle or stopped")
        }

        // Reset statistics
        statistics = StreamingPipelineStatistics()

        // Start source
        do {
            try await source.start()
        } catch {
            throw StreamingPipelineError.sourceError(error)
        }

        // Start sink
        if let sink = sink {
            do {
                try await sink.start()
            } catch {
                try? await source.stop()
                throw StreamingPipelineError.sinkError(error)
            }
        }

        // Update state
        state = .running
        await delegate?.pipelineDidChangeState(self, to: state)

        // Start processing loop
        pipelineTask = Task {
            try await runProcessingLoop()
        }
    }

    /// Stop the pipeline.
    ///
    /// Stops the processing loop and all components.
    public func stop() async {
        // Cancel processing task
        pipelineTask?.cancel()
        pipelineTask = nil

        // Stop components
        try? await source.stop()
        if let sink = sink {
            try? await sink.stop()
        }
        if let processor = processor {
            await processor.reset()
        }

        // Update state
        state = .stopped
        await delegate?.pipelineDidChangeState(self, to: state)
    }

    /// Pause the pipeline.
    ///
    /// Pauses audio capture while keeping sink running (with silence).
    public func pause() async {
        guard state == .running else { return }

        pipelineTask?.cancel()
        pipelineTask = nil

        try? await source.stop()

        state = .paused
        await delegate?.pipelineDidChangeState(self, to: state)
    }

    /// Resume the pipeline after pause.
    public func resume() async throws {
        guard state == .paused else {
            throw StreamingPipelineError.invalidState(current: state, expected: "paused")
        }

        try await source.start()

        state = .running
        await delegate?.pipelineDidChangeState(self, to: state)

        pipelineTask = Task {
            try await runProcessingLoop()
        }
    }

    /// Wait for the pipeline to complete.
    ///
    /// Blocks until the source ends or an error occurs.
    public func wait() async throws {
        guard let task = pipelineTask else { return }

        do {
            try await task.value
        } catch is CancellationError {
            // Normal cancellation
        } catch {
            throw error
        }
    }

    // MARK: - State Queries

    /// Current pipeline state.
    public var currentState: StreamingPipelineState {
        state
    }

    /// Current statistics.
    public var currentStatistics: StreamingPipelineStatistics {
        statistics
    }

    /// Whether the pipeline is actively processing.
    public var isRunning: Bool {
        state == .running
    }

    /// Estimated end-to-end latency in seconds.
    public var estimatedLatency: TimeInterval {
        // Estimate based on chunk size and sample rate
        let sampleRate = Double(source.sampleRate)
        let chunkLatency = Double(configuration.chunkSize) / sampleRate
        return chunkLatency * 2  // Input + output buffering
    }

    // MARK: - Processing Loop

    private func runProcessingLoop() async throws {
        let chunkSize = configuration.chunkSize
        let collectStats = configuration.collectStatistics
        var lastProgressTime = Date()

        while !Task.isCancelled && state == .running {
            let chunkStartTime = Date()

            // Read from source
            guard let inputAudio = try await source.read(count: chunkSize) else {
                // Source ended (EOF)
                break
            }

            // Process audio
            var outputAudio: MLXArray
            if let processor = processor {
                do {
                    outputAudio = try await processor.process(inputAudio)
                } catch {
                    state = .error(error.localizedDescription)
                    await delegate?.pipelineDidEncounterError(self, error: error)
                    throw StreamingPipelineError.processorError(error)
                }
            } else {
                outputAudio = inputAudio
            }

            // Write to sink
            if let sink = sink {
                do {
                    try await sink.write(outputAudio)
                } catch {
                    state = .error(error.localizedDescription)
                    await delegate?.pipelineDidEncounterError(self, error: error)
                    throw StreamingPipelineError.sinkError(error)
                }
            }

            // Update statistics
            if collectStats {
                let chunkDuration = Date().timeIntervalSince(chunkStartTime)
                let sampleRate = Double(source.sampleRate)

                statistics.chunksProcessed += 1
                statistics.samplesProcessed += chunkSize
                statistics.totalDuration += Double(chunkSize) / sampleRate
                statistics.processingTime += chunkDuration
            }

            // Progress callback
            if let interval = configuration.progressInterval {
                let now = Date()
                if now.timeIntervalSince(lastProgressTime) >= interval {
                    await delegate?.pipelineDidProgress(self, statistics: statistics)
                    lastProgressTime = now
                }
            }
        }

        // Source ended normally
        if state == .running {
            state = .stopped
            await delegate?.pipelineDidChangeState(self, to: state)
        }
    }
}

// MARK: - Convenience Extensions

extension StreamingPipeline {
    /// Create a microphone to speaker passthrough pipeline.
    public static func passthrough(
        micConfig: MicrophoneConfiguration = .mono,
        speakerConfig: SpeakerConfiguration = .stereo
    ) -> StreamingPipeline {
        StreamingPipeline(
            source: MicrophoneSource(configuration: micConfig),
            sink: SpeakerSink(configuration: speakerConfig)
        )
    }

    /// Create a file processing pipeline.
    public static func fileProcessor(
        inputURL: URL,
        outputURL: URL,
        processor: any StreamingProcessor
    ) -> StreamingPipeline {
        StreamingPipeline(
            source: FileSource(url: inputURL),
            processor: processor,
            sink: FileSink(url: outputURL, sampleRate: 44100, channels: 2)
        )
    }
}
