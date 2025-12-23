// StreamingPipelineTests.swift
// Tests for the StreamingPipeline orchestrator.

import XCTest
@preconcurrency import MLX
@testable import MLXAudioStreaming

// MARK: - Mock Audio Source

/// Mock audio source for testing that generates synthetic audio.
actor MockAudioSource: AudioSource {
    nonisolated let sampleRate: Int
    nonisolated let channels: Int

    var isActive: Bool { _isActive }

    private var _isActive = false
    private var samplesGenerated = 0
    private let totalSamples: Int
    private let chunkSize: Int

    init(sampleRate: Int = 44100, channels: Int = 1, durationSeconds: Double = 1.0, chunkSize: Int = 1024) {
        self.sampleRate = sampleRate
        self.channels = channels
        self.totalSamples = Int(Double(sampleRate) * durationSeconds)
        self.chunkSize = chunkSize
    }

    func start() async throws {
        _isActive = true
        samplesGenerated = 0
    }

    func stop() async throws {
        _isActive = false
    }

    func read(count: Int) async throws -> MLXArray? {
        guard _isActive else { return nil }

        if samplesGenerated >= totalSamples {
            return nil  // EOF
        }

        let remaining = totalSamples - samplesGenerated
        let actualCount = min(count, remaining)

        // Generate sine wave
        var samples = [Float]()
        for i in 0..<actualCount {
            let t = Double(samplesGenerated + i) / Double(sampleRate)
            let value = Float(sin(2.0 * .pi * 440.0 * t))
            for _ in 0..<channels {
                samples.append(value)
            }
        }

        samplesGenerated += actualCount

        if channels == 1 {
            return MLXArray(samples).reshaped([1, actualCount])
        } else {
            return MLXArray(samples).reshaped([channels, actualCount])
        }
    }
}

// MARK: - Mock Audio Sink

/// Mock audio sink for testing that accumulates received audio.
actor MockAudioSink: AudioSink {
    nonisolated let sampleRate: Int
    nonisolated let channels: Int

    var isActive: Bool { _isActive }

    private var _isActive = false
    private var receivedSamples: [[Float]] = []
    private var totalFrames = 0

    init(sampleRate: Int = 44100, channels: Int = 2) {
        self.sampleRate = sampleRate
        self.channels = channels
    }

    func start() async throws {
        _isActive = true
        receivedSamples = []
        totalFrames = 0
    }

    func stop() async throws {
        _isActive = false
    }

    func write(_ samples: MLXArray) async throws {
        guard _isActive else { return }

        let floats = samples.asArray(Float.self)
        receivedSamples.append(floats)
        totalFrames += samples.shape.last ?? 0
    }

    var receivedFrameCount: Int {
        totalFrames
    }

    var allReceivedSamples: [Float] {
        receivedSamples.flatMap { $0 }
    }
}

// MARK: - Mock Processor

/// Mock processor that passes through or modifies audio.
actor MockProcessor: StreamingProcessor {
    private var processCount = 0
    private let gain: Float

    init(gain: Float = 1.0) {
        self.gain = gain
    }

    func process(_ input: MLXArray) async throws -> MLXArray {
        processCount += 1
        return input * MLXArray(gain)
    }

    func reset() async {
        processCount = 0
    }

    var timesProcessed: Int {
        processCount
    }
}

// MARK: - Pipeline Tests

final class StreamingPipelineTests: XCTestCase {

    // MARK: - Basic Pipeline Tests

    func testPipelineCreation() async {
        let source = MockAudioSource()
        let sink = MockAudioSink()
        let pipeline = StreamingPipeline(source: source, sink: sink)

        let state = await pipeline.currentState
        XCTAssertEqual(state, .idle)
    }

    func testPipelineStartAndStop() async throws {
        let source = MockAudioSource(durationSeconds: 0.5)
        let sink = MockAudioSink()
        let pipeline = StreamingPipeline(source: source, sink: sink)

        try await pipeline.start()
        var state = await pipeline.currentState
        XCTAssertEqual(state, .running)

        // Let it run briefly
        try await Task.sleep(nanoseconds: 100_000_000)  // 100ms

        await pipeline.stop()
        state = await pipeline.currentState
        XCTAssertEqual(state, .stopped)
    }

    func testPipelineProcessesAudio() async throws {
        let source = MockAudioSource(durationSeconds: 0.2)
        let sink = MockAudioSink()
        let pipeline = StreamingPipeline(
            source: source,
            sink: sink,
            configuration: StreamingPipelineConfiguration(chunkSize: 512)
        )

        try await pipeline.start()
        try await pipeline.wait()

        let receivedFrames = await sink.receivedFrameCount
        XCTAssertGreaterThan(receivedFrames, 0, "Sink should receive audio")
    }

    func testPipelineWithProcessor() async throws {
        let source = MockAudioSource(durationSeconds: 0.2)
        let processor = MockProcessor(gain: 0.5)
        let sink = MockAudioSink()
        let pipeline = StreamingPipeline(
            source: source,
            processor: processor,
            sink: sink
        )

        try await pipeline.start()
        try await pipeline.wait()

        let timesProcessed = await processor.timesProcessed
        XCTAssertGreaterThan(timesProcessed, 0, "Processor should be called")
    }

    // MARK: - State Transition Tests

    func testPipelineStateTransitions() async throws {
        let source = MockAudioSource(durationSeconds: 0.5)
        let pipeline = StreamingPipeline(source: source, sink: nil)

        // Idle -> Running
        try await pipeline.start()
        var state = await pipeline.currentState
        XCTAssertEqual(state, .running)

        // Running -> Paused
        await pipeline.pause()
        state = await pipeline.currentState
        XCTAssertEqual(state, .paused)

        // Paused -> Running
        try await pipeline.resume()
        state = await pipeline.currentState
        XCTAssertEqual(state, .running)

        // Running -> Stopped
        await pipeline.stop()
        state = await pipeline.currentState
        XCTAssertEqual(state, .stopped)
    }

    func testCannotStartWhileRunning() async throws {
        let source = MockAudioSource(durationSeconds: 1.0)
        let pipeline = StreamingPipeline(source: source, sink: nil)

        try await pipeline.start()

        do {
            try await pipeline.start()
            XCTFail("Should throw when starting already running pipeline")
        } catch {
            // Expected
        }

        await pipeline.stop()
    }

    // MARK: - Statistics Tests

    func testStatisticsCollection() async throws {
        let source = MockAudioSource(durationSeconds: 0.3)
        let pipeline = StreamingPipeline(
            source: source,
            sink: nil,
            configuration: StreamingPipelineConfiguration(collectStatistics: true)
        )

        try await pipeline.start()
        try await pipeline.wait()

        let stats = await pipeline.currentStatistics
        XCTAssertGreaterThan(stats.chunksProcessed, 0)
        XCTAssertGreaterThan(stats.samplesProcessed, 0)
        XCTAssertGreaterThan(stats.totalDuration, 0)
    }

    func testRealtimeFactor() async throws {
        let source = MockAudioSource(durationSeconds: 0.5)
        let pipeline = StreamingPipeline(
            source: source,
            sink: nil,
            configuration: StreamingPipelineConfiguration(collectStatistics: true)
        )

        try await pipeline.start()
        try await pipeline.wait()

        let stats = await pipeline.currentStatistics
        // Should process faster than real-time (since no actual audio I/O)
        XCTAssertGreaterThan(stats.realtimeFactor, 1.0,
            "Mock pipeline should be faster than real-time")
    }

    // MARK: - EOF Handling

    func testPipelineStopsAtEOF() async throws {
        let source = MockAudioSource(durationSeconds: 0.1)  // Short audio
        let pipeline = StreamingPipeline(source: source, sink: nil)

        try await pipeline.start()

        // Should complete on its own when source ends
        try await pipeline.wait()

        let state = await pipeline.currentState
        XCTAssertEqual(state, .stopped)
    }

    // MARK: - Latency Tests

    func testEstimatedLatency() async {
        let source = MockAudioSource(sampleRate: 44100)
        let pipeline = StreamingPipeline(
            source: source,
            sink: nil,
            configuration: StreamingPipelineConfiguration(chunkSize: 1024)
        )

        let latency = await pipeline.estimatedLatency

        // Expected: chunkSize / sampleRate * 2 ≈ 46ms
        XCTAssertGreaterThan(latency, 0.01)
        XCTAssertLessThan(latency, 0.1)
    }

    // MARK: - Convenience Factory Tests

    func testPassthroughFactory() async {
        let pipeline = StreamingPipeline.passthrough()

        let state = await pipeline.currentState
        XCTAssertEqual(state, .idle)
    }
}
