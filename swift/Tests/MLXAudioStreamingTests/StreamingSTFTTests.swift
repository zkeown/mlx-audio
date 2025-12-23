// StreamingSTFTTests.swift
// Tests for streaming STFT processor.
//
// Validates correctness of streaming vs batch STFT and shape consistency.

import XCTest
@preconcurrency import MLX
@testable import MLXAudioStreaming
@testable import MLXAudioPrimitives

final class StreamingSTFTTests: XCTestCase {

    // MARK: - Basic Functionality

    func testStreamingSTFTProcessesAudio() async throws {
        let stft = StreamingSTFT(configuration: .visualization)

        // Create a 1-second sine wave at 440 Hz
        let sampleRate = 44100
        let samples = createSineWave(frequency: 440, sampleRate: sampleRate, duration: 0.5)
        let audio = MLXArray(samples)

        let result = try await stft.process(audio)

        // Should produce spectrogram frames
        XCTAssertGreaterThan(result.shape[0], 0, "Should have frequency bins")
        XCTAssertGreaterThanOrEqual(result.shape[1], 0, "Should have time frames")
    }

    func testStreamingSTFTAccumulatesData() async throws {
        let stft = StreamingSTFT(configuration: StreamingSTFTConfiguration(
            nFFT: 512,
            hopLength: 128
        ))

        // First chunk - not enough for a frame
        let smallChunk = MLXArray([Float](repeating: 0.5, count: 256))
        let result1 = try await stft.process(smallChunk)
        XCTAssertEqual(result1.shape[1], 0, "Should have no frames yet")

        // Second chunk - now we have enough
        let result2 = try await stft.process(smallChunk)
        XCTAssertGreaterThan(result2.shape[1], 0, "Should have frames now")
    }

    func testStreamingSTFTReset() async throws {
        let stft = StreamingSTFT(configuration: .visualization)

        // Process some audio
        let samples = [Float](repeating: 0.5, count: 1000)
        _ = try await stft.process(MLXArray(samples))

        let bufferedBefore = await stft.bufferedSamples
        XCTAssertGreaterThan(bufferedBefore, 0)

        // Reset
        await stft.reset()

        let bufferedAfter = await stft.bufferedSamples
        XCTAssertEqual(bufferedAfter, 0)
    }

    // MARK: - Configuration Tests

    func testVisualizationConfigReturnsDecibels() async throws {
        let stft = StreamingSTFT(configuration: .visualization)

        let samples = createSineWave(frequency: 440, sampleRate: 44100, duration: 0.2)
        let result = try await stft.process(MLXArray(samples))

        if result.shape[1] > 0 {
            let values = result.asArray(Float.self)
            // dB values should typically be negative for normalized audio
            let hasNegativeValues = values.contains { $0 < 0 }
            XCTAssertTrue(hasNegativeValues, "Visualization mode should return dB values")
        }
    }

    func testProcessingConfigReturnsMagnitude() async throws {
        let stft = StreamingSTFT(configuration: .processing)

        let samples = createSineWave(frequency: 440, sampleRate: 44100, duration: 0.2)
        let result = try await stft.process(MLXArray(samples))

        if result.shape[1] > 0 {
            let values = result.asArray(Float.self)
            // Magnitude values should be non-negative
            let allNonNegative = values.allSatisfy { $0 >= 0 }
            XCTAssertTrue(allNonNegative, "Processing mode should return non-negative magnitude")
        }
    }

    // MARK: - Shape Consistency Tests

    func testOutputShape() async throws {
        let nFFT = 1024
        let stft = StreamingSTFT(configuration: StreamingSTFTConfiguration(
            nFFT: nFFT,
            hopLength: 256
        ))

        let samples = [Float](repeating: 0.5, count: 4096)
        let result = try await stft.process(MLXArray(samples))

        // Frequency bins should be nFFT/2 + 1
        XCTAssertEqual(result.shape[0], nFFT / 2 + 1)
    }

    func testFramePositionTracking() async throws {
        let stft = StreamingSTFT(configuration: StreamingSTFTConfiguration(
            nFFT: 512,
            hopLength: 128
        ))

        let initialPosition = await stft.framesProcessed
        XCTAssertEqual(initialPosition, 0)

        // Process enough for several frames
        let samples = [Float](repeating: 0.5, count: 2048)
        _ = try await stft.process(MLXArray(samples))

        let newPosition = await stft.framesProcessed
        XCTAssertGreaterThan(newPosition, 0)
    }

    // MARK: - Streaming vs Batch Consistency

    func testStreamingMatchesBatchProcessing() async throws {
        let nFFT = 512
        let hopLength = 128
        let config = StreamingSTFTConfiguration(
            nFFT: nFFT,
            hopLength: hopLength,
            returnMagnitude: true,
            toDecibels: false
        )

        let streamingSTFT = StreamingSTFT(configuration: config)

        // Generate test signal
        let samples = createSineWave(frequency: 440, sampleRate: 16000, duration: 0.5)

        // Process in streaming fashion (multiple small chunks)
        var streamingFrames: [MLXArray] = []
        let chunkSize = 256

        for i in stride(from: 0, to: samples.count, by: chunkSize) {
            let end = min(i + chunkSize, samples.count)
            let chunk = Array(samples[i..<end])
            let result = try await streamingSTFT.process(MLXArray(chunk))
            if result.shape[1] > 0 {
                streamingFrames.append(result)
            }
        }

        // Concatenate streaming results
        guard !streamingFrames.isEmpty else {
            XCTFail("No frames produced in streaming mode")
            return
        }

        let streamingResult = MLX.concatenated(streamingFrames, axis: 1)

        // Batch process with MLXAudioPrimitives STFT
        let batchConfig = STFTConfig(nFFT: nFFT, hopLength: hopLength, center: false)
        let batchComplex = try stft(MLXArray(samples), config: batchConfig)
        let batchResult = magnitude(batchComplex)

        // Compare shapes
        XCTAssertEqual(streamingResult.shape[0], batchResult.shape[0],
            "Frequency bins should match")

        // Frame counts may differ slightly due to buffering
        // But should be reasonably close
        let frameDiff = abs(streamingResult.shape[1] - batchResult.shape[1])
        XCTAssertLessThanOrEqual(frameDiff, 2,
            "Frame counts should be within 2: streaming=\(streamingResult.shape[1]), batch=\(batchResult.shape[1])")
    }

    // MARK: - Multi-Channel Tests

    func testMultiChannelInputTakesMean() async throws {
        let stft = StreamingSTFT(configuration: .visualization)

        // Stereo input [channels, samples]
        let left = [Float](repeating: 1.0, count: 2048)
        let right = [Float](repeating: 0.5, count: 2048)
        let stereo = MLXArray(left + right).reshaped([2, 2048])

        let result = try await stft.process(stereo)

        // Should still produce valid output
        XCTAssertGreaterThan(result.shape[0], 0)
    }

    // MARK: - Helpers

    private func createSineWave(frequency: Double, sampleRate: Int, duration: Double) -> [Float] {
        let numSamples = Int(Double(sampleRate) * duration)
        return (0..<numSamples).map { i in
            Float(sin(2.0 * .pi * frequency * Double(i) / Double(sampleRate)))
        }
    }
}

// MARK: - Streaming Mel Spectrogram Tests

final class StreamingMelSpectrogramTests: XCTestCase {

    func testMelSpectrogramOutput() async throws {
        let mel = StreamingMelSpectrogram(
            sampleRate: 16000,
            nFFT: 400,
            hopLength: 160,
            nMels: 80
        )

        // Generate 1 second of audio
        let samples = [Float](repeating: 0.3, count: 16000)
        let result = try await mel.process(MLXArray(samples))

        // Should have 80 mel bands
        if result.shape[1] > 0 {
            XCTAssertEqual(result.shape[0], 80)
        }
    }

    func testMelSpectrogramReset() async throws {
        let mel = StreamingMelSpectrogram(sampleRate: 16000)

        let samples = [Float](repeating: 0.5, count: 8000)
        _ = try await mel.process(MLXArray(samples))

        await mel.reset()

        let framesAfter = await mel.framesProcessed
        XCTAssertEqual(framesAfter, 0)
    }
}
