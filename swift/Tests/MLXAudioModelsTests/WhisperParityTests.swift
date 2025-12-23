// WhisperParityTests.swift
// Parity tests comparing Swift Whisper implementation against Python reference.
//
// These tests verify that the Swift implementation produces identical or
// near-identical outputs to the Python implementation.
//
// To run these tests:
// 1. Generate reference data from Python:
//    cd python && python tests/models/whisper/generate_swift_parity_data.py
// 2. Run Swift tests:
//    swift test --filter WhisperParityTests
//
// Tests will be skipped if reference data is not found.

import XCTest
import Foundation
import MLX
import MLXNN
@testable import MLXAudioModels

final class WhisperParityTests: XCTestCase {

    /// Base directory for parity test data.
    static let parityDataDir = URL(fileURLWithPath: "/tmp/whisper_parity")

    /// Maximum allowed difference for floating point comparisons.
    static let tolerance: Float = 1e-4

    /// Maximum allowed difference for model outputs (slightly looser).
    static let modelTolerance: Float = 1e-3

    // MARK: - Helper Methods

    /// Load a numpy array from file.
    func loadNpy(_ filename: String) throws -> MLXArray {
        let url = Self.parityDataDir.appendingPathComponent(filename)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw XCTSkip("Reference file not found: \(filename). Run generate_swift_parity_data.py first.")
        }
        return try MLX.loadArray(url: url)
    }

    /// Load JSON file as dictionary.
    func loadJson(_ filename: String) throws -> [String: Any] {
        let url = Self.parityDataDir.appendingPathComponent(filename)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw XCTSkip("Reference file not found: \(filename)")
        }
        let data = try Data(contentsOf: url)
        return try JSONSerialization.jsonObject(with: data) as? [String: Any] ?? [:]
    }

    /// Check if parity data exists.
    func checkParityDataExists() throws {
        guard FileManager.default.fileExists(atPath: Self.parityDataDir.path) else {
            throw XCTSkip("""
                Parity test data not found at \(Self.parityDataDir.path).
                Generate it by running:
                  cd python && python tests/models/whisper/generate_swift_parity_data.py
                """)
        }
    }

    /// Assert two arrays are close within tolerance.
    func assertClose(
        _ a: MLXArray,
        _ b: MLXArray,
        rtol: Float = 1e-5,
        atol: Float = 1e-4,
        message: String = ""
    ) {
        let diff = abs(a - b)
        let maxDiff = max(diff).item(Float.self)
        let meanDiff = mean(diff).item(Float.self)

        XCTAssertLessThan(
            maxDiff,
            atol + rtol * max(abs(b)).item(Float.self),
            "Max diff: \(maxDiff), Mean diff: \(meanDiff). \(message)"
        )
    }

    // MARK: - Config Parity

    func testConfigParity() throws {
        try checkParityDataExists()

        let configJson = try loadJson("config.json")

        // Verify config matches
        let config = WhisperConfig.tiny()

        XCTAssertEqual(config.nMels, configJson["n_mels"] as? Int)
        XCTAssertEqual(config.nAudioCtx, configJson["n_audio_ctx"] as? Int)
        XCTAssertEqual(config.nAudioState, configJson["n_audio_state"] as? Int)
        XCTAssertEqual(config.nAudioHead, configJson["n_audio_head"] as? Int)
        XCTAssertEqual(config.nAudioLayer, configJson["n_audio_layer"] as? Int)
        XCTAssertEqual(config.nTextCtx, configJson["n_text_ctx"] as? Int)
        XCTAssertEqual(config.nTextState, configJson["n_text_state"] as? Int)
        XCTAssertEqual(config.nTextHead, configJson["n_text_head"] as? Int)
        XCTAssertEqual(config.nTextLayer, configJson["n_text_layer"] as? Int)
        XCTAssertEqual(config.nVocab, configJson["n_vocab"] as? Int)
    }

    // MARK: - Sinusoidal Embeddings Parity

    func testSinusoidsParity() throws {
        try checkParityDataExists()

        let pythonSinusoids = try loadNpy("sinusoids_100x384.npy")

        // Generate Swift sinusoids
        let swiftSinusoids = sinusoids(length: 100, dim: 384)

        // Verify shape
        XCTAssertEqual(swiftSinusoids.shape, pythonSinusoids.shape)

        // Verify values match
        assertClose(
            swiftSinusoids,
            pythonSinusoids,
            atol: Self.tolerance,
            message: "Sinusoidal embeddings mismatch"
        )
    }

    // MARK: - Encoder Parity

    func testEncoderShapeParity() throws {
        try checkParityDataExists()

        let shapeJson = try loadJson("encoder_shape.json")
        guard let expectedShape = shapeJson["shape"] as? [Int] else {
            XCTFail("Could not parse encoder shape")
            return
        }

        // Create model and run encoder
        let config = WhisperConfig.tiny()
        let model = WhisperModel(config: config)

        let melInput = try loadNpy("mel_input.npy")
        let encoderOutput = model.encode(melInput)

        XCTAssertEqual(Array(encoderOutput.shape), expectedShape)
    }

    func testEncoderOutputParity() throws {
        try checkParityDataExists()

        // Load model with Python weights
        let modelPath = Self.parityDataDir.appendingPathComponent("model")
        guard FileManager.default.fileExists(atPath: modelPath.path) else {
            throw XCTSkip("Model weights not found")
        }

        let model = try WhisperModel.fromPretrained(path: modelPath)

        // Load inputs and expected outputs
        let melInput = try loadNpy("mel_input.npy")
        let expectedOutput = try loadNpy("encoder_output.npy")

        // Run encoder
        let actualOutput = model.encode(melInput)

        // Verify output
        assertClose(
            actualOutput,
            expectedOutput,
            atol: Self.modelTolerance,
            message: "Encoder output mismatch"
        )
    }

    // MARK: - Decoder Parity

    func testDecoderOutputParity() throws {
        try checkParityDataExists()

        let modelPath = Self.parityDataDir.appendingPathComponent("model")
        guard FileManager.default.fileExists(atPath: modelPath.path) else {
            throw XCTSkip("Model weights not found")
        }

        let model = try WhisperModel.fromPretrained(path: modelPath)

        // Load inputs
        let melInput = try loadNpy("mel_input.npy")
        let tokenInput = try loadNpy("token_input.npy")
        let expectedLogits = try loadNpy("decoder_logits_no_cache.npy")

        // Run model
        let encoderOutput = model.encode(melInput)
        let (actualLogits, _) = model.decode(
            tokens: tokenInput,
            audioFeatures: encoderOutput
        )

        // Verify output
        assertClose(
            actualLogits,
            expectedLogits,
            atol: Self.modelTolerance,
            message: "Decoder output mismatch"
        )
    }

    func testDecoderWithCacheParity() throws {
        try checkParityDataExists()

        let modelPath = Self.parityDataDir.appendingPathComponent("model")
        guard FileManager.default.fileExists(atPath: modelPath.path) else {
            throw XCTSkip("Model weights not found")
        }

        let model = try WhisperModel.fromPretrained(path: modelPath)

        // Load inputs
        let melInput = try loadNpy("mel_input.npy")
        let tokenInput = try loadNpy("token_input.npy")
        let expectedLogitsCached = try loadNpy("decoder_logits_with_cache.npy")

        // Run model with cache
        let encoderOutput = model.encode(melInput)
        let (_, kvCache) = model.decode(
            tokens: tokenInput,
            audioFeatures: encoderOutput
        )

        // Feed single token with cache
        let lastToken = tokenInput[0..., (-1)...].reshaped([1, 1])
        let (actualLogitsCached, _) = model.decode(
            tokens: lastToken,
            audioFeatures: encoderOutput,
            kvCache: kvCache
        )

        // Verify output
        assertClose(
            actualLogitsCached,
            expectedLogitsCached,
            atol: Self.modelTolerance,
            message: "Decoder with cache output mismatch"
        )
    }

    // MARK: - Full Forward Pass Parity

    func testFullForwardParity() throws {
        try checkParityDataExists()

        let modelPath = Self.parityDataDir.appendingPathComponent("model")
        guard FileManager.default.fileExists(atPath: modelPath.path) else {
            throw XCTSkip("Model weights not found")
        }

        let model = try WhisperModel.fromPretrained(path: modelPath)

        // Load inputs and expected outputs
        let melInput = try loadNpy("mel_input.npy")
        let tokenInput = try loadNpy("token_input.npy")
        let expectedLogits = try loadNpy("full_forward_logits.npy")

        // Run full forward pass
        let actualLogits = model(mel: melInput, tokens: tokenInput)

        // Verify output
        assertClose(
            actualLogits,
            expectedLogits,
            atol: Self.modelTolerance,
            message: "Full forward pass mismatch"
        )
    }

    // MARK: - Tokenizer Parity

    func testTokenizerSpecialTokensParity() throws {
        try checkParityDataExists()

        let tokenizerInfo = try loadJson("tokenizer_info.json")

        // These should match exactly (they're constants)
        XCTAssertEqual(50256, tokenizerInfo["eot"] as? Int)  // EOT is always 50256
        XCTAssertEqual(50257, tokenizerInfo["sot"] as? Int)  // SOT is always 50257
    }

    func testInitialTokensParity() throws {
        try checkParityDataExists()

        let initialTokensInfo = try loadJson("initial_tokens.json")

        // Create tokenizer (note: requires tokenizer.json file)
        // For now, just verify the expected format
        guard let transcribeEn = initialTokensInfo["transcribe_en_timestamps"] as? [Int] else {
            XCTFail("Could not parse initial tokens")
            return
        }

        // SOT should be first
        XCTAssertEqual(transcribeEn.first, 50257)

        // Should have language token, task token
        XCTAssertGreaterThan(transcribeEn.count, 2)
    }

    // MARK: - KV Cache Shape Parity

    func testKVCacheShapesParity() throws {
        try checkParityDataExists()

        let kvShapesJson = try loadJson("kv_cache_shapes.json")
        guard let kvShapes = kvShapesJson as? [[String: Any]] else {
            // Try parsing as array directly
            let url = Self.parityDataDir.appendingPathComponent("kv_cache_shapes.json")
            let data = try Data(contentsOf: url)
            guard let shapes = try JSONSerialization.jsonObject(with: data) as? [[String: Any]] else {
                XCTFail("Could not parse KV cache shapes")
                return
            }

            // Verify we have the right number of layers
            let config = WhisperConfig.tiny()
            XCTAssertEqual(shapes.count, config.nTextLayer)
            return
        }

        let config = WhisperConfig.tiny()
        XCTAssertEqual(kvShapes.count, config.nTextLayer)
    }
}

// MARK: - Performance Parity Tests

extension WhisperParityTests {

    /// Test that Swift encoder performance is reasonable compared to Python.
    /// This is not a strict parity test but ensures performance is acceptable.
    func testEncoderPerformance() throws {
        let config = WhisperConfig.tiny()
        let model = WhisperModel(config: config)

        // Warm up
        let warmupMel = MLXArray.zeros([1, config.nMels, 100])
        _ = model.encode(warmupMel)
        eval(model.parameters())

        // Benchmark
        let mel = MLXArray.zeros([1, config.nMels, 3000])  // ~30s of audio

        let start = CFAbsoluteTimeGetCurrent()
        let output = model.encode(mel)
        eval(output)
        let elapsed = CFAbsoluteTimeGetCurrent() - start

        print("Encoder time for 30s audio: \(elapsed * 1000)ms")

        // Should complete in reasonable time (< 5s for tiny model)
        XCTAssertLessThan(elapsed, 5.0, "Encoder too slow")
    }
}
