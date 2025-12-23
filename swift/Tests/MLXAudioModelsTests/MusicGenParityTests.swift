// MusicGenParityTests.swift
// Parity tests comparing Swift MusicGen to Python implementation.

import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXAudioModels

/// Tests that verify Swift MusicGen matches Python implementation outputs.
/// Requires fixture files generated from Python.
final class MusicGenParityTests: XCTestCase {

    // MARK: - Test Configuration

    /// Path to fixtures directory (set via environment variable)
    static var fixturesPath: String? {
        ProcessInfo.processInfo.environment["MUSICGEN_FIXTURES_PATH"]
    }

    /// Skip tests if fixtures not available
    func skipIfNoFixtures(file: StaticString = #file, line: UInt = #line) throws {
        guard Self.fixturesPath != nil else {
            throw XCTSkip("MUSICGEN_FIXTURES_PATH not set - skipping parity tests")
        }
    }

    // MARK: - Delay Pattern Parity

    func testDelayPatternApplyParity() throws {
        try skipIfNoFixtures()
        let path = Self.fixturesPath!

        // Load Python fixture
        let inputPath = (path as NSString).appendingPathComponent("delay_pattern_input.npy")
        let expectedPath = (path as NSString).appendingPathComponent("delay_pattern_output.npy")

        guard FileManager.default.fileExists(atPath: inputPath),
            FileManager.default.fileExists(atPath: expectedPath)
        else {
            throw XCTSkip("Delay pattern fixtures not found")
        }

        let input = try MLX.loadArray(url: URL(fileURLWithPath: inputPath))
        let expected = try MLX.loadArray(url: URL(fileURLWithPath: expectedPath))

        // Run Swift implementation
        let scheduler = DelayPatternScheduler(numCodebooks: 4, padTokenId: 2048)
        let output = scheduler.applyDelayPattern(input)

        // Compare
        XCTAssertEqual(output.shape, expected.shape, "Shape mismatch")
        XCTAssertTrue(
            allClose(output.asType(.float32), expected.asType(.float32), atol: 1e-5).item(),
            "Delay pattern output mismatch"
        )
    }

    func testDelayPatternRevertParity() throws {
        try skipIfNoFixtures()
        let path = Self.fixturesPath!

        let inputPath = (path as NSString).appendingPathComponent("delay_pattern_delayed.npy")
        let expectedPath = (path as NSString).appendingPathComponent("delay_pattern_reverted.npy")

        guard FileManager.default.fileExists(atPath: inputPath),
            FileManager.default.fileExists(atPath: expectedPath)
        else {
            throw XCTSkip("Delay pattern revert fixtures not found")
        }

        let input = try MLX.loadArray(url: URL(fileURLWithPath: inputPath))
        let expected = try MLX.loadArray(url: URL(fileURLWithPath: expectedPath))

        let scheduler = DelayPatternScheduler(numCodebooks: 4, padTokenId: 2048)
        let output = scheduler.revertDelayPattern(input)

        XCTAssertEqual(output.shape, expected.shape, "Shape mismatch")
        XCTAssertTrue(
            allClose(output.asType(.float32), expected.asType(.float32), atol: 1e-5).item(),
            "Delay pattern revert mismatch"
        )
    }

    // MARK: - Embedding Parity

    func testSinusoidalEmbeddingParity() throws {
        try skipIfNoFixtures()
        let path = Self.fixturesPath!

        let expectedPath = (path as NSString).appendingPathComponent("sinusoidal_embedding.npy")

        guard FileManager.default.fileExists(atPath: expectedPath) else {
            throw XCTSkip("Sinusoidal embedding fixture not found")
        }

        let expected = try MLX.loadArray(url: URL(fileURLWithPath: expectedPath))

        // Create embedding with same parameters as Python
        let embedding = SinusoidalPositionalEmbedding(embeddingDim: 1024, maxLength: 8192)

        // Compare first 100 positions
        let positions = MLXArray(0 ..< 100)
        let output = embedding(positions)

        XCTAssertTrue(
            allClose(output, expected[0 ..< 100], atol: 1e-4).item(),
            "Sinusoidal embedding mismatch"
        )
    }

    func testCodebookEmbeddingsParity() throws {
        try skipIfNoFixtures()
        let path = Self.fixturesPath!

        let inputPath = (path as NSString).appendingPathComponent("codebook_input.npy")
        let expectedPath = (path as NSString).appendingPathComponent("codebook_output.npy")
        let weightsPath = (path as NSString).appendingPathComponent("codebook_weights.safetensors")

        guard FileManager.default.fileExists(atPath: inputPath),
            FileManager.default.fileExists(atPath: expectedPath),
            FileManager.default.fileExists(atPath: weightsPath)
        else {
            throw XCTSkip("Codebook embedding fixtures not found")
        }

        let input = try MLX.loadArray(url: URL(fileURLWithPath: inputPath))
        let expected = try MLX.loadArray(url: URL(fileURLWithPath: expectedPath))

        // Create embeddings and load weights
        let config = MusicGenConfig.small()
        let embeddings = CodebookEmbeddings(config: config)

        let weights = try loadArrays(url: URL(fileURLWithPath: weightsPath))
        let params = ModuleParameters.unflattened(weights)
        try embeddings.update(parameters: params, verify: .noUnusedKeys)

        let output = embeddings(input)

        XCTAssertEqual(output.shape, expected.shape, "Shape mismatch")
        XCTAssertTrue(
            allClose(output, expected, atol: 1e-4).item(),
            "Codebook embedding output mismatch"
        )
    }

    // MARK: - Attention Parity

    func testSelfAttentionParity() throws {
        try skipIfNoFixtures()
        let path = Self.fixturesPath!

        let inputPath = (path as NSString).appendingPathComponent("attention_input.npy")
        let expectedPath = (path as NSString).appendingPathComponent("attention_output.npy")
        let weightsPath = (path as NSString).appendingPathComponent("attention_weights.safetensors")

        guard FileManager.default.fileExists(atPath: inputPath),
            FileManager.default.fileExists(atPath: expectedPath),
            FileManager.default.fileExists(atPath: weightsPath)
        else {
            throw XCTSkip("Attention fixtures not found")
        }

        let input = try MLX.loadArray(url: URL(fileURLWithPath: inputPath))
        let expected = try MLX.loadArray(url: URL(fileURLWithPath: expectedPath))

        let config = MusicGenConfig.small()
        let attention = MusicGenAttention(config: config)

        let weights = try loadArrays(url: URL(fileURLWithPath: weightsPath))
        let params = ModuleParameters.unflattened(weights)
        try attention.update(parameters: params, verify: .noUnusedKeys)

        let (output, _) = attention(input)

        XCTAssertEqual(output.shape, expected.shape, "Shape mismatch")
        XCTAssertTrue(
            allClose(output, expected, atol: 1e-4).item(),
            "Attention output mismatch"
        )
    }

    // MARK: - Decoder Parity

    func testDecoderBlockParity() throws {
        try skipIfNoFixtures()
        let path = Self.fixturesPath!

        let inputPath = (path as NSString).appendingPathComponent("decoder_block_input.npy")
        let encoderPath = (path as NSString).appendingPathComponent("decoder_block_encoder.npy")
        let expectedPath = (path as NSString).appendingPathComponent("decoder_block_output.npy")
        let weightsPath = (path as NSString).appendingPathComponent(
            "decoder_block_weights.safetensors")

        guard FileManager.default.fileExists(atPath: inputPath),
            FileManager.default.fileExists(atPath: encoderPath),
            FileManager.default.fileExists(atPath: expectedPath),
            FileManager.default.fileExists(atPath: weightsPath)
        else {
            throw XCTSkip("Decoder block fixtures not found")
        }

        let input = try MLX.loadArray(url: URL(fileURLWithPath: inputPath))
        let encoder = try MLX.loadArray(url: URL(fileURLWithPath: encoderPath))
        let expected = try MLX.loadArray(url: URL(fileURLWithPath: expectedPath))

        let config = MusicGenConfig.small()
        let block = MusicGenDecoderBlock(config: config)

        let weights = try loadArrays(url: URL(fileURLWithPath: weightsPath))
        let params = ModuleParameters.unflattened(weights)
        try block.update(parameters: params, verify: .noUnusedKeys)

        let (output, _, _) = block(input, encoderHiddenStates: encoder)

        XCTAssertEqual(output.shape, expected.shape, "Shape mismatch")
        XCTAssertTrue(
            allClose(output, expected, atol: 1e-3).item(),
            "Decoder block output mismatch"
        )
    }

    // MARK: - LM Head Parity

    func testLMHeadParity() throws {
        try skipIfNoFixtures()
        let path = Self.fixturesPath!

        let inputPath = (path as NSString).appendingPathComponent("lm_head_input.npy")
        let expectedPath = (path as NSString).appendingPathComponent("lm_head_output.npy")
        let weightsPath = (path as NSString).appendingPathComponent("lm_head_weights.safetensors")

        guard FileManager.default.fileExists(atPath: inputPath),
            FileManager.default.fileExists(atPath: expectedPath),
            FileManager.default.fileExists(atPath: weightsPath)
        else {
            throw XCTSkip("LM head fixtures not found")
        }

        let input = try MLX.loadArray(url: URL(fileURLWithPath: inputPath))
        let expected = try MLX.loadArray(url: URL(fileURLWithPath: expectedPath))

        let config = MusicGenConfig.small()
        let lmHead = MusicGenLMHead(config: config)

        let weights = try loadArrays(url: URL(fileURLWithPath: weightsPath))
        let params = ModuleParameters.unflattened(weights)
        try lmHead.update(parameters: params, verify: .noUnusedKeys)

        let output = lmHead(input)

        XCTAssertEqual(output.shape, expected.shape, "Shape mismatch")
        XCTAssertTrue(
            allClose(output, expected, atol: 1e-4).item(),
            "LM head output mismatch"
        )
    }

    // MARK: - T5 Encoder Parity

    func testT5EncoderParity() throws {
        try skipIfNoFixtures()
        let path = Self.fixturesPath!

        let inputPath = (path as NSString).appendingPathComponent("t5_input.npy")
        let expectedPath = (path as NSString).appendingPathComponent("t5_output.npy")
        let weightsPath = (path as NSString).appendingPathComponent("t5_weights.safetensors")

        guard FileManager.default.fileExists(atPath: inputPath),
            FileManager.default.fileExists(atPath: expectedPath),
            FileManager.default.fileExists(atPath: weightsPath)
        else {
            throw XCTSkip("T5 encoder fixtures not found")
        }

        let input = try MLX.loadArray(url: URL(fileURLWithPath: inputPath))
        let expected = try MLX.loadArray(url: URL(fileURLWithPath: expectedPath))

        let config = T5Config.base()
        let encoder = T5Encoder(config: config)

        let weights = try loadArrays(url: URL(fileURLWithPath: weightsPath))
        let params = ModuleParameters.unflattened(weights)
        try encoder.update(parameters: params, verify: .noUnusedKeys)

        let output = encoder(inputIds: input)

        XCTAssertEqual(output.shape, expected.shape, "Shape mismatch")
        XCTAssertTrue(
            allClose(output, expected, atol: 1e-3).item(),
            "T5 encoder output mismatch"
        )
    }

    // MARK: - Full Model Parity

    func testMusicGenForwardParity() throws {
        try skipIfNoFixtures()
        let path = Self.fixturesPath!

        let inputIdsPath = (path as NSString).appendingPathComponent("musicgen_input_ids.npy")
        let encoderPath = (path as NSString).appendingPathComponent("musicgen_encoder_hidden.npy")
        let expectedPath = (path as NSString).appendingPathComponent("musicgen_logits.npy")
        let weightsPath = (path as NSString).appendingPathComponent("musicgen_weights.safetensors")

        guard FileManager.default.fileExists(atPath: inputIdsPath),
            FileManager.default.fileExists(atPath: encoderPath),
            FileManager.default.fileExists(atPath: expectedPath),
            FileManager.default.fileExists(atPath: weightsPath)
        else {
            throw XCTSkip("MusicGen forward fixtures not found")
        }

        let inputIds = try MLX.loadArray(url: URL(fileURLWithPath: inputIdsPath))
        let encoderHidden = try MLX.loadArray(url: URL(fileURLWithPath: encoderPath))
        let expected = try MLX.loadArray(url: URL(fileURLWithPath: expectedPath))

        let config = MusicGenConfig.small()
        let model = MusicGen(config: config)

        let weights = try loadArrays(url: URL(fileURLWithPath: weightsPath))
        // Map weight keys
        var mappedWeights: [String: MLXArray] = [:]
        for (key, value) in weights {
            var newKey = key
            newKey = newKey.replacingOccurrences(of: "self_attn", with: "selfAttn")
            newKey = newKey.replacingOccurrences(of: "encoder_attn", with: "crossAttn")
            newKey = newKey.replacingOccurrences(
                of: "self_attn_layer_norm", with: "selfAttnLayerNorm")
            newKey = newKey.replacingOccurrences(
                of: "encoder_attn_layer_norm", with: "crossAttnLayerNorm")
            newKey = newKey.replacingOccurrences(of: "final_layer_norm", with: "finalLayerNorm")
            newKey = newKey.replacingOccurrences(of: "layer_norm", with: "layerNorm")
            newKey = newKey.replacingOccurrences(of: "text_projection", with: "textProjection")
            newKey = newKey.replacingOccurrences(of: "lm_head", with: "lmHead")
            newKey = newKey.replacingOccurrences(of: "q_proj", with: "qProj")
            newKey = newKey.replacingOccurrences(of: "k_proj", with: "kProj")
            newKey = newKey.replacingOccurrences(of: "v_proj", with: "vProj")
            newKey = newKey.replacingOccurrences(of: "out_proj", with: "outProj")
            newKey = newKey.replacingOccurrences(of: "position_embedding", with: "positionEmbedding")
            mappedWeights[newKey] = value
        }

        let params = ModuleParameters.unflattened(mappedWeights)
        try model.update(parameters: params, verify: .noUnusedKeys)

        let (logits, _) = model(
            inputIds: inputIds,
            encoderHiddenStates: encoderHidden
        )

        XCTAssertEqual(logits.shape, expected.shape, "Shape mismatch")
        XCTAssertTrue(
            allClose(logits, expected, atol: 1e-3).item(),
            "MusicGen forward output mismatch"
        )
    }
}

// MARK: - Fixture Generation Script

/*
 To generate fixtures for parity tests, run this Python script:

 ```python
 import mlx.core as mx
 import numpy as np
 from pathlib import Path
 from mlx_audio.models.musicgen import MusicGen, MusicGenConfig

 fixtures_path = Path("fixtures/musicgen")
 fixtures_path.mkdir(parents=True, exist_ok=True)

 # Delay pattern fixtures
 config = MusicGenConfig.small()
 codes = mx.random.randint(0, 2048, (1, 4, 10))
 np.save(fixtures_path / "delay_pattern_input.npy", np.array(codes))

 # ... generate other fixtures ...

 # Save model weights
 model = MusicGen(config)
 mx.save_safetensors(str(fixtures_path / "musicgen_weights.safetensors"), dict(model.parameters()))
 ```
 */
