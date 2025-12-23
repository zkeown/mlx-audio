// MusicGenTests.swift
// Unit tests for MusicGen Swift implementation.

import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXAudioModels

final class MusicGenTests: XCTestCase {

    // MARK: - Configuration Tests

    func testConfigPresets() {
        // Test small preset
        let small = MusicGenConfig.small()
        XCTAssertEqual(small.hiddenSize, 1024)
        XCTAssertEqual(small.numHiddenLayers, 24)
        XCTAssertEqual(small.numAttentionHeads, 16)
        XCTAssertEqual(small.intermediateSize, 4096)

        // Test medium preset
        let medium = MusicGenConfig.medium()
        XCTAssertEqual(medium.hiddenSize, 1536)
        XCTAssertEqual(medium.numHiddenLayers, 48)
        XCTAssertEqual(medium.numAttentionHeads, 24)
        XCTAssertEqual(medium.intermediateSize, 6144)

        // Test large preset
        let large = MusicGenConfig.large()
        XCTAssertEqual(large.hiddenSize, 2048)
        XCTAssertEqual(large.numHiddenLayers, 48)
        XCTAssertEqual(large.numAttentionHeads, 32)
        XCTAssertEqual(large.intermediateSize, 8192)

        // Test melody preset (same as medium)
        let melody = MusicGenConfig.melody()
        XCTAssertEqual(melody.hiddenSize, medium.hiddenSize)
        XCTAssertEqual(melody.numHiddenLayers, medium.numHiddenLayers)
    }

    func testConfigComputedProperties() {
        let config = MusicGenConfig.small()

        // Test vocabSize (codebookSize + 1)
        XCTAssertEqual(config.vocabSize, config.codebookSize + 1)

        // Test headDim
        XCTAssertEqual(config.headDim, config.hiddenSize / config.numAttentionHeads)

        // Test maxNewTokens
        let expectedTokens = Int(config.maxDuration * Float(config.frameRate))
        XCTAssertEqual(config.maxNewTokens, expectedTokens)
    }

    func testConfigFromName() throws {
        let small = try MusicGenConfig.fromName("small")
        XCTAssertEqual(small.hiddenSize, 1024)

        let medium = try MusicGenConfig.fromName("musicgen-medium")
        XCTAssertEqual(medium.hiddenSize, 1536)

        XCTAssertThrowsError(try MusicGenConfig.fromName("unknown")) { error in
            guard case MusicGenError.unknownModel = error else {
                XCTFail("Expected unknownModel error")
                return
            }
        }
    }

    // MARK: - Delay Pattern Tests

    func testDelayPatternApplyRevert() {
        let scheduler = DelayPatternScheduler(numCodebooks: 4, padTokenId: 2048)

        // Create test codes: [B=1, K=4, T=4]
        let codes = MLXArray.zeros([1, 4, 4]).asType(.int32)

        // Apply delay pattern
        let delayed = scheduler.applyDelayPattern(codes)

        // Check shape: T_delayed = T + K - 1 = 4 + 4 - 1 = 7
        XCTAssertEqual(delayed.dim(2), 7)

        // Revert should recover original
        let reverted = scheduler.revertDelayPattern(delayed)
        XCTAssertEqual(reverted.shape, codes.shape)
    }

    func testDelayPatternValidCodebooks() {
        let scheduler = DelayPatternScheduler(numCodebooks: 4, padTokenId: 2048)

        // At step 0, only codebook 0 is valid
        XCTAssertEqual(scheduler.getValidCodebooks(step: 0), [0])

        // At step 1, codebooks 0 and 1 are valid
        XCTAssertEqual(scheduler.getValidCodebooks(step: 1), [0, 1])

        // At step 3+, all codebooks are valid
        XCTAssertEqual(scheduler.getValidCodebooks(step: 3), [0, 1, 2, 3])
        XCTAssertEqual(scheduler.getValidCodebooks(step: 10), [0, 1, 2, 3])
    }

    // MARK: - Embedding Tests

    func testSinusoidalPositionalEmbedding() {
        let embedding = SinusoidalPositionalEmbedding(embeddingDim: 64, maxLength: 100)

        // Test shape
        let positions = MLXArray([0, 1, 2, 3, 4])
        let output = embedding(positions)
        XCTAssertEqual(output.shape, [5, 64])

        // Embeddings at different positions should be different
        let pos0 = output[0]
        let pos1 = output[1]
        XCTAssertFalse(allClose(pos0, pos1, atol: 1e-5).item())
    }

    func testCodebookEmbeddings() {
        let config = MusicGenConfig(
            numCodebooks: 4,
            codebookSize: 128,
            hiddenSize: 64,
            numHiddenLayers: 2,
            numAttentionHeads: 4
        )

        let embeddings = CodebookEmbeddings(config: config)

        // Test forward pass shape
        let inputIds = MLXArray.zeros([2, 4, 10]).asType(.int32)  // [B, K, T]
        let output = embeddings(inputIds)

        // Output should be [B, T, D]
        XCTAssertEqual(output.shape, [2, 10, 64])
    }

    // MARK: - Attention Tests

    func testMusicGenAttention() {
        let config = MusicGenConfig(
            hiddenSize: 64,
            numHiddenLayers: 2,
            numAttentionHeads: 4
        )

        let attention = MusicGenAttention(config: config, isCrossAttention: false)

        // Test self-attention
        let hidden = MLXArray.zeros([2, 10, 64])
        let (output, cache) = attention(hidden)

        XCTAssertEqual(output.shape, [2, 10, 64])
        XCTAssertNotNil(cache)

        // Test with cache
        let nextHidden = MLXArray.zeros([2, 1, 64])
        let (nextOutput, _) = attention(nextHidden, kvCache: cache)
        XCTAssertEqual(nextOutput.shape, [2, 1, 64])
    }

    func testCausalMask() {
        let mask = createCausalMask(queryLength: 4, keyLength: 4)

        // Shape should be [1, 1, 4, 4]
        XCTAssertEqual(mask.shape, [1, 1, 4, 4])

        // Upper triangle should be -inf (masked)
        // Lower triangle and diagonal should be 0 (not masked)
        let maskValues = mask.squeezed().asArray(Float.self)

        // Check diagonal is not masked
        for i in 0 ..< 4 {
            XCTAssertEqual(maskValues[i * 4 + i], 0.0)
        }
    }

    // MARK: - Decoder Tests

    func testMusicGenDecoderBlock() {
        let config = MusicGenConfig(
            hiddenSize: 64,
            numHiddenLayers: 2,
            numAttentionHeads: 4,
            intermediateSize: 128
        )

        let block = MusicGenDecoderBlock(config: config)

        let hidden = MLXArray.zeros([2, 10, 64])
        let encoder = MLXArray.zeros([2, 5, 64])

        let (output, selfCache, crossCache) = block(
            hidden,
            encoderHiddenStates: encoder
        )

        XCTAssertEqual(output.shape, [2, 10, 64])
        XCTAssertNotNil(selfCache)
        XCTAssertNotNil(crossCache)
    }

    func testMusicGenDecoder() {
        let config = MusicGenConfig(
            hiddenSize: 64,
            numHiddenLayers: 2,
            numAttentionHeads: 4,
            intermediateSize: 128
        )

        let decoder = MusicGenDecoder(config: config)

        let hidden = MLXArray.zeros([2, 10, 64])
        let encoder = MLXArray.zeros([2, 5, 64])

        let (output, cache) = decoder(
            hidden,
            encoderHiddenStates: encoder
        )

        XCTAssertEqual(output.shape, [2, 10, 64])
        XCTAssertNotNil(cache)
        XCTAssertEqual(cache?.selfAttnCache.count, 2)  // 2 layers
    }

    // MARK: - LM Head Tests

    func testMusicGenLMHead() {
        let config = MusicGenConfig(
            numCodebooks: 4,
            codebookSize: 128,
            hiddenSize: 64,
            numHiddenLayers: 2,
            numAttentionHeads: 4
        )

        let lmHead = MusicGenLMHead(config: config)

        let hidden = MLXArray.zeros([2, 10, 64])
        let logits = lmHead(hidden)

        // Output shape: [B, K, T, V]
        XCTAssertEqual(logits.shape, [2, 4, 10, 129])  // vocabSize = codebookSize + 1
    }

    // MARK: - Sampling Tests

    func testSampleNextToken() {
        // Test greedy decoding (temperature = 0)
        let logits = MLXArray([0.1, 0.2, 0.9, 0.1]).expandedDimensions(axis: 0)  // [1, 4]
        let greedy = sampleNextToken(logits: logits, temperature: 0)
        XCTAssertEqual(greedy[0].item(Int.self), 2)  // Index of max value

        // Test that sampling works (non-deterministic, just check shape)
        let sampled = sampleNextToken(logits: logits, temperature: 1.0, topK: 2)
        XCTAssertEqual(sampled.shape, [1])
    }

    func testClassifierFreeGuidance() {
        let cond = MLXArray([1.0, 2.0, 3.0])
        let uncond = MLXArray([0.5, 1.0, 1.5])

        // Scale = 1.0 should return conditional
        let noGuidance = applyClassifierFreeGuidance(
            conditionalLogits: cond,
            unconditionalLogits: uncond,
            scale: 1.0
        )
        XCTAssertTrue(allClose(noGuidance, cond, atol: 1e-5).item())

        // Scale = 2.0: uncond + 2 * (cond - uncond) = uncond + 2*cond - 2*uncond = 2*cond - uncond
        let guided = applyClassifierFreeGuidance(
            conditionalLogits: cond,
            unconditionalLogits: uncond,
            scale: 2.0
        )
        let expected = 2 * cond - uncond
        XCTAssertTrue(allClose(guided, expected, atol: 1e-5).item())
    }

    // MARK: - T5 Config Tests

    func testT5ConfigPresets() {
        let base = T5Config.base()
        XCTAssertEqual(base.hiddenSize, 768)
        XCTAssertEqual(base.numHiddenLayers, 12)
        XCTAssertEqual(base.numAttentionHeads, 12)

        let small = T5Config.small()
        XCTAssertEqual(small.hiddenSize, 512)
        XCTAssertEqual(small.numHiddenLayers, 6)

        let large = T5Config.large()
        XCTAssertEqual(large.hiddenSize, 1024)
        XCTAssertEqual(large.numHiddenLayers, 24)
    }

    // MARK: - T5 Encoder Tests

    func testT5RMSNorm() {
        let norm = T5RMSNorm(dimensions: 64)
        let input = MLXArray.ones([2, 10, 64])
        let output = norm(input)

        XCTAssertEqual(output.shape, [2, 10, 64])
    }

    func testT5Attention() {
        let config = T5Config(
            hiddenSize: 64,
            numHiddenLayers: 2,
            numAttentionHeads: 4
        )

        let attention = T5Attention(config: config)

        let hidden = MLXArray.zeros([2, 10, 64])
        let output = attention(hidden)

        XCTAssertEqual(output.shape, [2, 10, 64])
    }

    func testT5EncoderBlock() {
        let config = T5Config(
            hiddenSize: 64,
            numHiddenLayers: 2,
            numAttentionHeads: 4,
            intermediateSize: 128
        )

        let block = T5EncoderBlock(config: config)

        let hidden = MLXArray.zeros([2, 10, 64])
        let output = block(hidden)

        XCTAssertEqual(output.shape, [2, 10, 64])
    }

    func testT5Encoder() {
        let config = T5Config(
            vocabSize: 100,
            hiddenSize: 64,
            numHiddenLayers: 2,
            numAttentionHeads: 4,
            intermediateSize: 128
        )

        let encoder = T5Encoder(config: config)

        let inputIds = MLXArray.zeros([2, 10]).asType(.int32)
        let output = encoder(inputIds: inputIds)

        XCTAssertEqual(output.shape, [2, 10, 64])
    }

    // MARK: - Integration Tests

    func testMusicGenForwardPass() {
        let config = MusicGenConfig(
            numCodebooks: 4,
            codebookSize: 128,
            textHiddenSize: 64,
            hiddenSize: 64,
            numHiddenLayers: 2,
            numAttentionHeads: 4,
            intermediateSize: 128
        )

        let model = MusicGen(config: config)

        // Prepare inputs
        let inputIds = MLXArray.zeros([1, 4, 5]).asType(.int32)  // [B, K, T]
        let encoderHidden = MLXArray.zeros([1, 10, 64])  // [B, S, D_text]

        // Forward pass
        let (logits, cache) = model(
            inputIds: inputIds,
            encoderHiddenStates: encoderHidden
        )

        // Check output shapes
        XCTAssertEqual(logits.shape, [1, 4, 5, 129])  // [B, K, T, V]
        XCTAssertNotNil(cache)
    }

    func testMusicGenGenerateFromEmbeddings() {
        let config = MusicGenConfig(
            numCodebooks: 4,
            codebookSize: 128,
            textHiddenSize: 64,
            hiddenSize: 64,
            numHiddenLayers: 2,
            numAttentionHeads: 4,
            intermediateSize: 128
        )

        let model = MusicGen(config: config)

        // Prepare encoder hidden states
        let encoderHidden = MLXArray.zeros([1, 10, 64])

        // Generate a short sequence
        let result = model.generate(
            encoderHiddenStates: encoderHidden,
            maxNewTokens: 5
        )

        // Check output shape: [B, K, T]
        XCTAssertEqual(result.codes.dim(0), 1)
        XCTAssertEqual(result.codes.dim(1), 4)
        XCTAssertGreaterThan(result.codes.dim(2), 0)
    }
}
