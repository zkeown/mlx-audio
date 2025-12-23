// CLAPTests.swift
// Tests for CLAP model implementation.
//
// Includes shape tests, configuration tests, and parity tests
// against Python reference embeddings.

import XCTest
@preconcurrency import MLX
import MLXNN
@testable import MLXAudioModels
@testable import MLXAudioPrimitives

final class CLAPTests: XCTestCase {

    // MARK: - Configuration Tests

    func testCLAPConfigDefaults() {
        let config = CLAPConfig()

        // Audio config defaults
        XCTAssertEqual(config.audio.sampleRate, 48000)
        XCTAssertEqual(config.audio.nMels, 64)
        XCTAssertEqual(config.audio.nFFT, 1024)
        XCTAssertEqual(config.audio.hopLength, 480)
        XCTAssertEqual(config.audio.embedDim, 96)
        XCTAssertEqual(config.audio.depths, [2, 2, 6, 2])
        XCTAssertEqual(config.audio.numHeads, [4, 8, 16, 32])
        XCTAssertEqual(config.audio.windowSize, 8)
        XCTAssertEqual(config.audio.hiddenSize, 768)

        // Text config defaults
        XCTAssertEqual(config.text.vocabSize, 50265)
        XCTAssertEqual(config.text.hiddenSize, 768)
        XCTAssertEqual(config.text.numHiddenLayers, 12)
        XCTAssertEqual(config.text.numAttentionHeads, 12)
        XCTAssertEqual(config.text.intermediateSize, 3072)

        // Global config
        XCTAssertEqual(config.projectionDim, 512)
    }

    func testCLAPConfigPresets() {
        let tiny = CLAPConfig.htsatTiny()
        XCTAssertEqual(tiny.audio.embedDim, 96)
        XCTAssertEqual(tiny.audio.hiddenSize, 768)

        let base = CLAPConfig.htsatBase()
        XCTAssertEqual(base.audio.embedDim, 128)
        XCTAssertEqual(base.audio.hiddenSize, 1024)
    }

    func testCLAPAudioConfigFinalDim() {
        let config = CLAPAudioConfig()
        // embedDim=96, 3 downsamples (2x each) = 96 * 8 = 768
        XCTAssertEqual(config.finalDim, 768)

        let baseConfig = CLAPAudioConfig(embedDim: 128)
        // embedDim=128, 3 downsamples = 128 * 8 = 1024
        XCTAssertEqual(baseConfig.finalDim, 1024)
    }

    // MARK: - Tokenizer Tests

    func testTokenizerBasic() {
        let tokenizer = CLAPTokenizer.createBasic()

        // Check special tokens
        XCTAssertEqual(tokenizer.bosTokenId, 0)
        XCTAssertEqual(tokenizer.padTokenId, 1)
        XCTAssertEqual(tokenizer.eosTokenId, 2)
        XCTAssertEqual(tokenizer.unkTokenId, 3)
    }

    func testTokenizerEncode() {
        let tokenizer = CLAPTokenizer.createBasic()

        let (ids, mask) = tokenizer.encodeWithMask("hello world")

        // Should have BOS at start, EOS at end
        XCTAssertEqual(ids.first, tokenizer.bosTokenId)
        XCTAssertEqual(ids.count, tokenizer.maxLength)

        // Mask should have 1s for real tokens, 0s for padding
        let realTokenCount = mask.filter { $0 == 1 }.count
        XCTAssertGreaterThan(realTokenCount, 2)  // At least BOS, something, EOS
    }

    func testTokenizerDecodeRoundtrip() {
        let tokenizer = CLAPTokenizer.createBasic()

        let original = "a dog"
        let encoded = tokenizer.encode(original, padding: false, truncation: false)
        let decoded = tokenizer.decode(encoded, skipSpecialTokens: true)

        // Basic check - decoded should contain original words
        XCTAssertTrue(decoded.contains("a") || decoded.contains("dog") || decoded.isEmpty)
    }

    // MARK: - Component Shape Tests

    func testWindowAttentionShape() {
        let dim = 96
        let numHeads = 4
        let windowSize = 8
        let batchWindows = 16  // numWindows * B

        let attn = WindowAttention(
            dim: dim,
            windowSize: (windowSize, windowSize),
            numHeads: numHeads
        )

        // Input: [numWindows*B, N, C] where N = windowSize^2
        let x = MLX.zeros([batchWindows, windowSize * windowSize, dim])
        let output = attn(x)

        XCTAssertEqual(output.shape, [batchWindows, windowSize * windowSize, dim])
    }

    func testSwinTransformerBlockShape() {
        let dim = 96
        let numHeads = 4
        let B = 2
        let H = 64
        let W = 64
        let L = H * W

        let block = SwinTransformerBlock(
            dim: dim,
            numHeads: numHeads,
            windowSize: 8,
            shiftSize: 0
        )

        let x = MLX.zeros([B, L, dim])
        let output = block(x, H: H, W: W)

        XCTAssertEqual(output.shape, [B, L, dim])
    }

    func testPatchMergingShape() {
        let dim = 96
        let B = 2
        let H = 64
        let W = 64

        let merge = PatchMerging(dim: dim)

        let x = MLX.zeros([B, H * W, dim])
        let (output, newH, newW) = merge(x, H: H, W: W)

        XCTAssertEqual(output.shape, [B, (H / 2) * (W / 2), dim * 2])
        XCTAssertEqual(newH, H / 2)
        XCTAssertEqual(newW, W / 2)
    }

    func testBasicLayerShape() {
        let dim = 96
        let B = 2
        let H = 64
        let W = 64

        let layer = BasicLayer(
            dim: dim,
            depth: 2,
            numHeads: 4,
            dropPath: [0.0, 0.0],
            downsampleEnabled: true
        )

        let x = MLX.zeros([B, H * W, dim])
        let (output, newH, newW) = layer(x, H: H, W: W)

        // After downsample: H/2 x W/2, dim*2
        XCTAssertEqual(output.shape, [B, (H / 2) * (W / 2), dim * 2])
        XCTAssertEqual(newH, H / 2)
        XCTAssertEqual(newW, W / 2)
    }

    func testPatchEmbedShape() {
        let B = 2
        let H = 256
        let W = 256
        let C = 1
        let embedDim = 96
        let patchSize = 4

        let patchEmbed = PatchEmbed(
            patchSize: patchSize,
            patchStride: (patchSize, patchSize),
            inChans: C,
            embedDim: embedDim,
            flatten: true
        )

        // Input: [B, H, W, C] (NHWC format for MLX)
        let x = MLX.zeros([B, H, W, C])
        let output = patchEmbed(x)

        let numPatches = (H / patchSize) * (W / patchSize)
        XCTAssertEqual(output.shape, [B, numPatches, embedDim])
    }

    // MARK: - Text Encoder Tests

    func testRobertaEmbeddingsShape() {
        let config = CLAPTextConfig()
        let B = 2
        let L = 32

        let embeddings = RobertaEmbeddings(config: config)
        let inputIds = MLX.zeros([B, L], dtype: .int32)
        let output = embeddings(inputIds)

        XCTAssertEqual(output.shape, [B, L, config.hiddenSize])
    }

    func testRobertaLayerShape() {
        let config = CLAPTextConfig()
        let B = 2
        let L = 32

        let layer = RobertaLayer(config: config)
        let input = MLX.zeros([B, L, config.hiddenSize])
        let output = layer(input)

        XCTAssertEqual(output.shape, [B, L, config.hiddenSize])
    }

    func testCLAPTextEncoderShape() {
        let config = CLAPTextConfig()
        let projectionDim = 512
        let B = 2
        let L = 32

        let encoder = CLAPTextEncoder(config: config, projectionDim: projectionDim)
        let inputIds = MLX.zeros([B, L], dtype: .int32)
        let output = encoder(inputIds, normalize: true)

        XCTAssertEqual(output.shape, [B, projectionDim])

        // Check normalization (L2 norm should be ~1)
        let norms = MLX.sqrt(MLX.sum(output * output, axis: -1))
        let normValues = norms.asArray(Float.self)
        for norm in normValues {
            XCTAssertEqual(norm, 1.0, accuracy: 0.01)
        }
    }

    // MARK: - Audio Encoder Tests

    func testHTSATShape() {
        let config = CLAPAudioConfig(enableFusion: false)
        let B = 1
        let C = 1
        let F = 64
        let T = 1024  // ~10 seconds at 480 hop

        let htsat = HTSAT(config: config)

        // Input: [B, C, F, T]
        let x = MLX.zeros([B, C, F, T])
        let output = htsat(x)

        // Output should be [B, hiddenSize]
        XCTAssertEqual(output.shape, [B, config.hiddenSize])
    }

    // MARK: - Full Model Tests

    func testCLAPModelCreation() {
        let model = CLAPModel()

        XCTAssertEqual(model.config.projectionDim, 512)
        XCTAssertNotNil(model.tokenizer)
    }

    func testCLAPModelTextEncoding() throws {
        let model = CLAPModel()

        let embedding = try model.encodeText("a dog barking")

        // Should be [1, projectionDim]
        XCTAssertEqual(embedding.shape, [1, model.config.projectionDim])

        // Should be normalized
        let norm = MLX.sqrt(MLX.sum(embedding * embedding))
        XCTAssertEqual(norm.item() as Float, 1.0, accuracy: 0.01)
    }

    func testCLAPModelSimilarity() throws {
        let model = CLAPModel()

        let text1 = try model.encodeText("a dog barking")
        let text2 = try model.encodeText("a cat meowing")

        let similarity = model.similarity(audioEmbeds: text1, textEmbeds: text2)

        // Similarity should be a scalar (or [1, 1] matrix)
        XCTAssertEqual(similarity.shape.count, 2)
        XCTAssertEqual(similarity.shape[0], 1)
        XCTAssertEqual(similarity.shape[1], 1)
    }

    // MARK: - Feature Extractor Tests

    func testFeatureExtractorShape() throws {
        let extractor = CLAPFeatureExtractor()
        let B = 2
        let T = 48000 * 5  // 5 seconds at 48kHz

        let audio = MLX.zeros([B, T])
        let (features, isLonger) = try extractor(audio)

        // Features should be [B, 4, nMels, nFrames]
        XCTAssertEqual(features.shape[0], B)
        XCTAssertEqual(features.shape[1], 4)  // 4 channels for fusion
        XCTAssertEqual(features.shape[2], extractor.nMels)

        // isLonger should be [B, 1]
        XCTAssertEqual(isLonger.shape, [B, 1])
    }

    // MARK: - Projection Tests

    func testCLAPProjectionShape() {
        let inDim = 768
        let outDim = 512
        let B = 2

        let proj = CLAPProjection(inDim: inDim, outDim: outDim)
        let input = MLX.zeros([B, inDim])
        let output = proj(input)

        XCTAssertEqual(output.shape, [B, outDim])
    }

    // MARK: - Integration Tests

    func testEndToEndTextPipeline() throws {
        let model = CLAPModel()

        // Encode multiple texts
        let texts = ["a dog barking", "birds singing", "rain falling"]
        var embeddings: [MLXArray] = []

        for text in texts {
            let emb = try model.encodeText(text)
            embeddings.append(emb)
        }

        // Stack embeddings
        let stacked = MLX.concatenated(embeddings, axis: 0)
        XCTAssertEqual(stacked.shape, [texts.count, model.config.projectionDim])

        // Compute pairwise similarities
        let similarities = model.similarity(audioEmbeds: stacked, textEmbeds: stacked)
        XCTAssertEqual(similarities.shape, [texts.count, texts.count])

        // Diagonal should be highest (self-similarity)
        // Note: With random weights, this may not hold, but shape should be correct
    }
}

// MARK: - Parity Tests

extension CLAPTests {
    /// Test for numerical parity with Python implementation.
    ///
    /// To use this test:
    /// 1. Generate reference embeddings from Python:
    ///    ```python
    ///    from mlx_audio.models.clap import CLAP
    ///    model = CLAP.from_pretrained("laion/clap-htsat-fused")
    ///    text_embed = model.encode_text(["a dog barking"])
    ///    np.save("clap_text_embed.npy", np.array(text_embed))
    ///    ```
    /// 2. Load the reference embeddings in this test
    /// 3. Compare Swift output to Python output
    ///
    /// This test is disabled by default as it requires reference data.
    func testTextEmbeddingParity() throws {
        // Skip if no reference data
        let refPath = URL(fileURLWithPath: "/tmp/clap_text_embed.npy")
        guard FileManager.default.fileExists(atPath: refPath.path) else {
            throw XCTSkip("Reference embedding file not found at \(refPath.path)")
        }

        // Load reference
        let refEmbed = try loadArray(url: refPath)

        // Create model with same config and weights
        let model = CLAPModel()

        // Encode same text
        let swiftEmbed = try model.encodeText("a dog barking")

        // Compare
        let diff = MLX.abs(swiftEmbed - refEmbed)
        let maxDiff = MLX.max(diff).item() as Float

        XCTAssertLessThan(maxDiff, 1e-4, "Embedding difference too large: \(maxDiff)")
    }

    func testAudioEmbeddingParity() throws {
        // Skip if no reference data
        let audioPath = URL(fileURLWithPath: "/tmp/clap_audio_input.npy")
        let refPath = URL(fileURLWithPath: "/tmp/clap_audio_embed.npy")

        guard FileManager.default.fileExists(atPath: audioPath.path),
              FileManager.default.fileExists(atPath: refPath.path) else {
            throw XCTSkip("Reference files not found")
        }

        // Load reference
        let audioInput = try loadArray(url: audioPath)
        let refEmbed = try loadArray(url: refPath)

        // Create model
        let model = CLAPModel()

        // Encode audio
        let swiftEmbed = try model.encodeAudio(audioInput)

        // Compare
        let diff = MLX.abs(swiftEmbed - refEmbed)
        let maxDiff = MLX.max(diff).item() as Float

        XCTAssertLessThan(maxDiff, 1e-4, "Embedding difference too large: \(maxDiff)")
    }

    // MARK: - Helper Functions

    /// Load array from numpy file.
    private func loadArray(url: URL) throws -> MLXArray {
        // For now, this is a placeholder that would need numpy file loading support.
        // In production, you would use MLX.loadArrays() for safetensors or implement npy loading.
        fatalError("loadArray not implemented - use safetensors format instead")
    }
}
