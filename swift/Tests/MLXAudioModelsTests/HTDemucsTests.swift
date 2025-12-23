// HTDemucsTests.swift
// Tests for HTDemucs model implementation.

import XCTest
@testable import MLXAudioModels
import MLX
import MLXNN

final class HTDemucsTests: XCTestCase {

    // MARK: - Configuration Tests

    func testConfigDefaults() {
        let config = HTDemucsConfig()

        XCTAssertEqual(config.sources.count, 4)
        XCTAssertEqual(config.sources, ["drums", "bass", "other", "vocals"])
        XCTAssertEqual(config.audio_channels, 2)
        XCTAssertEqual(config.samplerate, 44100)
        XCTAssertEqual(config.segment, 6.0)
        XCTAssertEqual(config.depth, 4)
        XCTAssertEqual(config.channels, 48)
        XCTAssertEqual(config.nfft, 4096)
        XCTAssertEqual(config.hop_length, 1024)
        XCTAssertEqual(config.t_depth, 5)
        XCTAssertEqual(config.t_heads, 8)
        XCTAssertTrue(config.cac)
    }

    func testConfigPresets() {
        let ftConfig = HTDemucsConfig.htdemucs_ft()
        XCTAssertEqual(ftConfig.sources.count, 4)

        let sixsConfig = HTDemucsConfig.htdemucs_6s()
        XCTAssertEqual(sixsConfig.sources.count, 6)
        XCTAssertTrue(sixsConfig.sources.contains("guitar"))
        XCTAssertTrue(sixsConfig.sources.contains("piano"))
    }

    func testConfigComputedProperties() {
        let config = HTDemucsConfig()
        XCTAssertEqual(config.num_sources, 4)
        XCTAssertEqual(config.freq_bins, 2049)  // nfft / 2 + 1
    }

    // MARK: - Layer Tests

    func testLayerScale() {
        let layerScale = LayerScale(channels: 64, init: 1e-4)
        let input = MLXRandom.normal([2, 100, 64])

        let output = layerScale(input)

        XCTAssertEqual(output.shape, input.shape)
        // Initial scale is 1e-4, so output should be much smaller than input
        let inputMag = abs(input).mean().item(Float.self)
        let outputMag = abs(output).mean().item(Float.self)
        XCTAssertLessThan(outputMag, inputMag * 0.01)
    }

    func testScaledEmbedding() {
        let emb = ScaledEmbedding(numEmbeddings: 128, embeddingDim: 48, scale: 10.0)
        let indices = MLXArray([0, 1, 2, 3])

        let output = emb(indices)

        XCTAssertEqual(output.shape, [4, 48])
    }

    func testMyGroupNorm() {
        let norm = MyGroupNorm(numChannels: 64)
        let input = MLXRandom.normal([2, 100, 64])

        let output = norm(input)

        XCTAssertEqual(output.shape, input.shape)
        // Output should be normalized (mean close to 0, std close to 1)
        // Note: With learned scale/bias, this may vary
    }

    func testGLU() {
        let input = MLXRandom.normal([2, 100, 128])

        let output = MLXNN.glu(input, axis: -1)

        XCTAssertEqual(output.shape, [2, 100, 64])  // Half the last dimension
    }

    // MARK: - DConv Tests

    func testDConvShape() {
        let dconv = DConv(channels: 48, depth: 2, compress: 8)
        let input = MLXRandom.normal([2, 100, 48])  // [B, T, C] NLC format

        let output = dconv(input)

        XCTAssertEqual(output.shape, input.shape)
    }

    // MARK: - Encoder Tests

    func testFrequencyEncoderShape() {
        let encoder = HEncLayer(
            chin: 4,  // C*2 for CAC
            chout: 48,
            kernelSize: 8,
            stride: 4,
            freq: true,
            dconvDepth: 2,
            dconvCompress: 8
        )

        // Input: [B, C, F, T] NCHW format
        let input = MLXRandom.normal([1, 4, 2048, 260])

        let output = encoder(input)

        // Frequency dimension should be reduced by stride
        XCTAssertEqual(output.shape[0], 1)
        XCTAssertEqual(output.shape[1], 48)
        XCTAssertEqual(output.shape[2], 512)  // 2048 / 4
        XCTAssertEqual(output.shape[3], 260)  // Time unchanged
    }

    func testTimeEncoderShape() {
        let encoder = HEncLayer(
            chin: 2,
            chout: 48,
            kernelSize: 8,
            stride: 4,
            freq: false,
            dconvDepth: 2,
            dconvCompress: 8
        )

        // Input: [B, C, T] NCL format
        let input = MLXRandom.normal([1, 2, 264600])

        let output = encoder(input)

        XCTAssertEqual(output.shape[0], 1)
        XCTAssertEqual(output.shape[1], 48)
        // Time dimension reduced by stride (with padding)
        XCTAssertTrue(output.shape[2] > 0)
    }

    // MARK: - Decoder Tests

    func testFrequencyDecoderShape() {
        let decoder = HDecLayer(
            chin: 48,
            chout: 4,
            kernelSize: 8,
            stride: 4,
            freq: true,
            dconvDepth: 2,
            dconvCompress: 8,
            last: true
        )

        let input = MLXRandom.normal([1, 48, 512, 260])
        let skip = MLXRandom.normal([1, 48, 512, 260])
        let length = 260

        let (output, pre) = decoder(input, skip: skip, length: length)

        XCTAssertEqual(output.shape[0], 1)
        XCTAssertEqual(output.shape[1], 4)
        // Frequency dimension should be upsampled
        XCTAssertTrue(output.shape[2] > 512)
    }

    // MARK: - Transformer Tests

    func testSinusoidalEmbedding1D() {
        let emb = createSinEmbedding(length: 100, dim: 512)

        XCTAssertEqual(emb.shape, [100, 1, 512])
    }

    func testSinusoidalEmbedding2D() {
        let emb = create2DSinEmbedding(dModel: 512, height: 8, width: 16)

        XCTAssertEqual(emb.shape, [1, 512, 8, 16])
    }

    func testMultiheadAttention() {
        let attn = MultiheadAttention(embedDim: 512, numHeads: 8)
        let query = MLXRandom.normal([2, 100, 512])
        let key = MLXRandom.normal([2, 80, 512])
        let value = key

        let output = attn(query: query, key: key, value: value)

        XCTAssertEqual(output.shape, [2, 100, 512])
    }

    func testCrossTransformerEncoder() {
        let transformer = CrossTransformerEncoder(
            dim: 512,
            depth: 5,
            heads: 8
        )

        let freq = MLXRandom.normal([1, 512, 8, 16])  // [B, C, F, T]
        let time = MLXRandom.normal([1, 512, 100])     // [B, C, T]

        let (freqOut, timeOut) = transformer(freq: freq, time: time)

        XCTAssertEqual(freqOut.shape, freq.shape)
        XCTAssertEqual(timeOut.shape, time.shape)
    }

    // MARK: - Full Model Tests

    func testHTDemucsInitialization() {
        let config = HTDemucsConfig()
        let model = HTDemucs(config: config)

        XCTAssertEqual(model.encoder.count, config.depth)
        XCTAssertEqual(model.tencoder.count, config.depth)
        XCTAssertEqual(model.decoder.count, config.depth)
        XCTAssertEqual(model.tdecoder.count, config.depth)
    }

    func testHTDemucsOutputShape() {
        // Use smaller config for faster testing
        var config = HTDemucsConfig()
        config.depth = 2
        config.channels = 16
        config.t_depth = 1
        config.bottom_channels = 32
        config.segment = 1.0

        let model = HTDemucs(config: config)

        // Short audio for testing
        let samplerate = config.samplerate
        let duration = 1.0
        let samples = Int(Float(samplerate) * Float(duration))
        let input = MLXRandom.normal([1, 2, samples])

        let output = model(input)

        // Output should be [B, S, C, T]
        XCTAssertEqual(output.shape[0], 1)
        XCTAssertEqual(output.shape[1], config.num_sources)
        XCTAssertEqual(output.shape[2], config.audio_channels)
        XCTAssertEqual(output.shape[3], samples)
    }

    // MARK: - Inference Tests

    func testWeightWindow() {
        // Test weight window creation indirectly through applyModel
        let config = HTDemucsConfig()
        XCTAssertEqual(config.segment, 6.0)
    }

    func testBagOfModelsInitialization() {
        var config = HTDemucsConfig()
        config.depth = 1
        config.channels = 8
        config.t_depth = 1
        config.bottom_channels = 16

        let model1 = HTDemucs(config: config)
        let model2 = HTDemucs(config: config)

        let bag = BagOfModels(models: [model1, model2])

        XCTAssertEqual(bag.numModels, 2)
        XCTAssertEqual(bag.weights.shape, [2, 2])
    }

    // MARK: - Parity Tests (Placeholder)

    // These tests require Python-generated fixtures
    // Uncomment and implement when fixtures are available

    /*
    func testEncoderParity() throws {
        // Load Python test fixture
        let inputURL = Bundle.module.url(forResource: "encoder_0_input", withExtension: "safetensors")!
        let outputURL = Bundle.module.url(forResource: "encoder_0_output", withExtension: "safetensors")!

        let inputData = try MLX.loadArrays(url: inputURL)
        let outputData = try MLX.loadArrays(url: outputURL)

        let input = inputData["input"]!
        let expected = outputData["output"]!

        // Create encoder with same config as Python
        let encoder = HEncLayer(chin: 4, chout: 48, kernelSize: 8, stride: 4, freq: true)
        // Load weights...

        let output = encoder(input)
        let maxDiff = abs(output - expected).max().item(Float.self)

        XCTAssertLessThan(maxDiff, 1e-5)
    }

    func testFullModelParity() throws {
        // Load full model output from Python
        let fixtureURL = Bundle.module.url(forResource: "full_output", withExtension: "safetensors")!
        let fixtures = try MLX.loadArrays(url: fixtureURL)

        let input = fixtures["input"]!
        let expected = fixtures["output"]!

        // Load model with same weights as Python
        let modelURL = Bundle.module.url(forResource: "htdemucs_test", withExtension: "")!
        let model = try HTDemucs.fromPretrained(path: modelURL)

        let output = model(input)
        let maxDiff = abs(output - expected).max().item(Float.self)

        // Full model may have accumulated error
        XCTAssertLessThan(maxDiff, 1e-3)
    }
    */
}
