// EnCodecTests.swift
// Unit tests for EnCodec Swift implementation.

import XCTest
@testable import MLXAudioModels
import MLX
import MLXNN

final class EnCodecTests: XCTestCase {

    // MARK: - Config Tests

    func testConfigDefaults() {
        let config = EnCodecConfig()

        XCTAssertEqual(config.sample_rate, 32000)
        XCTAssertEqual(config.channels, 1)
        XCTAssertEqual(config.num_codebooks, 4)
        XCTAssertEqual(config.codebook_size, 2048)
        XCTAssertEqual(config.codebook_dim, 128)
        XCTAssertEqual(config.num_filters, 32)
        XCTAssertEqual(config.num_residual_layers, 1)
        XCTAssertEqual(config.ratios, [8, 5, 4, 2])
        XCTAssertEqual(config.kernel_size, 7)
        XCTAssertEqual(config.dilation_base, 2)
        XCTAssertTrue(config.causal)
        XCTAssertEqual(config.lstm_layers, 2)
    }

    func testConfigHopLength() {
        let config = EnCodecConfig(ratios: [8, 5, 4, 2])
        XCTAssertEqual(config.hop_length, 320)  // 8 * 5 * 4 * 2

        let config32k = EnCodecConfig.encodec_32khz()
        XCTAssertEqual(config32k.hop_length, 640)  // 8 * 5 * 4 * 4
    }

    func testConfigFrameRate() {
        let config24k = EnCodecConfig.encodec_24khz()
        XCTAssertEqual(config24k.frame_rate, 75.0, accuracy: 0.1)  // 24000 / 320

        let config32k = EnCodecConfig.encodec_32khz()
        XCTAssertEqual(config32k.frame_rate, 50.0, accuracy: 0.1)  // 32000 / 640
    }

    func testConfigPreset24kHz() {
        let config = EnCodecConfig.encodec_24khz()

        XCTAssertEqual(config.sample_rate, 24000)
        XCTAssertEqual(config.channels, 1)
        XCTAssertEqual(config.num_codebooks, 8)
        XCTAssertEqual(config.codebook_size, 1024)
        XCTAssertEqual(config.num_filters, 32)
        XCTAssertEqual(config.ratios, [8, 5, 4, 2])
    }

    func testConfigPreset32kHz() {
        let config = EnCodecConfig.encodec_32khz()

        XCTAssertEqual(config.sample_rate, 32000)
        XCTAssertEqual(config.channels, 1)
        XCTAssertEqual(config.num_codebooks, 4)
        XCTAssertEqual(config.codebook_size, 2048)
        XCTAssertEqual(config.num_filters, 64)
        XCTAssertEqual(config.ratios, [8, 5, 4, 4])
    }

    func testConfigPreset48kHzStereo() {
        let config = EnCodecConfig.encodec_48khz_stereo()

        XCTAssertEqual(config.sample_rate, 48000)
        XCTAssertEqual(config.channels, 2)
        XCTAssertEqual(config.num_codebooks, 8)
        XCTAssertEqual(config.codebook_size, 1024)
        XCTAssertEqual(config.ratios, [8, 5, 4, 2])
    }

    func testConfigFromName() throws {
        let config1 = try EnCodecConfig.fromName("encodec_24khz")
        XCTAssertEqual(config1.sample_rate, 24000)

        let config2 = try EnCodecConfig.fromName("32khz")
        XCTAssertEqual(config2.sample_rate, 32000)

        XCTAssertThrowsError(try EnCodecConfig.fromName("invalid_model"))
    }

    func testConfigCodable() throws {
        let original = EnCodecConfig.encodec_24khz()

        let encoder = JSONEncoder()
        let data = try encoder.encode(original)

        let decoder = JSONDecoder()
        let decoded = try decoder.decode(EnCodecConfig.self, from: data)

        XCTAssertEqual(original, decoded)
    }

    // MARK: - Layer Tests

    func testConvBlockShape() {
        let conv = ConvBlock(
            inChannels: 32,
            outChannels: 64,
            kernelSize: 7,
            stride: 1,
            causal: true,
            activation: "elu"
        )

        let input = MLXRandom.normal([2, 32, 1000])  // [B, C, T]
        let output = conv(input)

        XCTAssertEqual(output.shape, [2, 64, 1000])
    }

    func testConvBlockDownsample() {
        let conv = ConvBlock(
            inChannels: 32,
            outChannels: 64,
            kernelSize: 16,  // 2x stride
            stride: 8,
            causal: true,
            activation: "none"
        )

        let input = MLXRandom.normal([2, 32, 1000])  // [B, C, T]
        let output = conv(input)

        // Output time dimension should be ~1000/8 = 125
        XCTAssertEqual(output.dim(0), 2)
        XCTAssertEqual(output.dim(1), 64)
        XCTAssertTrue(output.dim(2) >= 120 && output.dim(2) <= 130)
    }

    func testConvTransposeBlockShape() {
        let conv = ConvTransposeBlock(
            inChannels: 64,
            outChannels: 32,
            kernelSize: 16,  // 2x stride
            stride: 8,
            causal: true,
            activation: "elu"
        )

        let input = MLXRandom.normal([2, 64, 125])  // [B, C, T]
        let output = conv(input)

        // Output time dimension should be ~125*8 = 1000
        XCTAssertEqual(output.dim(0), 2)
        XCTAssertEqual(output.dim(1), 32)
        XCTAssertTrue(output.dim(2) >= 990 && output.dim(2) <= 1010)
    }

    func testResidualUnitShape() {
        let residual = ResidualUnit(
            channels: 64,
            kernelSize: 3,
            dilation: 1,
            causal: true
        )

        let input = MLXRandom.normal([2, 64, 1000])  // [B, C, T]
        let output = residual(input)

        // Residual unit preserves shape
        XCTAssertEqual(output.shape, input.shape)
    }

    func testResidualUnitDilated() {
        let residual = ResidualUnit(
            channels: 64,
            kernelSize: 3,
            dilation: 4,
            causal: true
        )

        let input = MLXRandom.normal([2, 64, 1000])  // [B, C, T]
        let output = residual(input)

        // Dilated residual unit also preserves shape
        XCTAssertEqual(output.shape, input.shape)
    }

    // MARK: - Quantizer Tests

    func testVectorQuantizerShape() {
        let vq = VectorQuantizer(codebookSize: 1024, codebookDim: 128)

        let input = MLXRandom.normal([2, 100, 128])  // [B, T, D]
        let (quantized, codes) = vq(input)

        XCTAssertEqual(quantized.shape, [2, 100, 128])
        XCTAssertEqual(codes.shape, [2, 100])
    }

    func testVectorQuantizerCodeRange() {
        let vq = VectorQuantizer(codebookSize: 1024, codebookDim: 128)

        let input = MLXRandom.normal([2, 100, 128])
        let codes = vq.encode(input)

        // Codes should be in range [0, codebook_size)
        let minCode = codes.min().item(Int32.self)
        let maxCode = codes.max().item(Int32.self)
        XCTAssertGreaterThanOrEqual(minCode, 0)
        XCTAssertLessThan(maxCode, 1024)
    }

    func testRVQShape() {
        let rvq = ResidualVectorQuantizer(
            numCodebooks: 4,
            codebookSize: 1024,
            codebookDim: 128
        )

        let input = MLXRandom.normal([2, 100, 128])  // [B, T, D]
        let (quantized, codes) = rvq(input)

        XCTAssertEqual(quantized.shape, [2, 100, 128])
        XCTAssertEqual(codes.shape, [2, 4, 100])  // [B, K, T]
    }

    func testRVQEncodeDecode() {
        let rvq = ResidualVectorQuantizer(
            numCodebooks: 4,
            codebookSize: 1024,
            codebookDim: 128
        )

        let input = MLXRandom.normal([2, 100, 128])
        let codes = rvq.encode(input)
        let decoded = rvq.decode(codes)

        XCTAssertEqual(decoded.shape, input.shape)
    }

    func testRVQCodebookAccess() {
        let rvq = ResidualVectorQuantizer(
            numCodebooks: 4,
            codebookSize: 1024,
            codebookDim: 128
        )

        let codebook0 = rvq.getCodebook(0)
        XCTAssertEqual(codebook0.shape, [1024, 128])

        let codebook3 = rvq.getCodebook(3)
        XCTAssertEqual(codebook3.shape, [1024, 128])
    }

    // MARK: - Encoder Tests

    func testEncoderShape() {
        let config = EnCodecConfig(
            channels: 1,
            num_filters: 32,
            ratios: [8, 5, 4, 2],
            lstm_layers: 0  // Disable LSTM for simpler test
        )
        let encoder = EnCodecEncoder(config: config)

        // Input: 1 second of audio at default sample rate
        let input = MLXRandom.normal([2, 1, 32000])  // [B, C, T]
        let output = encoder(input)

        // Output should be [B, T', D] where T' = T / hop_length
        let expectedTimeFrames = 32000 / config.hop_length
        XCTAssertEqual(output.dim(0), 2)
        XCTAssertTrue(abs(output.dim(1) - expectedTimeFrames) <= 2)  // Allow small variance
        XCTAssertEqual(output.dim(2), config.codebook_dim)
    }

    func testEncoderWithLSTM() {
        let config = EnCodecConfig(
            channels: 1,
            num_filters: 32,
            ratios: [8, 5, 4, 2],
            lstm_layers: 2
        )
        let encoder = EnCodecEncoder(config: config)

        let input = MLXRandom.normal([2, 1, 32000])
        let output = encoder(input)

        // Shape should still be correct with LSTM
        let expectedTimeFrames = 32000 / config.hop_length
        XCTAssertEqual(output.dim(0), 2)
        XCTAssertTrue(abs(output.dim(1) - expectedTimeFrames) <= 2)
        XCTAssertEqual(output.dim(2), config.codebook_dim)
    }

    func testEncoderMonoInput() {
        let config = EnCodecConfig(channels: 1, lstm_layers: 0)
        let encoder = EnCodecEncoder(config: config)

        // Mono input without channel dimension
        let input = MLXRandom.normal([2, 32000])  // [B, T]
        let output = encoder(input)

        XCTAssertEqual(output.ndim, 3)  // [B, T', D]
    }

    // MARK: - Decoder Tests

    func testDecoderShape() {
        let config = EnCodecConfig(
            channels: 1,
            num_filters: 32,
            ratios: [8, 5, 4, 2],
            lstm_layers: 0
        )
        let decoder = EnCodecDecoder(config: config)

        // Input: latent embeddings
        let numFrames = 100
        let input = MLXRandom.normal([2, numFrames, config.codebook_dim])  // [B, T', D]
        let output = decoder(input)

        // Output should be [B, C, T] where T = T' * hop_length
        let expectedTimeSamples = numFrames * config.hop_length
        XCTAssertEqual(output.dim(0), 2)
        XCTAssertEqual(output.dim(1), config.channels)
        XCTAssertTrue(abs(output.dim(2) - expectedTimeSamples) <= 10)  // Allow small variance
    }

    func testDecoderWithLSTM() {
        let config = EnCodecConfig(
            channels: 1,
            num_filters: 32,
            ratios: [8, 5, 4, 2],
            lstm_layers: 2
        )
        let decoder = EnCodecDecoder(config: config)

        let numFrames = 100
        let input = MLXRandom.normal([2, numFrames, config.codebook_dim])
        let output = decoder(input)

        // Shape should still be correct with LSTM
        XCTAssertEqual(output.dim(0), 2)
        XCTAssertEqual(output.dim(1), config.channels)
    }

    // MARK: - Full Model Tests

    func testEnCodecEncode() {
        let config = EnCodecConfig(
            channels: 1,
            num_codebooks: 4,
            num_filters: 32,
            ratios: [8, 5, 4, 2],
            lstm_layers: 0
        )
        let model = EnCodec(config: config)

        let audio = MLXRandom.normal([2, 1, 32000])  // [B, C, T]
        let codes = model.encode(audio)

        // Codes shape: [B, K, T']
        XCTAssertEqual(codes.dim(0), 2)
        XCTAssertEqual(codes.dim(1), config.num_codebooks)
    }

    func testEnCodecDecode() {
        let config = EnCodecConfig(
            channels: 1,
            num_codebooks: 4,
            num_filters: 32,
            ratios: [8, 5, 4, 2],
            lstm_layers: 0
        )
        let model = EnCodec(config: config)

        // Create random codes
        let numFrames = 100
        let codes = MLXRandom.randInt(
            low: 0,
            high: config.codebook_size,
            [2, config.num_codebooks, numFrames]
        )
        let audio = model.decode(codes)

        // Audio shape: [B, C, T]
        XCTAssertEqual(audio.dim(0), 2)
        XCTAssertEqual(audio.dim(1), config.channels)
    }

    func testEnCodecRoundtrip() {
        let config = EnCodecConfig(
            channels: 1,
            num_codebooks: 4,
            num_filters: 32,
            ratios: [8, 5, 4, 2],
            lstm_layers: 0
        )
        let model = EnCodec(config: config)

        // Encode audio
        let audio = MLXRandom.normal([1, 1, 32000])
        let codes = model.encode(audio)

        // Decode back
        let reconstructed = model.decode(codes)

        // Shapes should be compatible
        XCTAssertEqual(reconstructed.dim(0), audio.dim(0))
        XCTAssertEqual(reconstructed.dim(1), audio.dim(1))
    }

    func testEnCodecCallAsFunction() {
        let config = EnCodecConfig(
            channels: 1,
            num_codebooks: 4,
            num_filters: 32,
            ratios: [8, 5, 4, 2],
            lstm_layers: 0
        )
        let model = EnCodec(config: config)

        let audio = MLXRandom.normal([1, 1, 32000])
        let (reconstructed, codes) = model(audio)

        XCTAssertEqual(reconstructed.dim(0), 1)
        XCTAssertEqual(codes.dim(1), config.num_codebooks)
    }

    func testEnCodecHopLength() {
        let config = EnCodecConfig(ratios: [8, 5, 4, 2])
        let model = EnCodec(config: config)

        XCTAssertEqual(model.hopLength, 320)
    }

    // MARK: - Input Shape Flexibility Tests

    func testEnCodecFlexibleInput1D() {
        let config = EnCodecConfig(
            channels: 1,
            num_filters: 32,
            ratios: [8, 5, 4, 2],
            lstm_layers: 0
        )
        let model = EnCodec(config: config)

        // 1D input: [T]
        let audio = MLXRandom.normal([32000])
        let codes = model.encode(audio)

        XCTAssertEqual(codes.dim(0), 1)  // Batch = 1
        XCTAssertEqual(codes.dim(1), config.num_codebooks)
    }

    func testEnCodecFlexibleInput2D() {
        let config = EnCodecConfig(
            channels: 1,
            num_filters: 32,
            ratios: [8, 5, 4, 2],
            lstm_layers: 0
        )
        let model = EnCodec(config: config)

        // 2D input: [B, T]
        let audio = MLXRandom.normal([2, 32000])
        let codes = model.encode(audio)

        XCTAssertEqual(codes.dim(0), 2)  // Batch = 2
        XCTAssertEqual(codes.dim(1), config.num_codebooks)
    }

    // MARK: - Stereo Tests

    func testEnCodecStereo() {
        let config = EnCodecConfig(
            channels: 2,
            num_codebooks: 4,
            num_filters: 32,
            ratios: [8, 5, 4, 2],
            lstm_layers: 0
        )
        let model = EnCodec(config: config)

        let audio = MLXRandom.normal([1, 2, 32000])  // Stereo
        let (reconstructed, codes) = model(audio)

        XCTAssertEqual(reconstructed.dim(1), 2)  // Stereo output
        XCTAssertEqual(codes.dim(1), config.num_codebooks)
    }
}
