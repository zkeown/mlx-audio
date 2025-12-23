// WhisperTests.swift
// Tests for Whisper speech recognition model.

import XCTest
import MLX
import MLXNN
@testable import MLXAudioModels

final class WhisperTests: XCTestCase {

    // MARK: - Configuration Tests

    func testTinyConfig() {
        let config = WhisperConfig.tiny()

        XCTAssertEqual(config.nMels, 80)
        XCTAssertEqual(config.nAudioState, 384)
        XCTAssertEqual(config.nAudioHead, 6)
        XCTAssertEqual(config.nAudioLayer, 4)
        XCTAssertEqual(config.nTextState, 384)
        XCTAssertEqual(config.nTextHead, 6)
        XCTAssertEqual(config.nTextLayer, 4)
        XCTAssertEqual(config.nVocab, 51865)
        XCTAssertTrue(config.isMultilingual)
        XCTAssertFalse(config.isV3)
    }

    func testBaseConfig() {
        let config = WhisperConfig.base()

        XCTAssertEqual(config.nAudioState, 512)
        XCTAssertEqual(config.nAudioHead, 8)
        XCTAssertEqual(config.nAudioLayer, 6)
    }

    func testSmallConfig() {
        let config = WhisperConfig.small()

        XCTAssertEqual(config.nAudioState, 768)
        XCTAssertEqual(config.nAudioHead, 12)
        XCTAssertEqual(config.nAudioLayer, 12)
    }

    func testMediumConfig() {
        let config = WhisperConfig.medium()

        XCTAssertEqual(config.nAudioState, 1024)
        XCTAssertEqual(config.nAudioHead, 16)
        XCTAssertEqual(config.nAudioLayer, 24)
    }

    func testLargeConfig() {
        let config = WhisperConfig.large()

        XCTAssertEqual(config.nAudioState, 1280)
        XCTAssertEqual(config.nAudioHead, 20)
        XCTAssertEqual(config.nAudioLayer, 32)
        XCTAssertEqual(config.nTextLayer, 32)
    }

    func testLargeV3Config() {
        let config = WhisperConfig.largeV3()

        XCTAssertEqual(config.nMels, 128)
        XCTAssertEqual(config.nAudioState, 1280)
        XCTAssertTrue(config.isV3)
    }

    func testLargeV3TurboConfig() {
        let config = WhisperConfig.largeV3Turbo()

        XCTAssertEqual(config.nMels, 128)
        XCTAssertEqual(config.nAudioState, 1280)
        XCTAssertEqual(config.nAudioLayer, 32)
        XCTAssertEqual(config.nTextLayer, 4)  // Reduced decoder
        XCTAssertTrue(config.isV3)
    }

    func testEnglishOnlyConfig() {
        let config = WhisperConfig.tinyEn()

        XCTAssertEqual(config.nVocab, 51864)
        XCTAssertFalse(config.isMultilingual)
    }

    func testConfigFromName() throws {
        let tiny = try WhisperConfig.fromName("tiny")
        XCTAssertEqual(tiny.nAudioState, 384)

        let turbo = try WhisperConfig.fromName("large-v3-turbo")
        XCTAssertEqual(turbo.nTextLayer, 4)

        let whisperBase = try WhisperConfig.fromName("whisper-base")
        XCTAssertEqual(whisperBase.nAudioState, 512)
    }

    func testConfigFromNameInvalid() {
        XCTAssertThrowsError(try WhisperConfig.fromName("unknown-model"))
    }

    // MARK: - Sinusoidal Positional Embedding Tests

    func testSinusoids() {
        let length = 100
        let dim = 64
        let embeddings = sinusoids(length: length, dim: dim)

        XCTAssertEqual(embeddings.shape, [length, dim])

        // First half should be sin, second half should be cos
        // Values should be in [-1, 1]
        let minVal = min(embeddings).item(Float.self)
        let maxVal = max(embeddings).item(Float.self)
        XCTAssertGreaterThanOrEqual(minVal, -1.0)
        XCTAssertLessThanOrEqual(maxVal, 1.0)
    }

    // MARK: - Model Shape Tests

    func testEncoderOutputShape() {
        let config = WhisperConfig.tiny()
        let encoder = AudioEncoder(config: config)

        // Create dummy mel spectrogram [B, nMels, T]
        let batchSize = 2
        let nMels = config.nMels
        let frames = 100
        let mel = MLXArray.zeros([batchSize, nMels, frames])

        let output = encoder(mel)

        // Output should be [B, T//2, nState]
        XCTAssertEqual(output.shape[0], batchSize)
        XCTAssertEqual(output.shape[1], frames / 2)  // stride 2 downsampling
        XCTAssertEqual(output.shape[2], config.nAudioState)
    }

    func testEncoderUnbatchedInput() {
        let config = WhisperConfig.tiny()
        let encoder = AudioEncoder(config: config)

        // Create unbatched mel [nMels, T]
        let mel = MLXArray.zeros([config.nMels, 100])

        let output = encoder(mel)

        // Output should have batch dimension added
        XCTAssertEqual(output.ndim, 3)
        XCTAssertEqual(output.shape[0], 1)
    }

    func testDecoderOutputShape() {
        let config = WhisperConfig.tiny()
        let decoder = TextDecoder(config: config)

        // Create dummy inputs
        let batchSize = 2
        let seqLen = 10
        let audioLen = 50
        let tokens = MLXArray.zeros([batchSize, seqLen], dtype: .int32)
        let audioFeatures = MLXArray.zeros([batchSize, audioLen, config.nAudioState])

        let (logits, kvCache) = decoder(
            tokens: tokens,
            audioFeatures: audioFeatures
        )

        // Logits should be [B, T, nVocab]
        XCTAssertEqual(logits.shape[0], batchSize)
        XCTAssertEqual(logits.shape[1], seqLen)
        XCTAssertEqual(logits.shape[2], config.nVocab)

        // KV cache should have entries for each layer
        XCTAssertEqual(kvCache.count, config.nTextLayer)
    }

    func testDecoderKVCache() {
        let config = WhisperConfig.tiny()
        let decoder = TextDecoder(config: config)

        let batchSize = 1
        let audioLen = 50
        let audioFeatures = MLXArray.zeros([batchSize, audioLen, config.nAudioState])

        // First decode: 5 tokens
        let tokens1 = MLXArray.zeros([batchSize, 5], dtype: .int32)
        let (logits1, cache1) = decoder(
            tokens: tokens1,
            audioFeatures: audioFeatures
        )

        XCTAssertEqual(cache1[0].0.shape[1], 5)  // Cache should have 5 keys

        // Second decode: 1 token with cache
        let tokens2 = MLXArray.zeros([batchSize, 1], dtype: .int32)
        let (logits2, cache2) = decoder(
            tokens: tokens2,
            audioFeatures: audioFeatures,
            kvCache: cache1
        )

        XCTAssertEqual(logits2.shape[1], 1)  // Only 1 output position
        XCTAssertEqual(cache2[0].0.shape[1], 6)  // Cache should have 6 keys now
    }

    // MARK: - Multi-Head Attention Tests

    func testMultiHeadAttentionShape() {
        let nState = 256
        let nHead = 4
        let attn = MultiHeadAttention(nState: nState, nHead: nHead)

        let batchSize = 2
        let seqLen = 10
        let x = MLXArray.zeros([batchSize, seqLen, nState])

        let (output, kvCache) = attn(x)

        XCTAssertEqual(output.shape, x.shape)
        XCTAssertNotNil(kvCache)
        XCTAssertEqual(kvCache!.0.shape[1], seqLen)  // Keys
        XCTAssertEqual(kvCache!.1.shape[1], seqLen)  // Values
    }

    func testCrossAttention() {
        let nState = 256
        let nHead = 4
        let attn = MultiHeadAttention(nState: nState, nHead: nHead)

        let batchSize = 2
        let queryLen = 10
        let keyLen = 20
        let x = MLXArray.zeros([batchSize, queryLen, nState])
        let xa = MLXArray.zeros([batchSize, keyLen, nState])

        let (output, kvCache) = attn(x, xa: xa)

        XCTAssertEqual(output.shape, x.shape)
        XCTAssertNil(kvCache)  // No caching for cross-attention
    }

    // MARK: - Residual Attention Block Tests

    func testResidualAttentionBlockEncoder() {
        let nState = 256
        let nHead = 4
        let block = ResidualAttentionBlock(nState: nState, nHead: nHead, crossAttention: false)

        let batchSize = 2
        let seqLen = 10
        let x = MLXArray.zeros([batchSize, seqLen, nState])

        let (output, _) = block(x)

        XCTAssertEqual(output.shape, x.shape)
    }

    func testResidualAttentionBlockDecoder() {
        let nState = 256
        let nHead = 4
        let block = ResidualAttentionBlock(nState: nState, nHead: nHead, crossAttention: true)

        let batchSize = 2
        let seqLen = 10
        let encoderLen = 20
        let x = MLXArray.zeros([batchSize, seqLen, nState])
        let xa = MLXArray.zeros([batchSize, encoderLen, nState])

        let (output, _) = block(x, xa: xa)

        XCTAssertEqual(output.shape, x.shape)
    }

    // MARK: - Model Tests

    func testModelForwardPass() {
        let config = WhisperConfig.tiny()
        let model = WhisperModel(config: config)

        let batchSize = 1
        let nMels = config.nMels
        let frames = 100
        let seqLen = 5

        let mel = MLXArray.zeros([batchSize, nMels, frames])
        let tokens = MLXArray.zeros([batchSize, seqLen], dtype: .int32)

        let logits = model(mel: mel, tokens: tokens)

        XCTAssertEqual(logits.shape[0], batchSize)
        XCTAssertEqual(logits.shape[1], seqLen)
        XCTAssertEqual(logits.shape[2], config.nVocab)
    }

    func testModelEncode() {
        let config = WhisperConfig.tiny()
        let model = WhisperModel(config: config)

        let mel = MLXArray.zeros([1, config.nMels, 100])
        let features = model.encode(mel)

        XCTAssertEqual(features.shape[2], config.nAudioState)
    }

    func testModelDecode() {
        let config = WhisperConfig.tiny()
        let model = WhisperModel(config: config)

        let tokens = MLXArray.zeros([1, 5], dtype: .int32)
        let audioFeatures = MLXArray.zeros([1, 50, config.nAudioState])

        let (logits, cache) = model.decode(
            tokens: tokens,
            audioFeatures: audioFeatures
        )

        XCTAssertEqual(logits.shape[2], config.nVocab)
        XCTAssertEqual(cache.count, config.nTextLayer)
    }

    // MARK: - Audio Processing Tests

    func testPadOrTrimPad() {
        let audio = MLXArray.ones([100])
        let padded = padOrTrim(audio, length: 200)

        XCTAssertEqual(padded.shape[0], 200)
        // First 100 should be ones, rest zeros
        XCTAssertEqual(sum(padded).item(Float.self), 100.0, accuracy: 0.001)
    }

    func testPadOrTrimTrim() {
        let audio = MLXArray.ones([200])
        let trimmed = padOrTrim(audio, length: 100)

        XCTAssertEqual(trimmed.shape[0], 100)
        XCTAssertEqual(sum(trimmed).item(Float.self), 100.0, accuracy: 0.001)
    }

    func testPadOrTrimBatched() {
        let audio = MLXArray.ones([2, 100])
        let padded = padOrTrim(audio, length: 150)

        XCTAssertEqual(padded.shape, [2, 150])
    }

    func testToMono() {
        let stereo = MLXArray.ones([2, 100])
        let mono = toMono(stereo)

        XCTAssertEqual(mono.shape, [100])
        XCTAssertEqual(mean(mono).item(Float.self), 1.0, accuracy: 0.001)
    }

    func testNormalizeAudio() {
        let audio = MLXArray([0.5, -0.5, 0.25, -0.25])
        let normalized = normalizeAudio(audio)

        let maxVal = max(abs(normalized)).item(Float.self)
        XCTAssertEqual(maxVal, 1.0, accuracy: 0.001)
    }

    // MARK: - Decoding Options Tests

    func testDecodingOptionsDefaults() {
        let options = DecodingOptions()

        XCTAssertNil(options.language)
        XCTAssertEqual(options.task, "transcribe")
        XCTAssertEqual(options.temperature, 0.0)
        XCTAssertEqual(options.beamSize, 1)
        XCTAssertEqual(options.maxTokens, 448)
        XCTAssertTrue(options.isGreedy)
    }

    func testDecodingOptionsNotGreedy() {
        let options1 = DecodingOptions(temperature: 0.5)
        XCTAssertFalse(options1.isGreedy)

        let options2 = DecodingOptions(beamSize: 5)
        XCTAssertFalse(options2.isGreedy)
    }

    // MARK: - Transcription Result Tests

    func testTranscriptionResultSRT() {
        let segments = [
            TranscriptionSegment(text: "Hello world", start: 0.0, end: 1.5, tokens: []),
            TranscriptionSegment(text: "This is a test", start: 1.5, end: 3.0, tokens: [])
        ]
        let result = TranscriptionResult(text: "Hello world This is a test", segments: segments, language: "en")

        let srt = result.toSRT()

        XCTAssertTrue(srt.contains("1\n"))
        XCTAssertTrue(srt.contains("00:00:00,000 --> 00:00:01,500"))
        XCTAssertTrue(srt.contains("Hello world"))
        XCTAssertTrue(srt.contains("2\n"))
    }

    func testTranscriptionResultVTT() {
        let segments = [
            TranscriptionSegment(text: "Hello", start: 0.0, end: 1.0, tokens: [])
        ]
        let result = TranscriptionResult(text: "Hello", segments: segments, language: "en")

        let vtt = result.toVTT()

        XCTAssertTrue(vtt.starts(with: "WEBVTT"))
        XCTAssertTrue(vtt.contains("00:00:00.000 --> 00:00:01.000"))
    }
}

// MARK: - Tokenizer Tests

final class WhisperTokenizerTests: XCTestCase {

    func testLanguagesDict() {
        XCTAssertEqual(WHISPER_LANGUAGES["en"], "english")
        XCTAssertEqual(WHISPER_LANGUAGES["zh"], "chinese")
        XCTAssertEqual(WHISPER_LANGUAGES["ja"], "japanese")
        XCTAssertEqual(WHISPER_LANGUAGES.count, 98)
    }

    func testToLanguageCode() {
        XCTAssertEqual(WhisperTokenizer.toLanguageCode["english"], "en")
        XCTAssertEqual(WhisperTokenizer.toLanguageCode["chinese"], "zh")
    }
}
