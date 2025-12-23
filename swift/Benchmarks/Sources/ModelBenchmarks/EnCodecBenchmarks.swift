// EnCodecBenchmarks.swift
// Benchmarks for EnCodec neural audio codec.

import Foundation
import MLX
import MLXAudioModels
import BenchmarkKit

/// EnCodec benchmark suite.
public struct EnCodecBenchmarks {

    /// Run all EnCodec benchmarks.
    public static func runAll(config: BenchmarkConfig) throws -> [BenchmarkResult] {
        var results: [BenchmarkResult] = []

        // Encode benchmarks
        for duration in BenchmarkDurations.encodec {
            results.append(try benchmarkEncode(durationSec: duration, config: config))
        }

        // Decode benchmarks
        for duration in BenchmarkDurations.encodec {
            results.append(try benchmarkDecode(durationSec: duration, config: config))
        }

        // Roundtrip benchmark
        results.append(try benchmarkRoundtrip(config: config))

        // Batch decode benchmark
        for batchSize in BenchmarkBatchSizes.small {
            results.append(try benchmarkBatchDecode(batchSize: batchSize, config: config))
        }

        return results
    }

    /// Benchmark EnCodec encoding.
    public static func benchmarkEncode(
        durationSec: Double,
        config: BenchmarkConfig
    ) throws -> BenchmarkResult {
        let encodecConfig = EnCodecConfig.encodec_24khz()
        let model = EnCodec(config: encodecConfig)

        // Generate audio input
        let numSamples = Int(durationSec * Double(encodecConfig.sample_rate))
        MLXRandom.seed(42)
        let audio = MLXRandom.normal([1, encodecConfig.channels, numSamples])
        eval(audio)

        let timing = try BenchmarkMetrics.measureTime(
            warmup: config.warmup,
            iterations: config.iterations
        ) {
            let _ = model.encode(audio)
        }

        return BenchmarkMetrics.createResult(
            name: "encodec_encode_\(Int(durationSec))s",
            timing: timing,
            audioDurationSec: durationSec,
            sampleRate: encodecConfig.sample_rate,
            iterations: config.iterations,
            params: [
                "duration_sec": .double(durationSec),
                "sample_rate": .int(encodecConfig.sample_rate),
            ]
        )
    }

    /// Benchmark EnCodec decoding.
    public static func benchmarkDecode(
        durationSec: Double,
        config: BenchmarkConfig
    ) throws -> BenchmarkResult {
        let encodecConfig = EnCodecConfig.encodec_24khz()
        let model = EnCodec(config: encodecConfig)

        // Generate codes
        let frameRate = Int(encodecConfig.frame_rate)
        let numFrames = Int(durationSec * Double(frameRate))
        MLXRandom.seed(42)
        let codes = MLXRandom.randInt(
            low: 0,
            high: encodecConfig.codebook_size,
            [1, encodecConfig.num_codebooks, numFrames]
        )
        eval(codes)

        let timing = try BenchmarkMetrics.measureTime(
            warmup: config.warmup,
            iterations: config.iterations
        ) {
            let _ = model.decode(codes)
        }

        return BenchmarkMetrics.createResult(
            name: "encodec_decode_\(Int(durationSec))s",
            timing: timing,
            audioDurationSec: durationSec,
            sampleRate: encodecConfig.sample_rate,
            iterations: config.iterations,
            params: [
                "duration_sec": .double(durationSec),
                "num_codebooks": .int(encodecConfig.num_codebooks),
            ]
        )
    }

    /// Benchmark EnCodec roundtrip (encode + decode).
    public static func benchmarkRoundtrip(config: BenchmarkConfig) throws -> BenchmarkResult {
        let encodecConfig = EnCodecConfig.encodec_24khz()
        let model = EnCodec(config: encodecConfig)

        let durationSec = 5.0
        let numSamples = Int(durationSec * Double(encodecConfig.sample_rate))
        MLXRandom.seed(42)
        let audio = MLXRandom.normal([1, encodecConfig.channels, numSamples])
        eval(audio)

        let timing = try BenchmarkMetrics.measureTime(
            warmup: config.warmup,
            iterations: config.iterations
        ) {
            let (_, _) = model(audio)
        }

        return BenchmarkMetrics.createResult(
            name: "encodec_roundtrip_\(Int(durationSec))s",
            timing: timing,
            audioDurationSec: durationSec,
            sampleRate: encodecConfig.sample_rate,
            iterations: config.iterations,
            params: [
                "duration_sec": .double(durationSec),
            ]
        )
    }

    /// Benchmark batch decoding.
    public static func benchmarkBatchDecode(
        batchSize: Int,
        config: BenchmarkConfig
    ) throws -> BenchmarkResult {
        let encodecConfig = EnCodecConfig.encodec_24khz()
        let model = EnCodec(config: encodecConfig)

        let durationSec = 5.0
        let frameRate = Int(encodecConfig.frame_rate)
        let numFrames = Int(durationSec * Double(frameRate))

        MLXRandom.seed(42)
        let codes = MLXRandom.randInt(
            low: 0,
            high: encodecConfig.codebook_size,
            [batchSize, encodecConfig.num_codebooks, numFrames]
        )
        eval(codes)

        let timing = try BenchmarkMetrics.measureTime(
            warmup: config.warmup,
            iterations: config.iterations
        ) {
            let _ = model.decode(codes)
        }

        return BenchmarkMetrics.createResult(
            name: "encodec_batch_decode_b\(batchSize)",
            timing: timing,
            iterations: config.iterations,
            params: [
                "batch_size": .int(batchSize),
                "duration_sec": .double(durationSec),
            ]
        )
    }

    /// Check if target performance is met (5s decode < 50ms on M2 Pro).
    public static func checkTargetPerformance() throws -> (passed: Bool, timeMs: Double) {
        let result = try benchmarkDecode(
            durationSec: 5.0,
            config: BenchmarkConfig(warmup: 3, iterations: 5)
        )
        return (result.meanTimeMs < 50, result.meanTimeMs)
    }
}
