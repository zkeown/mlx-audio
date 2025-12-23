// CLAPBenchmarks.swift
// Benchmarks for CLAP audio-text embedding model.

import Foundation
import MLX
import MLXAudioModels
import BenchmarkKit

/// CLAP benchmark suite.
public struct CLAPBenchmarks {

    /// Run all CLAP benchmarks.
    public static func runAll(config: BenchmarkConfig) throws -> [BenchmarkResult] {
        var results: [BenchmarkResult] = []

        // Audio encoding at different durations
        for duration in BenchmarkDurations.clap {
            results.append(try benchmarkAudioEncode(durationSec: duration, config: config))
        }

        // Batch audio encoding
        for batchSize in BenchmarkBatchSizes.small {
            results.append(try benchmarkBatchAudioEncode(batchSize: batchSize, config: config))
        }

        return results
    }

    /// Benchmark CLAP audio encoding using HTSAT encoder.
    public static func benchmarkAudioEncode(
        durationSec: Double,
        config: BenchmarkConfig
    ) throws -> BenchmarkResult {
        let clapConfig = CLAPConfig()
        let audioEncoder = HTSAT(config: clapConfig.audio)

        // Create mel spectrogram (CLAP expects mel input)
        // HTSAT input format: [batch, 1, freq, time]
        let melFrames = Int(durationSec * 48000 / 480)  // hop_size = 480
        MLXRandom.seed(42)
        let melFeatures = MLXRandom.normal([1, 1, 64, melFrames])
        eval(melFeatures)

        let timing = try BenchmarkMetrics.measureTime(
            warmup: config.warmup,
            iterations: config.iterations
        ) {
            let _ = audioEncoder(melFeatures)
        }

        return BenchmarkMetrics.createResult(
            name: "clap_audio_embed_\(Int(durationSec))s",
            timing: timing,
            audioDurationSec: durationSec,
            sampleRate: 48000,
            iterations: config.iterations,
            params: [
                "duration_sec": .double(durationSec),
                "n_mels": .int(64),
            ]
        )
    }

    /// Benchmark batch audio encoding.
    public static func benchmarkBatchAudioEncode(
        batchSize: Int,
        config: BenchmarkConfig
    ) throws -> BenchmarkResult {
        let clapConfig = CLAPConfig()
        let audioEncoder = HTSAT(config: clapConfig.audio)

        // 10 seconds of audio per sample
        let durationSec = 10.0
        let melFrames = Int(durationSec * 48000 / 480)

        MLXRandom.seed(42)
        // HTSAT input format: [batch, 1, freq, time]
        let melFeatures = MLXRandom.normal([batchSize, 1, 64, melFrames])
        eval(melFeatures)

        let timing = try BenchmarkMetrics.measureTime(
            warmup: config.warmup,
            iterations: config.iterations
        ) {
            let _ = audioEncoder(melFeatures)
        }

        return BenchmarkMetrics.createResult(
            name: "clap_batch_audio_embed_b\(batchSize)",
            timing: timing,
            iterations: config.iterations,
            params: [
                "batch_size": .int(batchSize),
                "duration_sec": .double(durationSec),
            ]
        )
    }

    /// Check if target performance is met (10s embed < 100ms on M2 Pro).
    public static func checkTargetPerformance() throws -> (passed: Bool, timeMs: Double) {
        let result = try benchmarkAudioEncode(
            durationSec: 10.0,
            config: BenchmarkConfig(warmup: 3, iterations: 5)
        )
        return (result.meanTimeMs < 100, result.meanTimeMs)
    }
}
