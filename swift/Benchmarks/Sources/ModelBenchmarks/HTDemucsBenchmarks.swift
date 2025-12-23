// HTDemucsBenchmarks.swift
// Benchmarks for HTDemucs source separation model.

import Foundation
import MLX
import MLXAudioModels
import BenchmarkKit

/// HTDemucs benchmark suite.
public struct HTDemucsBenchmarks {

    /// Run all HTDemucs benchmarks.
    public static func runAll(config: BenchmarkConfig) throws -> [BenchmarkResult] {
        var results: [BenchmarkResult] = []

        // Layer-level benchmarks (fast, good for profiling)
        results.append(try benchmarkEncoderLayer(config: config))
        results.append(try benchmarkDecoderLayer(config: config))

        // Full model at different durations
        for duration in BenchmarkDurations.htdemucs {
            results.append(try benchmarkFullModel(durationSec: duration, config: config))
        }

        return results
    }

    /// Benchmark a single encoder layer.
    public static func benchmarkEncoderLayer(config: BenchmarkConfig) throws -> BenchmarkResult {
        let encoder = HEncLayer(
            chin: 4,
            chout: 48,
            kernelSize: 8,
            stride: 4,
            freq: true,
            dconvDepth: 2,
            dconvCompress: 8
        )

        // Input in NHWC format: [B, F, T, C]
        let input = MLXRandom.normal([1, 512, 216, 4])
        eval(input)

        let timing = try BenchmarkMetrics.measureTime(
            warmup: config.warmup,
            iterations: config.iterations
        ) {
            let _ = encoder(input)
        }

        return BenchmarkMetrics.createResult(
            name: "htdemucs_encoder_layer",
            timing: timing,
            iterations: config.iterations,
            params: [
                "chin": 4,
                "chout": 48,
                "input_shape": "[1, 512, 216, 4]",
            ]
        )
    }

    /// Benchmark a single decoder layer.
    public static func benchmarkDecoderLayer(config: BenchmarkConfig) throws -> BenchmarkResult {
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

        // Input in NHWC format: [B, F, T, C]
        let input = MLXRandom.normal([1, 128, 54, 48])
        let skip = MLXRandom.normal([1, 128, 54, 48])
        eval(input, skip)

        let timing = try BenchmarkMetrics.measureTime(
            warmup: config.warmup,
            iterations: config.iterations
        ) {
            let _ = decoder(input, skip: skip, length: 216)
        }

        return BenchmarkMetrics.createResult(
            name: "htdemucs_decoder_layer",
            timing: timing,
            iterations: config.iterations,
            params: [
                "chin": 48,
                "chout": 4,
                "input_shape": "[1, 128, 54, 48]",
            ]
        )
    }

    /// Benchmark full HTDemucs model at specified duration.
    public static func benchmarkFullModel(
        durationSec: Double,
        config: BenchmarkConfig
    ) throws -> BenchmarkResult {
        let modelConfig = HTDemucsConfig()
        let model = HTDemucs(config: modelConfig)

        let numSamples = Int(durationSec * Double(modelConfig.samplerate))
        let audio = BenchmarkFixtures.generateAudio(
            durationSec: durationSec,
            sampleRate: modelConfig.samplerate,
            channels: modelConfig.audio_channels
        )

        let timing = try BenchmarkMetrics.measureTime(
            warmup: min(config.warmup, 2),  // Fewer warmups for long audio
            iterations: config.iterations
        ) {
            let _ = model(audio)
        }

        return BenchmarkMetrics.createResult(
            name: "htdemucs_full_\(Int(durationSec))s",
            timing: timing,
            audioDurationSec: durationSec,
            sampleRate: modelConfig.samplerate,
            iterations: config.iterations,
            params: [
                "duration_sec": .double(durationSec),
                "sample_rate": .int(modelConfig.samplerate),
                "num_sources": .int(modelConfig.num_sources),
                "num_samples": .int(numSamples),
            ]
        )
    }

    /// Check if target performance is met (3min song < 5s on M2 Pro).
    public static func checkTargetPerformance() throws -> (passed: Bool, timeMs: Double) {
        let modelConfig = HTDemucsConfig()
        let model = HTDemucs(config: modelConfig)

        let durationSec = 180.0  // 3 minutes
        let audio = BenchmarkFixtures.generateAudio(
            durationSec: durationSec,
            sampleRate: modelConfig.samplerate,
            channels: modelConfig.audio_channels
        )

        // Warmup
        let shortAudio = BenchmarkFixtures.generateAudio(
            durationSec: 10.0,
            sampleRate: modelConfig.samplerate,
            channels: modelConfig.audio_channels
        )
        let _ = model(shortAudio)
        eval()

        // Measure
        let start = CFAbsoluteTimeGetCurrent()
        let _ = model(audio)
        eval()
        let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000

        return (elapsed < 5000, elapsed)
    }
}
