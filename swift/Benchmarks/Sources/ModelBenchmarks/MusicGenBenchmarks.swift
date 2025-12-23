// MusicGenBenchmarks.swift
// Benchmarks for MusicGen audio generation model.

import Foundation
import MLX
import MLXAudioModels
import BenchmarkKit

/// MusicGen benchmark suite.
public struct MusicGenBenchmarks {

    /// Run all MusicGen benchmarks.
    public static func runAll(config: BenchmarkConfig) throws -> [BenchmarkResult] {
        var results: [BenchmarkResult] = []

        // Decoder step benchmark
        results.append(try benchmarkDecoderStep(config: config))

        // Full generation at different durations
        for duration in BenchmarkDurations.musicgen {
            results.append(try benchmarkGeneration(durationSec: duration, config: config))
        }

        return results
    }

    /// Benchmark a single decoder step.
    public static func benchmarkDecoderStep(config: BenchmarkConfig) throws -> BenchmarkResult {
        let musicgenConfig = MusicGenConfig()
        let model = MusicGen(config: musicgenConfig)

        // Simulate encoder hidden states
        let encoderStates = BenchmarkFixtures.generateMusicGenEncoderStates(
            textLength: 50,
            hiddenSize: musicgenConfig.textHiddenSize
        )

        // Single token input
        MLXRandom.seed(42)
        let inputIds = MLXRandom.randInt(low: 0, high: 1024, [1, musicgenConfig.numCodebooks, 1])
        eval(inputIds, encoderStates)

        let timing = try BenchmarkMetrics.measureTime(
            warmup: config.warmup,
            iterations: config.iterations
        ) {
            let _ = model(inputIds: inputIds, encoderHiddenStates: encoderStates)
        }

        return BenchmarkMetrics.createResult(
            name: "musicgen_decoder_step",
            timing: timing,
            iterations: config.iterations,
            params: [
                "num_codebooks": .int(musicgenConfig.numCodebooks),
                "hidden_size": .int(musicgenConfig.hiddenSize),
            ]
        )
    }

    /// Benchmark full MusicGen generation.
    public static func benchmarkGeneration(
        durationSec: Double,
        config: BenchmarkConfig
    ) throws -> BenchmarkResult {
        let musicgenConfig = MusicGenConfig()
        let model = MusicGen(config: musicgenConfig)

        // Pre-compute encoder hidden states
        let encoderStates = BenchmarkFixtures.generateMusicGenEncoderStates(
            textLength: 50,
            hiddenSize: musicgenConfig.textHiddenSize
        )

        // Calculate number of tokens for target duration
        let maxTokens = Int(durationSec * Double(musicgenConfig.frameRate))

        let timing = try BenchmarkMetrics.measureTime(
            warmup: 1,  // Fewer warmups for slow generation
            iterations: min(config.iterations, 3)  // Limit iterations for long generations
        ) {
            // Simulate autoregressive generation
            var tokens = MLXRandom.randInt(
                low: 0,
                high: 1024,
                [1, musicgenConfig.numCodebooks, 1]
            )
            eval(tokens)

            for _ in 0..<maxTokens {
                let (logits, _) = model(inputIds: tokens, encoderHiddenStates: encoderStates)

                // Sample next token (simplified - just take argmax of last position)
                let lastLogits = logits[0..., 0..., .stride(from: -1), 0...]
                let nextTokens = argMax(lastLogits, axis: -1)
                tokens = concatenated([tokens, nextTokens.reshaped([1, musicgenConfig.numCodebooks, 1])], axis: 2)
                eval(tokens)
            }
        }

        let tokensPerSec = Double(maxTokens) / (timing.mean / 1000)

        return BenchmarkResult(
            name: "musicgen_generate_\(Int(durationSec))s",
            meanTimeMs: timing.mean,
            stdTimeMs: timing.std,
            minTimeMs: timing.min,
            maxTimeMs: timing.max,
            throughput: tokensPerSec,
            peakMemoryMB: timing.peakMemoryMB,
            realtimeFactor: (durationSec * 1000) / timing.mean,
            iterations: min(config.iterations, 3),
            params: [
                "duration_sec": .double(durationSec),
                "max_tokens": .int(maxTokens),
                "tokens_per_sec": .double(tokensPerSec),
            ]
        )
    }

    /// Check if target performance is met (10s generation < 15s on M2 Pro).
    public static func checkTargetPerformance() throws -> (passed: Bool, timeMs: Double) {
        let result = try benchmarkGeneration(
            durationSec: 10.0,
            config: BenchmarkConfig(warmup: 1, iterations: 1)
        )
        return (result.meanTimeMs < 15000, result.meanTimeMs)
    }
}
