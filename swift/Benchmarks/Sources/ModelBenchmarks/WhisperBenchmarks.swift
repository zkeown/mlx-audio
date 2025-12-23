// WhisperBenchmarks.swift
// Benchmarks for Whisper speech recognition model.

import Foundation
import MLX
import MLXAudioModels
import BenchmarkKit

/// Whisper benchmark suite.
public struct WhisperBenchmarks {

    /// Run all Whisper benchmarks.
    public static func runAll(config: BenchmarkConfig) throws -> [BenchmarkResult] {
        var results: [BenchmarkResult] = []

        // Encoder benchmark
        results.append(try benchmarkEncoder(config: config))

        // Full transcription at different durations
        for duration in BenchmarkDurations.whisper {
            results.append(try benchmarkTranscription(durationSec: duration, config: config))
        }

        return results
    }

    /// Benchmark Whisper audio encoder.
    public static func benchmarkEncoder(config: BenchmarkConfig) throws -> BenchmarkResult {
        let whisperConfig = WhisperConfig()
        let model = WhisperModel(config: whisperConfig)

        // Generate mel spectrogram input (30s audio)
        let mel = BenchmarkFixtures.generateWhisperMel(durationSec: 30.0)

        let timing = try BenchmarkMetrics.measureTime(
            warmup: config.warmup,
            iterations: config.iterations
        ) {
            let _ = model.encode(mel)
        }

        return BenchmarkMetrics.createResult(
            name: "whisper_encoder",
            timing: timing,
            audioDurationSec: 30.0,
            sampleRate: 16000,
            iterations: config.iterations,
            params: [
                "n_mels": .int(128),
                "duration_sec": .double(30.0),
            ]
        )
    }

    /// Benchmark full Whisper transcription (simulated decode loop).
    public static func benchmarkTranscription(
        durationSec: Double,
        config: BenchmarkConfig
    ) throws -> BenchmarkResult {
        let whisperConfig = WhisperConfig()
        let model = WhisperModel(config: whisperConfig)

        // Generate mel spectrogram
        let mel = BenchmarkFixtures.generateWhisperMel(durationSec: durationSec)

        // Estimate token count (roughly 1 token per 20ms of speech)
        let estimatedTokens = min(Int(durationSec * 50), 448)  // Max 448 tokens for Whisper

        let timing = try BenchmarkMetrics.measureTime(
            warmup: min(config.warmup, 2),
            iterations: config.iterations
        ) {
            // Encode audio
            let audioFeatures = model.encode(mel)

            // Simulate autoregressive decoding
            var tokens = MLXArray([50258])  // Start-of-transcript token
            var kvCache: [(MLXArray, MLXArray)]? = nil

            for _ in 0..<estimatedTokens {
                let tokenInput = tokens.reshaped([1, -1])
                let (logits, newCache) = model.decode(
                    tokens: tokenInput,
                    audioFeatures: audioFeatures,
                    kvCache: kvCache
                )
                kvCache = newCache

                // Greedy decode
                let nextToken = argMax(logits[0, -1, 0...], axis: -1)
                tokens = concatenated([tokens, nextToken.reshaped([1])], axis: 0)
                eval(tokens)
            }
        }

        return BenchmarkMetrics.createResult(
            name: "whisper_transcribe_\(Int(durationSec))s",
            timing: timing,
            audioDurationSec: durationSec,
            sampleRate: 16000,
            iterations: config.iterations,
            params: [
                "duration_sec": .double(durationSec),
                "estimated_tokens": .int(estimatedTokens),
            ]
        )
    }

    /// Check if target performance is met (30s audio < 3s on M2 Pro).
    public static func checkTargetPerformance() throws -> (passed: Bool, timeMs: Double) {
        let result = try benchmarkTranscription(
            durationSec: 30.0,
            config: BenchmarkConfig(warmup: 1, iterations: 1)
        )
        return (result.meanTimeMs < 3000, result.meanTimeMs)
    }
}
