// BenchmarkFixtures.swift
// Standardized test inputs for reproducible benchmarks.

import Foundation
import MLX

/// Standard audio fixture configurations.
public enum AudioFixture: String, CaseIterable, Sendable {
    case short10s = "audio_10s_44100hz_stereo"
    case medium30s = "audio_30s_16000hz_mono"
    case long3min = "audio_3min_44100hz_stereo"

    public var sampleRate: Int {
        switch self {
        case .short10s, .long3min: return 44100
        case .medium30s: return 16000
        }
    }

    public var channels: Int {
        switch self {
        case .short10s, .long3min: return 2
        case .medium30s: return 1
        }
    }

    public var durationSec: Double {
        switch self {
        case .short10s: return 10.0
        case .medium30s: return 30.0
        case .long3min: return 180.0
        }
    }

    public var numSamples: Int {
        Int(durationSec * Double(sampleRate))
    }
}

/// Fixture generation utilities.
public enum BenchmarkFixtures {

    /// Default random seed for reproducibility.
    public static let defaultSeed: UInt64 = 42

    /// Generate synthetic audio fixture with deterministic values.
    public static func generateAudio(
        fixture: AudioFixture,
        seed: UInt64 = defaultSeed
    ) -> MLXArray {
        MLXRandom.seed(seed)
        let audio = MLXRandom.normal([1, fixture.channels, fixture.numSamples])
        eval(audio)
        return audio
    }

    /// Generate audio with custom parameters.
    public static func generateAudio(
        durationSec: Double,
        sampleRate: Int = 44100,
        channels: Int = 2,
        seed: UInt64 = defaultSeed
    ) -> MLXArray {
        let numSamples = Int(durationSec * Double(sampleRate))
        MLXRandom.seed(seed)
        let audio = MLXRandom.normal([1, channels, numSamples])
        eval(audio)
        return audio
    }

    /// Generate Whisper mel spectrogram input.
    public static func generateWhisperMel(
        durationSec: Double = 30.0,
        nMels: Int = 128,
        hopLength: Int = 160,
        seed: UInt64 = defaultSeed
    ) -> MLXArray {
        let numFrames = Int(durationSec * 16000 / Double(hopLength))
        MLXRandom.seed(seed)
        let mel = MLXRandom.normal([1, nMels, numFrames])
        eval(mel)
        return mel
    }

    /// Generate CLAP audio input (48kHz mono).
    public static func generateCLAPAudio(
        durationSec: Double = 10.0,
        seed: UInt64 = defaultSeed
    ) -> MLXArray {
        let sampleRate = 48000
        let numSamples = Int(durationSec * Double(sampleRate))
        MLXRandom.seed(seed)
        let audio = MLXRandom.normal([1, numSamples])
        eval(audio)
        return audio
    }

    /// Generate EnCodec codes for decode benchmarks.
    public static func generateEnCodecCodes(
        durationSec: Double = 5.0,
        numCodebooks: Int = 8,
        codebookSize: Int = 1024,
        frameRate: Int = 75,
        seed: UInt64 = defaultSeed
    ) -> MLXArray {
        let numFrames = Int(durationSec * Double(frameRate))
        MLXRandom.seed(seed)
        let codes = MLXRandom.randInt(
            low: 0,
            high: codebookSize,
            [1, numCodebooks, numFrames]
        )
        eval(codes)
        return codes
    }

    /// Generate MusicGen encoder hidden states.
    public static func generateMusicGenEncoderStates(
        textLength: Int = 50,
        hiddenSize: Int = 1024,
        seed: UInt64 = defaultSeed
    ) -> MLXArray {
        MLXRandom.seed(seed)
        let states = MLXRandom.normal([1, textLength, hiddenSize])
        eval(states)
        return states
    }
}

/// Audio durations for benchmark sweeps.
public enum BenchmarkDurations {
    public static let htdemucs: [Double] = [10.0, 30.0, 60.0, 180.0]
    public static let whisper: [Double] = [10.0, 30.0, 60.0]
    public static let clap: [Double] = [1.0, 5.0, 10.0, 30.0]
    public static let musicgen: [Double] = [5.0, 10.0, 20.0]
    public static let encodec: [Double] = [1.0, 5.0, 10.0]
}

/// Batch sizes for throughput benchmarks.
public enum BenchmarkBatchSizes {
    public static let standard: [Int] = [1, 4, 8, 16]
    public static let small: [Int] = [1, 4, 8]
}
