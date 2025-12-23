// FeatureExtractor.swift
// Audio feature extraction for CLAP.
//
// Implements audio preprocessing compatible with HuggingFace's ClapFeatureExtractor,
// converting raw waveforms to mel spectrograms suitable for the CLAP audio encoder.

import Foundation
@preconcurrency import MLX
import MLXAudioPrimitives

// MARK: - CLAP Feature Extractor

/// Audio feature extractor for CLAP models.
///
/// Converts audio waveforms to log-mel spectrograms with the specific
/// preprocessing required by CLAP:
/// - 48kHz sample rate
/// - 64 mel bins
/// - 10ms hop length
/// - Up to 10 seconds of audio (with fusion for longer)
public struct CLAPFeatureExtractor: Sendable {
    /// Target sample rate.
    public let sampleRate: Int

    /// FFT window size.
    public let nFFT: Int

    /// Hop length for STFT.
    public let hopLength: Int

    /// Number of mel bins.
    public let nMels: Int

    /// Maximum audio length in seconds.
    public let maxLengthS: Float

    /// Maximum number of samples.
    public var maxSamples: Int {
        Int(maxLengthS * Float(sampleRate))
    }

    /// Expected number of frames for max length audio.
    public var maxFrames: Int {
        maxSamples / hopLength + 1
    }

    /// Creates a feature extractor with the given configuration.
    public init(
        sampleRate: Int = 48000,
        nFFT: Int = 1024,
        hopLength: Int = 480,
        nMels: Int = 64,
        maxLengthS: Float = 10.0
    ) {
        self.sampleRate = sampleRate
        self.nFFT = nFFT
        self.hopLength = hopLength
        self.nMels = nMels
        self.maxLengthS = maxLengthS
    }

    /// Create feature extractor from CLAP audio config.
    public static func fromConfig(_ config: CLAPAudioConfig) -> CLAPFeatureExtractor {
        CLAPFeatureExtractor(
            sampleRate: config.sampleRate,
            nFFT: config.nFFT,
            hopLength: config.hopLength,
            nMels: config.nMels,
            maxLengthS: config.maxLengthS
        )
    }

    /// Process audio waveform into CLAP-compatible features.
    ///
    /// - Parameters:
    ///   - audio: Audio waveform [T] or [B, T]
    ///   - inputSampleRate: Sample rate of input audio (resamples if different)
    /// - Returns: Tuple of (inputFeatures, isLonger)
    ///   - inputFeatures: Mel spectrogram [B, 4, F, T] for fusion mode
    ///   - isLonger: Boolean array [B] indicating which samples need fusion
    public func callAsFunction(
        _ audio: MLXArray,
        inputSampleRate: Int? = nil
    ) throws -> (inputFeatures: MLXArray, isLonger: MLXArray) {
        var processedAudio = audio

        // Handle 1D input
        if processedAudio.ndim == 1 {
            processedAudio = processedAudio.expandedDimensions(axis: 0)
        }

        let batchSize = processedAudio.shape[0]
        let originalLength = processedAudio.shape[1]

        // Resample if needed
        if let sr = inputSampleRate, sr != sampleRate {
            processedAudio = resample(processedAudio, fromRate: sr, toRate: sampleRate)
        }

        let currentLength = processedAudio.shape[1]

        // Determine which samples are longer than max
        let isLongerBool = currentLength > maxSamples
        let isLongerArray = MLXArray(
            Array(repeating: isLongerBool ? Float(1) : Float(0), count: batchSize)
        ).reshaped([batchSize, 1])

        // Process each sample
        var allFeatures: [MLXArray] = []

        for b in 0..<batchSize {
            let sample = processedAudio[b]
            let sampleLength = sample.shape[0]

            let mel: MLXArray
            let channel0: MLXArray  // Global/downsampled
            let channel1: MLXArray  // Local chunk 1
            let channel2: MLXArray  // Local chunk 2
            let channel3: MLXArray  // Local chunk 3

            if sampleLength > maxSamples {
                // Long audio: create fusion features
                let fusionResult = try computeFusionMel(sample)
                channel0 = fusionResult.global
                channel1 = fusionResult.local1
                channel2 = fusionResult.local2
                channel3 = fusionResult.local3
            } else {
                // Short audio: pad/repeat and duplicate
                let paddedMel = try computePaddedMel(sample)
                channel0 = paddedMel
                channel1 = paddedMel
                channel2 = paddedMel
                channel3 = paddedMel
            }

            // Stack channels: [4, F, T]
            let stacked = MLX.stacked([channel0, channel1, channel2, channel3], axis: 0)
            allFeatures.append(stacked.expandedDimensions(axis: 0))
        }

        // Concatenate batch: [B, 4, F, T]
        let inputFeatures = MLX.concatenated(allFeatures, axis: 0)

        return (inputFeatures, isLongerArray)
    }

    /// Compute log-mel spectrogram.
    private func computeMel(_ audio: MLXArray) throws -> MLXArray {
        // Create mel config for CLAP
        let melConfig = MelConfig(
            sampleRate: sampleRate,
            nMels: nMels,
            fMin: 50.0,
            fMax: 14000.0,
            htk: true,  // CLAP uses HTK mel scale
            norm: .slaney
        )

        // Compute mel spectrogram
        var mel = try melspectrogram(
            audio,
            nFFT: nFFT,
            hopLength: hopLength,
            melConfig: melConfig,
            power: 2.0
        )

        // Convert to dB
        mel = powerToDb(mel, ref: 1.0, amin: 1e-10, topDb: 80.0)

        return mel  // [nMels, nFrames]
    }

    /// Compute mel for short audio with padding.
    private func computePaddedMel(_ audio: MLXArray) throws -> MLXArray {
        var processedAudio = audio
        let length = processedAudio.shape[0]

        // Pad with repeat if too short
        if length < maxSamples {
            let nRepeat = (maxSamples + length - 1) / length
            var repeated = processedAudio
            for _ in 1..<nRepeat {
                repeated = MLX.concatenated([repeated, processedAudio], axis: 0)
            }
            processedAudio = repeated
        }

        // Truncate or pad to exact length
        let currentLength = processedAudio.shape[0]
        if currentLength > maxSamples {
            processedAudio = processedAudio[0..<maxSamples]
        } else if currentLength < maxSamples {
            let padding = MLX.zeros([maxSamples - currentLength])
            processedAudio = MLX.concatenated([processedAudio, padding], axis: 0)
        }

        return try computeMel(processedAudio)
    }

    /// Compute fusion mel for long audio.
    private func computeFusionMel(_ audio: MLXArray) throws -> (
        global: MLXArray,
        local1: MLXArray,
        local2: MLXArray,
        local3: MLXArray
    ) {
        // Compute full mel
        let fullMel = try computeMel(audio)  // [nMels, totalFrames]
        let totalFrames = fullMel.shape[1]
        let chunkFrames = maxFrames

        // If audio is shorter than expected, just repeat
        if chunkFrames >= totalFrames {
            return (fullMel, fullMel, fullMel, fullMel)
        }

        // Get 3 random crops from different parts
        let rangeSize = (totalFrames - chunkFrames) / 3

        // Front chunk
        let frontStart = 0
        let frontEnd = min(frontStart + chunkFrames, totalFrames)
        let local1 = fullMel[0..., frontStart..<frontEnd]

        // Middle chunk
        let middleStart = rangeSize
        let middleEnd = min(middleStart + chunkFrames, totalFrames)
        let local2 = fullMel[0..., middleStart..<middleEnd]

        // Back chunk
        let backStart = 2 * rangeSize
        let backEnd = min(backStart + chunkFrames, totalFrames)
        let local3 = fullMel[0..., backStart..<backEnd]

        // Global: downsample full mel to chunkFrames
        // Transpose to [1, 1, F, T] for interpolation
        let fullMelExpanded = fullMel.expandedDimensions(axes: [0, 1])
        let globalMel = interpolate2DSimple(
            fullMelExpanded,
            targetShape: (nMels, chunkFrames)
        )
        let global = globalMel.squeezed(axes: [0, 1])

        return (global, local1, local2, local3)
    }

    /// Simple 2D bilinear interpolation for mel downsampling.
    private func interpolate2DSimple(_ x: MLXArray, targetShape: (Int, Int)) -> MLXArray {
        let shape = x.shape
        let H = shape[2]
        let W = shape[3]
        let (targetH, targetW) = targetShape

        if H == targetH && W == targetW {
            return x
        }

        // Simple bilinear interpolation
        let yCoords = MLX.linspace(Float(0), Float(H - 1), count: targetH)
        let xCoords = MLX.linspace(Float(0), Float(W - 1), count: targetW)

        let y0 = MLX.floor(yCoords).asType(.int32)
        let y1 = MLX.minimum(y0 + 1, MLXArray(Int32(H - 1)))
        let x0 = MLX.floor(xCoords).asType(.int32)
        let x1 = MLX.minimum(x0 + 1, MLXArray(Int32(W - 1)))

        let wy = (yCoords - y0.asType(.float32)).reshaped([1, 1, targetH, 1])
        let wx = (xCoords - x0.asType(.float32)).reshaped([1, 1, 1, targetW])

        // Sample corners
        let val00 = x[0..., 0..., y0, 0...][0..., 0..., 0..., x0]
        let val01 = x[0..., 0..., y0, 0...][0..., 0..., 0..., x1]
        let val10 = x[0..., 0..., y1, 0...][0..., 0..., 0..., x0]
        let val11 = x[0..., 0..., y1, 0...][0..., 0..., 0..., x1]

        let result = val00 * (1 - wy) * (1 - wx) +
                     val01 * (1 - wy) * wx +
                     val10 * wy * (1 - wx) +
                     val11 * wy * wx

        return result
    }

    /// Simple linear resampling.
    private func resample(_ audio: MLXArray, fromRate: Int, toRate: Int) -> MLXArray {
        if fromRate == toRate {
            return audio
        }

        let batchSize = audio.shape[0]
        let length = audio.shape[1]
        let newLength = length * toRate / fromRate

        var results: [MLXArray] = []
        for b in 0..<batchSize {
            let sample = audio[b]
            let indices = MLX.linspace(Float(0), Float(length - 1), count: newLength)
            let i0 = MLX.floor(indices).asType(.int32)
            let i1 = MLX.minimum(i0 + 1, MLXArray(Int32(length - 1)))
            let w = indices - i0.asType(.float32)

            let v0 = sample[i0]
            let v1 = sample[i1]
            let resampled = v0 * (1 - w) + v1 * w

            results.append(resampled.expandedDimensions(axis: 0))
        }

        return MLX.concatenated(results, axis: 0)
    }
}
