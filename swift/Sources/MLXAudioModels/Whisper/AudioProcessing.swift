// AudioProcessing.swift
// Audio processing utilities for Whisper.

import Foundation
import MLX
import MLXAudioPrimitives

// MARK: - Mel Spectrogram

/// Compute log-mel spectrogram for Whisper.
///
/// - Parameters:
///   - audio: Audio waveform [T] or [B, T]
///   - nMels: Number of mel filterbank bins (80 or 128)
///   - nFft: FFT window size
///   - hopLength: STFT hop length
///   - sampleRate: Audio sample rate
/// - Returns: Log-mel spectrogram [B, nMels, T] or [nMels, T]
public func computeLogMelSpectrogram(
    audio: MLXArray,
    nMels: Int = 80,
    nFft: Int = 400,
    hopLength: Int = 160,
    sampleRate: Int = 16000
) -> MLXArray {
    // Ensure audio is 2D [B, T]
    var x = audio
    let wasBatched = x.ndim == 2
    if !wasBatched {
        x = x.expandedDimensions(axis: 0)
    }

    // Compute mel spectrogram using MLXAudioPrimitives
    let melConfig = MelConfig(
        sampleRate: sampleRate,
        nMels: nMels
    )
    var mel = try! melspectrogram(
        x,
        nFFT: nFft,
        hopLength: hopLength,
        melConfig: melConfig
    )

    // Apply log scaling with clamping
    // log10(max(mel, 1e-10))
    let clampedMel = maximum(mel, MLXArray(1e-10))
    mel = log10(clampedMel)

    // Normalize to [-1, 1] range (Whisper-specific normalization)
    // This matches the Python implementation's log_mel normalization
    let maxVal: Float = 4.0  // Approximate max log mel value
    mel = clip(mel, min: -maxVal, max: maxVal) / maxVal

    // Return in original batch format
    if !wasBatched {
        mel = mel.squeezed(axis: 0)
    }

    return mel
}

// MARK: - Audio Utilities

/// Pad or trim audio to a specific length.
///
/// - Parameters:
///   - audio: Audio waveform [T] or [B, T]
///   - length: Target length in samples
/// - Returns: Padded or trimmed audio
public func padOrTrim(_ audio: MLXArray, length: Int) -> MLXArray {
    var x = audio

    // Handle batched input
    let wasBatched = x.ndim == 2
    if !wasBatched {
        x = x.expandedDimensions(axis: 0)
    }

    let currentLength = x.dim(-1)

    if currentLength > length {
        // Trim to length
        x = x[0..., 0..<length]
    } else if currentLength < length {
        // Pad with zeros
        let padAmount = length - currentLength
        let padding = MLXArray.zeros([x.dim(0), padAmount], dtype: x.dtype)
        x = concatenated([x, padding], axis: -1)
    }

    // Return in original format
    if !wasBatched {
        x = x.squeezed(axis: 0)
    }

    return x
}

/// Pad or trim mel spectrogram to a specific number of frames.
///
/// - Parameters:
///   - mel: Mel spectrogram [nMels, T] or [B, nMels, T]
///   - frames: Target number of frames
/// - Returns: Padded or trimmed mel spectrogram
public func padOrTrimMel(_ mel: MLXArray, frames: Int) -> MLXArray {
    var x = mel

    // Handle batched input
    let wasBatched = x.ndim == 3
    if !wasBatched {
        x = x.expandedDimensions(axis: 0)
    }

    let currentFrames = x.dim(-1)

    if currentFrames > frames {
        // Trim to frames
        x = x[0..., 0..., 0..<frames]
    } else if currentFrames < frames {
        // Pad with zeros (or minimum value for log mel)
        let padAmount = frames - currentFrames
        let nMels = x.dim(1)
        let batchSize = x.dim(0)
        let padding = MLXArray.full([batchSize, nMels, padAmount], values: MLXArray(-1.0), dtype: x.dtype)
        x = concatenated([x, padding], axis: -1)
    }

    // Return in original format
    if !wasBatched {
        x = x.squeezed(axis: 0)
    }

    return x
}

/// Resample audio to a target sample rate.
///
/// - Parameters:
///   - audio: Audio waveform
///   - originalRate: Original sample rate
///   - targetRate: Target sample rate
/// - Returns: Resampled audio
public func resampleAudio(
    _ audio: MLXArray,
    originalRate: Int,
    targetRate: Int
) -> MLXArray {
    guard originalRate != targetRate else { return audio }

    let ratio = Float(targetRate) / Float(originalRate)
    let originalLength = audio.dim(-1)
    let targetLength = Int(Float(originalLength) * ratio)

    // Simple linear interpolation resampling
    // For production use, consider proper sinc interpolation
    let indices = MLXArray(0..<targetLength).asType(.float32) / ratio
    let floorIndices = floor(indices).asType(.int32)
    let ceilIndices = minimum(floorIndices + 1, MLXArray(Int32(originalLength - 1)))
    let weights = indices - floor(indices)

    let floorValues = audio[floorIndices]
    let ceilValues = audio[ceilIndices]

    return floorValues * (1 - weights) + ceilValues * weights
}

/// Convert stereo audio to mono.
///
/// - Parameter audio: Audio waveform [C, T] where C is number of channels
/// - Returns: Mono audio [T]
public func toMono(_ audio: MLXArray) -> MLXArray {
    guard audio.ndim == 2 else { return audio }
    return mean(audio, axis: 0)
}

/// Normalize audio to [-1, 1] range.
///
/// - Parameter audio: Audio waveform
/// - Returns: Normalized audio
public func normalizeAudio(_ audio: MLXArray) -> MLXArray {
    let maxVal = max(abs(audio))
    let maxScalar = maxVal.item(Float.self)
    guard maxScalar > 0 else { return audio }
    return audio / maxScalar
}

// MARK: - Whisper Constants

/// Default chunk length in samples (30 seconds at 16kHz).
public let whisperChunkSamples = 30 * 16000

/// Default number of mel frames per chunk (after STFT).
public let whisperMelFrames = whisperChunkSamples / 160

/// Default number of mel frames after encoder conv (with stride 2).
public let whisperEncoderFrames = whisperMelFrames / 2
