// AudioData.swift
// Core audio data type for mlx-audio.
//
// Provides a standardized container for audio data with common operations.

import Foundation
@preconcurrency import MLX

/// A container for audio data with associated sample rate.
///
/// AudioData is the fundamental type for representing audio in mlx-audio.
/// It wraps an MLX array containing audio samples and provides common
/// operations like resampling, channel conversion, and file I/O.
///
/// Example:
/// ```swift
/// let audio = AudioData(array: samples, sampleRate: 44100)
/// let mono = audio.toMono()
/// let resampled = audio.resample(to: 16000)
/// ```
///
/// ## Thread Safety
///
/// `AudioData` is marked `@unchecked Sendable` because:
/// 1. **Immutable properties**: Both `array` and `sampleRate` are `let` constants
/// 2. **MLXArray thread safety**: MLXArray operations are serialized through MLX's
///    GPU command queue. All array operations are dispatched to the GPU and execute
///    in-order, making concurrent access from multiple threads safe.
/// 3. **Value semantics**: All methods return new `AudioData` instances rather than
///    mutating self, ensuring no shared mutable state.
///
/// The `@unchecked` annotation is required because MLXArray is marked `@preconcurrency`
/// in the MLX-Swift bindings, but the underlying implementation is thread-safe.
public struct AudioData: @unchecked Sendable {
    /// The audio samples as an MLX array.
    ///
    /// Shape is typically:
    /// - (samples,) for mono audio
    /// - (channels, samples) for multi-channel audio
    public let array: MLXArray

    /// The sample rate in Hz.
    public let sampleRate: Int

    /// Creates a new AudioData instance.
    ///
    /// - Parameters:
    ///   - array: The audio samples.
    ///   - sampleRate: The sample rate in Hz.
    public init(array: MLXArray, sampleRate: Int) {
        self.array = array
        self.sampleRate = sampleRate
    }

    // MARK: - Properties

    /// Duration of the audio in seconds.
    public var duration: TimeInterval {
        let samples = array.shape.last ?? 0
        return Double(samples) / Double(sampleRate)
    }

    /// Number of audio channels.
    public var channels: Int {
        if array.ndim == 1 {
            return 1
        } else {
            return array.shape[0]
        }
    }

    /// Number of samples (per channel).
    public var sampleCount: Int {
        return array.shape.last ?? 0
    }

    /// Whether the audio is mono (single channel).
    public var isMono: Bool {
        return channels == 1
    }

    /// Whether the audio is stereo (two channels).
    public var isStereo: Bool {
        return channels == 2
    }

    // MARK: - Channel Operations

    /// Convert to mono by averaging channels.
    ///
    /// If already mono, returns self unchanged.
    ///
    /// - Returns: Mono audio data.
    public func toMono() -> AudioData {
        if isMono {
            return self
        }

        // Average across channel dimension (axis 0)
        let mono = MLX.mean(array, axis: 0)
        return AudioData(array: mono, sampleRate: sampleRate)
    }

    /// Convert mono to stereo by duplicating the channel.
    ///
    /// If already stereo or multi-channel, returns self unchanged.
    ///
    /// - Returns: Stereo audio data.
    public func toStereo() -> AudioData {
        if !isMono {
            return self
        }

        // Duplicate mono channel
        let stereo = MLX.stacked([array, array], axis: 0)
        return AudioData(array: stereo, sampleRate: sampleRate)
    }

    // MARK: - Resampling

    /// Resample audio to a target sample rate.
    ///
    /// Uses linear interpolation for resampling.
    /// For high-quality resampling, consider using specialized libraries.
    ///
    /// - Parameter targetRate: The target sample rate in Hz.
    /// - Returns: Resampled audio data.
    public func resample(to targetRate: Int) -> AudioData {
        if targetRate == sampleRate {
            return self
        }

        let ratio = Double(targetRate) / Double(sampleRate)
        let newLength = Int(Double(sampleCount) * ratio)

        // Simple linear interpolation
        let resampled = linearResample(array, newLength: newLength)

        return AudioData(array: resampled, sampleRate: targetRate)
    }

    /// Simple linear interpolation for resampling.
    private func linearResample(_ x: MLXArray, newLength: Int) -> MLXArray {
        let oldLength = x.shape.last ?? 0
        guard oldLength > 1 && newLength > 1 else {
            return x
        }

        // Create interpolation indices
        var indices = [Float](repeating: 0, count: newLength)
        let scale = Float(oldLength - 1) / Float(newLength - 1)
        for i in 0..<newLength {
            indices[i] = Float(i) * scale
        }

        let indicesArray = MLXArray(indices)

        // Floor and ceil indices
        let floorIndices = MLX.floor(indicesArray).asType(.int32)
        let ceilIndices = MLX.minimum(floorIndices + 1, MLXArray(Int32(oldLength - 1)))

        // Interpolation weights
        let weights = indicesArray - MLX.floor(indicesArray)

        // Gather values and interpolate
        if x.ndim == 1 {
            let floorValues = x.take(floorIndices, axis: 0)
            let ceilValues = x.take(ceilIndices, axis: 0)
            return floorValues * (1 - weights) + ceilValues * weights
        } else {
            // Multi-channel: interpolate each channel
            var channels = [MLXArray]()
            for c in 0..<x.shape[0] {
                let channel = x[c]
                let floorValues = channel.take(floorIndices, axis: 0)
                let ceilValues = channel.take(ceilIndices, axis: 0)
                let interpolated = floorValues * (1 - weights) + ceilValues * weights
                channels.append(interpolated.expandedDimensions(axis: 0))
            }
            return MLX.concatenated(channels, axis: 0)
        }
    }

    // MARK: - Slicing

    /// Extract a time slice of the audio.
    ///
    /// - Parameters:
    ///   - start: Start time in seconds.
    ///   - end: End time in seconds. If nil, extends to end of audio.
    /// - Returns: Sliced audio data.
    public func slice(start: TimeInterval, end: TimeInterval? = nil) -> AudioData {
        let startSample = Int(start * Double(sampleRate))
        let endSample: Int
        if let end = end {
            endSample = min(Int(end * Double(sampleRate)), sampleCount)
        } else {
            endSample = sampleCount
        }

        guard startSample < endSample else {
            // Return empty audio
            return AudioData(array: MLXArray.zeros([0]), sampleRate: sampleRate)
        }

        let sliced: MLXArray
        if array.ndim == 1 {
            sliced = array[startSample..<endSample]
        } else {
            sliced = array[0..., startSample..<endSample]
        }

        return AudioData(array: sliced, sampleRate: sampleRate)
    }

    // MARK: - Normalization

    /// Normalize audio to a peak value.
    ///
    /// - Parameter peak: Target peak value. Default: 1.0.
    /// - Returns: Normalized audio data.
    public func normalize(peak: Float = 1.0) -> AudioData {
        let maxAbs = MLX.max(MLX.abs(array))
        let scale = MLXArray(peak) / MLX.maximum(maxAbs, MLXArray(Float(1e-10)))
        return AudioData(array: array * scale, sampleRate: sampleRate)
    }

    // MARK: - Conversion

    /// Convert to NumPy-compatible array (as [Float]).
    ///
    /// - Returns: Audio samples as a flat Float array.
    public func toFloatArray() -> [Float] {
        return array.asArray(Float.self)
    }
}

// MARK: - CustomStringConvertible

extension AudioData: CustomStringConvertible {
    public var description: String {
        let channelStr = isMono ? "mono" : "\(channels) channels"
        return "AudioData(\(channelStr), \(sampleRate) Hz, \(String(format: "%.2f", duration))s)"
    }
}

// MARK: - Equatable

extension AudioData: Equatable {
    public static func == (lhs: AudioData, rhs: AudioData) -> Bool {
        return lhs.sampleRate == rhs.sampleRate &&
            lhs.array.shape == rhs.array.shape
        // Note: Full array comparison would require MLX.allClose
    }
}
