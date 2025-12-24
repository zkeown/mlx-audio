// AudioLoader.swift
// Utilities for loading and resampling audio files.

import Foundation
import AVFoundation
import MLX

/// Errors that can occur during audio loading.
public enum AudioLoadError: Error, LocalizedError {
    case fileNotFound(URL)
    case unsupportedFormat(String)
    case readError(Error)
    case resampleError(String)
    case emptyFile

    public var errorDescription: String? {
        switch self {
        case .fileNotFound(let url):
            return "Audio file not found: \(url.lastPathComponent)"
        case .unsupportedFormat(let format):
            return "Unsupported audio format: \(format)"
        case .readError(let error):
            return "Failed to read audio: \(error.localizedDescription)"
        case .resampleError(let message):
            return "Resample error: \(message)"
        case .emptyFile:
            return "Audio file is empty"
        }
    }
}

/// Information about a loaded audio file.
public struct AudioInfo: Sendable {
    public let sampleRate: Int
    public let channels: Int
    public let duration: TimeInterval
    public let frameCount: Int
}

/// Utilities for loading audio files.
public struct AudioLoader {

    /// Load an audio file and return as MLXArray.
    ///
    /// - Parameters:
    ///   - url: Path to the audio file.
    ///   - targetSampleRate: Desired sample rate (nil = use file's native rate).
    ///   - mono: Whether to mix down to mono.
    /// - Returns: Audio data as `[channels, samples]` MLXArray.
    public static func load(
        url: URL,
        targetSampleRate: Int? = nil,
        mono: Bool = false
    ) async throws -> MLXArray {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw AudioLoadError.fileNotFound(url)
        }

        // Load with AVAudioFile
        let audioFile: AVAudioFile
        do {
            audioFile = try AVAudioFile(forReading: url)
        } catch {
            throw AudioLoadError.readError(error)
        }

        let fileFormat = audioFile.processingFormat
        let fileSampleRate = Int(fileFormat.sampleRate)
        let frameCount = AVAudioFrameCount(audioFile.length)

        guard frameCount > 0 else {
            throw AudioLoadError.emptyFile
        }

        // Read file into buffer
        guard let buffer = AVAudioPCMBuffer(
            pcmFormat: fileFormat,
            frameCapacity: frameCount
        ) else {
            throw AudioLoadError.readError(NSError(domain: "AudioLoader", code: -1))
        }

        do {
            try audioFile.read(into: buffer)
        } catch {
            throw AudioLoadError.readError(error)
        }

        // Convert to Float array
        guard let floatData = buffer.floatChannelData else {
            throw AudioLoadError.unsupportedFormat("Non-float format")
        }

        let channelCount = Int(fileFormat.channelCount)
        let sampleCount = Int(buffer.frameLength)

        // Create MLXArray from buffer
        var samples: [[Float]] = []
        for ch in 0..<channelCount {
            let channelData = Array(UnsafeBufferPointer(start: floatData[ch], count: sampleCount))
            samples.append(channelData)
        }

        var audio: MLXArray
        if channelCount == 1 {
            audio = MLXArray(samples[0]).reshaped([1, -1])
        } else {
            // Stack channels: [C, T]
            let channelArrays = samples.map { MLXArray($0) }
            audio = MLX.stacked(channelArrays, axis: 0)
        }

        // Mix to mono if requested
        if mono && channelCount > 1 {
            audio = audio.mean(axis: 0, keepDims: true)
        }

        // Resample if needed
        if let targetRate = targetSampleRate, targetRate != fileSampleRate {
            audio = try await resample(
                audio,
                fromRate: fileSampleRate,
                toRate: targetRate
            )
        }

        return audio
    }

    /// Get information about an audio file without loading it.
    public static func info(url: URL) throws -> AudioInfo {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw AudioLoadError.fileNotFound(url)
        }

        let audioFile = try AVAudioFile(forReading: url)
        let format = audioFile.processingFormat

        return AudioInfo(
            sampleRate: Int(format.sampleRate),
            channels: Int(format.channelCount),
            duration: Double(audioFile.length) / format.sampleRate,
            frameCount: Int(audioFile.length)
        )
    }

    /// Resample audio to a target sample rate.
    ///
    /// Uses linear interpolation for simplicity. For production use,
    /// consider using a proper resampling algorithm.
    public static func resample(
        _ audio: MLXArray,
        fromRate: Int,
        toRate: Int
    ) async throws -> MLXArray {
        guard fromRate != toRate else {
            return audio
        }

        let ratio = Double(toRate) / Double(fromRate)
        let inputLength = audio.shape[1]
        let outputLength = Int(Double(inputLength) * ratio)

        guard outputLength > 0 else {
            throw AudioLoadError.resampleError("Invalid output length")
        }

        // Simple linear interpolation resampling
        // For each output sample, find the corresponding input position
        let indices = (0..<outputLength).map { Double($0) / ratio }

        var outputChannels: [MLXArray] = []
        let numChannels = audio.shape[0]

        for ch in 0..<numChannels {
            let channelData = audio[ch].asArray(Float.self)
            var outputSamples = [Float](repeating: 0, count: outputLength)

            for (outIdx, inPos) in indices.enumerated() {
                let inIdx = Int(inPos)
                let frac = Float(inPos - Double(inIdx))

                if inIdx + 1 < inputLength {
                    // Linear interpolation
                    outputSamples[outIdx] = channelData[inIdx] * (1 - frac) + channelData[inIdx + 1] * frac
                } else if inIdx < inputLength {
                    outputSamples[outIdx] = channelData[inIdx]
                }
            }

            outputChannels.append(MLXArray(outputSamples))
        }

        return MLX.stacked(outputChannels, axis: 0)
    }

    /// Save audio to a WAV file.
    public static func save(
        audio: MLXArray,
        to url: URL,
        sampleRate: Int
    ) throws {
        let channels = audio.shape[0]
        let samples = audio.shape[1]

        // Create audio format
        guard let format = AVAudioFormat(
            standardFormatWithSampleRate: Double(sampleRate),
            channels: AVAudioChannelCount(channels)
        ) else {
            throw AudioLoadError.unsupportedFormat("Cannot create format")
        }

        // Create buffer
        guard let buffer = AVAudioPCMBuffer(
            pcmFormat: format,
            frameCapacity: AVAudioFrameCount(samples)
        ) else {
            throw AudioLoadError.unsupportedFormat("Cannot create buffer")
        }

        buffer.frameLength = AVAudioFrameCount(samples)

        // Copy data to buffer
        guard let floatData = buffer.floatChannelData else {
            throw AudioLoadError.unsupportedFormat("Cannot get float data")
        }

        for ch in 0..<channels {
            let channelSamples = audio[ch].asArray(Float.self)
            for (i, sample) in channelSamples.enumerated() {
                floatData[ch][i] = sample
            }
        }

        // Write to file
        let audioFile = try AVAudioFile(
            forWriting: url,
            settings: format.settings,
            commonFormat: .pcmFormatFloat32,
            interleaved: false
        )

        try audioFile.write(from: buffer)
    }

    /// Convert audio samples to waveform data for visualization.
    ///
    /// - Parameters:
    ///   - audio: Audio array `[C, T]` or `[T]`.
    ///   - targetPoints: Number of points for the waveform.
    /// - Returns: Array of amplitude values normalized to [-1, 1].
    public static func waveformData(
        from audio: MLXArray,
        targetPoints: Int = 1000
    ) -> [Float] {
        // Take first channel if multi-channel
        let samples: MLXArray
        if audio.ndim == 2 {
            samples = audio[0]
        } else {
            samples = audio
        }

        let sampleCount = samples.shape[0]
        let samplesPerPoint = max(1, sampleCount / targetPoints)

        var waveform: [Float] = []
        let allSamples = samples.asArray(Float.self)

        for i in stride(from: 0, to: sampleCount, by: samplesPerPoint) {
            let end = min(i + samplesPerPoint, sampleCount)
            let chunk = allSamples[i..<end]

            // Use max absolute value for each chunk
            let maxAbs = chunk.map { abs($0) }.max() ?? 0
            waveform.append(maxAbs)
        }

        return waveform
    }
}
