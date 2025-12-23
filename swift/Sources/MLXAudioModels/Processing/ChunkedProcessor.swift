// ChunkedProcessor.swift
// Chunked audio processing for memory-efficient inference.
//
// Processes long audio in overlapping chunks with smooth blending,
// enabling arbitrarily long audio within a fixed memory budget.

import Foundation
import MLX

// MARK: - Configuration

/// Window function for overlap-add blending.
public enum WindowFunction: String, Codable, Sendable {
    /// Linear ramp (triangular window)
    case triangular
    /// Hann window (raised cosine)
    case hann
    /// Rectangular (no windowing)
    case rectangular

    /// Create window array of given size.
    public func createWindow(size: Int) -> MLXArray {
        switch self {
        case .triangular:
            // Triangular: ramp up then down
            let half = size / 2
            let rampUp = MLXArray(stride(from: 0, to: half, by: 1).map { Float($0) / Float(half) })
            let rampDown = MLXArray(stride(from: half, to: 0, by: -1).map { Float($0) / Float(half) })
            return concatenated([rampUp, rampDown], axis: 0)

        case .hann:
            // Hann: 0.5 * (1 - cos(2*pi*n/N))
            let n = MLXArray(0..<size)
            return 0.5 * (1.0 - MLX.cos(2.0 * Float.pi * n / Float(size)))

        case .rectangular:
            return MLXArray.ones([size])
        }
    }
}

/// Configuration for chunked processing.
public struct ChunkConfig: Codable, Sendable {

    /// Duration of each chunk in seconds.
    public var chunkDurationSec: Float

    /// Overlap ratio between adjacent chunks (0.0 to 0.5).
    public var overlapRatio: Float

    /// Window function for blending overlapping regions.
    public var windowFunction: WindowFunction

    /// Maximum memory budget for processing (in MB).
    public var maxMemoryMB: UInt64?

    /// Sample rate (for calculating samples from seconds).
    public var sampleRate: Int

    public init(
        chunkDurationSec: Float = 6.0,
        overlapRatio: Float = 0.25,
        windowFunction: WindowFunction = .triangular,
        maxMemoryMB: UInt64? = nil,
        sampleRate: Int = 44100
    ) {
        precondition(overlapRatio >= 0 && overlapRatio <= 0.5,
                     "Overlap ratio must be between 0 and 0.5")
        self.chunkDurationSec = chunkDurationSec
        self.overlapRatio = overlapRatio
        self.windowFunction = windowFunction
        self.maxMemoryMB = maxMemoryMB
        self.sampleRate = sampleRate
    }

    /// Number of samples per chunk.
    public var chunkSamples: Int {
        Int(chunkDurationSec * Float(sampleRate))
    }

    /// Number of overlap samples.
    public var overlapSamples: Int {
        Int(Float(chunkSamples) * overlapRatio)
    }

    /// Stride between chunk starts.
    public var strideSamples: Int {
        chunkSamples - overlapSamples
    }

    // MARK: - Presets

    /// Default configuration for HTDemucs (6s chunks, 25% overlap).
    public static let htdemucs = ChunkConfig(
        chunkDurationSec: 6.0,
        overlapRatio: 0.25,
        windowFunction: .triangular,
        sampleRate: 44100
    )

    /// Configuration for Whisper (30s chunks for segment processing).
    public static let whisper = ChunkConfig(
        chunkDurationSec: 30.0,
        overlapRatio: 0.1,
        windowFunction: .hann,
        sampleRate: 16000
    )

    /// Configuration for CLAP (10s chunks for long audio embedding).
    public static let clap = ChunkConfig(
        chunkDurationSec: 10.0,
        overlapRatio: 0.2,
        windowFunction: .hann,
        sampleRate: 48000
    )

    /// Memory-constrained configuration for iPhone.
    public static let phoneConstrained = ChunkConfig(
        chunkDurationSec: 3.0,
        overlapRatio: 0.25,
        windowFunction: .triangular,
        maxMemoryMB: 2048,
        sampleRate: 44100
    )
}

// MARK: - Progress Tracking

/// Progress information for chunked processing.
public struct ChunkProgress: Sendable {
    /// Current chunk index (0-based).
    public let currentChunk: Int

    /// Total number of chunks.
    public let totalChunks: Int

    /// Fraction complete (0.0 to 1.0).
    public var progress: Float {
        Float(currentChunk + 1) / Float(totalChunks)
    }

    /// Estimated time remaining in seconds (if duration known).
    public var estimatedTimeRemaining: Float?
}

/// Progress callback type.
public typealias ChunkProgressCallback = (ChunkProgress) -> Void

// MARK: - Chunked Processor

/// Processes audio in chunks with overlap-add blending.
///
/// Example:
/// ```swift
/// let processor = ChunkedProcessor(config: .htdemucs)
///
/// let output = processor.process(audio) { chunk in
///     return model(chunk)
/// } progressCallback: { progress in
///     print("Progress: \(progress.progress * 100)%")
/// }
/// ```
public class ChunkedProcessor {

    /// Configuration for chunking.
    public let config: ChunkConfig

    /// Cached window for blending.
    private var window: MLXArray?

    public init(config: ChunkConfig = .htdemucs) {
        self.config = config
    }

    // MARK: - Main Processing

    /// Process audio in chunks with overlap-add.
    ///
    /// - Parameters:
    ///   - audio: Input audio [B, C, T] or [C, T] or [T]
    ///   - processChunk: Closure to process each chunk, receives [B, C, chunkT] returns same shape
    ///   - progressCallback: Optional callback for progress updates
    /// - Returns: Processed audio with same shape as input
    public func process(
        _ audio: MLXArray,
        processChunk: (MLXArray) throws -> MLXArray,
        progressCallback: ChunkProgressCallback? = nil
    ) rethrows -> MLXArray {
        // Normalize input shape to [B, C, T]
        let (normalizedAudio, originalShape) = normalizeShape(audio)
        let T = normalizedAudio.dim(2)

        // If audio is shorter than one chunk, process directly
        if T <= config.chunkSamples {
            let result = try processChunk(normalizedAudio)
            return denormalizeShape(result, originalShape: originalShape)
        }

        // Calculate number of chunks
        let numChunks = (T - config.overlapSamples + config.strideSamples - 1) / config.strideSamples
        let numChunksInt = max(1, numChunks)

        // Pre-allocate output buffer
        var output = MLXArray.zeros(normalizedAudio.shape)
        var weightSum = MLXArray.zeros(normalizedAudio.shape)

        // Get or create window
        let chunkWindow = getWindow(size: config.chunkSamples)

        let startTime = CFAbsoluteTimeGetCurrent()

        for chunkIdx in 0..<numChunksInt {
            let offset = chunkIdx * config.strideSamples
            let chunkEnd = min(offset + config.chunkSamples, T)
            let actualChunkSize = chunkEnd - offset

            // Extract chunk
            var chunk = normalizedAudio[0..., 0..., offset..<chunkEnd]

            // Pad if necessary (last chunk might be shorter)
            if actualChunkSize < config.chunkSamples {
                let padAmount = config.chunkSamples - actualChunkSize
                chunk = MLX.padded(chunk, widths: [[0, 0], [0, 0], [0, padAmount]])
            }

            // Process chunk
            var processedChunk = try processChunk(chunk)

            // Trim if we padded
            if actualChunkSize < config.chunkSamples {
                processedChunk = processedChunk[0..., 0..., 0..<actualChunkSize]
            }

            // Apply window for blending (trim window if chunk was trimmed)
            let chunkWindowTrimmed = actualChunkSize < config.chunkSamples
                ? chunkWindow[0..<actualChunkSize]
                : chunkWindow
            let windowExpanded = chunkWindowTrimmed.expandedDimensions(axes: [0, 1])

            let weightedChunk = processedChunk * windowExpanded

            // Accumulate using at[].add() pattern for memory efficiency
            output = output.at[0..., 0..., offset..<chunkEnd].add(weightedChunk)
            weightSum = weightSum.at[0..., 0..., offset..<chunkEnd].add(windowExpanded)

            // Evaluate to release intermediate tensors
            eval(output, weightSum)

            // Progress callback
            if let callback = progressCallback {
                let elapsed = CFAbsoluteTimeGetCurrent() - startTime
                let avgTimePerChunk = elapsed / Double(chunkIdx + 1)
                let remaining = Float(avgTimePerChunk) * Float(numChunksInt - chunkIdx - 1)

                callback(ChunkProgress(
                    currentChunk: chunkIdx,
                    totalChunks: numChunksInt,
                    estimatedTimeRemaining: remaining
                ))
            }
        }

        // Normalize by weight sum (avoid division by zero)
        let safeWeightSum = MLX.maximum(weightSum, MLXArray(1e-8))
        let result = output / safeWeightSum

        return denormalizeShape(result, originalShape: originalShape)
    }

    /// Process audio with async chunk processor.
    public func processAsync(
        _ audio: MLXArray,
        processChunk: @Sendable (MLXArray) async throws -> MLXArray,
        progressCallback: ChunkProgressCallback? = nil
    ) async rethrows -> MLXArray {
        // Normalize input shape to [B, C, T]
        let (normalizedAudio, originalShape) = normalizeShape(audio)
        let T = normalizedAudio.dim(2)

        // If audio is shorter than one chunk, process directly
        if T <= config.chunkSamples {
            let result = try await processChunk(normalizedAudio)
            return denormalizeShape(result, originalShape: originalShape)
        }

        // Calculate number of chunks
        let numChunks = (T - config.overlapSamples + config.strideSamples - 1) / config.strideSamples
        let numChunksInt = max(1, numChunks)

        // Pre-allocate output buffer
        var output = MLXArray.zeros(normalizedAudio.shape)
        var weightSum = MLXArray.zeros(normalizedAudio.shape)

        let chunkWindow = getWindow(size: config.chunkSamples)
        let startTime = CFAbsoluteTimeGetCurrent()

        for chunkIdx in 0..<numChunksInt {
            let offset = chunkIdx * config.strideSamples
            let chunkEnd = min(offset + config.chunkSamples, T)
            let actualChunkSize = chunkEnd - offset

            var chunk = normalizedAudio[0..., 0..., offset..<chunkEnd]

            if actualChunkSize < config.chunkSamples {
                let padAmount = config.chunkSamples - actualChunkSize
                chunk = MLX.padded(chunk, widths: [[0, 0], [0, 0], [0, padAmount]])
            }

            var processedChunk = try await processChunk(chunk)

            if actualChunkSize < config.chunkSamples {
                processedChunk = processedChunk[0..., 0..., 0..<actualChunkSize]
            }

            let chunkWindowTrimmed = actualChunkSize < config.chunkSamples
                ? chunkWindow[0..<actualChunkSize]
                : chunkWindow
            let windowExpanded = chunkWindowTrimmed.expandedDimensions(axes: [0, 1])

            let weightedChunk = processedChunk * windowExpanded

            output = output.at[0..., 0..., offset..<chunkEnd].add(weightedChunk)
            weightSum = weightSum.at[0..., 0..., offset..<chunkEnd].add(windowExpanded)

            eval(output, weightSum)

            if let callback = progressCallback {
                let elapsed = CFAbsoluteTimeGetCurrent() - startTime
                let avgTimePerChunk = elapsed / Double(chunkIdx + 1)
                let remaining = Float(avgTimePerChunk) * Float(numChunksInt - chunkIdx - 1)

                callback(ChunkProgress(
                    currentChunk: chunkIdx,
                    totalChunks: numChunksInt,
                    estimatedTimeRemaining: remaining
                ))
            }
        }

        let safeWeightSum = MLX.maximum(weightSum, MLXArray(1e-8))
        let result = output / safeWeightSum

        return denormalizeShape(result, originalShape: originalShape)
    }

    // MARK: - Helpers

    /// Get or create window array.
    private func getWindow(size: Int) -> MLXArray {
        if let w = window, w.dim(0) == size {
            return w
        }
        let w = config.windowFunction.createWindow(size: size)
        window = w
        return w
    }

    /// Normalize audio shape to [B, C, T].
    private func normalizeShape(_ audio: MLXArray) -> (MLXArray, [Int]) {
        let originalShape = audio.shape

        switch audio.ndim {
        case 1:
            // [T] -> [1, 1, T]
            return (audio.expandedDimensions(axes: [0, 1]), originalShape)
        case 2:
            // [C, T] -> [1, C, T]
            return (audio.expandedDimensions(axis: 0), originalShape)
        case 3:
            // Already [B, C, T]
            return (audio, originalShape)
        default:
            fatalError("Unsupported audio shape: \(originalShape)")
        }
    }

    /// Denormalize shape back to original.
    private func denormalizeShape(_ audio: MLXArray, originalShape: [Int]) -> MLXArray {
        switch originalShape.count {
        case 1:
            // [1, 1, T] -> [T]
            return audio.squeezed(axes: [0, 1])
        case 2:
            // [1, C, T] -> [C, T]
            return audio.squeezed(axis: 0)
        case 3:
            return audio
        default:
            return audio
        }
    }

    /// Calculate number of chunks for given audio length.
    public func numChunks(audioLength: Int) -> Int {
        if audioLength <= config.chunkSamples {
            return 1
        }
        return max(1, (audioLength - config.overlapSamples + config.strideSamples - 1) / config.strideSamples)
    }

    /// Estimate memory usage for processing.
    public func estimateMemory(audioLength: Int, channels: Int, sources: Int = 1) -> UInt64 {
        // Input buffer
        let inputBytes = UInt64(channels * audioLength * 4)  // float32

        // Output buffer (with sources)
        let outputBytes = UInt64(sources * channels * audioLength * 4)

        // Weight sum buffer
        let weightBytes = UInt64(sources * channels * audioLength * 4)

        // Chunk buffers (2x for input and output)
        let chunkBytes = UInt64(sources * channels * config.chunkSamples * 4 * 2)

        return inputBytes + outputBytes + weightBytes + chunkBytes
    }
}

// MARK: - Model-Specific Extensions

extension ChunkedProcessor {

    /// Create processor optimized for HTDemucs.
    public static func forHTDemucs(sampleRate: Int = 44100) -> ChunkedProcessor {
        ChunkedProcessor(config: ChunkConfig(
            chunkDurationSec: 6.0,
            overlapRatio: 0.25,
            windowFunction: .triangular,
            sampleRate: sampleRate
        ))
    }

    /// Create processor optimized for Whisper.
    public static func forWhisper() -> ChunkedProcessor {
        ChunkedProcessor(config: ChunkConfig(
            chunkDurationSec: 30.0,
            overlapRatio: 0.1,
            windowFunction: .hann,
            sampleRate: 16000
        ))
    }

    /// Create processor optimized for CLAP.
    public static func forCLAP() -> ChunkedProcessor {
        ChunkedProcessor(config: ChunkConfig(
            chunkDurationSec: 10.0,
            overlapRatio: 0.2,
            windowFunction: .hann,
            sampleRate: 48000
        ))
    }

    /// Create processor based on device profile.
    public static func forDevice(_ profile: DeviceProfile, model: String = "default") -> ChunkedProcessor {
        let config: ChunkConfig
        switch profile {
        case .phone:
            config = .phoneConstrained
        case .tablet:
            config = ChunkConfig(
                chunkDurationSec: 6.0,
                overlapRatio: 0.25,
                windowFunction: .triangular,
                maxMemoryMB: 4096,
                sampleRate: 44100
            )
        case .mac, .macPro:
            config = .htdemucs  // Full size chunks on Mac/Mac Pro
        }
        return ChunkedProcessor(config: config)
    }
}
