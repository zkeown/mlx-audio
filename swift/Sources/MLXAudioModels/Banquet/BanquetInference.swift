// BanquetInference.swift
// Inference utilities for Banquet query-based source separation.
//
// Provides chunked inference for processing long audio with overlap-add blending.

import Foundation
import MLX
import MLXAudioPrimitives

// MARK: - Query Preparation

/// Prepare query mel spectrogram for PaSST encoder.
///
/// Converts query audio to mel spectrogram in the format expected by PaSST.
///
/// - Parameters:
///   - queryAudio: Query audio [C, T] or [T] (at sampleRate)
///   - sampleRate: Input sample rate
///   - targetSampleRate: PaSST expected sample rate (32kHz)
///   - nMels: Number of mel bins
///   - nFFT: FFT size
///   - hopLength: Hop length
///   - winLength: Window length
///   - fMax: Maximum frequency for mel filterbank
/// - Returns: Mel spectrogram [1, 1, n_mels, time]
public func prepareQueryMel(
    _ queryAudio: MLXArray,
    sampleRate: Int = 44100,
    targetSampleRate: Int = 32000,
    nMels: Int = 128,
    nFFT: Int = 1024,
    hopLength: Int = 320,
    winLength: Int = 800,
    fMax: Float = 16000.0
) throws -> MLXArray {
    var audio = queryAudio

    // Convert to mono if stereo
    if audio.ndim == 2 {
        audio = mean(audio, axis: 0)
    }

    // Resample to target sample rate if needed
    if sampleRate != targetSampleRate {
        audio = resampleAudio(audio, originalRate: sampleRate, targetRate: targetSampleRate)
    }

    // Compute mel spectrogram
    let melConfig = MelConfig(
        sampleRate: targetSampleRate,
        nMels: nMels,
        fMin: 0.0,
        fMax: fMax
    )

    let stftConfig = STFTConfig(
        nFFT: nFFT,
        hopLength: hopLength,
        winLength: winLength
    )

    var mel = try melspectrogram(
        audio,
        nFFT: nFFT,
        hopLength: hopLength,
        stftConfig: stftConfig,
        melConfig: melConfig,
        power: 2.0
    )

    // Add batch and channel dimensions: [n_mels, time] -> [1, 1, n_mels, time]
    mel = mel.expandedDimensions(axes: [0, 1])

    return mel
}

// MARK: - Chunked Inference

/// Apply Banquet model with chunked inference.
///
/// For long audio, the input is split into overlapping segments,
/// processed independently, and blended using overlap-add.
///
/// - Parameters:
///   - model: Banquet model
///   - mixture: Input mixture [C, T] or [B, C, T]
///   - queryEmbedding: Pre-computed query embedding [768] or [B, 768]
///   - segment: Segment duration in seconds (nil = 6.0 seconds default)
///   - overlap: Overlap ratio between segments (default: 0.25)
///   - split: Enable chunking (false = process entire audio at once)
///   - progressCallback: Optional progress callback function
/// - Returns: BanquetOutput with separated audio, spectrogram, and mask
public func applyBanquetModel(
    _ model: Banquet,
    mixture: MLXArray,
    queryEmbedding: MLXArray,
    segment: Float? = nil,
    overlap: Float = 0.25,
    split: Bool = true,
    progressCallback: ((Float) -> Void)? = nil
) throws -> BanquetOutput {
    // Add batch dimension if needed
    var mix = mixture
    var queryEmb = queryEmbedding
    var squeezeBatch = false

    if mix.ndim == 2 {
        mix = mix.expandedDimensions(axis: 0)
        squeezeBatch = true
    }

    if queryEmb.ndim == 1 {
        queryEmb = queryEmb.expandedDimensions(axis: 0)
    }

    let shape = mix.shape
    let T = shape[2]

    let segmentSeconds = segment ?? 6.0
    let segmentSamples = Int(segmentSeconds * Float(model.config.sampleRate))

    let result: BanquetOutput
    if !split || T <= segmentSamples {
        // Process in single pass
        result = try model(mix, queryEmbedding: queryEmb)
    } else {
        // Chunked processing with overlap-add
        result = try chunkedInference(
            model: model,
            mixture: mix,
            queryEmbedding: queryEmb,
            segmentSamples: segmentSamples,
            overlap: overlap,
            progressCallback: progressCallback
        )
    }

    if squeezeBatch {
        return BanquetOutput(
            audio: result.audio[0],
            spectrogram: result.spectrogram[0],
            mask: result.mask[0]
        )
    }

    return result
}

/// Process long audio in overlapping chunks.
///
/// Uses overlap-add blending strategy to ensure smooth transitions.
private func chunkedInference(
    model: Banquet,
    mixture: MLXArray,
    queryEmbedding: MLXArray,
    segmentSamples: Int,
    overlap: Float,
    progressCallback: ((Float) -> Void)?
) throws -> BanquetOutput {
    let shape = mixture.shape
    let B = shape[0]
    let C = shape[1]
    let T = shape[2]

    let overlapSamples = Int(Float(segmentSamples) * overlap)
    let stride = segmentSamples - overlapSamples

    // Output buffer for audio
    var out = MLXArray.zeros([B, C, T])
    var weightSum = MLXArray.zeros([B, C, T])

    // Triangular window for overlap-add
    let weight = createWeightWindow(length: segmentSamples)

    // Calculate number of chunks
    let numChunks: Int
    if T <= segmentSamples {
        numChunks = 1
    } else {
        numChunks = (T - overlapSamples + stride - 1) / stride
    }

    // Store last mask and spectrogram
    var lastMask: MLXArray = MLXArray.zeros([1])
    var lastSpec: MLXArray = MLXArray.zeros([1])

    for chunkIdx in 0..<numChunks {
        let offset = chunkIdx * stride
        let chunkEnd = Swift.min(offset + segmentSamples, T)
        let chunkLen = chunkEnd - offset

        // Extract chunk
        var chunk = mixture[0..., 0..., offset..<chunkEnd]

        // Pad if needed
        if chunkLen < segmentSamples {
            let padAmount = segmentSamples - chunkLen
            chunk = MLX.padded(chunk, widths: [.init((0, 0)), .init((0, 0)), .init((0, padAmount))])
        }

        // Apply model
        let chunkResult = try model(chunk, queryEmbedding: queryEmbedding)

        // Trim audio to actual length if we padded
        var chunkAudio = chunkResult.audio
        var chunkWeight: MLXArray
        if chunkLen < segmentSamples {
            chunkAudio = chunkAudio[0..., 0..., 0..<chunkLen]
            chunkWeight = weight[0..<chunkLen]
        } else {
            chunkWeight = weight
        }

        // Reshape weight for broadcasting: [1, 1, chunkLen]
        let chunkWeight3d = chunkWeight.reshaped([1, 1, -1])

        // Weighted accumulation
        let weightedChunk = chunkAudio * chunkWeight3d

        // Accumulate using slice update
        let currentOut = out[0..., 0..., offset..<chunkEnd]
        let currentWeightSum = weightSum[0..., 0..., offset..<chunkEnd]

        out = updateSlice3D(out, with: currentOut + weightedChunk, start: offset, end: chunkEnd)
        weightSum = updateSlice3D(
            weightSum,
            with: currentWeightSum + MLX.broadcast(chunkWeight3d, to: weightedChunk.shape),
            start: offset,
            end: chunkEnd
        )

        // Store last mask/spec for output
        lastMask = chunkResult.mask
        lastSpec = chunkResult.spectrogram

        // Progress callback
        if let callback = progressCallback {
            callback(Float(chunkIdx + 1) / Float(numChunks))
        }

        // Evaluate to avoid memory buildup
        eval(out, weightSum)
    }

    // Normalize by weights
    out = out / MLX.maximum(weightSum, MLXArray(1e-8))

    return BanquetOutput(
        audio: out,
        spectrogram: lastSpec,
        mask: lastMask
    )
}

/// Helper to update a slice of a 3D array along the last axis.
private func updateSlice3D(
    _ array: MLXArray,
    with values: MLXArray,
    start: Int,
    end: Int
) -> MLXArray {
    let shape = array.shape
    let T = shape[2]

    if start == 0 && end == T {
        return values
    }

    var parts: [MLXArray] = []

    if start > 0 {
        parts.append(array[0..., 0..., 0..<start])
    }

    parts.append(values)

    if end < T {
        parts.append(array[0..., 0..., end...])
    }

    return MLX.concatenated(parts, axis: 2)
}

/// Create triangular weight window for overlap-add.
///
/// - Parameters:
///   - length: Window length
///   - power: Power to raise the window to
/// - Returns: Weight window
private func createWeightWindow(length: Int, power: Float = 1.0) -> MLXArray {
    let half = length / 2

    let rampUp = MLXArray(Array(1...half).map { Float($0) })
    let rampDown = MLXArray(Array(stride(from: length - half, to: 0, by: -1)).map { Float($0) })

    var weight = MLX.concatenated([rampUp, rampDown], axis: 0)
    weight = MLX.pow(weight / MLX.max(weight), MLXArray(power))

    return weight
}

/// Separate audio using query mel spectrogram.
///
/// High-level API that encodes the query and runs separation.
///
/// - Parameters:
///   - model: Banquet model
///   - mixture: Input mixture [C, T] or [B, C, T]
///   - queryMel: Query mel spectrogram [1, n_mels, time] or [B, 1, n_mels, time]
///   - segment: Segment duration in seconds (nil = 6.0 seconds default)
///   - overlap: Overlap ratio between segments
///   - split: Enable chunking
///   - progressCallback: Optional progress callback
/// - Returns: BanquetOutput with separated audio, spectrogram, and mask
public func separateWithQuery(
    model: Banquet,
    mixture: MLXArray,
    queryMel: MLXArray,
    segment: Float? = nil,
    overlap: Float = 0.25,
    split: Bool = true,
    progressCallback: ((Float) -> Void)? = nil
) throws -> BanquetOutput {
    // Add batch dimension to query if needed
    var qMel = queryMel
    if qMel.ndim == 3 {
        qMel = qMel.expandedDimensions(axis: 0)
    }

    // Encode query
    let queryEmbedding = model.encodeQuery(qMel)

    // Run separation
    return try applyBanquetModel(
        model,
        mixture: mixture,
        queryEmbedding: queryEmbedding,
        segment: segment,
        overlap: overlap,
        split: split,
        progressCallback: progressCallback
    )
}
