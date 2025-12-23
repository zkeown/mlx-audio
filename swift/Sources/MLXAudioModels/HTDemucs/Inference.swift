// Inference.swift
// Inference utilities for HTDemucs.

import Foundation
@preconcurrency import MLX

// MARK: - Model Protocol

/// Protocol for models that can perform source separation.
public protocol SeparationModel {
    var config: HTDemucsConfig { get }
    func callAsFunction(_ mix: MLXArray) -> MLXArray
}

extension HTDemucs: SeparationModel {}
extension BagOfModels: SeparationModel {}

// MARK: - Inference

/// Apply HTDemucs model with chunked inference.
///
/// For long audio, the input is split into overlapping segments,
/// processed independently, and blended using overlap-add.
///
/// Works with both single HTDemucs models and BagOfModels ensembles.
///
/// - Parameters:
///   - model: HTDemucs model or BagOfModels ensemble.
///   - mix: Input mixture `[C, T]` or `[B, C, T]`.
///   - segment: Segment duration in seconds (nil = use model default).
///   - overlap: Overlap ratio between segments.
///   - split: Enable chunking (false = process entire audio at once).
///   - progressCallback: Optional progress callback function.
/// - Returns: Separated stems `[S, C, T]` or `[B, S, C, T]`.
public func applyModel(
    _ model: SeparationModel,
    mix: MLXArray,
    segment: Float? = nil,
    overlap: Float = 0.25,
    split: Bool = true,
    progressCallback: ((Float) -> Void)? = nil
) -> MLXArray {
    var input = mix
    var squeezeBatch = false

    // Add batch dimension if needed
    if input.ndim == 2 {
        input = input.expandedDimensions(axis: 0)
        squeezeBatch = true
    }

    let T = input.shape[2]
    let segmentValue = segment ?? model.config.segment
    let segmentSamples = Int(segmentValue * Float(model.config.samplerate))

    var stems: MLXArray

    if !split || T <= segmentSamples {
        // Process in single pass
        stems = model.callAsFunction(input)
    } else {
        // Chunked processing with overlap-add
        stems = chunkedInference(
            model: model,
            mix: input,
            segmentSamples: segmentSamples,
            overlap: overlap,
            progressCallback: progressCallback
        )
    }

    if squeezeBatch {
        stems = stems.squeezed(axis: 0)
    }

    return stems
}

/// Process long audio in overlapping chunks.
///
/// Uses overlap-add blending strategy to ensure smooth transitions
/// between chunks.
///
/// - Parameters:
///   - model: HTDemucs model or ensemble.
///   - mix: Input mixture `[B, C, T]`.
///   - segmentSamples: Samples per segment.
///   - overlap: Overlap ratio.
///   - progressCallback: Optional progress callback.
/// - Returns: Separated stems `[B, S, C, T]`.
private func chunkedInference(
    model: SeparationModel,
    mix: MLXArray,
    segmentSamples: Int,
    overlap: Float,
    progressCallback: ((Float) -> Void)?
) -> MLXArray {
    let B = mix.shape[0]
    let C = mix.shape[1]
    let T = mix.shape[2]
    let S = model.config.num_sources

    let overlapSamples = Int(Float(segmentSamples) * overlap)
    let stride = segmentSamples - overlapSamples

    // Output buffer
    var out = MLXArray.zeros([B, S, C, T])
    var weightSum = MLXArray.zeros([B, S, C, T])

    // Triangular window for overlap-add
    let weight = createWeightWindow(length: segmentSamples)

    // Calculate number of chunks
    let numChunks: Int
    if T <= segmentSamples {
        numChunks = 1
    } else {
        numChunks = (T - overlapSamples + stride - 1) / stride
    }

    for chunkIdx in 0..<numChunks {
        let offset = chunkIdx * stride
        let chunkEnd = min(offset + segmentSamples, T)
        let chunkLen = chunkEnd - offset

        // Extract chunk
        var chunk = mix[0..., 0..., offset..<chunkEnd]

        // Pad if needed
        if chunkLen < segmentSamples {
            let padAmount = segmentSamples - chunkLen
            chunk = MLX.padded(chunk, widths: [[0, 0], [0, 0], [0, padAmount]])
        }

        // Apply model
        var chunkOut = model.callAsFunction(chunk)

        // Trim to actual length if we padded
        var chunkWeight: MLXArray
        if chunkLen < segmentSamples {
            chunkOut = chunkOut[0..., 0..., 0..., 0..<chunkLen]
            chunkWeight = weight[0..<chunkLen]
        } else {
            chunkWeight = weight
        }

        // Reshape weight for broadcasting: [1, 1, 1, chunkLen]
        let chunkWeight4D = chunkWeight.reshaped([1, 1, 1, -1])

        // Weighted accumulation
        let weightedChunk = chunkOut * chunkWeight4D

        // Accumulate into output buffers
        // Note: MLX Swift may not have .at().add() - use workaround
        let slicedOut = out[0..., 0..., 0..., offset..<chunkEnd]
        let newSlice = slicedOut + weightedChunk

        // Reconstruct output with updated slice
        if offset == 0 {
            let rest = out[0..., 0..., 0..., chunkEnd...]
            out = concatenated([newSlice, rest], axis: 3)
        } else if chunkEnd == T {
            let prefix = out[0..., 0..., 0..., 0..<offset]
            out = concatenated([prefix, newSlice], axis: 3)
        } else {
            let prefix = out[0..., 0..., 0..., 0..<offset]
            let suffix = out[0..., 0..., 0..., chunkEnd...]
            out = concatenated([prefix, newSlice, suffix], axis: 3)
        }

        // Same for weight sum
        let broadcastWeight = MLX.broadcast(chunkWeight4D, to: weightedChunk.shape)
        let slicedWeight = weightSum[0..., 0..., 0..., offset..<chunkEnd]
        let newWeightSlice = slicedWeight + broadcastWeight

        if offset == 0 {
            let rest = weightSum[0..., 0..., 0..., chunkEnd...]
            weightSum = concatenated([newWeightSlice, rest], axis: 3)
        } else if chunkEnd == T {
            let prefix = weightSum[0..., 0..., 0..., 0..<offset]
            weightSum = concatenated([prefix, newWeightSlice], axis: 3)
        } else {
            let prefix = weightSum[0..., 0..., 0..., 0..<offset]
            let suffix = weightSum[0..., 0..., 0..., chunkEnd...]
            weightSum = concatenated([prefix, newWeightSlice, suffix], axis: 3)
        }

        // Progress callback
        progressCallback?(Float(chunkIdx + 1) / Float(numChunks))

        // Evaluate to avoid memory buildup
        eval(out, weightSum)
    }

    // Normalize by weights
    out = out / maximum(weightSum, MLXArray(1e-8))

    return out
}

/// Create triangular weight window for overlap-add.
///
/// - Parameters:
///   - length: Window length.
///   - power: Power to raise the window to.
/// - Returns: Weight window.
private func createWeightWindow(length: Int, power: Float = 1.0) -> MLXArray {
    let half = length / 2

    // Ramp up from 1 to half
    let rampUp = MLXArray(1...half).asType(.float32)

    // Ramp down from (length - half) to 1
    let rampDownValues = stride(from: length - half, through: 1, by: -1).map { Float($0) }
    let rampDown = MLXArray(rampDownValues)

    var weight = concatenated([rampUp, rampDown], axis: 0)
    weight = pow(weight / weight.max(), power)

    return weight
}
