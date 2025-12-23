// Inference.swift
// Generation utilities for MusicGen music synthesis.

import Foundation
import MLX
import MLXFast
import MLXRandom

/// Sampling parameters for music generation.
public struct MusicGenSamplingConfig {
    public var temperature: Float
    public var topK: Int
    public var topP: Float
    public var classifierFreeGuidance: Float

    public init(
        temperature: Float = 1.0,
        topK: Int = 250,
        topP: Float = 0.0,
        classifierFreeGuidance: Float = 3.0
    ) {
        self.temperature = temperature
        self.topK = topK
        self.topP = topP
        self.classifierFreeGuidance = classifierFreeGuidance
    }

    public static var `default`: MusicGenSamplingConfig { MusicGenSamplingConfig() }
}

/// Sample next token from logits using temperature, top-k, and top-p filtering.
/// - Parameters:
///   - logits: Unnormalized log probabilities [B, V] or [V]
///   - temperature: Temperature for scaling logits (higher = more random)
///   - topK: Keep only top-k highest probability tokens (0 = disabled)
///   - topP: Keep tokens with cumulative probability >= topP (0 = disabled)
/// - Returns: Sampled token indices [B] or scalar
public func sampleNextToken(
    logits: MLXArray,
    temperature: Float = 1.0,
    topK: Int = 0,
    topP: Float = 0.0
) -> MLXArray {
    var processedLogits = logits

    // Apply temperature
    if temperature != 1.0 && temperature > 0 {
        processedLogits = processedLogits / temperature
    }

    // Apply top-k filtering
    if topK > 0 {
        processedLogits = applyTopK(processedLogits, k: topK)
    }

    // Apply top-p (nucleus) filtering
    if topP > 0 && topP < 1.0 {
        processedLogits = applyTopP(processedLogits, p: topP)
    }

    // Sample from distribution
    if temperature == 0 {
        // Greedy decoding
        return argMax(processedLogits, axis: -1)
    } else {
        // Categorical sampling
        return categorical(softmax(processedLogits, axis: -1))
    }
}

/// Apply top-k filtering to logits.
/// Sets all logits outside top-k to -infinity.
private func applyTopK(_ logits: MLXArray, k: Int) -> MLXArray {
    let vocabSize = logits.dim(-1)
    let actualK = min(k, vocabSize)

    // Get the k-th largest value using sorted
    let sortedVals = MLX.sorted(logits, axis: -1)
    let kthValue = sortedVals[0..., (-actualK)]

    // Mask values below threshold
    let mask = logits .< kthValue.expandedDimensions(axis: -1)
    let negInf = MLXArray(-Float.infinity)
    return MLX.where(mask, negInf, logits)
}

/// Apply top-p (nucleus) filtering to logits.
/// Keeps tokens until cumulative probability exceeds p.
private func applyTopP(_ logits: MLXArray, p: Float) -> MLXArray {
    // Sort in descending order
    let sortedIndices = argSort(logits, axis: -1)
    // Reverse to get descending order
    let reversedIndices = sortedIndices[0..., .stride(from: -1, to: nil, by: -1)]

    // Get sorted probabilities
    let probs = softmax(logits, axis: -1)
    let sortedProbs = take(probs, reversedIndices, axis: -1)

    // Compute cumulative probabilities
    let cumProbs = cumsum(sortedProbs, axis: -1)

    // Find cutoff (first position where cumsum > p)
    let mask = cumProbs .> p

    // Shift mask right by 1 to keep the token that crosses the threshold
    let shiftedMask = concatenated(
        [MLXArray.zeros([logits.dim(0), 1]).asType(.bool), mask[0..., ..<(-1)]],
        axis: -1
    )

    // Create mask in original order
    let unsortMask = scatter(shiftedMask, reversedIndices, axis: -1)

    let negInf = MLXArray(-Float.infinity)
    return MLX.where(unsortMask, negInf, logits)
}

/// Apply classifier-free guidance to conditional and unconditional logits.
/// - Parameters:
///   - conditionalLogits: Logits from conditioning [B, V]
///   - unconditionalLogits: Logits without conditioning [B, V]
///   - scale: Guidance scale (1.0 = no guidance, higher = stronger conditioning)
/// - Returns: Guided logits [B, V]
public func applyClassifierFreeGuidance(
    conditionalLogits: MLXArray,
    unconditionalLogits: MLXArray,
    scale: Float
) -> MLXArray {
    if scale == 1.0 {
        return conditionalLogits
    }
    // CFG formula: uncond + scale * (cond - uncond)
    return unconditionalLogits + scale * (conditionalLogits - unconditionalLogits)
}

/// Scatter values to original indices (inverse of gather/take).
private func scatter(_ values: MLXArray, _ indices: MLXArray, axis: Int) -> MLXArray {
    // Create output array
    let shape = values.shape
    var output = MLXArray.zeros(shape).asType(values.dtype)

    // Use advanced indexing to scatter
    // This is a simplified version - for production, use MLX's scatter operations
    for i in 0 ..< shape[0] {
        for j in 0 ..< shape[1] {
            let idx = indices[i, j].item(Int.self)
            output[i, idx] = values[i, j]
        }
    }

    return output
}

/// Categorical sampling from probability distribution.
/// - Parameter probs: Probability distribution [B, V] (should sum to 1)
/// - Returns: Sampled indices [B]
private func categorical(_ probs: MLXArray) -> MLXArray {
    // Generate uniform random values
    let uniform = MLXRandom.uniform(low: 0, high: 1, probs.shape.dropLast())

    // Compute cumulative probabilities
    let cumProbs = cumsum(probs, axis: -1)

    // Find first index where cumsum > uniform
    let expanded = uniform.expandedDimensions(axis: -1)
    let mask = cumProbs .> expanded

    // Get first True index for each batch
    return argMax(mask.asType(.int32), axis: -1)
}

/// Generation result containing audio codes.
public struct MusicGenGenerationResult {
    /// Generated audio codes [B, K, T]
    public let codes: MLXArray

    /// Number of tokens generated per codebook
    public var sequenceLength: Int { codes.dim(2) }

    /// Duration in seconds (approximate, based on frame rate)
    public func duration(frameRate: Int) -> Float {
        Float(sequenceLength) / Float(frameRate)
    }
}

/// Progress callback for generation.
public typealias MusicGenProgressCallback = (Int, Int) -> Void
