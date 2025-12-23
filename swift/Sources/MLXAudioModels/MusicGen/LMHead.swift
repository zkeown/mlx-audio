// LMHead.swift
// Language model head and delay pattern scheduler for MusicGen.

import Foundation
import MLX
import MLXFast
import MLXNN

/// Delay pattern scheduler for multi-codebook generation.
/// MusicGen uses a delay pattern where each codebook is offset by 1 timestep.
/// This creates hierarchical dependencies: codebook k depends on codebooks 0..k-1 at the same step.
public struct DelayPatternScheduler {

    let numCodebooks: Int
    let padTokenId: Int

    public init(numCodebooks: Int, padTokenId: Int) {
        self.numCodebooks = numCodebooks
        self.padTokenId = padTokenId
    }

    /// Apply delay pattern to input codes.
    /// Each codebook k is shifted right by k positions.
    /// - Parameter codes: Input codes [B, K, T]
    /// - Returns: Delayed codes [B, K, T + K - 1]
    public func applyDelayPattern(_ codes: MLXArray) -> MLXArray {
        let batchSize = codes.dim(0)
        let seqLength = codes.dim(2)
        let delayedLength = seqLength + numCodebooks - 1

        // Initialize with pad tokens
        var delayed = MLXArray.full([batchSize, numCodebooks, delayedLength], values: MLXArray(padTokenId))

        // Apply delays
        for k in 0 ..< numCodebooks {
            // Codebook k starts at position k
            let startIdx = k
            let endIdx = k + seqLength
            delayed[0..., k, startIdx ..< endIdx] = codes[0..., k, 0...]
        }

        return delayed
    }

    /// Revert delay pattern to recover original timing.
    /// - Parameter delayedCodes: Delayed codes [B, K, T_delayed]
    /// - Returns: Original codes [B, K, T] where T = T_delayed - K + 1
    public func revertDelayPattern(_ delayedCodes: MLXArray) -> MLXArray {
        let batchSize = delayedCodes.dim(0)
        let delayedLength = delayedCodes.dim(2)
        let originalLength = delayedLength - numCodebooks + 1

        guard originalLength > 0 else {
            return MLXArray.zeros([batchSize, numCodebooks, 0])
        }

        var codes = MLXArray.zeros([batchSize, numCodebooks, originalLength])

        // Extract undelayed codes
        for k in 0 ..< numCodebooks {
            let startIdx = k
            let endIdx = k + originalLength
            codes[0..., k, 0...] = delayedCodes[0..., k, startIdx ..< endIdx]
        }

        return codes
    }

    /// Get which codebooks are valid (not padding) at a given step.
    /// At step t, codebooks 0..min(t, K-1) have valid tokens.
    /// - Parameter step: Current generation step (0-indexed in delayed space)
    /// - Returns: Array of valid codebook indices
    public func getValidCodebooks(step: Int) -> [Int] {
        var valid: [Int] = []
        for k in 0 ..< numCodebooks {
            if step >= k {
                valid.append(k)
            }
        }
        return valid
    }

    /// Build the input for the next generation step.
    /// Combines previously generated tokens with appropriate delays.
    /// - Parameters:
    ///   - generatedCodes: All generated codes so far [B, K, T]
    ///   - step: Current step (0-indexed)
    /// - Returns: Input codes for the decoder [B, K, 1]
    public func buildInput(generatedCodes: MLXArray, step: Int) -> MLXArray {
        let batchSize = generatedCodes.dim(0)

        // Initialize with pad tokens
        var input = MLXArray.full([batchSize, numCodebooks, 1], values: MLXArray(padTokenId))

        // For step t, codebook k uses the token from position (t - k) if t >= k
        for k in 0 ..< numCodebooks {
            let sourceStep = step - k
            if sourceStep >= 0 && sourceStep < generatedCodes.dim(2) {
                input[0..., k, 0] = generatedCodes[0..., k, sourceStep]
            }
        }

        return input
    }
}

/// Language model head for MusicGen.
/// Uses K separate linear projections, one per codebook.
public class MusicGenLMHead: Module {

    @ModuleInfo(key: "linears") var linears: [Linear]

    let numCodebooks: Int
    let vocabSize: Int

    public init(config: MusicGenConfig) {
        self.numCodebooks = config.numCodebooks
        self.vocabSize = config.vocabSize

        var heads: [Linear] = []
        for _ in 0 ..< numCodebooks {
            heads.append(Linear(config.hiddenSize, vocabSize, bias: false))
        }
        self._linears.wrappedValue = heads

        super.init()
    }

    /// Compute logits for all codebooks.
    /// - Parameter hiddenStates: Decoder output [B, T, D]
    /// - Returns: Logits [B, K, T, V]
    public func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        let batchSize = hiddenStates.dim(0)
        let seqLength = hiddenStates.dim(1)

        // Compute logits for each codebook
        var allLogits: [MLXArray] = []
        for linear in linears {
            let logits = linear(hiddenStates)  // [B, T, V]
            allLogits.append(logits)
        }

        // Stack along codebook dimension: [K, B, T, V] -> [B, K, T, V]
        let stacked = stacked(allLogits, axis: 0)  // [K, B, T, V]
        return stacked.transposed(1, 0, 2, 3)  // [B, K, T, V]
    }

    /// Compute logits for specific codebooks only.
    /// Used during generation when only some codebooks are valid.
    /// - Parameters:
    ///   - hiddenStates: Decoder output [B, T, D]
    ///   - codebookIndices: Which codebooks to compute logits for
    /// - Returns: Dictionary mapping codebook index to logits [B, T, V]
    public func callAsFunction(
        _ hiddenStates: MLXArray,
        codebookIndices: [Int]
    ) -> [Int: MLXArray] {
        var result: [Int: MLXArray] = [:]
        for k in codebookIndices {
            result[k] = linears[k](hiddenStates)
        }
        return result
    }
}
