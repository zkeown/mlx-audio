// ParlerTTSLMHead.swift
// Language model head with delay pattern for Parler-TTS.

import Foundation
@preconcurrency import MLX
import MLXNN

// MARK: - Delay Pattern Scheduler

/// Manages the delay pattern for multi-codebook generation in Parler-TTS.
///
/// Different codebooks are offset in time during generation.
/// Codebook k has a delay of k timesteps. This allows the model to
/// generate multiple codebooks while respecting dependencies.
///
/// Example with 4 codebooks and 6 timesteps:
/// ```
/// With delay pattern:
///     t=0  t=1  t=2  t=3  t=4  t=5  t=6  t=7  t=8
/// k=0:  0    1    2    3    4    5    -    -    -
/// k=1:  -    0    1    2    3    4    5    -    -
/// k=2:  -    -    0    1    2    3    4    5    -
/// k=3:  -    -    -    0    1    2    3    4    5
/// ```
/// Where '-' represents padding tokens.
public struct ParlerTTSDelayPattern: Sendable {
    public let numCodebooks: Int
    public let padTokenId: Int32

    public init(numCodebooks: Int, padTokenId: Int32 = 1024) {
        self.numCodebooks = numCodebooks
        self.padTokenId = padTokenId
    }

    /// Apply delay pattern to codes.
    ///
    /// - Parameter codes: Input codes [B, K, T] where K is numCodebooks
    /// - Returns: Delayed codes [B, K, T + K - 1]
    public func applyDelayPattern(_ codes: MLXArray) -> MLXArray {
        let batchSize = codes.dim(0)
        let numCB = codes.dim(1)
        let seqLength = codes.dim(2)

        var delayedList: [MLXArray] = []

        for k in 0..<numCB {
            // Pad left with k padding tokens, right with (numCodebooks - 1 - k) tokens
            let leftPadCount = k
            let rightPadCount = numCodebooks - 1 - k

            let leftPad = MLXArray.full(
                [batchSize, 1, leftPadCount],
                values: MLXArray(padTokenId)
            )
            let rightPad = MLXArray.full(
                [batchSize, 1, rightPadCount],
                values: MLXArray(padTokenId)
            )

            let codebookCodes = codes[0..., k..<(k + 1), 0...]  // [B, 1, T]
            let delayed = MLX.concatenated([leftPad, codebookCodes, rightPad], axis: 2)
            delayedList.append(delayed)
        }

        return MLX.concatenated(delayedList, axis: 1)
    }

    /// Remove delay pattern from codes.
    ///
    /// - Parameter delayedCodes: Delayed codes [B, K, T_delayed]
    /// - Returns: Original codes [B, K, T] where T = T_delayed - K + 1
    public func revertDelayPattern(_ delayedCodes: MLXArray) -> MLXArray {
        let numCB = delayedCodes.dim(1)
        let delayedLength = delayedCodes.dim(2)
        let seqLength = delayedLength - numCodebooks + 1

        if seqLength <= 0 {
            // Not enough tokens to revert
            return delayedCodes[0..., 0..., 0..<1]
        }

        var codesList: [MLXArray] = []

        for k in 0..<numCB {
            // Codebook k's data starts at position k
            let codebookCodes = delayedCodes[0..., k..<(k + 1), k..<(k + seqLength)]
            codesList.append(codebookCodes)
        }

        return MLX.concatenated(codesList, axis: 1)
    }

    /// Get which codebook positions are valid at a given step.
    ///
    /// - Parameter step: Current generation step (0-indexed in delayed space)
    /// - Returns: List of (codebookIdx, originalPosition) tuples for valid tokens
    public func getNextTokenPositions(step: Int) -> [(Int, Int)] {
        var positions: [(Int, Int)] = []
        for k in 0..<numCodebooks {
            // Codebook k has valid tokens at positions >= k
            if step >= k {
                let originalPos = step - k
                positions.append((k, originalPos))
            }
        }
        return positions
    }
}

// MARK: - Language Model Head

/// Language model head for Parler-TTS with multiple codebooks.
///
/// Projects hidden states to logits for each codebook independently.
public class ParlerTTSLMHead: Module, @unchecked Sendable {
    public let numCodebooks: Int
    public let vocabSize: Int

    @ModuleInfo var linears: [Linear]

    public init(config: ParlerTTSConfig) {
        self.numCodebooks = config.numCodebooks
        self.vocabSize = config.codebookSize + 2  // +2 for special tokens

        var heads: [Linear] = []
        for _ in 0..<config.numCodebooks {
            heads.append(Linear(config.hiddenSize, vocabSize, bias: false))
        }
        self._linears.wrappedValue = heads

        super.init()
    }

    /// Project hidden states to logits.
    ///
    /// - Parameters:
    ///   - hiddenStates: Hidden states [B, T, D]
    ///   - codebookIdx: Optional specific codebook index
    /// - Returns: Logits [B, T, V] if codebookIdx specified, else [B, K, T, V]
    public func callAsFunction(
        _ hiddenStates: MLXArray,
        codebookIdx: Int? = nil
    ) -> MLXArray {
        if let idx = codebookIdx {
            return linears[idx](hiddenStates)
        }

        // All codebooks
        var logits: [MLXArray] = []
        for linear in linears {
            logits.append(linear(hiddenStates))
        }

        // Stack to [B, K, T, V]
        return MLX.stacked(logits, axis: 1)
    }

    /// Get logits for a specific codebook.
    public func getCodebookLogits(_ hiddenStates: MLXArray, codebookIdx: Int) -> MLXArray {
        return linears[codebookIdx](hiddenStates)
    }
}
