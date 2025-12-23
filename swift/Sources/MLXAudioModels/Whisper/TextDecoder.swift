// TextDecoder.swift
// Whisper text decoder.

import Foundation
import MLX
import MLXNN

/// Whisper text decoder.
///
/// Generates text tokens autoregressively using:
/// 1. Token embeddings + learned positional embeddings
/// 2. Stack of transformer blocks with causal self-attention
///    and cross-attention to encoder output
/// 3. Final layer norm and projection to vocabulary
///
/// The decoder uses KV caching for efficient incremental decoding.
public class TextDecoder: Module {

    /// Hidden dimension.
    let nState: Int

    /// Text context length.
    let nCtx: Int

    /// Token embedding lookup table.
    @ModuleInfo var tokenEmbedding: Embedding

    /// Learned positional embeddings [nTextCtx, nState].
    var positionalEmbedding: MLXArray

    /// Transformer decoder blocks.
    @ModuleInfo var blocks: [ResidualAttentionBlock]

    /// Final layer normalization.
    @ModuleInfo var ln: LayerNorm

    /// Model configuration.
    let config: WhisperConfig

    /// Initialize text decoder.
    ///
    /// - Parameter config: Whisper configuration
    public init(config: WhisperConfig) {
        self.config = config

        let nVocab = config.nVocab
        let nCtx = config.nTextCtx
        let nState = config.nTextState
        let nHead = config.nTextHead
        let nLayer = config.nTextLayer

        self.nState = nState
        self.nCtx = nCtx

        // Token embedding
        self._tokenEmbedding.wrappedValue = Embedding(embeddingCount: nVocab, dimensions: nState)

        // Learned positional embedding (initialized to zeros, loaded from weights)
        self.positionalEmbedding = MLXArray.zeros([nCtx, nState])

        // Transformer blocks (with cross-attention)
        self._blocks.wrappedValue = (0 ..< nLayer).map { _ in
            ResidualAttentionBlock(nState: nState, nHead: nHead, crossAttention: true)
        }

        // Final layer norm
        self._ln.wrappedValue = LayerNorm(dimensions: nState)
    }

    /// Generate logits for next token prediction.
    ///
    /// - Parameters:
    ///   - tokens: Input token IDs [B, T]
    ///   - audioFeatures: Encoder output [B, S, D]
    ///   - kvCache: List of (key, value) tuples for each layer,
    ///              from previous decoding steps. Length = nLayers.
    /// - Returns: Tuple of logits [B, T, nVocab] and updated KV cache
    public func callAsFunction(
        tokens: MLXArray,
        audioFeatures: MLXArray,
        kvCache: [(MLXArray, MLXArray)]? = nil
    ) -> (MLXArray, [(MLXArray, MLXArray)]) {
        let T = tokens.dim(1)

        // Token + positional embeddings
        // For incremental decoding, offset is based on cache length
        let offset: Int
        if let cache = kvCache, !cache.isEmpty {
            offset = cache[0].0.dim(1)  // Length of cached keys
        } else {
            offset = 0
        }

        var x = tokenEmbedding(tokens)
        x = x + positionalEmbedding[offset ..< (offset + T)]

        // Create causal mask for self-attention
        // Mask shape: [T, T + offset] where cached positions are visible
        let mask = createCausalMask(T: T, offset: offset)

        // Apply transformer blocks
        var newKvCache: [(MLXArray, MLXArray)] = []
        for (i, block) in blocks.enumerated() {
            let layerCache: (MLXArray, MLXArray)?
            if let cache = kvCache, i < cache.count {
                layerCache = cache[i]
            } else {
                layerCache = nil
            }

            let (output, newCache) = block(x, xa: audioFeatures, mask: mask, kvCache: layerCache)
            x = output
            if let newCache = newCache {
                newKvCache.append(newCache)
            }
        }

        // Final layer norm
        x = ln(x)

        // Project to vocabulary using tied embeddings
        // x: [B, T, nState], tokenEmbedding.weight: [nVocab, nState]
        let logits = x.matmul(tokenEmbedding.weight.T)

        return (logits, newKvCache)
    }

    /// Create causal attention mask.
    ///
    /// - Parameters:
    ///   - T: Query sequence length
    ///   - offset: Number of cached positions
    /// - Returns: Causal mask [T, T + offset] where -inf indicates masked positions
    private func createCausalMask(T: Int, offset: Int = 0) -> MLXArray {
        // Total key/value length including cache
        let S = T + offset

        // Create mask where True means "attend to this position"
        // For position i, can attend to positions 0..i+offset
        let queryPos = MLXArray(0 ..< T).reshaped([T, 1]) + offset
        let keyPos = MLXArray(0 ..< S).reshaped([1, S])

        // Allow attending to positions <= current position
        let mask = lessEqual(keyPos, queryPos)

        // Convert to additive mask (0 for attend, -inf for masked)
        return which(mask, MLXArray(0.0), MLXArray(-.infinity))
    }

    /// Optimized forward pass using pre-allocated KV cache.
    ///
    /// This version avoids O(n) concatenation by using the `at[].add()` pattern
    /// for cache updates. It also caches cross-attention K/V which are fixed
    /// for the entire decode sequence.
    ///
    /// - Parameters:
    ///   - tokens: Input token IDs [B, T]
    ///   - audioFeatures: Encoder output [B, S, D]
    ///   - kvCache: Pre-allocated KV cache (WhisperKVCache or compatible)
    ///   - crossAttnCache: Cached cross-attention K/V per layer (computed on first step)
    ///   - offset: Position offset for positional embeddings
    /// - Returns: Tuple of (logits, newKs, newVs, crossKs, crossVs) for cache updates
    public func forwardOptimized(
        tokens: MLXArray,
        audioFeatures: MLXArray,
        selfAttnKVs: [(MLXArray, MLXArray)]?,
        crossAttnKVs: [(MLXArray, MLXArray)]?,
        offset: Int
    ) -> (logits: MLXArray, newKs: [MLXArray], newVs: [MLXArray], crossKs: [MLXArray], crossVs: [MLXArray]) {
        let T = tokens.dim(1)

        // Token + positional embeddings
        var x = tokenEmbedding(tokens)
        x = x + positionalEmbedding[offset ..< (offset + T)]

        // Create causal mask
        let totalLen = offset + T
        let mask = createCausalMask(T: T, offset: offset)

        // Process blocks with optimized caching
        var newKs: [MLXArray] = []
        var newVs: [MLXArray] = []
        var crossKs: [MLXArray] = []
        var crossVs: [MLXArray] = []

        for (i, block) in blocks.enumerated() {
            let selfKV = selfAttnKVs.flatMap { i < $0.count ? $0[i] : nil }
            let crossKV = crossAttnKVs.flatMap { i < $0.count ? $0[i] : nil }

            let (output, newK, newV, crossK, crossV) = block.forwardOptimized(
                x,
                xa: audioFeatures,
                mask: mask,
                precomputedKV: selfKV,
                crossAttnKV: crossKV
            )
            x = output

            newKs.append(newK)
            newVs.append(newV)

            if let ck = crossK {
                crossKs.append(ck)
            }
            if let cv = crossV {
                crossVs.append(cv)
            }
        }

        // Final layer norm and projection
        x = ln(x)
        let logits = x.matmul(tokenEmbedding.weight.T)

        return (logits, newKs, newVs, crossKs, crossVs)
    }
}
