// Attention.swift
// Multi-head attention with KV caching for Whisper.

import Foundation
import MLX
import MLXNN

/// Multi-head attention with optional KV caching.
///
/// Supports both self-attention and cross-attention modes.
/// KV caching is used for efficient autoregressive decoding.
public class MultiHeadAttention: Module {

    /// Hidden dimension.
    let nState: Int

    /// Number of attention heads.
    let nHead: Int

    /// Dimension per head.
    let headDim: Int

    /// Attention scaling factor.
    let scale: Float

    /// Query projection.
    @ModuleInfo var query: Linear

    /// Key projection (no bias).
    @ModuleInfo var key: Linear

    /// Value projection.
    @ModuleInfo var value: Linear

    /// Output projection.
    @ModuleInfo var out: Linear

    /// Initialize multi-head attention.
    ///
    /// - Parameters:
    ///   - nState: Hidden dimension (must be divisible by nHead)
    ///   - nHead: Number of attention heads
    public init(nState: Int, nHead: Int) {
        self.nState = nState
        self.nHead = nHead
        self.headDim = nState / nHead
        self.scale = Float(pow(Double(headDim), -0.5))

        // Separate projections for Q, K, V (Whisper style)
        self._query.wrappedValue = Linear(nState, nState)
        self._key.wrappedValue = Linear(nState, nState, bias: false)
        self._value.wrappedValue = Linear(nState, nState)
        self._out.wrappedValue = Linear(nState, nState)
    }

    /// Forward pass with optional KV caching.
    ///
    /// - Parameters:
    ///   - x: Query input [B, T, D]
    ///   - xa: Key/value source for cross-attention [B, S, D].
    ///         If nil, performs self-attention.
    ///   - mask: Attention mask [T, S] or [B, T, S]
    ///   - kvCache: Cached (key, value) from previous steps for
    ///              incremental decoding. Only used for self-attention.
    /// - Returns: Tuple of output tensor [B, T, D] and updated KV cache
    public func callAsFunction(
        _ x: MLXArray,
        xa: MLXArray? = nil,
        mask: MLXArray? = nil,
        kvCache: (MLXArray, MLXArray)? = nil
    ) -> (MLXArray, (MLXArray, MLXArray)?) {
        let B = x.dim(0)
        let T = x.dim(1)

        // Compute query
        var q = query(x)

        var k: MLXArray
        var v: MLXArray
        var newKvCache: (MLXArray, MLXArray)?

        if let xa = xa {
            // Cross-attention (no caching needed, encoder output is fixed)
            k = key(xa)
            v = value(xa)
            newKvCache = nil
        } else {
            // Self-attention
            k = key(x)
            v = value(x)

            // Handle KV cache for incremental decoding
            if let (kCache, vCache) = kvCache {
                k = concatenated([kCache, k], axis: 1)
                v = concatenated([vCache, v], axis: 1)
            }

            newKvCache = (k, v)
        }

        let S = k.dim(1)

        // Reshape for multi-head attention
        // [B, T, D] -> [B, T, nHead, headDim] -> [B, nHead, T, headDim]
        q = q.reshaped([B, T, nHead, headDim]).transposed(0, 2, 1, 3)
        k = k.reshaped([B, S, nHead, headDim]).transposed(0, 2, 1, 3)
        v = v.reshaped([B, S, nHead, headDim]).transposed(0, 2, 1, 3)

        // Compute attention scores
        // [B, nHead, T, headDim] @ [B, nHead, headDim, S] -> [B, nHead, T, S]
        var attn = (q.matmul(k.transposed(0, 1, 3, 2))) * scale

        // Apply mask if provided
        if let mask = mask {
            attn = attn + mask
        }

        // Softmax and apply to values
        attn = softmax(attn, axis: -1)

        // [B, nHead, T, S] @ [B, nHead, S, headDim] -> [B, nHead, T, headDim]
        var output = attn.matmul(v)

        // Reshape back
        // [B, nHead, T, headDim] -> [B, T, nHead, headDim] -> [B, T, D]
        output = output.transposed(0, 2, 1, 3).reshaped([B, T, nState])

        // Output projection
        output = out(output)

        return (output, newKvCache)
    }
}

/// Residual attention block for Whisper transformer.
///
/// Structure:
/// - LayerNorm -> Multi-head Self-Attention -> Residual
/// - (Optional) LayerNorm -> Multi-head Cross-Attention -> Residual
/// - LayerNorm -> MLP -> Residual
public class ResidualAttentionBlock: Module {

    /// Hidden dimension.
    let nState: Int

    /// Number of attention heads.
    let nHead: Int

    /// Whether this block has cross-attention.
    let hasCrossAttention: Bool

    /// Self-attention layer.
    @ModuleInfo var attn: MultiHeadAttention

    /// Layer norm for self-attention.
    @ModuleInfo var attnLn: LayerNorm

    /// Cross-attention layer (decoder only).
    @ModuleInfo var crossAttn: MultiHeadAttention?

    /// Layer norm for cross-attention.
    @ModuleInfo var crossAttnLn: LayerNorm?

    /// First linear layer of MLP.
    @ModuleInfo var mlp0: Linear

    /// Second linear layer of MLP.
    @ModuleInfo var mlp1: Linear

    /// Layer norm for MLP.
    @ModuleInfo var mlpLn: LayerNorm

    /// Initialize residual attention block.
    ///
    /// - Parameters:
    ///   - nState: Hidden dimension
    ///   - nHead: Number of attention heads
    ///   - crossAttention: Whether to include cross-attention (decoder)
    public init(nState: Int, nHead: Int, crossAttention: Bool = false) {
        self.nState = nState
        self.nHead = nHead
        self.hasCrossAttention = crossAttention

        // Self-attention
        self._attn.wrappedValue = MultiHeadAttention(nState: nState, nHead: nHead)
        self._attnLn.wrappedValue = LayerNorm(dimensions: nState)

        // Cross-attention (decoder only)
        if crossAttention {
            self._crossAttn.wrappedValue = MultiHeadAttention(nState: nState, nHead: nHead)
            self._crossAttnLn.wrappedValue = LayerNorm(dimensions: nState)
        }

        // MLP with 4x expansion
        self._mlp0.wrappedValue = Linear(nState, nState * 4)
        self._mlp1.wrappedValue = Linear(nState * 4, nState)
        self._mlpLn.wrappedValue = LayerNorm(dimensions: nState)
    }

    /// Forward pass.
    ///
    /// - Parameters:
    ///   - x: Input tensor [B, T, D]
    ///   - xa: Encoder output for cross-attention [B, S, D]
    ///   - mask: Causal attention mask
    ///   - kvCache: Cached (key, value) for incremental decoding
    /// - Returns: Tuple of output tensor [B, T, D] and updated KV cache
    public func callAsFunction(
        _ x: MLXArray,
        xa: MLXArray? = nil,
        mask: MLXArray? = nil,
        kvCache: (MLXArray, MLXArray)? = nil
    ) -> (MLXArray, (MLXArray, MLXArray)?) {
        var x = x

        // Self-attention
        let (attnOut, newKvCache) = attn(attnLn(x), mask: mask, kvCache: kvCache)
        x = x + attnOut

        // Cross-attention (if applicable)
        if hasCrossAttention, let crossAttn = crossAttn, let crossAttnLn = crossAttnLn, let xa = xa {
            let (crossOut, _) = crossAttn(crossAttnLn(x), xa: xa)
            x = x + crossOut
        }

        // MLP
        let mlpOut = mlp1(gelu(mlp0(mlpLn(x))))
        x = x + mlpOut

        return (x, newKvCache)
    }
}
