// Transformer.swift
// Transformer layers for HTDemucs.

import Foundation
@preconcurrency import MLX
import MLXNN

// MARK: - Positional Embeddings

/// Create 1D sinusoidal positional embedding.
///
/// Matches PyTorch demucs create_sin_embedding exactly.
/// - Parameters:
///   - length: Sequence length
///   - dim: Embedding dimension
///   - shift: Position shift (default 0)
///   - maxPeriod: Maximum period for sinusoidal encoding
/// - Returns: Positional embedding `[length, 1, dim]`
public func createSinEmbedding(
    length: Int,
    dim: Int,
    shift: Int = 0,
    maxPeriod: Float = 10000.0
) -> MLXArray {
    // pos: [T, 1, 1]
    let positions = MLXArray((shift..<(shift + length)).map { Float($0) })
    let pos = positions.expandedDimensions(axes: [1, 2])

    let halfDim = dim / 2
    let adim = MLXArray(0..<halfDim).asType(.float32).expandedDimensions(axes: [0, 1])

    // phase = pos / (maxPeriod ** (adim / (halfDim - 1)))
    let exponent = adim / Float(halfDim - 1)
    let divisor = pow(MLXArray(maxPeriod), exponent)
    let phase = pos / divisor

    let cosPhase = cos(phase)
    let sinPhase = sin(phase)
    return concatenated([cosPhase, sinPhase], axis: -1)  // [T, 1, D]
}

/// Create 2D sinusoidal positional embedding.
///
/// Matches PyTorch demucs create_2d_sin_embedding exactly:
/// - First half of channels: encodes width with sin/cos interleaved
/// - Second half of channels: encodes height with sin/cos interleaved
///
/// - Parameters:
///   - dModel: Embedding dimension (must be divisible by 4)
///   - height: Height (freq bins)
///   - width: Width (time frames)
///   - maxPeriod: Maximum period
/// - Returns: Positional embedding `[1, dModel, height, width]`
public func create2DSinEmbedding(
    dModel: Int,
    height: Int,
    width: Int,
    maxPeriod: Float = 10000.0
) -> MLXArray {
    let halfD = dModel / 2

    // div_term: step by 2 to match PyTorch
    let indices = MLXArray(stride(from: 0, to: halfD, by: 2).map { Float($0) })
    let divTerm = exp(indices * Float(-log(Double(maxPeriod)) / Double(halfD)))  // [D/4]

    let posW = MLXArray(0..<width).asType(.float32).expandedDimensions(axis: 1)  // [W, 1]
    let posH = MLXArray(0..<height).asType(.float32).expandedDimensions(axis: 1)  // [H, 1]

    // Width encoding
    let wArgs = posW * divTerm  // [W, D/4]
    let sinW = sin(wArgs)
    let cosW = cos(wArgs)

    // Height encoding
    let hArgs = posH * divTerm  // [H, D/4]
    let sinH = sin(hArgs)
    let cosH = cos(hArgs)

    // Interleave sin/cos: [W, D/4] -> stack -> [W, D/4, 2] -> reshape [W, D/2]
    var widthEmb = stacked([sinW, cosW], axis: -1)  // [W, D/4, 2]
    widthEmb = widthEmb.reshaped([width, halfD])  // [W, D/2]
    widthEmb = widthEmb.T  // [D/2, W]
    // Broadcast over height: [D/2, W] -> [D/2, H, W]
    widthEmb = widthEmb.expandedDimensions(axis: 1)
    widthEmb = MLX.broadcast(widthEmb, to: [halfD, height, width])

    var heightEmb = stacked([sinH, cosH], axis: -1)  // [H, D/4, 2]
    heightEmb = heightEmb.reshaped([height, halfD])  // [H, D/2]
    heightEmb = heightEmb.T  // [D/2, H]
    // Broadcast over width: [D/2, H] -> [D/2, H, W]
    heightEmb = heightEmb.expandedDimensions(axis: 2)
    heightEmb = MLX.broadcast(heightEmb, to: [halfD, height, width])

    let pe = concatenated([widthEmb, heightEmb], axis: 0)  // [D, H, W]
    return pe.expandedDimensions(axis: 0)  // [1, D, H, W]
}

// MARK: - MultiheadAttention

/// Multihead attention matching PyTorch's nn.MultiheadAttention.
///
/// Uses combined in_proj for Q, K, V (in_proj_weight, in_proj_bias).
public class MultiheadAttention: Module, @unchecked Sendable {
    let embedDim: Int
    let numHeads: Int
    let headDim: Int
    let scale: Float

    @ParameterInfo(key: "in_proj_weight") var in_proj_weight: MLXArray
    @ParameterInfo(key: "in_proj_bias") var in_proj_bias: MLXArray
    let out_proj: Linear

    /// Creates a MultiheadAttention layer.
    /// - Parameters:
    ///   - embedDim: Embedding dimension.
    ///   - numHeads: Number of attention heads.
    ///   - dropout: Dropout rate (unused, kept for API compatibility).
    public init(embedDim: Int, numHeads: Int, dropout: Float = 0.0) {
        self.embedDim = embedDim
        self.numHeads = numHeads
        self.headDim = embedDim / numHeads
        self.scale = pow(Float(headDim), -0.5)

        // Combined QKV projection (MLX format: [in, out])
        // PyTorch [3*D, D] -> transposed to MLX [D, 3*D]
        self._in_proj_weight.wrappedValue = MLXArray.zeros([embedDim, 3 * embedDim])
        self._in_proj_bias.wrappedValue = MLXArray.zeros([3 * embedDim])

        // Output projection
        self.out_proj = Linear(embedDim, embedDim)
    }

    /// Forward pass.
    /// - Parameters:
    ///   - query: Query tensor `[B, T, D]`
    ///   - key: Key tensor `[B, S, D]`
    ///   - value: Value tensor `[B, S, D]`
    /// - Returns: Output tensor `[B, T, D]`
    public func callAsFunction(
        query: MLXArray,
        key: MLXArray,
        value: MLXArray
    ) -> MLXArray {
        let B = query.shape[0]
        let T = query.shape[1]
        let S = key.shape[1]

        // Project Q, K, V using combined weight
        let wQ = in_proj_weight[0..., 0..<embedDim]
        let wK = in_proj_weight[0..., embedDim..<(2 * embedDim)]
        let wV = in_proj_weight[0..., (2 * embedDim)...]

        let bQ = in_proj_bias[0..<embedDim]
        let bK = in_proj_bias[embedDim..<(2 * embedDim)]
        let bV = in_proj_bias[(2 * embedDim)...]

        var q = matmul(query, wQ) + bQ
        var k = matmul(key, wK) + bK
        var v = matmul(value, wV) + bV

        // Reshape for multi-head attention
        q = q.reshaped([B, T, numHeads, headDim]).transposed(0, 2, 1, 3)
        k = k.reshaped([B, S, numHeads, headDim]).transposed(0, 2, 1, 3)
        v = v.reshaped([B, S, numHeads, headDim]).transposed(0, 2, 1, 3)

        // Attention
        var attn = matmul(q, k.transposed(0, 1, 3, 2)) * scale
        attn = softmax(attn, axis: -1)

        // Output
        var out = matmul(attn, v)
        out = out.transposed(0, 2, 1, 3).reshaped([B, T, embedDim])

        return out_proj(out)
    }
}

// MARK: - MyTransformerEncoderLayer

/// Transformer encoder layer matching PyTorch's MyTransformerEncoderLayer.
///
/// Structure (self-attention variant):
/// - self_attn: MultiheadAttention
/// - linear1, linear2: FFN
/// - norm1, norm2: LayerNorm
/// - norm_out: MyGroupNorm
/// - gamma_1, gamma_2: LayerScale
///
/// Structure (cross-attention variant, adds):
/// - cross_attn: MultiheadAttention
/// - norm3: LayerNorm
public class MyTransformerEncoderLayer: Module, @unchecked Sendable {
    let crossAttention: Bool

    var self_attn: MultiheadAttention?
    var cross_attn: MultiheadAttention?

    let linear1: Linear
    let linear2: Linear

    let norm1: LayerNorm
    let norm2: LayerNorm
    var norm3: LayerNorm?

    let norm_out: MyGroupNorm
    let gamma_1: LayerScale
    let gamma_2: LayerScale

    /// Creates a transformer encoder layer.
    /// - Parameters:
    ///   - dModel: Model dimension.
    ///   - nhead: Number of attention heads.
    ///   - dimFeedforward: FFN hidden dimension.
    ///   - dropout: Dropout rate (unused).
    ///   - crossAttention: If true, use cross-attention instead of self-attention.
    public init(
        dModel: Int,
        nhead: Int,
        dimFeedforward: Int,
        dropout: Float = 0.0,
        crossAttention: Bool = false
    ) {
        self.crossAttention = crossAttention

        if crossAttention {
            self.cross_attn = MultiheadAttention(embedDim: dModel, numHeads: nhead, dropout: dropout)
            self.self_attn = nil
        } else {
            self.self_attn = MultiheadAttention(embedDim: dModel, numHeads: nhead, dropout: dropout)
            self.cross_attn = nil
        }

        self.linear1 = Linear(dModel, dimFeedforward)
        self.linear2 = Linear(dimFeedforward, dModel)

        self.norm1 = LayerNorm(dimensions: dModel)
        self.norm2 = LayerNorm(dimensions: dModel)
        if crossAttention {
            self.norm3 = LayerNorm(dimensions: dModel)
        }

        self.norm_out = MyGroupNorm(numChannels: dModel)
        self.gamma_1 = LayerScale(channels: dModel, init: 1e-4)
        self.gamma_2 = LayerScale(channels: dModel, init: 1e-4)
    }

    /// Forward pass.
    /// - Parameters:
    ///   - x: Input `[B, T, D]`
    ///   - cross: Cross-attention source `[B, S, D]` (only for cross-attn layers)
    /// - Returns: Output `[B, T, D]`
    public func callAsFunction(_ x: MLXArray, cross: MLXArray? = nil) -> MLXArray {
        var out = x

        if crossAttention {
            // Cross-attention
            let xNorm = norm1(out)
            let crossNorm = norm2(cross!)
            let attnOut = cross_attn!(query: xNorm, key: crossNorm, value: crossNorm)
            out = out + gamma_1(attnOut)

            // FFN
            let xNorm2 = norm3!(out)
            let ffnOut = linear2(gelu(linear1(xNorm2)))
            out = out + gamma_2(ffnOut)
        } else {
            // Self-attention
            let xNorm = norm1(out)
            let attnOut = self_attn!(query: xNorm, key: xNorm, value: xNorm)
            out = out + gamma_1(attnOut)

            // FFN
            let xNorm2 = norm2(out)
            let ffnOut = linear2(gelu(linear1(xNorm2)))
            out = out + gamma_2(ffnOut)
        }

        // Final norm
        out = norm_out(out)

        return out
    }
}

// MARK: - CrossTransformerEncoder

/// Cross-domain transformer matching PyTorch's CrossTransformerEncoder.
///
/// Structure:
/// - norm_in: LayerNorm (freq input)
/// - norm_in_t: LayerNorm (time input)
/// - layers: list of MyTransformerEncoderLayer (freq branch)
/// - layers_t: list of MyTransformerEncoderLayer (time branch)
///
/// Pattern: alternating self-attention (0,2,4) and cross-attention (1,3)
public class CrossTransformerEncoder: Module, @unchecked Sendable {
    let dim: Int
    let depth: Int
    let maxPeriod: Float

    let norm_in: LayerNorm
    let norm_in_t: LayerNorm

    let layers: [MyTransformerEncoderLayer]
    let layers_t: [MyTransformerEncoderLayer]

    /// Creates a cross-domain transformer.
    /// - Parameters:
    ///   - dim: Model dimension.
    ///   - depth: Number of layers.
    ///   - heads: Number of attention heads.
    ///   - dimFeedforward: FFN hidden dimension (nil = 4*dim).
    ///   - dropout: Dropout rate.
    ///   - maxPeriod: Maximum period for positional embeddings.
    public init(
        dim: Int,
        depth: Int = 5,
        heads: Int = 8,
        dimFeedforward: Int? = nil,
        dropout: Float = 0.0,
        maxPeriod: Float = 10000.0
    ) {
        self.dim = dim
        self.depth = depth
        self.maxPeriod = maxPeriod

        let feedforward = dimFeedforward ?? (4 * dim)

        self.norm_in = LayerNorm(dimensions: dim)
        self.norm_in_t = LayerNorm(dimensions: dim)

        // Build layers - alternating self and cross attention
        var freqLayers: [MyTransformerEncoderLayer] = []
        var timeLayers: [MyTransformerEncoderLayer] = []

        for i in 0..<depth {
            let cross = (i % 2 == 1)  // odd layers are cross-attention
            freqLayers.append(
                MyTransformerEncoderLayer(
                    dModel: dim,
                    nhead: heads,
                    dimFeedforward: feedforward,
                    dropout: dropout,
                    crossAttention: cross
                )
            )
            timeLayers.append(
                MyTransformerEncoderLayer(
                    dModel: dim,
                    nhead: heads,
                    dimFeedforward: feedforward,
                    dropout: dropout,
                    crossAttention: cross
                )
            )
        }

        self.layers = freqLayers
        self.layers_t = timeLayers
    }

    /// Apply cross-domain transformer.
    /// - Parameters:
    ///   - freq: Frequency features `[B, C, F, T]` (4D)
    ///   - time: Time features `[B, C, T]` (3D)
    /// - Returns: Tuple of (freq_out, time_out)
    public func callAsFunction(
        freq: MLXArray,
        time: MLXArray
    ) -> (freq: MLXArray, time: MLXArray) {
        let B = freq.shape[0]
        let C = freq.shape[1]
        let Fr = freq.shape[2]
        let T1 = freq.shape[3]
        let T2 = time.shape[2]

        // Create 2D positional embedding for freq branch
        var posEmb2D = create2DSinEmbedding(dModel: C, height: Fr, width: T1, maxPeriod: maxPeriod)

        // Rearrange: [B, C, Fr, T1] -> [B, (T1*Fr), C]
        posEmb2D = posEmb2D.transposed(0, 3, 2, 1)  // [1, T1, Fr, C]
        posEmb2D = posEmb2D.reshaped([1, T1 * Fr, C])  // [1, T1*Fr, C]

        var freqFlat = freq.transposed(0, 3, 2, 1)  // [B, T1, Fr, C]
        freqFlat = freqFlat.reshaped([B, T1 * Fr, C])  // [B, T1*Fr, C]

        // Apply norm and add position embedding
        freqFlat = norm_in(freqFlat)
        freqFlat = freqFlat + posEmb2D

        // Create 1D positional embedding for time branch
        var posEmb1D = createSinEmbedding(length: T2, dim: C, shift: 0, maxPeriod: maxPeriod)
        posEmb1D = posEmb1D.transposed(1, 0, 2)  // [1, T2, C]

        // Rearrange: [B, C, T2] -> [B, T2, C]
        var timeFlat = time.transposed(0, 2, 1)

        // Apply norm and add position embedding
        timeFlat = norm_in_t(timeFlat)
        timeFlat = timeFlat + posEmb1D

        // Apply layers
        for i in 0..<depth {
            if i % 2 == 0 {
                // Self-attention
                freqFlat = layers[i](freqFlat)
                timeFlat = layers_t[i](timeFlat)
            } else {
                // Cross-attention (freq attends to time, time attends to freq)
                let oldFreq = freqFlat
                freqFlat = layers[i](freqFlat, cross: timeFlat)
                timeFlat = layers_t[i](timeFlat, cross: oldFreq)
            }
        }

        // Rearrange back: [B, (T1*Fr), C] -> [B, C, Fr, T1]
        var freqOut = freqFlat.reshaped([B, T1, Fr, C])
        freqOut = freqOut.transposed(0, 3, 2, 1)

        // Rearrange back: [B, T2, C] -> [B, C, T2]
        let timeOut = timeFlat.transposed(0, 2, 1)

        return (freqOut, timeOut)
    }
}
