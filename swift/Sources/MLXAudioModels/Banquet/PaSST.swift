// PaSST.swift
// PaSST (Patchout Spectrogram Transformer) query encoder for Banquet.
//
// PaSST encodes reference audio into a 768-dimensional embedding for query-based
// source separation. This implementation ports the hear21passt model to MLX.
//
// Architecture:
//     Input: mel spectrogram [batch, 1, 128, 998]
//     Patch embedding: Conv2d (1 → 768, kernel=16x16, stride=10x10)
//     Position: Separate time (99) and frequency (12) embeddings
//     Tokens: CLS + DIST tokens prepended
//     Transformer: 12 blocks, 12 heads, mlp_ratio=4
//     Output: Average of CLS and DIST tokens [batch, 768]

import Foundation
import MLX
import MLXNN

/// Multi-head self-attention for PaSST.
public class PaSSTAttention: Module, @unchecked Sendable {
    @ModuleInfo var qkv: Linear
    @ModuleInfo var proj: Linear

    let numHeads: Int
    let headDim: Int
    let scale: Float

    /// Creates a PaSST attention module.
    ///
    /// - Parameters:
    ///   - dim: Embedding dimension
    ///   - numHeads: Number of attention heads
    ///   - qkvBias: Whether to use bias in QKV projection
    public init(dim: Int, numHeads: Int = 12, qkvBias: Bool = true) {
        self.numHeads = numHeads
        self.headDim = dim / numHeads
        self.scale = pow(Float(headDim), -0.5)

        _qkv.wrappedValue = Linear(dim, dim * 3, bias: qkvBias)
        _proj.wrappedValue = Linear(dim, dim)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let shape = x.shape
        let B = shape[0]
        let N = shape[1]
        let C = shape[2]

        // QKV projection
        var qkvOut = qkv(x)
        qkvOut = qkvOut.reshaped([B, N, 3, numHeads, headDim])
        qkvOut = qkvOut.transposed(2, 0, 3, 1, 4)  // [3, B, numHeads, N, headDim]

        let q = qkvOut[0]
        let k = qkvOut[1]
        let v = qkvOut[2]

        // Attention: q @ k.T * scale
        let kT = k.transposed(0, 1, 3, 2)
        var attn = MLX.matmul(q, kT) * scale
        attn = softmax(attn, axis: -1)

        // Output: attn @ v
        var output = MLX.matmul(attn, v)  // [B, numHeads, N, headDim]
        output = output.transposed(0, 2, 1, 3)  // [B, N, numHeads, headDim]
        output = output.reshaped([B, N, C])

        return proj(output)
    }
}

/// MLP block with GELU activation.
public class PaSSTMLP: Module, @unchecked Sendable {
    @ModuleInfo var fc1: Linear
    @ModuleInfo var fc2: Linear

    /// Creates a PaSST MLP block.
    ///
    /// - Parameters:
    ///   - inFeatures: Input dimension
    ///   - hiddenFeatures: Hidden dimension
    ///   - outFeatures: Output dimension (defaults to inFeatures)
    public init(inFeatures: Int, hiddenFeatures: Int? = nil, outFeatures: Int? = nil) {
        let hidden = hiddenFeatures ?? inFeatures * 4
        let out = outFeatures ?? inFeatures

        _fc1.wrappedValue = Linear(inFeatures, hidden)
        _fc2.wrappedValue = Linear(hidden, out)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = fc1(x)
        out = gelu(out)
        out = fc2(out)
        return out
    }
}

/// Transformer block with pre-norm architecture.
///
/// Architecture: x = x + attn(norm1(x)); x = x + mlp(norm2(x))
public class PaSSTBlock: Module, @unchecked Sendable {
    @ModuleInfo var norm1: LayerNorm
    @ModuleInfo var attn: PaSSTAttention
    @ModuleInfo var norm2: LayerNorm
    @ModuleInfo var mlp: PaSSTMLP

    /// Creates a PaSST transformer block.
    ///
    /// - Parameters:
    ///   - dim: Embedding dimension
    ///   - numHeads: Number of attention heads
    ///   - mlpRatio: MLP hidden dimension ratio
    ///   - qkvBias: Whether to use bias in QKV
    public init(dim: Int, numHeads: Int, mlpRatio: Float = 4.0, qkvBias: Bool = true) {
        _norm1.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6)
        _attn.wrappedValue = PaSSTAttention(dim: dim, numHeads: numHeads, qkvBias: qkvBias)
        _norm2.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6)
        _mlp.wrappedValue = PaSSTMLP(
            inFeatures: dim,
            hiddenFeatures: Int(Float(dim) * mlpRatio)
        )
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = x + attn(norm1(x))
        out = out + mlp(norm2(out))
        return out
    }
}

/// Patch embedding using Conv2d.
///
/// Converts mel spectrogram to patch embeddings.
public class PaSSTEmbed: Module, @unchecked Sendable {
    @ModuleInfo var proj: Conv2d

    /// Creates a patch embedding layer.
    ///
    /// - Parameters:
    ///   - inChannels: Input channels (1 for mel spectrogram)
    ///   - embedDim: Embedding dimension
    ///   - patchSize: Patch size (height, width)
    ///   - stride: Stride for patch extraction
    public init(
        inChannels: Int = 1,
        embedDim: Int = 768,
        patchSize: (Int, Int) = (16, 16),
        stride: (Int, Int) = (10, 10)
    ) {
        _proj.wrappedValue = Conv2d(
            inputChannels: inChannels,
            outputChannels: embedDim,
            kernelSize: .init((patchSize.0, patchSize.1)),
            stride: .init((stride.0, stride.1))
        )
    }

    /// Forward pass.
    ///
    /// - Parameter x: Input [batch, channels, height, width]
    /// - Returns: Patch embeddings [batch, embed_dim, n_freq_patches, n_time_patches]
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // MLX Conv2d expects [batch, height, width, channels]
        var out = x.transposed(0, 2, 3, 1)
        out = proj(out)
        // Back to [batch, channels, height, width]
        out = out.transposed(0, 3, 1, 2)
        return out
    }
}

/// PaSST (Patchout Spectrogram Transformer) encoder.
///
/// Encodes mel spectrogram into a 768-dimensional embedding for query-based
/// source separation.
///
/// Input: mel spectrogram [batch, 1, 128, 998]
/// Output: embedding [batch, 768]
public class PaSST: Module, @unchecked Sendable {
    @ModuleInfo(key: "patch_embed") var patchEmbed: PaSSTEmbed
    @ModuleInfo var blocks: [PaSSTBlock]
    @ModuleInfo var norm: LayerNorm

    // Parameters for special tokens and position embeddings
    var clsToken: MLXArray
    var distToken: MLXArray
    var timePosEmbed: MLXArray
    var freqPosEmbed: MLXArray
    var posEmbed: MLXArray

    let config: PaSSTConfig
    let embedDim: Int

    /// Creates a PaSST encoder.
    ///
    /// - Parameter config: PaSST configuration
    public init(config: PaSSTConfig = PaSSTConfig()) {
        self.config = config
        self.embedDim = config.embedDim

        // Patch embedding
        _patchEmbed.wrappedValue = PaSSTEmbed(
            inChannels: 1,
            embedDim: config.embedDim,
            patchSize: config.patchSize,
            stride: (10, 10)  // Fixed stride for hear21passt compatibility
        )

        // Number of patches: 12 x 99 = 1188 for (128, 998) input
        let nFreqPatches = 12
        let nTimePatches = 99

        // Special tokens
        self.clsToken = MLXArray.zeros([1, 1, config.embedDim])
        self.distToken = MLXArray.zeros([1, 1, config.embedDim])

        // Separate time and frequency position embeddings
        self.timePosEmbed = MLXArray.zeros([1, config.embedDim, 1, nTimePatches])
        self.freqPosEmbed = MLXArray.zeros([1, config.embedDim, nFreqPatches, 1])

        // Position embedding for CLS and DIST tokens
        self.posEmbed = MLXArray.zeros([1, 2, config.embedDim])

        // Transformer blocks
        _blocks.wrappedValue = (0..<config.numLayers).map { _ in
            PaSSTBlock(
                dim: config.embedDim,
                numHeads: config.numHeads,
                mlpRatio: config.mlpRatio,
                qkvBias: true
            )
        }

        // Final layer norm
        _norm.wrappedValue = LayerNorm(dimensions: config.embedDim, eps: 1e-6)
    }

    /// Forward pass.
    ///
    /// - Parameter x: Mel spectrogram [batch, 1, 128, 998]
    /// - Returns: Embedding [batch, 768]
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let B = x.shape[0]

        // Patch embedding: [B, 768, 12, 99]
        var out = patchEmbed(x)

        // Add position embeddings (broadcast over spatial dimensions)
        out = out + timePosEmbed + freqPosEmbed

        // Flatten patches: [B, 768, 12, 99] -> [B, 1188, 768]
        out = out.reshaped([B, embedDim, -1])
        out = out.transposed(0, 2, 1)

        // Expand special tokens for batch
        let clsTokens = MLX.broadcast(clsToken, to: [B, 1, embedDim])
        let distTokens = MLX.broadcast(distToken, to: [B, 1, embedDim])

        // Concatenate: [B, 1190, 768]
        out = MLX.concatenated([clsTokens, distTokens, out], axis: 1)

        // Add position embedding for special tokens
        let posEmbedBroadcast = MLX.broadcast(posEmbed, to: [B, 2, embedDim])
        let specialTokens = out[0..., 0..<2, 0...] + posEmbedBroadcast
        let patches = out[0..., 2..., 0...]
        out = MLX.concatenated([specialTokens, patches], axis: 1)

        // Transformer blocks
        for block in blocks {
            out = block(out)
        }

        out = norm(out)

        // Average CLS and DIST tokens for final embedding
        let embedding = (out[0..., 0, 0...] + out[0..., 1, 0...]) / 2

        return embedding
    }

    /// Creates PaSST from configuration.
    ///
    /// - Parameter config: PaSST configuration
    /// - Returns: PaSST model
    public static func fromConfig(_ config: PaSSTConfig) -> PaSST {
        return PaSST(config: config)
    }
}
