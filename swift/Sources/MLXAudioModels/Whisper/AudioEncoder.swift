// AudioEncoder.swift
// Whisper audio encoder.

import Foundation
import MLX
import MLXNN

/// Create sinusoidal positional embeddings.
///
/// - Parameters:
///   - length: Sequence length
///   - dim: Embedding dimension
///   - maxTimescale: Maximum timescale for encoding
/// - Returns: Positional embeddings [length, dim]
public func sinusoids(length: Int, dim: Int, maxTimescale: Float = 10000.0) -> MLXArray {
    let halfDim = dim / 2
    let logTimescale = log(maxTimescale) / Float(halfDim - 1)

    // inv_timescales = exp(-log_timescale * arange(half_dim))
    let invTimescales = exp(-logTimescale * MLXArray(0 ..< halfDim).asType(.float32))

    // positions = arange(length)[:, None]
    let positions = MLXArray(0 ..< length).asType(.float32).reshaped([length, 1])

    // scaled_time = positions * inv_timescales[None, :]
    let scaledTime = positions * invTimescales.reshaped([1, halfDim])

    // concatenate([sin(scaled_time), cos(scaled_time)], axis=-1)
    return concatenated([sin(scaledTime), cos(scaledTime)], axis: -1)
}

/// Whisper audio encoder.
///
/// Processes log-mel spectrograms into audio features using:
/// 1. Two 1D convolutions for initial processing and downsampling
/// 2. Sinusoidal positional encoding
/// 3. Stack of transformer blocks
///
/// The encoder output is used as cross-attention context for the decoder.
public class AudioEncoder: Module {

    /// First convolution (nMels -> nState).
    @ModuleInfo var conv1: Conv1d

    /// Second convolution with stride 2 (nState -> nState).
    @ModuleInfo var conv2: Conv1d

    /// Sinusoidal positional embeddings [nAudioCtx, nState].
    var positionalEmbedding: MLXArray

    /// Transformer encoder blocks.
    @ModuleInfo var blocks: [ResidualAttentionBlock]

    /// Final layer normalization.
    @ModuleInfo var lnPost: LayerNorm

    /// Model configuration.
    let config: WhisperConfig

    /// Initialize audio encoder.
    ///
    /// - Parameter config: Whisper configuration
    public init(config: WhisperConfig) {
        self.config = config

        let nMels = config.nMels
        let nState = config.nAudioState
        let nHead = config.nAudioHead
        let nLayer = config.nAudioLayer
        let nCtx = config.nAudioCtx

        // Initial convolutions
        // Conv1: (nMels, nState, kernel=3, padding=1)
        self._conv1.wrappedValue = Conv1d(
            inputChannels: nMels,
            outputChannels: nState,
            kernelSize: 3,
            padding: 1
        )

        // Conv2: (nState, nState, kernel=3, stride=2, padding=1)
        self._conv2.wrappedValue = Conv1d(
            inputChannels: nState,
            outputChannels: nState,
            kernelSize: 3,
            stride: 2,
            padding: 1
        )

        // Positional embedding (not learned, sinusoidal)
        self.positionalEmbedding = sinusoids(length: nCtx, dim: nState)

        // Transformer blocks
        self._blocks.wrappedValue = (0 ..< nLayer).map { _ in
            ResidualAttentionBlock(nState: nState, nHead: nHead, crossAttention: false)
        }

        // Final layer norm
        self._lnPost.wrappedValue = LayerNorm(dimensions: nState)
    }

    /// Encode audio features.
    ///
    /// - Parameter mel: Log-mel spectrogram [B, nMels, T] or [nMels, T]
    /// - Returns: Audio features [B, T//2, nState]
    public func callAsFunction(_ mel: MLXArray) -> MLXArray {
        var x = mel

        // Handle unbatched input
        if x.ndim == 2 {
            x = x.expandedDimensions(axis: 0)
        }

        // MLX Conv1d expects [B, T, C] (channels-last)
        // Input mel is [B, nMels, T], so transpose to [B, T, nMels]
        x = x.transposed(0, 2, 1)

        // First conv + GELU
        x = gelu(conv1(x))

        // Second conv (stride 2 for downsampling) + GELU
        x = gelu(conv2(x))

        // x is now [B, T//2, nState] (channels-last after conv)

        // Add positional embedding
        // x: [B, T, nState], positionalEmbedding: [nCtx, nState]
        // Only use the first T positions
        let T = x.dim(1)
        x = x + positionalEmbedding[0 ..< T]

        // Apply transformer blocks
        for block in blocks {
            let (output, _) = block(x)
            x = output
        }

        // Final layer norm
        x = lnPost(x)

        return x
    }
}
