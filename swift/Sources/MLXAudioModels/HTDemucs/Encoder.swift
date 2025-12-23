// Encoder.swift
// Encoder layers for HTDemucs.

import Foundation
@preconcurrency import MLX
import MLXNN

// MARK: - HEncLayer

/// Hybrid encoder layer for HTDemucs.
///
/// Structure matches PyTorch exactly:
/// - conv: Conv2d (freq) or Conv1d (time) - main downsampling
/// - rewrite: Conv2d (freq) or Conv1d (time) - GLU rewrite
/// - dconv: DConv - dilated residual block
///
/// For frequency branch (freq=true):
/// - Uses Conv2d with kernel (K, 1), stride (S, 1)
/// - Input: `[B, C, F, T]` (batch, channels, freq bins, time frames)
///
/// For time branch (freq=false):
/// - Uses Conv1d with kernel K, stride S
/// - Input: `[B, C, T]` (batch, channels, time samples)
public class HEncLayer: Module, @unchecked Sendable {

    let chin: Int
    let chout: Int
    let kernelSize: Int
    let stride: Int
    let freq: Bool

    // Frequency branch (Conv2d)
    var conv2d: Conv2d?
    var rewrite2d: Conv2d?

    // Time branch (Conv1d)
    var conv1d: Conv1d?
    var rewrite1d: Conv1d?

    // Shared
    let dconv: DConv

    /// Creates an encoder layer.
    /// - Parameters:
    ///   - chin: Input channels.
    ///   - chout: Output channels.
    ///   - kernelSize: Convolution kernel size.
    ///   - stride: Convolution stride.
    ///   - freq: If true, use Conv2d for frequency branch.
    ///   - dconvDepth: Depth of DConv block.
    ///   - dconvCompress: Compression factor for DConv.
    ///   - dconvInit: Initial LayerScale value for DConv.
    public init(
        chin: Int,
        chout: Int,
        kernelSize: Int = 8,
        stride: Int = 4,
        freq: Bool = true,
        dconvDepth: Int = 2,
        dconvCompress: Int = 8,
        dconvInit: Float = 1e-4
    ) {
        self.chin = chin
        self.chout = chout
        self.kernelSize = kernelSize
        self.stride = stride
        self.freq = freq

        // Padding for "same" output (roughly)
        let pad = (kernelSize - stride) / 2

        if freq {
            // Frequency branch uses Conv2d
            self.conv2d = Conv2d(
                inputChannels: chin,
                outputChannels: chout,
                kernelSize: .init((kernelSize, 1)),
                stride: .init((stride, 1)),
                padding: .init((pad, 0))
            )
            // Rewrite conv: 1x1 that doubles channels for GLU
            self.rewrite2d = Conv2d(
                inputChannels: chout,
                outputChannels: chout * 2,
                kernelSize: 1
            )
            self.conv1d = nil
            self.rewrite1d = nil
        } else {
            // Time branch uses Conv1d
            self.conv1d = Conv1d(
                inputChannels: chin,
                outputChannels: chout,
                kernelSize: kernelSize,
                stride: stride,
                padding: pad
            )
            // Rewrite conv: 1x1 that doubles channels for GLU
            self.rewrite1d = Conv1d(
                inputChannels: chout,
                outputChannels: chout * 2,
                kernelSize: 1
            )
            self.conv2d = nil
            self.rewrite2d = nil
        }

        // DConv residual block
        self.dconv = DConv(
            channels: chout,
            depth: dconvDepth,
            compress: dconvCompress,
            initScale: dconvInit
        )
    }

    /// Forward pass.
    ///
    /// - Parameter x: Input tensor
    ///   - freq=true: `[B, C, F, T]` (NCHW format)
    ///   - freq=false: `[B, C, T]` (NCL format)
    /// - Returns: Downsampled output with same format as input.
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        if freq {
            return forwardFreq(x)
        } else {
            return forwardTime(x)
        }
    }

    /// Frequency branch forward pass.
    private func forwardFreq(_ input: MLXArray) -> MLXArray {
        var x = input

        // 1. Conv: NCHW -> NHWC for MLX Conv2d
        x = x.transposed(0, 2, 3, 1)  // [B, F, T, C]
        x = conv2d!(x)
        x = x.transposed(0, 3, 1, 2)  // back to [B, C, F, T]

        // 2. GELU
        x = gelu(x)

        // 3. DConv: collapse freq into batch, keep C channels
        let shape = x.shape
        let B = shape[0]
        let C = shape[1]
        let Fr = shape[2]
        let T = shape[3]

        x = x.transposed(0, 2, 1, 3)  // [B, Fr, C, T]
        x = x.reshaped([B * Fr, C, T])  // [B*Fr, C, T]
        x = x.transposed(0, 2, 1)  // NCL -> NLC for MLX
        x = dconv(x)
        x = x.transposed(0, 2, 1)  // NLC -> NCL
        x = x.reshaped([B, Fr, C, T])
        x = x.transposed(0, 2, 1, 3)  // [B, C, Fr, T]

        // 4. Rewrite: NCHW -> NHWC -> NCHW
        x = x.transposed(0, 2, 3, 1)
        x = rewrite2d!(x)
        x = x.transposed(0, 3, 1, 2)

        // 5. GLU: split along channel dim (axis=1 in NCHW)
        x = glu(x, axis: 1)

        return x
    }

    /// Time branch forward pass.
    private func forwardTime(_ input: MLXArray) -> MLXArray {
        var x = input

        // Pad input to be divisible by stride
        let le = x.shape[x.ndim - 1]
        if le % stride != 0 {
            let padAmount = stride - (le % stride)
            x = MLX.padded(x, widths: [[0, 0], [0, 0], [0, padAmount]])
        }

        // 1. Conv: NCL -> NLC for MLX
        x = x.transposed(0, 2, 1)  // [B, T, C]
        x = conv1d!(x)
        x = x.transposed(0, 2, 1)  // [B, C, T]

        // 2. GELU
        x = gelu(x)

        // 3. DConv
        x = x.transposed(0, 2, 1)
        x = dconv(x)
        x = x.transposed(0, 2, 1)

        // 4. Rewrite
        x = x.transposed(0, 2, 1)
        x = rewrite1d!(x)
        x = x.transposed(0, 2, 1)

        // 5. GLU
        x = glu(x, axis: 1)

        return x
    }
}

