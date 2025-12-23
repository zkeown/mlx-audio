// Encoder.swift
// Encoder layers for HTDemucs.
//
// OPTIMIZATION: Uses MLX-native NHWC/NLC format internally to avoid transposes.
// Format conversion happens at model boundaries in HTDemucs.swift.

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
/// INTERNAL FORMAT (optimized for MLX):
/// - freq=true: `[B, F, T, C]` (NHWC format)
/// - freq=false: `[B, T, C]` (NLC format)
///
/// NOTE: model.py converts from PyTorch format (NCHW/NCL) at entry
/// and converts back at exit. Layers operate in MLX-native format.
public class HEncLayer: Module, @unchecked Sendable {

    let chin: Int
    let chout: Int
    let kernelSize: Int
    let stride: Int
    let freq: Bool

    // Main conv layer - stores Conv2d for freq, Conv1d for time
    // Key "conv" matches Python weight keys
    let conv: Module

    // Rewrite conv layer - stores Conv2d for freq, Conv1d for time
    // Key "rewrite" matches Python weight keys
    let rewrite: Module

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
            self.conv = Conv2d(
                inputChannels: chin,
                outputChannels: chout,
                kernelSize: .init((kernelSize, 1)),
                stride: .init((stride, 1)),
                padding: .init((pad, 0))
            )
            // Rewrite conv: 1x1 that doubles channels for GLU
            self.rewrite = Conv2d(
                inputChannels: chout,
                outputChannels: chout * 2,
                kernelSize: 1
            )
        } else {
            // Time branch uses Conv1d
            self.conv = Conv1d(
                inputChannels: chin,
                outputChannels: chout,
                kernelSize: kernelSize,
                stride: stride,
                padding: pad
            )
            // Rewrite conv: 1x1 that doubles channels for GLU
            self.rewrite = Conv1d(
                inputChannels: chout,
                outputChannels: chout * 2,
                kernelSize: 1
            )
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
    /// - Parameter x: Input tensor (NHWC/NLC format for MLX efficiency)
    ///   - freq=true: `[B, F, T, C]` (NHWC format)
    ///   - freq=false: `[B, T, C]` (NLC format)
    /// - Returns: Downsampled output with same format as input.
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        if freq {
            return forwardFreq(x)
        } else {
            return forwardTime(x)
        }
    }

    /// Frequency branch forward pass.
    /// Input: [B, F, T, C] (NHWC) - no transposes needed!
    private func forwardFreq(_ input: MLXArray) -> MLXArray {
        var x = input
        let conv2d = conv as! Conv2d
        let rewrite2d = rewrite as! Conv2d

        // 1. Conv2d (MLX uses NHWC natively, no transpose needed)
        x = conv2d(x)

        // 2. GELU
        x = gelu(x)

        // 3. DConv: collapse freq into batch
        // [B, F, T, C] -> [B*F, T, C] - already NLC for DConv!
        let shape = x.shape
        let B = shape[0]
        let Fr = shape[1]
        let T = shape[2]
        let C = shape[3]

        x = x.reshaped([B * Fr, T, C])
        x = dconv(x)
        x = x.reshaped([B, Fr, T, C])

        // 4. Rewrite (NHWC, no transpose)
        x = rewrite2d(x)

        // 5. GLU: split along channel dim (axis=-1 in NHWC)
        x = glu(x, axis: -1)

        return x
    }

    /// Time branch forward pass.
    /// Input: [B, T, C] (NLC) - no transposes needed!
    private func forwardTime(_ input: MLXArray) -> MLXArray {
        var x = input
        let conv1d = conv as! Conv1d
        let rewrite1d = rewrite as! Conv1d

        // Pad input to be divisible by stride (T is axis 1 in NLC)
        let le = x.shape[1]
        if le % stride != 0 {
            let padAmount = stride - (le % stride)
            x = MLX.padded(x, widths: [[0, 0], [0, padAmount], [0, 0]])
        }

        // 1. Conv1d (MLX uses NLC natively, no transpose needed)
        x = conv1d(x)

        // 2. GELU
        x = gelu(x)

        // 3. DConv (already NLC)
        x = dconv(x)

        // 4. Rewrite (NLC, no transpose)
        x = rewrite1d(x)

        // 5. GLU: split along channel dim (axis=-1 in NLC)
        x = glu(x, axis: -1)

        return x
    }
}

