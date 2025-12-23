// Decoder.swift
// Decoder layers for HTDemucs.

import Foundation
@preconcurrency import MLX
import MLXNN

// MARK: - HDecLayer

/// Hybrid decoder layer for HTDemucs.
///
/// Structure matches PyTorch exactly:
/// - rewrite: Conv2d (freq) or Conv1d (time) - GLU rewrite (kernel 3)
/// - dconv: DConv - dilated residual block
/// - conv_tr: ConvTranspose2d (freq) or ConvTranspose1d (time) - upsampling
///
/// For frequency branch (freq=true):
/// - Uses ConvTranspose2d with kernel (K, 1), stride (S, 1)
/// - Input: `[B, C, F, T]`
///
/// For time branch (freq=false):
/// - Uses ConvTranspose1d with kernel K, stride S
/// - Input: `[B, C, T]`
///
/// The decoder uses length-based trimming to ensure output matches
/// the expected encoder input length.
public class HDecLayer: Module, @unchecked Sendable {

    let chin: Int
    let chout: Int
    let kernelSize: Int
    let stride: Int
    let freq: Bool
    let last: Bool
    let pad: Int

    // Frequency branch (ConvTransposed2d)
    var conv_tr2d: ConvTransposed2d?
    var rewrite2d: Conv2d?

    // Time branch (ConvTransposed1d)
    var conv_tr1d: ConvTransposed1d?
    var rewrite1d: Conv1d?

    // Shared
    let dconv: DConv

    /// Creates a decoder layer.
    /// - Parameters:
    ///   - chin: Input channels.
    ///   - chout: Output channels.
    ///   - kernelSize: Convolution kernel size.
    ///   - stride: Convolution stride.
    ///   - freq: If true, use Conv2d for frequency branch.
    ///   - dconvDepth: Depth of DConv block.
    ///   - dconvCompress: Compression factor for DConv.
    ///   - dconvInit: Initial LayerScale value for DConv.
    ///   - last: If true, skip final GELU activation.
    public init(
        chin: Int,
        chout: Int,
        kernelSize: Int = 8,
        stride: Int = 4,
        freq: Bool = true,
        dconvDepth: Int = 2,
        dconvCompress: Int = 8,
        dconvInit: Float = 1e-4,
        last: Bool = false
    ) {
        self.chin = chin
        self.chout = chout
        self.kernelSize = kernelSize
        self.stride = stride
        self.freq = freq
        self.last = last

        // PyTorch uses pad = kernel_size // 4 for output trimming
        self.pad = kernelSize / 4

        if freq {
            // Frequency branch uses ConvTransposed2d
            self.conv_tr2d = ConvTransposed2d(
                inputChannels: chin,
                outputChannels: chout,
                kernelSize: .init((kernelSize, 1)),
                stride: .init((stride, 1)),
                padding: 0
            )
            // Rewrite conv: 3x3 that doubles channels for GLU
            self.rewrite2d = Conv2d(
                inputChannels: chin,
                outputChannels: chin * 2,
                kernelSize: 3,
                padding: 1
            )
            self.conv_tr1d = nil
            self.rewrite1d = nil
        } else {
            // Time branch uses ConvTransposed1d
            self.conv_tr1d = ConvTransposed1d(
                inputChannels: chin,
                outputChannels: chout,
                kernelSize: kernelSize,
                stride: stride,
                padding: 0
            )
            // Rewrite conv: kernel 3 that doubles channels for GLU
            self.rewrite1d = Conv1d(
                inputChannels: chin,
                outputChannels: chin * 2,
                kernelSize: 3,
                padding: 1
            )
            self.conv_tr2d = nil
            self.rewrite2d = nil
        }

        // DConv residual block (operates on chin, before upsampling)
        self.dconv = DConv(
            channels: chin,
            depth: dconvDepth,
            compress: dconvCompress,
            initScale: dconvInit
        )
    }

    /// Forward pass.
    ///
    /// - Parameters:
    ///   - x: Input tensor from previous decoder layer
    ///     - freq=true: `[B, C, F, T]` (NCHW format)
    ///     - freq=false: `[B, C, T]` (NCL format)
    ///   - skip: Skip connection from corresponding encoder layer.
    ///   - length: Expected output time dimension length (from encoder input).
    /// - Returns: Tuple of (output, pre_output):
    ///   - output: Upsampled and trimmed output
    ///   - pre: Output before transposed conv (used for branch merge)
    public func callAsFunction(
        _ x: MLXArray,
        skip: MLXArray?,
        length: Int
    ) -> (output: MLXArray, pre: MLXArray) {
        if freq {
            return forwardFreq(x, skip: skip, length: length)
        } else {
            return forwardTime(x, skip: skip, length: length)
        }
    }

    /// Frequency branch forward pass.
    private func forwardFreq(
        _ input: MLXArray,
        skip: MLXArray?,
        length: Int
    ) -> (output: MLXArray, pre: MLXArray) {
        var x = input
        var skipTrimmed = skip

        // 1. Add skip connection
        if let s = skipTrimmed {
            // Handle length mismatch by trimming to shorter length
            if x.shape[2] != s.shape[2] {
                let minF = min(x.shape[2], s.shape[2])
                x = x[0..., 0..., 0..<minF, 0...]
                skipTrimmed = s[0..., 0..., 0..<minF, 0...]
            }
            x = x + skipTrimmed!
        }

        let shape = x.shape
        let B = shape[0]

        // 2. Rewrite -> GLU: NCHW -> NHWC -> NCHW
        x = x.transposed(0, 2, 3, 1)
        x = rewrite2d!(x)
        x = x.transposed(0, 3, 1, 2)
        var y = gluNCHW(x)

        // 3. DConv: collapse freq into batch
        let yShape = y.shape
        let C = yShape[1]
        let Fr = yShape[2]
        let T = yShape[3]

        y = y.transposed(0, 2, 1, 3)  // [B, Fr, C, T]
        y = y.reshaped([B * Fr, C, T])  // [B*Fr, C, T]
        y = y.transposed(0, 2, 1)  // NCL -> NLC
        y = dconv(y)
        y = y.transposed(0, 2, 1)  // NLC -> NCL
        y = y.reshaped([B, Fr, C, T])
        y = y.transposed(0, 2, 1, 3)  // [B, C, Fr, T]

        // Save pre-output for branch merge
        let pre = y

        // 4. ConvTranspose2d: NCHW -> NHWC -> NCHW
        y = y.transposed(0, 2, 3, 1)
        var z = conv_tr2d!(y)
        z = z.transposed(0, 3, 1, 2)

        // 5. Trim: z[..., pad:-pad, :] for freq
        if pad > 0 {
            let freqDim = z.shape[2]
            z = z[0..., 0..., pad..<(freqDim - pad), 0...]
        }

        // 6. GELU if not last
        if !last {
            z = gelu(z)
        }

        return (z, pre)
    }

    /// Time branch forward pass.
    private func forwardTime(
        _ input: MLXArray,
        skip: MLXArray?,
        length: Int
    ) -> (output: MLXArray, pre: MLXArray) {
        var x = input
        var skipTrimmed = skip

        // 1. Add skip connection
        if let s = skipTrimmed {
            // Handle length mismatch by trimming to shorter length
            if x.shape[2] != s.shape[2] {
                let minT = min(x.shape[2], s.shape[2])
                x = x[0..., 0..., 0..<minT]
                skipTrimmed = s[0..., 0..., 0..<minT]
            }
            x = x + skipTrimmed!
        }

        // 2. Rewrite -> GLU
        x = x.transposed(0, 2, 1)  // NCL -> NLC
        x = rewrite1d!(x)
        x = x.transposed(0, 2, 1)  // NLC -> NCL
        var y = gluNCL(x)

        // 3. DConv
        y = y.transposed(0, 2, 1)
        y = dconv(y)
        y = y.transposed(0, 2, 1)

        // Save pre-output for branch merge
        let pre = y

        // 4. ConvTranspose1d
        y = y.transposed(0, 2, 1)
        var z = conv_tr1d!(y)
        z = z.transposed(0, 2, 1)

        // 5. Trim: z[..., pad:pad+length] for time
        z = z[0..., 0..., pad..<(pad + length)]

        // 6. GELU if not last
        if !last {
            z = gelu(z)
        }

        return (z, pre)
    }

    /// GLU for NCHW format (split on axis 1).
    private func gluNCHW(_ x: MLXArray) -> MLXArray {
        let parts = x.split(parts: 2, axis: 1)
        return parts[0] * sigmoid(parts[1])
    }

    /// GLU for NCL format (split on axis 1).
    private func gluNCL(_ x: MLXArray) -> MLXArray {
        let parts = x.split(parts: 2, axis: 1)
        return parts[0] * sigmoid(parts[1])
    }
}
