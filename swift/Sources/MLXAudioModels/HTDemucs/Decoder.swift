// Decoder.swift
// Decoder layers for HTDemucs.
//
// OPTIMIZATION: Uses MLX-native NHWC/NLC format internally to avoid transposes.
// Format conversion happens at model boundaries in HTDemucs.swift.

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
/// INTERNAL FORMAT (optimized for MLX):
/// - freq=true: `[B, F, T, C]` (NHWC format)
/// - freq=false: `[B, T, C]` (NLC format)
///
/// NOTE: model.py converts from PyTorch format (NCHW/NCL) at entry
/// and converts back at exit. Layers operate in MLX-native format.
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

    // Transposed conv layer - stores ConvTransposed2d for freq, ConvTransposed1d for time
    // Key "conv_tr" matches Python weight keys
    let conv_tr: Module

    // Rewrite conv layer - stores Conv2d for freq, Conv1d for time
    // Key "rewrite" matches Python weight keys
    let rewrite: Module

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
            self.conv_tr = ConvTransposed2d(
                inputChannels: chin,
                outputChannels: chout,
                kernelSize: .init((kernelSize, 1)),
                stride: .init((stride, 1)),
                padding: 0
            )
            // Rewrite conv: 3x3 that doubles channels for GLU
            self.rewrite = Conv2d(
                inputChannels: chin,
                outputChannels: chin * 2,
                kernelSize: 3,
                padding: 1
            )
        } else {
            // Time branch uses ConvTransposed1d
            self.conv_tr = ConvTransposed1d(
                inputChannels: chin,
                outputChannels: chout,
                kernelSize: kernelSize,
                stride: stride,
                padding: 0
            )
            // Rewrite conv: kernel 3 that doubles channels for GLU
            self.rewrite = Conv1d(
                inputChannels: chin,
                outputChannels: chin * 2,
                kernelSize: 3,
                padding: 1
            )
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
    ///   - x: Input tensor from previous decoder layer (NHWC/NLC format)
    ///     - freq=true: `[B, F, T, C]` (NHWC format)
    ///     - freq=false: `[B, T, C]` (NLC format)
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
    /// Input: [B, F, T, C] (NHWC) - no transposes needed!
    private func forwardFreq(
        _ input: MLXArray,
        skip: MLXArray?,
        length: Int
    ) -> (output: MLXArray, pre: MLXArray) {
        var x = input
        var skipTrimmed = skip
        let rewrite2d = rewrite as! Conv2d
        let conv_tr2d = conv_tr as! ConvTransposed2d

        // 1. Add skip connection
        if let s = skipTrimmed {
            // Handle length mismatch by trimming to shorter length
            // In NHWC: F is axis 1
            if x.shape[1] != s.shape[1] {
                let minF = min(x.shape[1], s.shape[1])
                x = x[0..., 0..<minF, 0..., 0...]
                skipTrimmed = s[0..., 0..<minF, 0..., 0...]
            }
            x = x + skipTrimmed!
        }

        let B = x.shape[0]

        // 2. Rewrite -> GLU (no transpose needed, already NHWC)
        x = rewrite2d(x)
        var y = glu(x, axis: -1)

        // 3. DConv: collapse freq into batch
        // [B, F, T, C] -> [B*F, T, C] - already NLC!
        let yShape = y.shape
        let Fr2 = yShape[1]
        let T2 = yShape[2]
        let C2 = yShape[3]

        y = y.reshaped([B * Fr2, T2, C2])
        y = dconv(y)
        y = y.reshaped([B, Fr2, T2, C2])

        // Save pre-output for branch merge
        let pre = y

        // 4. ConvTranspose2d (no transpose, already NHWC)
        var z = conv_tr2d(y)

        // 5. Trim: z[:, pad:-pad, :, :] for freq (F is axis 1 in NHWC)
        if pad > 0 {
            let freqDim = z.shape[1]
            z = z[0..., pad..<(freqDim - pad), 0..., 0...]
        }

        // 6. GELU if not last
        if !last {
            z = gelu(z)
        }

        return (z, pre)
    }

    /// Time branch forward pass.
    /// Input: [B, T, C] (NLC) - no transposes needed!
    private func forwardTime(
        _ input: MLXArray,
        skip: MLXArray?,
        length: Int
    ) -> (output: MLXArray, pre: MLXArray) {
        var x = input
        var skipTrimmed = skip
        let rewrite1d = rewrite as! Conv1d
        let conv_tr1d = conv_tr as! ConvTransposed1d

        // 1. Add skip connection
        if let s = skipTrimmed {
            // Handle length mismatch (T is axis 1 in NLC)
            if x.shape[1] != s.shape[1] {
                let minT = min(x.shape[1], s.shape[1])
                x = x[0..., 0..<minT, 0...]
                skipTrimmed = s[0..., 0..<minT, 0...]
            }
            x = x + skipTrimmed!
        }

        // 2. Rewrite -> GLU (no transpose, already NLC)
        x = rewrite1d(x)
        var y = glu(x, axis: -1)

        // 3. DConv (already NLC)
        y = dconv(y)

        // Save pre-output for branch merge
        let pre = y

        // 4. ConvTranspose1d (no transpose, already NLC)
        var z = conv_tr1d(y)

        // 5. Trim: z[:, pad:pad+length, :] for time (T is axis 1 in NLC)
        z = z[0..., pad..<(pad + length), 0...]

        // 6. GELU if not last
        if !last {
            z = gelu(z)
        }

        return (z, pre)
    }
}

