// Layers.swift
// Convolutional building blocks for EnCodec encoder and decoder.

import Foundation
import MLX
import MLXNN

// MARK: - ConvBlock

/// Convolutional block with causal/same padding and optional activation.
///
/// Applies: Pad -> Conv1d -> Activation
///
/// MLX Conv1d expects [B, T, C] format internally, but the external API
/// uses [B, C, T] format (PyTorch convention). This block handles the
/// format conversion internally.
class ConvBlock: Module, @unchecked Sendable {
    let conv: Conv1d
    let causal: Bool
    let kernelSize: Int
    let dilation: Int
    let activation: String
    let paddingLeft: Int
    let paddingRight: Int

    init(
        inChannels: Int,
        outChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        dilation: Int = 1,
        groups: Int = 1,
        causal: Bool = true,
        activation: String = "elu"
    ) {
        self.causal = causal
        self.kernelSize = kernelSize
        self.dilation = dilation
        self.activation = activation

        // Calculate padding for causal or same convolution
        let effectiveKernel = (kernelSize - 1) * dilation + 1
        if causal {
            self.paddingLeft = effectiveKernel - 1
            self.paddingRight = 0
        } else {
            self.paddingLeft = (effectiveKernel - 1) / 2
            self.paddingRight = effectiveKernel - 1 - paddingLeft
        }

        // Convolution layer (we handle padding manually)
        self.conv = Conv1d(
            inputChannels: inChannels,
            outputChannels: outChannels,
            kernelSize: kernelSize,
            stride: stride,
            padding: 0,
            dilation: dilation
        )

        super.init()
    }

    /// Apply conv block.
    ///
    /// - Parameter x: Input tensor [B, C, T]
    /// - Returns: Output tensor [B, C', T']
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // MLX Conv1d expects [B, T, C] format, we have [B, C, T]
        var out = x.transposed(0, 2, 1)  // [B, C, T] -> [B, T, C]

        // Apply padding (on time dimension, which is now axis 1)
        if paddingLeft > 0 || paddingRight > 0 {
            out = padded(out, widths: [[0, 0], [paddingLeft, paddingRight], [0, 0]])
        }

        // Apply convolution
        out = conv(out)

        // Back to [B, C, T] format
        out = out.transposed(0, 2, 1)  // [B, T, C] -> [B, C, T]

        // Apply activation
        if activation == "elu" {
            out = elu(out)
        }

        return out
    }
}

// MARK: - ConvTransposeBlock

/// Transposed convolutional block for upsampling.
///
/// Applies: ConvTranspose1d -> Trim -> Activation
///
/// Handles the output trimming required for proper output length after
/// transposed convolution.
class ConvTransposeBlock: Module, @unchecked Sendable {
    let convTranspose: ConvTransposed1d
    let causal: Bool
    let kernelSize: Int
    let stride: Int
    let activation: String
    let trimAmount: Int

    init(
        inChannels: Int,
        outChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        causal: Bool = true,
        activation: String = "elu"
    ) {
        self.causal = causal
        self.kernelSize = kernelSize
        self.stride = stride
        self.activation = activation

        // Calculate trim amount for transposed convolution
        // For transposed conv, output_size = (input_size - 1) * stride + kernel_size
        // We need to trim to get the right output size
        self.trimAmount = kernelSize - stride

        // Transposed convolution
        self.convTranspose = ConvTransposed1d(
            inputChannels: inChannels,
            outputChannels: outChannels,
            kernelSize: kernelSize,
            stride: stride,
            padding: 0
        )

        super.init()
    }

    /// Apply transposed conv block.
    ///
    /// - Parameter x: Input tensor [B, C, T]
    /// - Returns: Output tensor [B, C', T * stride]
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // MLX ConvTranspose1d expects [B, T, C] format, we have [B, C, T]
        var out = x.transposed(0, 2, 1)  // [B, C, T] -> [B, T, C]

        // Apply transposed convolution
        out = convTranspose(out)

        // Back to [B, C, T] format
        out = out.transposed(0, 2, 1)  // [B, T, C] -> [B, C, T]

        // Trim excess samples for proper output length
        if trimAmount > 0 {
            let timeLen = out.dim(2)
            if causal {
                // Trim from the right for causal
                out = out[0..., 0..., 0..<(timeLen - trimAmount)]
            } else {
                // Trim equally from both sides
                let trimLeft = trimAmount / 2
                let trimRight = trimAmount - trimLeft
                if trimRight > 0 {
                    out = out[0..., 0..., trimLeft..<(timeLen - trimRight)]
                } else {
                    out = out[0..., 0..., trimLeft...]
                }
            }
        }

        // Apply activation
        if activation == "elu" {
            out = elu(out)
        }

        return out
    }
}

// MARK: - ResidualUnit

/// Residual unit with dilated convolutions.
///
/// Architecture: Conv(dilated) -> ELU -> Conv(1x1) + skip connection
class ResidualUnit: Module, @unchecked Sendable {
    let conv1: ConvBlock
    let conv2: ConvBlock

    init(
        channels: Int,
        kernelSize: Int = 3,
        dilation: Int = 1,
        causal: Bool = true
    ) {
        // Dilated convolution with activation
        self.conv1 = ConvBlock(
            inChannels: channels,
            outChannels: channels,
            kernelSize: kernelSize,
            dilation: dilation,
            causal: causal,
            activation: "elu"
        )

        // 1x1 convolution (no activation before residual)
        self.conv2 = ConvBlock(
            inChannels: channels,
            outChannels: channels,
            kernelSize: 1,
            causal: causal,
            activation: "none"
        )

        super.init()
    }

    /// Apply residual unit.
    ///
    /// - Parameter x: Input tensor [B, C, T]
    /// - Returns: Output tensor [B, C, T]
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let residual = x
        var out = conv1(x)
        out = conv2(out)
        return out + residual
    }
}

// MARK: - Helper Functions

/// Apply ELU activation function.
func elu(_ x: MLXArray, alpha: Float = 1.0) -> MLXArray {
    // ELU: x if x > 0, alpha * (exp(x) - 1) if x <= 0
    let positive = maximum(x, MLXArray(0))
    let negative = minimum(x, MLXArray(0))
    return positive + alpha * (exp(negative) - 1)
}
