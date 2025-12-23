// DConv.swift
// Dilated convolution residual block for HTDemucs.

import Foundation
@preconcurrency import MLX
import MLXNN

// MARK: - DConv Layer

/// A single dilated convolution layer within DConv.
///
/// Structure:
/// - Conv1d (dilated, compress channels)
/// - GroupNorm
/// - GELU
/// - Conv1d (expand to 2x channels for GLU)
/// - GroupNorm
/// - GLU
/// - LayerScale
class DConvLayer: Module, @unchecked Sendable {
    // Weight keys match Python: layers.{i}.0, layers.{i}.1, etc.
    // We use numeric indices to match the Sequential structure

    /// Dilated convolution (index 0 in Python Sequential)
    let conv1: Conv1d

    /// GroupNorm after first conv (index 1)
    let norm1: GroupNorm

    /// Expand convolution (index 3 in Python - indices 2 and 5 are placeholders)
    let conv2: Conv1d

    /// GroupNorm after second conv (index 4)
    let norm2: GroupNorm

    /// LayerScale (index 6)
    let layerScale: LayerScale

    /// Creates a DConv layer.
    /// - Parameters:
    ///   - channels: Number of input/output channels.
    ///   - compress: Compression factor for hidden channels.
    ///   - dilation: Dilation factor for the first convolution.
    ///   - initScale: Initial LayerScale value.
    init(channels: Int, compress: Int = 8, dilation: Int = 1, initScale: Float = 1e-4) {
        let hidden = channels / compress
        // Padding = dilation * (kernel // 2) for same-size output
        let padding = dilation * (3 / 2)  // = dilation * 1 = dilation

        self.conv1 = Conv1d(
            inputChannels: channels,
            outputChannels: hidden,
            kernelSize: 3,
            padding: padding,
            dilation: dilation
        )
        self.norm1 = GroupNorm(groupCount: 1, dimensions: hidden)

        self.conv2 = Conv1d(
            inputChannels: hidden,
            outputChannels: channels * 2,
            kernelSize: 1
        )
        self.norm2 = GroupNorm(groupCount: 1, dimensions: channels * 2)

        self.layerScale = LayerScale(channels: channels, init: initScale)
    }

    /// Forward pass with residual connection.
    /// - Parameter x: Input tensor `[B, T, C]` (NLC format).
    /// - Returns: Output tensor `[B, T, C]`.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = conv1(x)
        out = norm1(out)
        out = gelu(out)
        out = conv2(out)
        out = norm2(out)
        out = glu(out, axis: -1)
        out = layerScale(out)
        return x + out  // Residual connection
    }
}

// MARK: - DConv

/// Multi-depth dilated convolution with residual connections.
///
/// Matches PyTorch demucs DConv exactly.
public class DConv: Module, @unchecked Sendable {
    /// Dilated conv layers with exponentially increasing dilation.
    let layers: [DConvLayer]

    /// Creates a DConv block.
    /// - Parameters:
    ///   - channels: Number of channels.
    ///   - depth: Number of dilated conv layers.
    ///   - compress: Compression factor for hidden channels.
    ///   - initScale: Initial LayerScale value.
    public init(
        channels: Int,
        depth: Int = 2,
        compress: Int = 8,
        initScale: Float = 1e-4
    ) {
        var layerList: [DConvLayer] = []
        for d in 0..<depth {
            let dilation = 1 << d  // 2^d
            layerList.append(
                DConvLayer(
                    channels: channels,
                    compress: compress,
                    dilation: dilation,
                    initScale: initScale
                )
            )
        }
        self.layers = layerList
    }

    /// Forward pass through all dilated conv layers.
    /// - Parameter x: Input tensor `[B, T, C]` (NLC format).
    /// - Returns: Output tensor `[B, T, C]`.
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = x
        for layer in layers {
            out = layer(out)
        }
        return out
    }
}
