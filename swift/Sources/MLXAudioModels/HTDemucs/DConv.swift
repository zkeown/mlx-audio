// DConv.swift
// Dilated convolution residual block for HTDemucs.

import Foundation
@preconcurrency import MLX
import MLXNN

// MARK: - DConv

/// Multi-depth dilated convolution with residual connections.
///
/// Matches PyTorch demucs DConv exactly.
///
/// Structure: layers is a list of lists, where each inner list has 7 elements
/// matching Python's Sequential structure:
///   - 0: Conv1d (dilated)
///   - 1: GroupNorm
///   - 2: (placeholder for GELU - not stored)
///   - 3: Conv1d (expand)
///   - 4: GroupNorm
///   - 5: (placeholder for GLU - not stored)
///   - 6: LayerScale
///
/// Weight keys: layers.{i}.{j}.weight where i is layer index, j is sublayer index.
public class DConv: Module, @unchecked Sendable {
    /// Each layer is represented as a tuple of its parametric components.
    /// We use a list of lists to match Python's nested structure exactly.
    /// Structure: [[Conv1d, GroupNorm, nil, Conv1d, GroupNorm, nil, LayerScale], ...]
    var layers: [[Module?]]

    let channels: Int
    let depth: Int

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
        self.channels = channels
        self.depth = depth

        let hidden = channels / compress

        var layerList: [[Module?]] = []
        for d in 0..<depth {
            let dilation = 1 << d  // 2^d
            let padding = dilation * (3 / 2)

            let sublayers: [Module?] = [
                Conv1d(
                    inputChannels: channels,
                    outputChannels: hidden,
                    kernelSize: 3,
                    padding: padding,
                    dilation: dilation
                ),  // 0
                GroupNorm(groupCount: 1, dimensions: hidden),  // 1
                nil,  // 2: GELU placeholder
                Conv1d(
                    inputChannels: hidden,
                    outputChannels: channels * 2,
                    kernelSize: 1
                ),  // 3
                GroupNorm(groupCount: 1, dimensions: channels * 2),  // 4
                nil,  // 5: GLU placeholder
                LayerScale(channels: channels, init: initScale),  // 6
            ]
            layerList.append(sublayers)
        }
        self.layers = layerList
    }

    /// Forward pass through all dilated conv layers.
    /// - Parameter x: Input tensor `[B, T, C]` (NLC format).
    /// - Returns: Output tensor `[B, T, C]`.
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = x
        for layer in layers {
            out = applyLayer(layer, to: out)
        }
        return out
    }

    /// Apply a single DConv layer.
    private func applyLayer(_ layer: [Module?], to x: MLXArray) -> MLXArray {
        let residual = x
        var out = x

        // 0: Dilated conv
        out = (layer[0] as! Conv1d)(out)
        // 1: GroupNorm
        out = (layer[1] as! GroupNorm)(out)
        // 2: GELU (inline)
        out = gelu(out)
        // 3: Expand conv
        out = (layer[3] as! Conv1d)(out)
        // 4: GroupNorm
        out = (layer[4] as! GroupNorm)(out)
        // 5: GLU (inline)
        out = glu(out, axis: -1)
        // 6: LayerScale
        out = (layer[6] as! LayerScale)(out)

        return residual + out
    }
}
