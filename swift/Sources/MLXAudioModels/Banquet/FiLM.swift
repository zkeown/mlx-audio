// FiLM.swift
// Feature-wise Linear Modulation for Banquet query-based source separation.

import Foundation
import MLX
import MLXNN

// MARK: - ELU Activation Layer

/// ELU activation function as a UnaryLayer.
private class ELU: Module, UnaryLayer {
    let alpha: Float

    init(alpha: Float = 1.0) {
        self.alpha = alpha
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        return elu(x, alpha: alpha)
    }
}

/// Feature-wise Linear Modulation (FiLM) conditioning.
///
/// Applies query conditioning to band embeddings using learnable
/// scale (gamma) and shift (beta) parameters.
///
/// Architecture:
/// ```
/// x = GroupNorm(x)
/// if multiplicative: x = gamma(w) * x
/// if additive: x = x + beta(w)
/// ```
/// Where w is the conditioning embedding (from PaSST).
public class FiLM: Module, @unchecked Sendable {
    @ModuleInfo var gn: GroupNorm
    @ModuleInfo var gamma: Sequential?
    @ModuleInfo var beta: Sequential?

    let condEmbeddingDim: Int
    let channels: Int
    let additive: Bool
    let multiplicative: Bool

    /// Creates a FiLM module.
    ///
    /// - Parameters:
    ///   - condEmbeddingDim: Conditioning embedding dimension (768 for PaSST)
    ///   - channels: Number of channels in input features
    ///   - additive: Whether to use additive modulation (beta)
    ///   - multiplicative: Whether to use multiplicative modulation (gamma)
    ///   - depth: Depth of modulation networks
    ///   - channelsPerGroup: Channels per group for GroupNorm
    public init(
        condEmbeddingDim: Int,
        channels: Int,
        additive: Bool = true,
        multiplicative: Bool = true,
        depth: Int = 2,
        channelsPerGroup: Int = 16
    ) {
        self.condEmbeddingDim = condEmbeddingDim
        self.channels = channels
        self.additive = additive
        self.multiplicative = multiplicative

        // GroupNorm for input normalization
        let numGroups = max(1, channels / channelsPerGroup)
        _gn.wrappedValue = GroupNorm(groupCount: numGroups, dimensions: channels)

        // Build gamma (multiplicative) network
        if multiplicative {
            if depth == 1 {
                _gamma.wrappedValue = Sequential {
                    Linear(condEmbeddingDim, channels)
                }
            } else {
                var layers: [any UnaryLayer] = [Linear(condEmbeddingDim, channels)]
                for _ in 0..<(depth - 1) {
                    layers.append(ELU())
                    layers.append(Linear(channels, channels))
                }
                _gamma.wrappedValue = Sequential(layers: layers)
            }
        } else {
            _gamma.wrappedValue = nil
        }

        // Build beta (additive) network
        if additive {
            if depth == 1 {
                _beta.wrappedValue = Sequential {
                    Linear(condEmbeddingDim, channels)
                }
            } else {
                var layers: [any UnaryLayer] = [Linear(condEmbeddingDim, channels)]
                for _ in 0..<(depth - 1) {
                    layers.append(ELU())
                    layers.append(Linear(channels, channels))
                }
                _beta.wrappedValue = Sequential(layers: layers)
            }
        } else {
            _beta.wrappedValue = nil
        }
    }

    /// Forward pass.
    ///
    /// - Parameters:
    ///   - x: Input features [batch, channels, n_bands, n_time] (NCHW format)
    ///   - w: Conditioning embedding [batch, cond_embedding_dim]
    /// - Returns: Modulated features with same shape as input
    public func callAsFunction(_ x: MLXArray, conditioning w: MLXArray) -> MLXArray {
        let ndim = x.ndim

        // MLX GroupNorm expects channels-last format (NHWC)
        // Convert NCHW -> NHWC for GroupNorm, then back
        var output: MLXArray
        if ndim == 4 {
            // [B, C, H, W] -> [B, H, W, C]
            let xTransposed = x.transposed(0, 2, 3, 1)
            let normed = gn(xTransposed)
            // [B, H, W, C] -> [B, C, H, W]
            output = normed.transposed(0, 3, 1, 2)
        } else if ndim == 3 {
            // [B, C, L] -> [B, L, C]
            let xTransposed = x.transposed(0, 2, 1)
            let normed = gn(xTransposed)
            // [B, L, C] -> [B, C, L]
            output = normed.transposed(0, 2, 1)
        } else {
            output = gn(x)
        }

        // Multiplicative modulation (gamma)
        if multiplicative, let gammaNet = gamma {
            var gammaVal = gammaNet(w)
            // Broadcast gamma to match input dimensions
            if ndim == 4 {
                gammaVal = gammaVal.expandedDimensions(axes: [2, 3])
            } else if ndim == 3 {
                gammaVal = gammaVal.expandedDimensions(axis: 2)
            }
            output = gammaVal * output
        }

        // Additive modulation (beta)
        if additive, let betaNet = beta {
            var betaVal = betaNet(w)
            // Broadcast beta to match input dimensions
            if ndim == 4 {
                betaVal = betaVal.expandedDimensions(axes: [2, 3])
            } else if ndim == 3 {
                betaVal = betaVal.expandedDimensions(axis: 2)
            }
            output = output + betaVal
        }

        return output
    }

    /// Creates a FiLM module from configuration.
    ///
    /// - Parameter config: Banquet configuration
    /// - Returns: Configured FiLM module
    public static func fromConfig(_ config: BanquetConfig) -> FiLM {
        return FiLM(
            condEmbeddingDim: config.condEmbDim,
            channels: config.embDim,
            additive: config.filmAdditive,
            multiplicative: config.filmMultiplicative,
            depth: config.filmDepth,
            channelsPerGroup: config.channelsPerGroup
        )
    }
}
