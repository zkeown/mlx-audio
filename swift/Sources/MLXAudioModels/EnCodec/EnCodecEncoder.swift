// Encoder.swift
// EnCodec convolutional encoder.

import Foundation
import MLX
import MLXNN

// MARK: - EncoderBlock

/// Encoder block with residual units and downsampling.
///
/// Architecture: [ResidualUnits] -> ELU -> DownsampleConv
class EncoderBlock: Module, @unchecked Sendable {
    let residuals: [ResidualUnit]
    let downsample: ConvBlock

    init(
        inChannels: Int,
        outChannels: Int,
        stride: Int,
        kernelSize: Int = 7,
        residualKernelSize: Int = 3,
        numResidualLayers: Int = 1,
        dilationBase: Int = 2,
        causal: Bool = true
    ) {
        // Residual units with increasing dilation
        var residuals: [ResidualUnit] = []
        for i in 0..<numResidualLayers {
            let dilation = Int(pow(Double(dilationBase), Double(i)))
            residuals.append(
                ResidualUnit(
                    channels: inChannels,
                    kernelSize: residualKernelSize,
                    dilation: dilation,
                    causal: causal
                )
            )
        }
        self.residuals = residuals

        // Downsampling strided convolution
        // EnCodec uses 2x stride for kernel size
        self.downsample = ConvBlock(
            inChannels: inChannels,
            outChannels: outChannels,
            kernelSize: stride * 2,
            stride: stride,
            causal: causal,
            activation: "none"
        )

        super.init()
    }

    /// Apply encoder block.
    ///
    /// - Parameter x: Input tensor [B, C, T]
    /// - Returns: Output tensor [B, C', T//stride]
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = x

        // Apply residual units
        for residual in residuals {
            out = residual(out)
        }

        // ELU before downsampling
        out = elu(out)

        // Downsample
        out = downsample(out)

        return out
    }
}

// MARK: - EnCodecEncoder

/// EnCodec convolutional encoder.
///
/// Encodes audio waveform to latent embeddings with progressive downsampling.
///
/// Architecture:
///     Conv1d(initial) -> ELU -> [EncoderBlock(stride) for stride in ratios] ->
///     LSTM -> ELU -> Conv1d(final)
///
/// Input format: [B, C, T] where C is audio channels
/// Output format: [B, T', D] where T' = T / hop_length, D = codebook_dim
class EnCodecEncoder: Module, @unchecked Sendable {
    let config: EnCodecConfig
    let initialConv: ConvBlock
    let blocks: [EncoderBlock]
    let lstmLayersList: [LSTM]
    let finalConv: ConvBlock

    init(config: EnCodecConfig) {
        self.config = config

        // Initial convolution
        self.initialConv = ConvBlock(
            inChannels: config.channels,
            outChannels: config.num_filters,
            kernelSize: config.kernel_size,
            causal: config.causal,
            activation: "none"
        )

        // Encoder blocks with progressive downsampling
        var blocks: [EncoderBlock] = []
        var inChannels = config.num_filters
        for ratio in config.ratios {
            let outChannels = inChannels * 2
            blocks.append(
                EncoderBlock(
                    inChannels: inChannels,
                    outChannels: outChannels,
                    stride: ratio,
                    kernelSize: config.kernel_size,
                    residualKernelSize: config.residual_kernel_size,
                    numResidualLayers: config.num_residual_layers,
                    dilationBase: config.dilation_base,
                    causal: config.causal
                )
            )
            inChannels = outChannels
        }
        self.blocks = blocks

        // LSTM layers (stack manually since MLX LSTM doesn't support num_layers)
        var lstmLayers: [LSTM] = []
        if config.lstm_layers > 0 {
            for _ in 0..<config.lstm_layers {
                lstmLayers.append(LSTM(inputSize: inChannels, hiddenSize: inChannels))
            }
        }
        self.lstmLayersList = lstmLayers

        // Final convolution to codebook dimension
        self.finalConv = ConvBlock(
            inChannels: inChannels,
            outChannels: config.codebook_dim,
            kernelSize: config.last_kernel_size,
            causal: config.causal,
            activation: "none"
        )

        super.init()
    }

    /// Encode audio waveform.
    ///
    /// - Parameter x: Audio waveform [B, C, T] or [B, T] (mono)
    /// - Returns: Latent embeddings [B, T', D] where T' = T / hop_length
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = x

        // Handle mono input [B, T] -> [B, 1, T]
        if out.ndim == 2 {
            out = out.expandedDimensions(axis: 1)
        }

        // Initial convolution
        out = initialConv(out)
        out = elu(out)

        // Encoder blocks
        for block in blocks {
            out = block(out)
        }

        // LSTM (if enabled)
        if !lstmLayersList.isEmpty {
            // LSTM expects [B, T, C], we have [B, C, T]
            out = out.transposed(0, 2, 1)
            for lstm in lstmLayersList {
                let (output, _) = lstm(out)
                out = output
            }
            // Back to [B, C, T]
            out = out.transposed(0, 2, 1)
        }

        // Final convolution
        out = elu(out)
        out = finalConv(out)

        // Return [B, T', D] format (transpose from [B, D, T'])
        return out.transposed(0, 2, 1)
    }
}
