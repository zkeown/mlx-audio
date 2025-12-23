// Decoder.swift
// EnCodec convolutional decoder.

import Foundation
import MLX
import MLXNN

// MARK: - DecoderBlock

/// Decoder block with upsampling and residual units.
///
/// Architecture: UpsampleConv -> ELU -> [ResidualUnits]
class DecoderBlock: Module, @unchecked Sendable {
    let upsample: ConvTransposeBlock
    let residuals: [ResidualUnit]

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
        // Upsampling transposed convolution
        // EnCodec uses 2x stride for kernel size
        self.upsample = ConvTransposeBlock(
            inChannels: inChannels,
            outChannels: outChannels,
            kernelSize: stride * 2,
            stride: stride,
            causal: causal,
            activation: "none"
        )

        // Residual units with increasing dilation
        var residuals: [ResidualUnit] = []
        for i in 0..<numResidualLayers {
            let dilation = Int(pow(Double(dilationBase), Double(i)))
            residuals.append(
                ResidualUnit(
                    channels: outChannels,
                    kernelSize: residualKernelSize,
                    dilation: dilation,
                    causal: causal
                )
            )
        }
        self.residuals = residuals

        super.init()
    }

    /// Apply decoder block.
    ///
    /// - Parameter x: Input tensor [B, C, T]
    /// - Returns: Output tensor [B, C', T * stride]
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = x

        // Upsample
        out = upsample(out)

        // ELU after upsampling
        out = elu(out)

        // Apply residual units
        for residual in residuals {
            out = residual(out)
        }

        return out
    }
}

// MARK: - EnCodecDecoder

/// EnCodec convolutional decoder.
///
/// Decodes latent embeddings back to audio waveform with progressive upsampling.
///
/// Architecture:
///     Conv1d(initial) -> ELU -> LSTM -> [DecoderBlock(stride) for stride in reversed(ratios)]
///     -> ELU -> Conv1d(final)
///
/// Input format: [B, T', D] where T' is compressed time, D = codebook_dim
/// Output format: [B, C, T] where C is audio channels, T = T' * hop_length
class EnCodecDecoder: Module, @unchecked Sendable {
    let config: EnCodecConfig
    let initialConv: ConvBlock
    let lstmLayersList: [LSTM]
    let blocks: [DecoderBlock]
    let finalConv: ConvBlock

    init(config: EnCodecConfig) {
        self.config = config

        // Calculate channels at each stage (reverse of encoder)
        // Encoder: filters -> filters*2 -> filters*4 -> filters*8 -> ...
        // Decoder: ... -> filters*4 -> filters*2 -> filters
        let mult = 1 << config.ratios.count  // 2^num_stages
        let hiddenChannels = config.num_filters * mult

        // Initial convolution from codebook dimension
        self.initialConv = ConvBlock(
            inChannels: config.codebook_dim,
            outChannels: hiddenChannels,
            kernelSize: config.kernel_size,
            causal: config.causal,
            activation: "none"
        )

        // LSTM layers (stack manually since MLX LSTM doesn't support num_layers)
        var lstmLayers: [LSTM] = []
        if config.lstm_layers > 0 {
            for _ in 0..<config.lstm_layers {
                lstmLayers.append(LSTM(inputSize: hiddenChannels, hiddenSize: hiddenChannels))
            }
        }
        self.lstmLayersList = lstmLayers

        // Decoder blocks with progressive upsampling (reversed ratios)
        var blocks: [DecoderBlock] = []
        var inChannels = hiddenChannels
        for ratio in config.ratios.reversed() {
            let outChannels = inChannels / 2
            blocks.append(
                DecoderBlock(
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

        // Final convolution to audio channels
        self.finalConv = ConvBlock(
            inChannels: config.num_filters,
            outChannels: config.channels,
            kernelSize: config.last_kernel_size,
            causal: config.causal,
            activation: "none"
        )

        super.init()
    }

    /// Decode latent embeddings to audio.
    ///
    /// - Parameter x: Latent embeddings [B, T', D]
    /// - Returns: Audio waveform [B, C, T] where T = T' * hop_length
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Convert from [B, T', D] to [B, D, T']
        var out = x.transposed(0, 2, 1)

        // Initial convolution
        out = initialConv(out)
        out = elu(out)

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

        // Decoder blocks
        for block in blocks {
            out = block(out)
        }

        // Final convolution
        out = elu(out)
        out = finalConv(out)

        return out
    }
}
