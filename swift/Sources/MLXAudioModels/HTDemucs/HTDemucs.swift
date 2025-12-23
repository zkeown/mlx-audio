// HTDemucs.swift
// HTDemucs model implementation for source separation.
//
// OPTIMIZATION: Uses MLX-native NHWC/NLC format internally to avoid transposes.
// Format conversion happens at model boundaries.

import Foundation
@preconcurrency import MLX
import MLXNN
import MLXAudioPrimitives

// MARK: - HTDemucs Model

/// Hybrid Transformer Demucs model for source separation.
///
/// Structure matches PyTorch exactly:
/// - encoder: list of HEncLayer (frequency branch, Conv2d)
/// - decoder: list of HDecLayer (frequency branch, ConvTranspose2d)
/// - tencoder: list of HEncLayer (time branch, Conv1d)
/// - tdecoder: list of HDecLayer (time branch, ConvTranspose1d)
/// - freq_emb: ScaledEmbedding
/// - channel_upsampler: Conv1d
/// - channel_downsampler: Conv1d
/// - channel_upsampler_t: Conv1d
/// - channel_downsampler_t: Conv1d
/// - crosstransformer: CrossTransformerEncoder
public class HTDemucs: Module, @unchecked Sendable {

    public let config: HTDemucsConfig

    // Channel progression
    let channels: [Int]

    // Frequency branch encoders/decoders
    let encoder: [HEncLayer]
    let decoder: [HDecLayer]

    // Time branch encoders/decoders
    let tencoder: [HEncLayer]
    let tdecoder: [HDecLayer]

    // Frequency embedding
    let freq_emb: ScaledEmbedding
    let nFreqs: Int
    var cachedFreqEmb: MLXArray?

    // Channel up/downsamplers
    let channel_upsampler: Conv1d
    let channel_downsampler: Conv1d
    let channel_upsampler_t: Conv1d
    let channel_downsampler_t: Conv1d

    // Cross-domain transformer
    let crosstransformer: CrossTransformerEncoder

    /// Creates an HTDemucs model.
    /// - Parameter config: Model configuration.
    public init(config: HTDemucsConfig = HTDemucsConfig()) {
        self.config = config

        // Calculate channel progression
        var channelList: [Int] = [config.channels]
        for _ in 0..<(config.depth - 1) {
            channelList.append(Int(Float(channelList.last!) * config.growth))
        }
        self.channels = channelList

        let chin = config.audio_channels
        let chinZ = config.cac ? chin * 2 : chin

        // Frequency branch encoders
        var encoderLayers: [HEncLayer] = []
        for i in 0..<channelList.count {
            let chinEnc = i == 0 ? chinZ : channelList[i - 1]
            encoderLayers.append(
                HEncLayer(
                    chin: chinEnc,
                    chout: channelList[i],
                    kernelSize: config.kernel_size,
                    stride: config.stride,
                    freq: true,
                    dconvDepth: config.dconv_depth,
                    dconvCompress: config.dconv_comp
                )
            )
        }
        self.encoder = encoderLayers

        // Time branch encoders
        var tencoderLayers: [HEncLayer] = []
        for i in 0..<channelList.count {
            let chinEnc = i == 0 ? chin : channelList[i - 1]
            tencoderLayers.append(
                HEncLayer(
                    chin: chinEnc,
                    chout: channelList[i],
                    kernelSize: config.kernel_size,
                    stride: config.stride,
                    freq: false,
                    dconvDepth: config.dconv_depth,
                    dconvCompress: config.dconv_comp
                )
            )
        }
        self.tencoder = tencoderLayers

        // Frequency branch decoders (reversed)
        var decoderLayers: [HDecLayer] = []
        for i in stride(from: channelList.count - 1, through: 0, by: -1) {
            let chinDec = channelList[i]
            let choutDec = i > 0 ? channelList[i - 1] : chinZ * config.num_sources
            decoderLayers.append(
                HDecLayer(
                    chin: chinDec,
                    chout: choutDec,
                    kernelSize: config.kernel_size,
                    stride: config.stride,
                    freq: true,
                    dconvDepth: config.dconv_depth,
                    dconvCompress: config.dconv_comp,
                    last: i == 0
                )
            )
        }
        self.decoder = decoderLayers

        // Time branch decoders (reversed)
        var tdecoderLayers: [HDecLayer] = []
        for i in stride(from: channelList.count - 1, through: 0, by: -1) {
            let chinDec = channelList[i]
            let choutDec = i > 0 ? channelList[i - 1] : chin * config.num_sources
            tdecoderLayers.append(
                HDecLayer(
                    chin: chinDec,
                    chout: choutDec,
                    kernelSize: config.kernel_size,
                    stride: config.stride,
                    freq: false,
                    dconvDepth: config.dconv_depth,
                    dconvCompress: config.dconv_comp,
                    last: i == 0
                )
            )
        }
        self.tdecoder = tdecoderLayers

        // Frequency embedding
        self.nFreqs = (config.nfft / 2) / config.stride
        self.freq_emb = ScaledEmbedding(
            numEmbeddings: nFreqs,
            embeddingDim: channelList[0],
            scale: 10.0
        )
        self.cachedFreqEmb = nil

        // Channel up/downsamplers
        let encoderChannels = channelList.last!
        let transformerChannels = config.bottom_channels > 0 ? config.bottom_channels : encoderChannels

        self.channel_upsampler = Conv1d(
            inputChannels: encoderChannels,
            outputChannels: transformerChannels,
            kernelSize: 1
        )
        self.channel_downsampler = Conv1d(
            inputChannels: transformerChannels,
            outputChannels: encoderChannels,
            kernelSize: 1
        )
        self.channel_upsampler_t = Conv1d(
            inputChannels: encoderChannels,
            outputChannels: transformerChannels,
            kernelSize: 1
        )
        self.channel_downsampler_t = Conv1d(
            inputChannels: transformerChannels,
            outputChannels: encoderChannels,
            kernelSize: 1
        )

        // Cross-domain transformer
        self.crosstransformer = CrossTransformerEncoder(
            dim: transformerChannels,
            depth: config.t_depth,
            heads: config.t_heads,
            dimFeedforward: Int(Float(transformerChannels) * config.t_hidden_scale),
            dropout: config.t_dropout
        )
    }

    /// Forward pass.
    /// - Parameter mix: Input mixture `[B, C, T]` where C=audio_channels.
    /// - Returns: Separated stems `[B, S, C, T]` where S=num_sources.
    public func callAsFunction(_ mix: MLXArray) -> MLXArray {
        var inputMix = mix
        let B = inputMix.shape[0]
        let C = inputMix.shape[1]
        var T = inputMix.shape[2]
        let S = config.num_sources

        // Pad to training length if needed
        let trainingLength = Int(config.segment * Float(config.samplerate))
        var lengthPrePad: Int? = nil
        if T < trainingLength {
            lengthPrePad = T
            let padAmount = trainingLength - T
            inputMix = MLX.padded(inputMix, widths: [[0, 0], [0, 0], [0, padAmount]])
            T = trainingLength
        }

        // Compute STFT for frequency branch
        let spec = computeSTFT(inputMix)

        // Prepare frequency input - convert complex to real
        var mag: MLXArray
        if config.cac {
            mag = complexToCAC(spec)
        } else {
            mag = spec.magnitude()
        }

        // Normalize frequency input
        let specMean = mag.mean(axes: [1, 2, 3], keepDims: true)
        let specStd = MLX.sqrt(mag.variance(axes: [1, 2, 3], keepDims: true)) + 1e-5
        let freqIn = (mag - specMean) / specStd

        // Normalize time input
        let mixMean = inputMix.mean(axes: [1, 2], keepDims: true)
        let mixStd = MLX.sqrt(inputMix.variance(axes: [1, 2], keepDims: true)) + 1e-5
        let mixNorm = (inputMix - mixMean) / mixStd

        // ===== FORMAT CONVERSION: NCHW/NCL -> NHWC/NLC =====
        // Convert to MLX-native format ONCE at entry (avoids ~124 transposes)
        // Freq: [B, C, F, T] -> [B, F, T, C]
        var x = freqIn.transposed(0, 2, 3, 1)
        // Time: [B, C, T] -> [B, T, C]
        var xt = mixNorm.transposed(0, 2, 1)

        // Encode frequency branch
        var freqLengths: [Int] = []
        var freqSkips: [MLXArray] = []

        for (idx, enc) in encoder.enumerated() {
            // In NHWC, T is at axis 2
            freqLengths.append(x.shape[2])
            x = enc(x)

            // Add frequency embedding after first encoder
            if idx == 0 {
                if cachedFreqEmb == nil {
                    let frs = MLXArray(0..<nFreqs)
                    var emb = freq_emb(frs)  // [F, C]
                    // For NHWC format: [1, F, 1, C]
                    emb = emb.expandedDimensions(axes: [0, 2])  // [1, F, 1, C]
                    cachedFreqEmb = emb
                }
                x = x + config.freq_emb * cachedFreqEmb!
            }

            freqSkips.append(x)
        }

        // Encode time branch
        var timeLengths: [Int] = []
        var timeSkips: [MLXArray] = []

        for enc in tencoder {
            // In NLC, T is at axis 1
            timeLengths.append(xt.shape[1])
            xt = enc(xt)
            timeSkips.append(xt)
        }

        // Evaluate after encoder phase to release intermediate tensors
        eval(x, xt)

        // Channel upsample before transformer
        // x is [B, F, T, C] in NHWC format
        let Bx = x.shape[0]
        let Fx = x.shape[1]
        let Tx = x.shape[2]
        let Cx = x.shape[3]

        // Flatten freq: [B, F, T, C] -> [B, F*T, C] (already NLC!)
        var xFlat = x.reshaped([Bx, Fx * Tx, Cx])

        // Upsample channels (Conv1d in NLC format, no transpose needed)
        xFlat = channel_upsampler(xFlat)

        // Unflatten back to 4D: [B, F*T, C'] -> [B, F, T, C']
        x = xFlat.reshaped([Bx, Fx, Tx, -1])

        // Upsample time channels (already NLC, no transpose needed)
        xt = channel_upsampler_t(xt)

        // Cross-domain transformer
        // Transformer expects NCHW format, convert temporarily
        let xNCHW = x.transposed(0, 3, 1, 2)  // [B, F, T, C] -> [B, C, F, T]
        let xtNCL = xt.transposed(0, 2, 1)   // [B, T, C] -> [B, C, T]
        let transformerOutput = crosstransformer(freq: xNCHW, time: xtNCL)
        // Convert back to NHWC/NLC
        x = transformerOutput.freq.transposed(0, 2, 3, 1)  // [B, C, F, T] -> [B, F, T, C]
        xt = transformerOutput.time.transposed(0, 2, 1)   // [B, C, T] -> [B, T, C]

        // Evaluate after transformer to release attention intermediates
        eval(x, xt)

        // Flatten freq for downsample: [B, F, T, C] -> [B, F*T, C]
        xFlat = x.reshaped([Bx, Fx * Tx, -1])

        // Downsample channels (already NLC, no transpose needed)
        xFlat = channel_downsampler(xFlat)

        // Downsample time channels (already NLC, no transpose needed)
        xt = channel_downsampler_t(xt)

        // Unflatten back to 4D: [B, F*T, C] -> [B, F, T, C]
        x = xFlat.reshaped([Bx, Fx, Tx, Cx])

        // Decode frequency branch (with skips, reversed order)
        // x is in NHWC format: [B, F, T, C]
        let reversedFreqSkips = freqSkips.reversed()
        let reversedFreqLengths = freqLengths.reversed()
        for (idx, dec) in decoder.enumerated() {
            let skip = Array(reversedFreqSkips)[idx]
            let length = Array(reversedFreqLengths)[idx]
            let result = dec(x, skip: skip, length: length)
            x = result.output
        }

        // Decode time branch (with skips, reversed order)
        // xt is in NLC format: [B, T, C]
        let reversedTimeSkips = timeSkips.reversed()
        let reversedTimeLengths = timeLengths.reversed()
        for (idx, dec) in tdecoder.enumerated() {
            let skip = Array(reversedTimeSkips)[idx]
            let length = Array(reversedTimeLengths)[idx]
            let result = dec(xt, skip: skip, length: length)
            xt = result.output
        }

        // Evaluate after decoder phase to release skip connections
        eval(x, xt)

        // ===== FORMAT CONVERSION: NHWC/NLC -> NCHW/NCL =====
        // Convert back to PyTorch format for output processing
        // Freq: [B, F, T, C] -> [B, C, F, T]
        x = x.transposed(0, 3, 1, 2)
        // Time: [B, T, C] -> [B, C, T]
        xt = xt.transposed(0, 2, 1)

        // Process frequency branch output
        let FOut = x.shape[2]
        let TSpec = x.shape[3]
        let CReal = config.cac ? C * 2 : C
        x = x.reshaped([B, S, CReal, FOut, TSpec])

        // Denormalize frequency output
        x = x * specStd.expandedDimensions(axis: 1) + specMean.expandedDimensions(axis: 1)

        // Apply iSTFT
        let freqOut = maskAndISTFT(spec: spec, m: x, length: T)

        // Process time branch output
        let TOut = xt.shape[xt.ndim - 1]
        var timeOut = xt.reshaped([B, S, C, TOut])

        // Pad or trim to original length
        if TOut < T {
            let padAmount = T - TOut
            timeOut = MLX.padded(timeOut, widths: [[0, 0], [0, 0], [0, 0], [0, padAmount]])
        } else if TOut > T {
            timeOut = timeOut[0..., 0..., 0..., 0..<T]
        }

        // Denormalize time output
        timeOut = timeOut * mixStd.expandedDimensions(axis: -1) + mixMean.expandedDimensions(axis: -1)

        // Combine frequency and time outputs
        var output = timeOut + freqOut

        // Trim to original length if we padded
        if let originalLength = lengthPrePad {
            output = output[0..., 0..., 0..., 0..<originalLength]
        }

        return output
    }

    // MARK: - STFT/iSTFT Helpers

    /// Compute STFT for frequency branch.
    private func computeSTFT(_ x: MLXArray) -> ComplexArray {
        let B = x.shape[0]
        let C = x.shape[1]
        let T = x.shape[2]
        let nFft = config.nfft
        let hopLength = config.hop_length

        // Calculate expected output length
        let le = Int(ceil(Double(T) / Double(hopLength)))
        let pad = hopLength / 2 * 3

        // Pad input
        let xPadded = MLX.padded(x, widths: [[0, 0], [0, 0], [pad, pad + le * hopLength - T]])

        // Flatten batch and channels
        let xFlat = xPadded.reshaped([B * C, -1])

        // Compute STFT using primitives (use .edge since .reflect unavailable)
        let stftConfig = STFTConfig(
            nFFT: nFft,
            hopLength: hopLength,
            padMode: .edge
        )
        var spec = try! stft(xFlat, config: stftConfig)

        // Normalize by sqrt(nfft) to match PyTorch normalized=True
        let normFactor = Float(sqrt(Double(nFft)))
        spec = ComplexArray(
            real: spec.real / normFactor,
            imag: spec.imag / normFactor
        )

        // Remove Nyquist bin
        let freqBins = spec.shape[1]
        spec = ComplexArray(
            real: spec.real[0..., 0..<(freqBins - 1), 0...],
            imag: spec.imag[0..., 0..<(freqBins - 1), 0...]
        )

        // Trim time frames
        spec = ComplexArray(
            real: spec.real[0..., 0..., 2..<(2 + le)],
            imag: spec.imag[0..., 0..., 2..<(2 + le)]
        )

        // Reshape back to [B, C, F, T]
        let F = spec.shape[1]
        let Tf = spec.shape[2]
        return ComplexArray(
            real: spec.real.reshaped([B, C, F, Tf]),
            imag: spec.imag.reshaped([B, C, F, Tf])
        )
    }

    /// Convert complex spectrogram to CAC (Complex-As-Channels) format.
    private func complexToCAC(_ spec: ComplexArray) -> MLXArray {
        // spec: [B, C, F, T] complex -> [B, C*2, F, T] real
        let realPart = spec.real
        let imagPart = spec.imag

        // Stack and reshape to interleave
        let stacked = MLX.stacked([realPart, imagPart], axis: 2)  // [B, C, 2, F, T]
        let shape = stacked.shape
        let B = shape[0]
        let C = shape[1]
        let F = shape[3]
        let T = shape[4]
        return stacked.reshaped([B, C * 2, F, T])
    }

    /// Apply mask to STFT and compute iSTFT.
    private func maskAndISTFT(spec: ComplexArray, m: MLXArray, length: Int) -> MLXArray {
        let B = m.shape[0]
        let S = m.shape[1]
        let CReal = m.shape[2]
        let F = m.shape[3]
        let Tf = m.shape[4]
        let C = CReal / 2
        let hopLength = config.hop_length
        let nFft = config.nfft

        var zOut: ComplexArray

        if config.cac {
            // CAC mode: m is the full spectrogram in real format
            // Convert from [B, S, C*2, F, T] to complex [B, S, C, F, T]
            let mReshaped = m.reshaped([B, S, C, 2, F, Tf])
            let mPerm = mReshaped.transposed(0, 1, 2, 4, 5, 3)  // [B, S, C, F, T, 2]
            zOut = ComplexArray(
                real: mPerm[0..., 0..., 0..., 0..., 0..., 0],
                imag: mPerm[0..., 0..., 0..., 0..., 0..., 1]
            )
        } else {
            // Magnitude mask mode
            let zExpanded = ComplexArray(
                real: spec.real.expandedDimensions(axis: 1),
                imag: spec.imag.expandedDimensions(axis: 1)
            )
            let mag = zExpanded.magnitude() + 1e-8
            zOut = ComplexArray(
                real: zExpanded.real / mag * m,
                imag: zExpanded.imag / mag * m
            )
        }

        // Pad freq axis (add Nyquist bin)
        zOut = ComplexArray(
            real: MLX.padded(zOut.real, widths: [[0, 0], [0, 0], [0, 0], [0, 1], [0, 0]]),
            imag: MLX.padded(zOut.imag, widths: [[0, 0], [0, 0], [0, 0], [0, 1], [0, 0]])
        )

        // Pad time axis
        zOut = ComplexArray(
            real: MLX.padded(zOut.real, widths: [[0, 0], [0, 0], [0, 0], [0, 0], [2, 2]]),
            imag: MLX.padded(zOut.imag, widths: [[0, 0], [0, 0], [0, 0], [0, 0], [2, 2]])
        )

        // Calculate output length
        let pad = hopLength / 2 * 3
        let le = hopLength * Int(ceil(Double(length) / Double(hopLength))) + 2 * pad

        // Flatten for iSTFT: [B, S, C, F, T] -> [B*S*C, F, T]
        let zFlatReal = zOut.real.reshaped([B * S * C, zOut.shape[3], zOut.shape[4]])
        let zFlatImag = zOut.imag.reshaped([B * S * C, zOut.shape[3], zOut.shape[4]])
        let zFlat = ComplexArray(real: zFlatReal, imag: zFlatImag)

        // Compute iSTFT
        let istftConfig = STFTConfig(nFFT: nFft, hopLength: hopLength)
        var audio = try! istft(zFlat, config: istftConfig, length: le)

        // Apply normalization factor
        let normFactor = Float(sqrt(Double(nFft)))
        audio = audio * normFactor

        // Reshape and trim
        audio = audio.reshaped([B, S, C, -1])
        audio = audio[0..., 0..., 0..., pad..<(pad + length)]

        return audio
    }

    // MARK: - Loading

    /// Load pretrained model from path.
    /// - Parameter path: Path to model directory containing config.json and model.safetensors.
    /// - Returns: Loaded HTDemucs model.
    public static func fromPretrained(path: URL) throws -> HTDemucs {
        let configURL = path.appendingPathComponent("config.json")
        let config: HTDemucsConfig

        if FileManager.default.fileExists(atPath: configURL.path) {
            let data = try Data(contentsOf: configURL)
            config = try JSONDecoder().decode(HTDemucsConfig.self, from: data)
        } else {
            config = HTDemucsConfig()
        }

        let model = HTDemucs(config: config)

        // Load weights
        let weightsURL = path.appendingPathComponent("model.safetensors")
        if FileManager.default.fileExists(atPath: weightsURL.path) {
            try model.loadWeights(from: weightsURL)
        }

        return model
    }

    /// Load weights from safetensors file.
    private func loadWeights(from url: URL) throws {
        let weights = try MLX.loadArrays(url: url)
        // The weights are already in MLX format (transformed by Python convert.py)
        // Apply them to the model using Module's update mechanism
        try update(parameters: ModuleParameters.unflattened(weights), verify: .noUnusedKeys)
    }
}
