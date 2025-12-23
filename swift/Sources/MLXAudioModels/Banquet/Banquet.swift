// Banquet.swift
// Banquet query-based source separation model.
//
// Banquet uses a reference audio query to extract matching sounds from a mixture.
// This is the main model that combines all components.
//
// Pipeline:
//     Query Audio → PaSST → 768-dim embedding
//                               ↓
//     Mixture → STFT → BandSplit → SeqBand (24 LSTMs) → FiLM Conditioning → MaskEstimation → iSTFT → Output

import Foundation
import MLX
import MLXAudioPrimitives
import MLXNN

/// Output from Banquet separation.
public struct BanquetOutput: @unchecked Sendable {
    /// Separated audio [batch, channels, samples]
    public let audio: MLXArray

    /// Separated spectrogram [batch, channels, freq, time] (complex)
    public let spectrogram: MLXArray

    /// Estimated mask [batch, channels, freq, time, 2] or [batch, channels, freq, time]
    public let mask: MLXArray

    public init(audio: MLXArray, spectrogram: MLXArray, mask: MLXArray) {
        self.audio = audio
        self.spectrogram = spectrogram
        self.mask = mask
    }
}

/// Banquet query-based source separation model.
///
/// Uses a reference audio query (encoded by PaSST) to extract matching sounds
/// from a mixture audio signal.
public class Banquet: Module, @unchecked Sendable {
    @ModuleInfo(key: "query_encoder") var queryEncoder: PaSST
    @ModuleInfo(key: "band_split") var bandSplit: BandSplitModule
    @ModuleInfo(key: "tf_model") var tfModel: SeqBandModellingModule
    @ModuleInfo var film: FiLM
    @ModuleInfo(key: "mask_estim") var maskEstim: OverlappingMaskEstimationModule

    let config: BanquetConfig
    let passtConfig: PaSSTConfig
    let bandSpecs: [BandSpec]
    let freqWeights: [MLXArray]
    let bandWidths: [Int]

    /// Creates a Banquet model.
    ///
    /// - Parameters:
    ///   - config: Banquet model configuration
    ///   - passtConfig: PaSST encoder configuration
    public init(config: BanquetConfig = BanquetConfig(), passtConfig: PaSSTConfig = PaSSTConfig()) {
        self.config = config
        self.passtConfig = passtConfig

        // Generate band specifications
        let bandSpec = MusicalBandsplitSpecificationWithWeights(
            nFFT: config.nFFT,
            sampleRate: config.sampleRate,
            nBands: config.nBands
        )
        self.bandSpecs = bandSpec.bandSpecs
        self.freqWeights = bandSpec.freqWeights
        self.bandWidths = bandSpecs.map { $0.bandwidth }

        // Query encoder (PaSST)
        _queryEncoder.wrappedValue = PaSST(config: passtConfig)

        // Band split module
        _bandSplit.wrappedValue = BandSplitModule(
            inChannel: config.inChannel,
            bandSpecs: bandSpecs,
            embDim: config.embDim
        )

        // Time-frequency modelling (BiLSTM)
        _tfModel.wrappedValue = SeqBandModellingModule(
            nModules: config.nSQMModules,
            embDim: config.embDim,
            rnnDim: config.rnnDim,
            bidirectional: config.bidirectional,
            rnnType: config.rnnType
        )

        // FiLM conditioning
        _film.wrappedValue = FiLM(
            condEmbeddingDim: config.condEmbDim,
            channels: config.embDim,
            additive: config.filmAdditive,
            multiplicative: config.filmMultiplicative,
            depth: config.filmDepth,
            channelsPerGroup: config.channelsPerGroup
        )

        // Mask estimation
        _maskEstim.wrappedValue = OverlappingMaskEstimationModule(
            inChannel: config.inChannel,
            bandSpecs: bandSpecs,
            freqWeights: freqWeights,
            nFreq: config.freqBins,
            embDim: config.embDim,
            mlpDim: config.mlpDim,
            hiddenActivation: config.hiddenActivation,
            complexMask: config.complexMask,
            useFreqWeights: config.useFreqWeights
        )
    }

    /// Compute STFT of audio signal.
    ///
    /// - Parameter x: Audio signal [batch, channels, samples]
    /// - Returns: Complex spectrogram [batch, channels, freq, time]
    private func computeSTFT(_ x: MLXArray) throws -> ComplexArray {
        let shape = x.shape
        let B = shape[0]
        let C = shape[1]
        let T = shape[2]

        // Flatten batch and channels
        let xFlat = x.reshaped([B * C, T])

        // Compute STFT using MLXAudioPrimitives
        let stftConfig = STFTConfig(
            nFFT: config.nFFT,
            hopLength: config.hopLength,
            winLength: config.effectiveWinLength
        )
        var spec = try MLXAudioPrimitives.stft(xFlat, config: stftConfig)

        // Normalize
        let normFactor = sqrt(Float(config.nFFT))
        spec = ComplexArray(
            real: spec.real / normFactor,
            imag: spec.imag / normFactor
        )

        // Reshape back: [B*C, freq, time] -> [B, C, freq, time]
        let specShape = spec.shape
        let F = specShape[0]
        let Tf = specShape[1]
        spec = ComplexArray(
            real: spec.real.reshaped([B, C, F, Tf]),
            imag: spec.imag.reshaped([B, C, F, Tf])
        )

        return spec
    }

    /// Compute inverse STFT.
    ///
    /// - Parameters:
    ///   - spec: Complex spectrogram [batch, channels, freq, time]
    ///   - length: Target output length
    /// - Returns: Audio signal [batch, channels, samples]
    private func computeISTFT(_ spec: ComplexArray, length: Int) throws -> MLXArray {
        let shape = spec.shape
        let B = shape[0]
        let C = shape[1]
        let F = shape[2]
        let Tf = shape[3]

        // Flatten batch and channels
        let specFlat = ComplexArray(
            real: spec.real.reshaped([B * C, F, Tf]),
            imag: spec.imag.reshaped([B * C, F, Tf])
        )

        // Compute iSTFT
        let stftConfig = STFTConfig(
            nFFT: config.nFFT,
            hopLength: config.hopLength,
            winLength: config.effectiveWinLength
        )
        var x = try MLXAudioPrimitives.istft(specFlat, config: stftConfig, length: length)

        // Denormalize
        let normFactor = sqrt(Float(config.nFFT))
        x = x * normFactor

        // Reshape back
        x = x.reshaped([B, C, -1])

        // Trim to exact length
        x = x[0..., 0..., 0..<length]

        return x
    }

    /// Apply complex mask to spectrogram.
    ///
    /// - Parameters:
    ///   - spec: Complex spectrogram [batch, channels, freq, time]
    ///   - mask: Complex mask [batch, channels, freq, time, 2] (real, imag)
    /// - Returns: Masked spectrogram [batch, channels, freq, time]
    private func applyComplexMask(_ spec: ComplexArray, mask: MLXArray) -> ComplexArray {
        if config.complexMask {
            // Convert mask from [B, C, F, T, 2] to complex components
            let maskReal = mask[0..., 0..., 0..., 0..., 0]
            let maskImag = mask[0..., 0..., 0..., 0..., 1]

            // Complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
            let specReal = spec.real
            let specImag = spec.imag

            let outReal = specReal * maskReal - specImag * maskImag
            let outImag = specReal * maskImag + specImag * maskReal

            // Combine to complex
            return ComplexArray(real: outReal, imag: outImag)
        } else {
            // Real mask: simple multiplication
            return ComplexArray(real: spec.real * mask, imag: spec.imag * mask)
        }
    }

    /// Encode query audio using PaSST.
    ///
    /// - Parameter queryMel: Query mel spectrogram [batch, 1, n_mels, time]
    /// - Returns: Query embedding [batch, 768]
    public func encodeQuery(_ queryMel: MLXArray) -> MLXArray {
        return queryEncoder(queryMel)
    }

    /// Forward pass for Banquet separation.
    ///
    /// - Parameters:
    ///   - mixture: Mixture audio [batch, channels, samples]
    ///   - queryEmbedding: Pre-computed query embedding [batch, 768]
    /// - Returns: BanquetOutput with separated audio, spectrogram, and mask
    public func callAsFunction(_ mixture: MLXArray, queryEmbedding: MLXArray) throws -> BanquetOutput {
        // Store original length for reconstruction
        let originalLength = mixture.shape[2]

        // Compute STFT of mixture
        let spec = try computeSTFT(mixture)  // [B, C, F, T]

        // Band split: spec -> band embeddings
        var z = bandSplit(spec)  // [B, n_bands, n_time, emb_dim]

        // Time-frequency modelling
        z = tfModel(z)  // [B, n_bands, n_time, emb_dim]

        // Prepare for FiLM: [B, n_bands, n_time, emb_dim] -> [B, emb_dim, n_bands, n_time]
        z = z.transposed(0, 3, 1, 2)

        // Apply FiLM conditioning
        z = film(z, conditioning: queryEmbedding)

        // Back to [B, n_bands, n_time, emb_dim]
        z = z.transposed(0, 2, 3, 1)

        // Mask estimation
        let mask = maskEstim(z)  // [B, C, F, T, 2] or [B, C, F, T]

        // Apply mask to spectrogram
        let maskedSpec = applyComplexMask(spec, mask: mask)

        // Inverse STFT to get audio
        let audio = try computeISTFT(maskedSpec, length: originalLength)

        // Combine real and imag for output spectrogram
        let specOut = MLX.stacked([maskedSpec.real, maskedSpec.imag], axis: -1)

        return BanquetOutput(
            audio: audio,
            spectrogram: specOut,
            mask: mask
        )
    }

    /// Separate audio using query.
    ///
    /// Convenience method that encodes the query and runs separation.
    ///
    /// - Parameters:
    ///   - mixture: Mixture audio [batch, channels, samples]
    ///   - queryMel: Query mel spectrogram [batch, 1, n_mels, time]
    /// - Returns: BanquetOutput with separated audio, spectrogram, and mask
    public func separate(mixture: MLXArray, queryMel: MLXArray) throws -> BanquetOutput {
        // Encode query
        let queryEmbedding = encodeQuery(queryMel)

        // Run separation
        return try callAsFunction(mixture, queryEmbedding: queryEmbedding)
    }

    /// Creates Banquet from configuration.
    ///
    /// - Parameters:
    ///   - config: Banquet model configuration
    ///   - passtConfig: Optional PaSST configuration
    /// - Returns: Banquet model
    public static func fromConfig(
        _ config: BanquetConfig,
        passtConfig: PaSSTConfig? = nil
    ) -> Banquet {
        return Banquet(config: config, passtConfig: passtConfig ?? PaSSTConfig())
    }
}
