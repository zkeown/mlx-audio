// BandSplit.swift
// Band splitting module for Banquet query-based source separation.

import Foundation
import MLX
import MLXAudioPrimitives
import MLXNN

/// Normalization + Fully Connected layer for band processing.
///
/// Applies LayerNorm over flattened band input, then projects to embedding dimension.
public class NormFC: Module, @unchecked Sendable {
    @ModuleInfo var norm: LayerNorm
    @ModuleInfo var fc: Linear

    let treatChannelAsFeature: Bool

    /// Creates a NormFC layer.
    ///
    /// - Parameters:
    ///   - embDim: Embedding dimension output
    ///   - bandwidth: Width of frequency band in bins
    ///   - inChannel: Number of input channels
    ///   - treatChannelAsFeature: Whether to treat channels as features (default: true)
    public init(
        embDim: Int,
        bandwidth: Int,
        inChannel: Int,
        treatChannelAsFeature: Bool = true
    ) {
        self.treatChannelAsFeature = treatChannelAsFeature
        let reim = 2  // Real and imaginary parts

        // LayerNorm over flattened input
        _norm.wrappedValue = LayerNorm(dimensions: inChannel * bandwidth * reim)

        // FC input dimension depends on treatChannelAsFeature
        var fcIn = bandwidth * reim
        var outputDim = embDim
        if treatChannelAsFeature {
            fcIn *= inChannel
        } else {
            precondition(embDim % inChannel == 0, "embDim must be divisible by inChannel")
            outputDim = embDim / inChannel
        }

        _fc.wrappedValue = Linear(fcIn, outputDim)
    }

    /// Forward pass.
    ///
    /// - Parameter xb: Band input [batch, n_time, in_chan, reim * bandwidth]
    /// - Returns: Band embeddings [batch, n_time, emb_dim]
    public func callAsFunction(_ xb: MLXArray) -> MLXArray {
        let shape = xb.shape
        let batch = shape[0]
        let nTime = shape[1]
        let inChan = shape[2]
        let ribw = shape[3]

        // Flatten for LayerNorm: [batch, n_time, in_chan * reim * bandwidth]
        var x = xb.reshaped([batch, nTime, inChan * ribw])
        x = norm(x)

        if !treatChannelAsFeature {
            // Reshape back: [batch, n_time, in_chan, reim * bandwidth]
            x = x.reshaped([batch, nTime, inChan, ribw])
        }

        // Project to embedding dimension
        var z = fc(x)

        if !treatChannelAsFeature {
            // Flatten channel dimension into embedding
            let zShape = z.shape
            z = z.reshaped([zShape[0], zShape[1], zShape[2] * zShape[3]])
        }

        return z  // [batch, n_time, emb_dim]
    }
}

/// Band splitting module for frequency decomposition.
///
/// Splits complex spectrogram into frequency bands and projects each band
/// to an embedding dimension using per-band NormFC modules.
public class BandSplitModule: Module, @unchecked Sendable {
    @ModuleInfo(key: "norm_fc_modules") var normFCModules: [NormFC]

    let bandSpecs: [BandSpec]
    let bandWidths: [Int]
    let nBands: Int
    let embDim: Int

    /// Creates a BandSplitModule.
    ///
    /// - Parameters:
    ///   - inChannel: Number of input audio channels
    ///   - bandSpecs: List of band specifications (start_bin, end_bin)
    ///   - embDim: Embedding dimension for band features
    ///   - treatChannelAsFeature: Whether to treat channels as features
    public init(
        inChannel: Int,
        bandSpecs: [BandSpec],
        embDim: Int = 128,
        treatChannelAsFeature: Bool = true
    ) {
        self.bandSpecs = bandSpecs
        self.bandWidths = bandSpecs.map { $0.bandwidth }
        self.nBands = bandSpecs.count
        self.embDim = embDim

        // Create per-band NormFC modules
        _normFCModules.wrappedValue = bandWidths.map { bw in
            NormFC(
                embDim: embDim,
                bandwidth: bw,
                inChannel: inChannel,
                treatChannelAsFeature: treatChannelAsFeature
            )
        }
    }

    /// Forward pass.
    ///
    /// - Parameter x: Complex spectrogram [batch, in_chan, n_freq, n_time]
    /// - Returns: Band embeddings [batch, n_bands, n_time, emb_dim]
    public func callAsFunction(_ x: ComplexArray) -> MLXArray {
        let shape = x.shape
        let batch = shape[0]
        let inChan = shape[1]
        let nTime = shape[3]

        // Convert complex to real/imag: [batch, in_chan, n_freq, n_time, 2]
        let xr = MLX.stacked([x.real, x.imag], axis: -1)

        // Permute: [batch, n_time, in_chan, 2, n_freq]
        let xrT = xr.transposed(0, 3, 1, 4, 2)

        // Process each band
        var outputs: [MLXArray] = []
        for (i, nfm) in normFCModules.enumerated() {
            let fstart = bandSpecs[i].startBin
            let fend = bandSpecs[i].endBin

            // Extract band: [batch, n_time, in_chan, reim, bandwidth]
            let xb = xrT[0..., 0..., 0..., 0..., fstart..<fend]

            // Flatten reim and bandwidth: [batch, n_time, in_chan, reim * bandwidth]
            let xbShape = xb.shape
            let xbFlat = xb.reshaped([xbShape[0], xbShape[1], xbShape[2], xbShape[3] * xbShape[4]])

            // Apply NormFC
            let zb = nfm(xbFlat)  // [batch, n_time, emb_dim]
            outputs.append(zb)
        }

        // Stack bands: [batch, n_bands, n_time, emb_dim]
        let z = MLX.stacked(outputs, axis: 1)

        return z
    }

    /// Creates a BandSplitModule from configuration.
    ///
    /// - Parameters:
    ///   - config: Banquet configuration
    /// - Returns: Configured BandSplitModule
    public static func fromConfig(_ config: BanquetConfig) -> BandSplitModule {
        let bandSpecs = MusicalBandsplitSpecification.generate(
            nFFT: config.nFFT,
            sampleRate: config.sampleRate,
            nBands: config.nBands
        )

        return BandSplitModule(
            inChannel: config.inChannel,
            bandSpecs: bandSpecs,
            embDim: config.embDim
        )
    }
}

/// Musical band specification generator.
///
/// Generates frequency bands using MIDI note spacing for perceptually uniform bands.
public struct MusicalBandsplitSpecification {

    /// Generates band specifications for the given parameters.
    ///
    /// - Parameters:
    ///   - nFFT: FFT size
    ///   - sampleRate: Sample rate in Hz
    ///   - nBands: Number of frequency bands
    ///   - fMin: Minimum frequency (default: sampleRate / nFFT)
    ///   - fMax: Maximum frequency (default: sampleRate / 2)
    /// - Returns: Array of BandSpec
    public static func generate(
        nFFT: Int,
        sampleRate: Int,
        nBands: Int = 64,
        fMin: Float? = nil,
        fMax: Float? = nil
    ) -> [BandSpec] {
        let fs = Float(sampleRate)
        let nFreqs = nFFT / 2 + 1
        let df = fs / Float(nFFT)

        let fMaxVal = fMax ?? fs / 2
        let fMinVal = fMin ?? fs / Float(nFFT)

        // Calculate octave spacing
        let nOctaves = log2(fMaxVal / fMinVal)
        let nOctavesPerBand = nOctaves / Float(nBands)
        let bandwidthMult = pow(2.0, nOctavesPerBand)

        // Convert to MIDI for linear spacing
        let lowMidi = max(0, hzToMidi(fMinVal))
        let highMidi = hzToMidi(fMaxVal)

        var midiPoints: [Float] = []
        for i in 0..<nBands {
            let t = Float(i) / Float(nBands - 1)
            midiPoints.append(lowMidi + t * (highMidi - lowMidi))
        }

        let hzPts = midiPoints.map { midiToHz($0) }

        // Calculate band boundaries
        let lowPts = hzPts.map { $0 / bandwidthMult }
        let highPts = hzPts.map { $0 * bandwidthMult }

        let lowBins = lowPts.map { Int(floor($0 / df)) }
        let highBins = highPts.map { Int(ceil($0 / df)) }

        // Create filterbank mask
        var fb = [[Float]](repeating: [Float](repeating: 0, count: nFreqs), count: nBands)
        for i in 0..<nBands {
            let start = max(0, lowBins[i])
            let end = min(nFreqs, highBins[i] + 1)
            for j in start..<end {
                fb[i][j] = 1.0
            }
        }

        // Extend first and last bands
        for j in 0..<max(0, lowBins[0]) {
            fb[0][j] = 1.0
        }
        for j in min(nFreqs, highBins[nBands - 1] + 1)..<nFreqs {
            fb[nBands - 1][j] = 1.0
        }

        // Extract band specs
        var bandSpecs: [BandSpec] = []
        for i in 0..<nBands {
            var startIdx: Int?
            var endIdx: Int?
            for j in 0..<nFreqs {
                if fb[i][j] > 0 {
                    if startIdx == nil {
                        startIdx = j
                    }
                    endIdx = j + 1
                }
            }
            if let start = startIdx, let end = endIdx {
                bandSpecs.append(BandSpec(startBin: start, endBin: end))
            }
        }

        return bandSpecs
    }

    /// Convert frequency in Hz to MIDI note number.
    private static func hzToMidi(_ hz: Float) -> Float {
        return 12.0 * log2(hz / 440.0) + 69.0
    }

    /// Convert MIDI note number to frequency in Hz.
    private static func midiToHz(_ midi: Float) -> Float {
        return 440.0 * pow(2.0, (midi - 69.0) / 12.0)
    }
}
