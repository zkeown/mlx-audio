// MaskEstimation.swift
// Mask estimation module for Banquet query-based source separation.

import Foundation
import MLX
import MLXNN

/// Gated Linear Unit activation.
public class GLU: Module, UnaryLayer, @unchecked Sendable {
    let dim: Int

    public init(dim: Int = -1) {
        self.dim = dim
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let parts = x.split(parts: 2, axis: dim)
        return parts[0] * sigmoid(parts[1])
    }
}

/// Normalization + MLP for per-band mask estimation.
///
/// Architecture:
/// ```
/// Input -> LayerNorm -> Linear -> Tanh -> Linear -> GLU -> Reshape
/// ```
public class NormMLP: Module, @unchecked Sendable {
    @ModuleInfo var norm: LayerNorm
    @ModuleInfo(key: "hidden_linear") var hiddenLinear: Linear
    @ModuleInfo(key: "output_linear") var outputLinear: Linear

    let bandwidth: Int
    let inChannel: Int
    let complexMask: Bool
    let reim: Int
    let hiddenActivation: String

    /// Creates a NormMLP layer.
    ///
    /// - Parameters:
    ///   - embDim: Embedding dimension
    ///   - mlpDim: MLP hidden dimension
    ///   - bandwidth: Width of frequency band
    ///   - inChannel: Number of input channels
    ///   - hiddenActivation: Activation function name
    ///   - complexMask: Whether to output complex mask
    public init(
        embDim: Int,
        mlpDim: Int,
        bandwidth: Int,
        inChannel: Int,
        hiddenActivation: String = "Tanh",
        complexMask: Bool = true
    ) {
        self.bandwidth = bandwidth
        self.inChannel = inChannel
        self.complexMask = complexMask
        self.reim = complexMask ? 2 : 1
        self.hiddenActivation = hiddenActivation

        _norm.wrappedValue = LayerNorm(dimensions: embDim)
        _hiddenLinear.wrappedValue = Linear(embDim, mlpDim)

        // Output with GLU (doubles output dim, then halves with GLU)
        let outputDim = bandwidth * inChannel * reim * 2
        _outputLinear.wrappedValue = Linear(mlpDim, outputDim)
    }

    /// Forward pass.
    ///
    /// - Parameter qb: Band embeddings [batch, n_time, emb_dim]
    /// - Returns: Band mask [batch, in_channel, bandwidth, n_time] or with reim dimension
    public func callAsFunction(_ qb: MLXArray) -> MLXArray {
        let shape = qb.shape
        let batch = shape[0]
        let nTime = shape[1]

        // Apply norm and hidden layer
        var x = norm(qb)
        x = hiddenLinear(x)
        x = tanh(x)

        // Output with GLU
        var mb = outputLinear(x)
        // GLU: split and gate
        let parts = mb.split(parts: 2, axis: -1)
        mb = parts[0] * sigmoid(parts[1])

        // Reshape to [batch, n_time, in_channel, bandwidth, reim]
        if complexMask {
            mb = mb.reshaped([batch, nTime, inChannel, bandwidth, reim])
            // Permute to [batch, in_channel, bandwidth, n_time, reim]
            mb = mb.transposed(0, 2, 3, 1, 4)
        } else {
            mb = mb.reshaped([batch, nTime, inChannel, bandwidth])
            mb = mb.transposed(0, 2, 3, 1)
        }

        return mb
    }
}

/// Overlapping mask estimation with frequency weighting.
///
/// Processes each band through a NormMLP and accumulates masks
/// with frequency weights for overlapping bands.
public class OverlappingMaskEstimationModule: Module, @unchecked Sendable {
    @ModuleInfo(key: "norm_mlp") var normMLP: [NormMLP]

    let bandSpecs: [BandSpec]
    let bandWidths: [Int]
    let nBands: Int
    let nFreq: Int
    let inChannel: Int
    let complexMask: Bool
    let useFreqWeights: Bool
    let freqWeights: [MLXArray]?

    /// Creates an OverlappingMaskEstimationModule.
    ///
    /// - Parameters:
    ///   - inChannel: Number of input channels
    ///   - bandSpecs: Band specifications (start, end bins)
    ///   - freqWeights: Frequency weights for each band
    ///   - nFreq: Number of frequency bins
    ///   - embDim: Embedding dimension
    ///   - mlpDim: MLP hidden dimension
    ///   - hiddenActivation: Activation function
    ///   - complexMask: Whether to output complex mask
    ///   - useFreqWeights: Whether to apply frequency weights
    public init(
        inChannel: Int,
        bandSpecs: [BandSpec],
        freqWeights: [MLXArray]?,
        nFreq: Int,
        embDim: Int,
        mlpDim: Int,
        hiddenActivation: String = "Tanh",
        complexMask: Bool = true,
        useFreqWeights: Bool = true
    ) {
        self.bandSpecs = bandSpecs
        self.bandWidths = bandSpecs.map { $0.bandwidth }
        self.nBands = bandSpecs.count
        self.nFreq = nFreq
        self.inChannel = inChannel
        self.complexMask = complexMask
        self.useFreqWeights = useFreqWeights
        self.freqWeights = freqWeights

        // Per-band NormMLP modules
        _normMLP.wrappedValue = bandWidths.map { bw in
            NormMLP(
                embDim: embDim,
                mlpDim: mlpDim,
                bandwidth: bw,
                inChannel: inChannel,
                hiddenActivation: hiddenActivation,
                complexMask: complexMask
            )
        }
    }

    /// Forward pass.
    ///
    /// - Parameter q: Band embeddings [batch, n_bands, n_time, emb_dim]
    /// - Returns: Full-frequency mask
    public func callAsFunction(_ q: MLXArray) -> MLXArray {
        let shape = q.shape
        let batch = shape[0]
        let nTime = shape[2]

        // Compute per-band masks
        var maskList: [MLXArray] = []
        for (b, nmlp) in normMLP.enumerated() {
            let qb = q[0..., b, 0..., 0...]  // [batch, n_time, emb_dim]
            let mb = nmlp(qb)
            maskList.append(mb)
        }

        // Initialize output mask
        var masks: MLXArray
        if complexMask {
            masks = MLXArray.zeros([batch, inChannel, nFreq, nTime, 2])
        } else {
            masks = MLXArray.zeros([batch, inChannel, nFreq, nTime])
        }

        // Accumulate masks into full frequency range
        for (im, mask) in maskList.enumerated() {
            let fstart = bandSpecs[im].startBin
            let fend = bandSpecs[im].endBin

            var weightedMask = mask
            if useFreqWeights, let fw = freqWeights {
                let freqWeight = fw[im]
                if complexMask {
                    // fw: [bandwidth] -> [1, 1, bandwidth, 1, 1]
                    let fwReshaped = freqWeight.reshaped([1, 1, -1, 1, 1])
                    weightedMask = mask * fwReshaped
                } else {
                    let fwReshaped = freqWeight.reshaped([1, 1, -1, 1])
                    weightedMask = mask * fwReshaped
                }
            }

            // Add to the appropriate frequency range
            // Using index-based update
            if complexMask {
                let current = masks[0..., 0..., fstart..<fend, 0..., 0...]
                let updated = current + weightedMask
                // Create new array with updated slice
                masks = updateSlice(masks, with: updated, at: fstart, end: fend, axis: 2)
            } else {
                let current = masks[0..., 0..., fstart..<fend, 0...]
                let updated = current + weightedMask
                masks = updateSlice(masks, with: updated, at: fstart, end: fend, axis: 2)
            }
        }

        return masks
    }

    /// Helper to update a slice of an array along axis 2.
    private func updateSlice(
        _ array: MLXArray,
        with values: MLXArray,
        at start: Int,
        end: Int,
        axis: Int
    ) -> MLXArray {
        // Build the result by concatenating slices
        let shape = array.shape
        let beforeSlice = array[0..., 0..., 0..<start]
        let afterSlice = array[0..., 0..., end...]

        if complexMask {
            return MLX.concatenated([beforeSlice, values, afterSlice], axis: 2)
        } else {
            return MLX.concatenated([beforeSlice, values, afterSlice], axis: 2)
        }
    }

    /// Creates an OverlappingMaskEstimationModule from configuration.
    ///
    /// - Parameter config: Banquet configuration
    /// - Returns: Configured OverlappingMaskEstimationModule
    public static func fromConfig(_ config: BanquetConfig) -> OverlappingMaskEstimationModule {
        let bandSpecs = MusicalBandsplitSpecification.generate(
            nFFT: config.nFFT,
            sampleRate: config.sampleRate,
            nBands: config.nBands
        )

        // Generate frequency weights
        let spec = MusicalBandsplitSpecificationWithWeights(
            nFFT: config.nFFT,
            sampleRate: config.sampleRate,
            nBands: config.nBands
        )

        return OverlappingMaskEstimationModule(
            inChannel: config.inChannel,
            bandSpecs: bandSpecs,
            freqWeights: spec.freqWeights,
            nFreq: config.freqBins,
            embDim: config.embDim,
            mlpDim: config.mlpDim,
            hiddenActivation: config.hiddenActivation,
            complexMask: config.complexMask,
            useFreqWeights: config.useFreqWeights
        )
    }
}

/// Extended band specification that also generates frequency weights.
public struct MusicalBandsplitSpecificationWithWeights {
    public let bandSpecs: [BandSpec]
    public let freqWeights: [MLXArray]

    public init(nFFT: Int, sampleRate: Int, nBands: Int = 64) {
        let fs = Float(sampleRate)
        let nFreqs = nFFT / 2 + 1
        let df = fs / Float(nFFT)

        let fMax = fs / 2
        let fMin = fs / Float(nFFT)

        // Calculate octave spacing
        let nOctaves = log2(fMax / fMin)
        let nOctavesPerBand = nOctaves / Float(nBands)
        let bandwidthMult = pow(2.0, nOctavesPerBand)

        // Convert to MIDI for linear spacing
        let lowMidi = max(0, Self.hzToMidi(fMin))
        let highMidi = Self.hzToMidi(fMax)

        var midiPoints: [Float] = []
        for i in 0..<nBands {
            let t = Float(i) / Float(nBands - 1)
            midiPoints.append(lowMidi + t * (highMidi - lowMidi))
        }

        let hzPts = midiPoints.map { Self.midiToHz($0) }

        // Calculate band boundaries
        let lowPts = hzPts.map { $0 / bandwidthMult }
        let highPts = hzPts.map { $0 * bandwidthMult }

        let lowBins = lowPts.map { Int(floor($0 / df)) }
        let highBins = highPts.map { Int(ceil($0 / df)) }

        // Create filterbank
        var fb = [[Float]](repeating: [Float](repeating: 0, count: nFreqs), count: nBands)
        for i in 0..<nBands {
            let start = max(0, lowBins[i])
            let end = min(nFreqs, highBins[i] + 1)
            for j in start..<end {
                fb[i][j] = 1.0
            }
        }

        fb[0] = fb[0].enumerated().map { $0.offset < lowBins[0] ? 1.0 : $0.element }
        fb[nBands - 1] = fb[nBands - 1].enumerated().map {
            $0.offset > highBins[nBands - 1] ? 1.0 : $0.element
        }

        // Compute weight per bin
        var weightPerBin = [Float](repeating: 0, count: nFreqs)
        for i in 0..<nBands {
            for j in 0..<nFreqs {
                weightPerBin[j] += fb[i][j]
            }
        }
        weightPerBin = weightPerBin.map { max($0, 1e-8) }

        // Normalize filterbank
        var normalizedFb = fb
        for i in 0..<nBands {
            for j in 0..<nFreqs {
                normalizedFb[i][j] = fb[i][j] / weightPerBin[j]
            }
        }

        // Extract band specs and frequency weights
        var specs: [BandSpec] = []
        var weights: [MLXArray] = []

        for i in 0..<nBands {
            var startIdx: Int?
            var endIdx: Int?
            for j in 0..<nFreqs {
                if fb[i][j] > 0 {
                    if startIdx == nil { startIdx = j }
                    endIdx = j + 1
                }
            }
            if let start = startIdx, let end = endIdx {
                specs.append(BandSpec(startBin: start, endBin: end))
                let bandWeights = Array(normalizedFb[i][start..<end])
                weights.append(MLXArray(bandWeights))
            }
        }

        self.bandSpecs = specs
        self.freqWeights = weights
    }

    private static func hzToMidi(_ hz: Float) -> Float {
        return 12.0 * log2(hz / 440.0) + 69.0
    }

    private static func midiToHz(_ midi: Float) -> Float {
        return 440.0 * pow(2.0, (midi - 69.0) / 12.0)
    }
}
