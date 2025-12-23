// SeqBandModelling.swift
// Sequential Band Modelling module for Banquet query-based source separation.

import Foundation
import MLX
import MLXNN

/// Residual RNN block with LayerNorm and bidirectional processing.
///
/// Architecture: Input -> LayerNorm -> BiLSTM -> FC -> + Input (residual)
public class ResidualRNN: Module, @unchecked Sendable {
    @ModuleInfo var norm: LayerNorm
    @ModuleInfo var rnn: LSTM
    @ModuleInfo(key: "rnn_reverse") var rnnReverse: LSTM?
    @ModuleInfo var fc: Linear

    let bidirectional: Bool
    let useLayerNorm: Bool

    /// Creates a ResidualRNN block.
    ///
    /// - Parameters:
    ///   - embDim: Embedding dimension
    ///   - rnnDim: RNN hidden dimension
    ///   - bidirectional: Whether to use bidirectional RNN
    ///   - rnnType: Type of RNN ("LSTM" or "GRU") - currently only LSTM supported
    ///   - useLayerNorm: Whether to use LayerNorm (vs GroupNorm)
    public init(
        embDim: Int,
        rnnDim: Int,
        bidirectional: Bool = true,
        rnnType: String = "LSTM",
        useLayerNorm: Bool = true
    ) {
        precondition(rnnType == "LSTM", "Only LSTM is currently supported in Swift")

        self.bidirectional = bidirectional
        self.useLayerNorm = useLayerNorm

        _norm.wrappedValue = LayerNorm(dimensions: embDim)
        _rnn.wrappedValue = LSTM(inputSize: embDim, hiddenSize: rnnDim)

        if bidirectional {
            _rnnReverse.wrappedValue = LSTM(inputSize: embDim, hiddenSize: rnnDim)
        } else {
            _rnnReverse.wrappedValue = nil
        }

        // Output projection
        let fcIn = rnnDim * (bidirectional ? 2 : 1)
        _fc.wrappedValue = Linear(fcIn, embDim)
    }

    /// Forward pass.
    ///
    /// - Parameter z: Input [batch, n_uncrossed, n_across, emb_dim]
    /// - Returns: Output with same shape as input
    public func callAsFunction(_ z: MLXArray) -> MLXArray {
        let z0 = z  // Save for residual

        // Apply normalization
        var x = norm(z)

        let shape = x.shape
        let batch = shape[0]
        let nUncrossed = shape[1]
        let nAcross = shape[2]
        let embDim = shape[3]

        // Use batch trick: flatten first two dims for RNN processing
        x = x.reshaped([batch * nUncrossed, nAcross, embDim])

        // Apply forward RNN
        let (rnnOut, _) = rnn(x)

        var output: MLXArray
        if bidirectional, let rnnRev = rnnReverse {
            // Process reversed sequence along axis 1 (nAcross dimension)
            let xRev = x[0..., .stride(from: -1, to: nil, by: -1), 0...]
            let (rnnRevOut, _) = rnnRev(xRev)
            // Reverse back
            let rnnRevOutReversed = rnnRevOut[0..., .stride(from: -1, to: nil, by: -1), 0...]
            // Concatenate
            output = MLX.concatenated([rnnOut, rnnRevOutReversed], axis: -1)
        } else {
            output = rnnOut
        }

        // Reshape back
        output = output.reshaped([batch, nUncrossed, nAcross, -1])

        // Project to embedding dimension
        output = fc(output)

        // Residual connection
        output = output + z0

        return output
    }
}

/// Sequential Band Modelling module.
///
/// Alternates between time and frequency RNNs to model temporal and
/// cross-band dependencies.
///
/// Architecture:
/// - n_modules pairs of RNNs
/// - Each pair: Time RNN -> transpose -> Freq RNN -> transpose
/// - Total: 2 * n_modules ResidualRNN layers
public class SeqBandModellingModule: Module, @unchecked Sendable {
    @ModuleInfo var seqband: [ResidualRNN]

    let nModules: Int

    /// Creates a SeqBandModellingModule.
    ///
    /// - Parameters:
    ///   - nModules: Number of RNN module pairs
    ///   - embDim: Embedding dimension
    ///   - rnnDim: RNN hidden dimension
    ///   - bidirectional: Whether to use bidirectional RNNs
    ///   - rnnType: Type of RNN ("LSTM" or "GRU")
    public init(
        nModules: Int = 12,
        embDim: Int = 128,
        rnnDim: Int = 256,
        bidirectional: Bool = true,
        rnnType: String = "LSTM"
    ) {
        self.nModules = nModules

        // Create 2 * n_modules RNN layers (alternating time and freq)
        _seqband.wrappedValue = (0..<(2 * nModules)).map { _ in
            ResidualRNN(
                embDim: embDim,
                rnnDim: rnnDim,
                bidirectional: bidirectional,
                rnnType: rnnType
            )
        }
    }

    /// Forward pass.
    ///
    /// - Parameter z: Band embeddings [batch, n_bands, n_time, emb_dim]
    /// - Returns: Processed embeddings [batch, n_bands, n_time, emb_dim]
    public func callAsFunction(_ z: MLXArray) -> MLXArray {
        var output = z

        // Process through alternating time and freq RNNs
        for sbm in seqband {
            output = sbm(output)
            // Transpose between time and freq dimensions
            // [batch, n_bands, n_time, emb_dim] <-> [batch, n_time, n_bands, emb_dim]
            output = output.transposed(0, 2, 1, 3)
        }

        return output  // [batch, n_bands, n_time, emb_dim]
    }

    /// Creates a SeqBandModellingModule from configuration.
    ///
    /// - Parameter config: Banquet configuration
    /// - Returns: Configured SeqBandModellingModule
    public static func fromConfig(_ config: BanquetConfig) -> SeqBandModellingModule {
        return SeqBandModellingModule(
            nModules: config.nSQMModules,
            embDim: config.embDim,
            rnnDim: config.rnnDim,
            bidirectional: config.bidirectional,
            rnnType: config.rnnType
        )
    }
}
