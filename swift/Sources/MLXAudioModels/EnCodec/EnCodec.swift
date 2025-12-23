// EnCodec.swift
// Main EnCodec neural audio codec model.

import Foundation
import MLX
import MLXNN

// MARK: - EnCodec Model

/// EnCodec neural audio codec.
///
/// EnCodec is a neural audio codec that compresses audio into discrete tokens
/// using a convolutional encoder, residual vector quantization, and a
/// convolutional decoder.
///
/// ## Architecture
/// ```
/// Audio [B, C, T] -> Encoder -> Latent [B, T', D] -> RVQ -> Codes [B, K, T']
///                                                      |
/// Codes [B, K, T'] -> RVQ.decode -> Latent [B, T', D] -> Decoder -> Audio [B, C, T]
/// ```
///
/// Where:
/// - T' = T / hop_length (compressed time dimension)
/// - D = codebook_dim (latent dimension, typically 128)
/// - K = num_codebooks (number of RVQ codebooks, controls bitrate)
///
/// ## Usage
/// ```swift
/// // Create model
/// let config = EnCodecConfig.encodec_24khz()
/// let model = EnCodec(config: config)
///
/// // Encode audio to codes
/// let audio = MLXArray(...)  // [B, C, T]
/// let codes = model.encode(audio)  // [B, K, T']
///
/// // Decode codes back to audio
/// let reconstructed = model.decode(codes)  // [B, C, T]
/// ```
public class EnCodec: Module, @unchecked Sendable {

    // MARK: - Properties

    public let config: EnCodecConfig
    let encoder: EnCodecEncoder
    let quantizer: ResidualVectorQuantizer
    let decoder: EnCodecDecoder

    // MARK: - Initialization

    public init(config: EnCodecConfig = EnCodecConfig()) {
        self.config = config
        self.encoder = EnCodecEncoder(config: config)
        self.quantizer = ResidualVectorQuantizer(
            numCodebooks: config.num_codebooks,
            codebookSize: config.codebook_size,
            codebookDim: config.codebook_dim
        )
        self.decoder = EnCodecDecoder(config: config)
        super.init()
    }

    // MARK: - Forward Pass

    /// Encode audio waveform to discrete codes.
    ///
    /// - Parameter audio: Audio waveform with flexible input shapes:
    ///   - [T]: Single mono audio (promoted to [1, 1, T])
    ///   - [B, T]: Batch of mono audio (promoted to [B, 1, T])
    ///   - [C, T]: Multi-channel audio (promoted to [1, C, T])
    ///   - [B, C, T]: Batch of multi-channel audio (canonical)
    /// - Returns: Codebook indices [B, K, T'] where K=num_codebooks, T'=T/hop_length
    public func encode(_ audio: MLXArray) -> MLXArray {
        // Normalize input shape to [B, C, T]
        var x = audio
        if x.ndim == 1 {
            x = x.expandedDimensions(axes: [0, 1])  // [T] -> [1, 1, T]
        } else if x.ndim == 2 {
            // Could be [B, T] or [C, T]
            // Assume [B, T] for mono audio, add channel dim
            x = x.expandedDimensions(axis: 1)  // [B, T] -> [B, 1, T]
        }

        // Encode to latent embeddings
        let embeddings = encoder(x)  // [B, T', D]

        // Quantize to codes
        let (_, codes) = quantizer(embeddings)  // [B, K, T']

        return codes
    }

    /// Decode discrete codes to audio waveform.
    ///
    /// - Parameter codes: Codebook indices [B, K, T'] where K=num_codebooks
    /// - Returns: Audio waveform [B, C, T] where T = T' * hop_length
    public func decode(_ codes: MLXArray) -> MLXArray {
        // Decode codes to latent embeddings
        let embeddings = quantizer.decode(codes)  // [B, T', D]

        // Decode to audio
        let audio = decoder(embeddings)  // [B, C, T]

        return audio
    }

    /// Full encode-decode cycle.
    ///
    /// - Parameter audio: Audio waveform [B, C, T]
    /// - Returns: Tuple of (reconstructed audio [B, C, T], codes [B, K, T'])
    public func callAsFunction(_ audio: MLXArray) -> (MLXArray, MLXArray) {
        let codes = encode(audio)
        let reconstructed = decode(codes)
        return (reconstructed, codes)
    }

    // MARK: - Properties

    /// Total downsampling factor.
    public var hopLength: Int {
        config.hop_length
    }

    /// Get codebook weights for a specific layer.
    ///
    /// - Parameter layerIdx: Index of the codebook layer
    /// - Returns: Codebook weights [codebook_size, codebook_dim]
    public func getCodebook(_ layerIdx: Int) -> MLXArray {
        quantizer.getCodebook(layerIdx)
    }

    // MARK: - Loading

    /// Load EnCodec model from a pretrained checkpoint directory.
    ///
    /// The directory should contain:
    /// - config.json: Model configuration
    /// - model.safetensors: Model weights
    ///
    /// - Parameter path: Path to the model directory
    /// - Returns: Loaded EnCodec model
    public static func fromPretrained(path: URL) throws -> EnCodec {
        // Load config
        let configURL = path.appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: configURL.path) else {
            throw EnCodecError.configLoadFailed("config.json not found at \(path.path)")
        }

        let configData = try Data(contentsOf: configURL)
        let config = try JSONDecoder().decode(EnCodecConfig.self, from: configData)

        // Create model
        let model = EnCodec(config: config)

        // Load weights
        let weightsURL = path.appendingPathComponent("model.safetensors")
        if FileManager.default.fileExists(atPath: weightsURL.path) {
            try model.loadWeights(from: weightsURL)
        } else {
            // Try weights.safetensors as alternative
            let altWeightsURL = path.appendingPathComponent("weights.safetensors")
            if FileManager.default.fileExists(atPath: altWeightsURL.path) {
                try model.loadWeights(from: altWeightsURL)
            } else {
                throw EnCodecError.weightsLoadFailed("No weights file found at \(path.path)")
            }
        }

        return model
    }

    /// Load weights from a safetensors file.
    ///
    /// - Parameter url: Path to the weights file
    public func loadWeights(from url: URL) throws {
        let weights = try MLX.loadArrays(url: url)
        try update(parameters: ModuleParameters.unflattened(weights), verify: .noUnusedKeys)
    }

    /// Save model to a directory.
    ///
    /// - Parameter path: Directory path to save to
    public func savePretrained(path: URL) throws {
        // Create directory if needed
        try FileManager.default.createDirectory(at: path, withIntermediateDirectories: true)

        // Save config
        let configURL = path.appendingPathComponent("config.json")
        let configData = try JSONEncoder().encode(config)
        try configData.write(to: configURL)

        // Save weights
        let weightsURL = path.appendingPathComponent("model.safetensors")
        let flattenedParams = parameters().flattened()
        var paramsDict: [String: MLXArray] = [:]
        for (key, value) in flattenedParams {
            paramsDict[key] = value
        }
        try save(arrays: paramsDict, url: weightsURL)
    }
}
