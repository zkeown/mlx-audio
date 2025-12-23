// EnCodecConfig.swift
// Configuration for EnCodec neural audio codec.

import Foundation

/// Configuration for the EnCodec neural audio codec.
///
/// EnCodec is a neural audio codec that compresses audio into discrete tokens
/// using a convolutional encoder, residual vector quantization, and a
/// convolutional decoder.
///
/// Mirrors the Python `EnCodecConfig` exactly for weight compatibility.
public struct EnCodecConfig: Codable, Sendable, Equatable {

    // MARK: - Audio Settings

    /// Audio sample rate in Hz.
    public var sample_rate: Int

    /// Number of audio channels (1 for mono, 2 for stereo).
    public var channels: Int

    // MARK: - Quantization Settings

    /// Number of codebooks for residual vector quantization.
    public var num_codebooks: Int

    /// Number of entries per codebook (vocabulary size).
    public var codebook_size: Int

    /// Dimension of codebook vectors.
    public var codebook_dim: Int

    // MARK: - Architecture Settings

    /// Base hidden dimension for encoder/decoder.
    public var hidden_size: Int

    /// Number of output filters in first conv layer.
    public var num_filters: Int

    /// Number of residual layers in each block.
    public var num_residual_layers: Int

    /// Downsampling/upsampling ratios for each encoder/decoder stage.
    public var ratios: [Int]

    /// Normalization type ("weight_norm" or "time_group_norm").
    public var norm_type: String

    /// Kernel size for convolutional layers.
    public var kernel_size: Int

    /// Kernel size for final conv layer.
    public var last_kernel_size: Int

    /// Kernel size for residual layers.
    public var residual_kernel_size: Int

    /// Base for dilated convolutions.
    public var dilation_base: Int

    /// Whether to use causal convolutions.
    public var causal: Bool

    /// Padding mode for convolutions.
    public var pad_mode: String

    /// Compression factor for hidden dimension.
    public var compress: Int

    /// Number of LSTM layers (0 to disable).
    public var lstm_layers: Int

    /// Number of outer blocks to skip normalization.
    public var disable_norm_outer_blocks: Int

    /// Ratio of right padding to trim.
    public var trim_right_ratio: Float

    // MARK: - Computed Properties

    /// Total downsampling factor (product of ratios).
    public var hop_length: Int {
        ratios.reduce(1, *)
    }

    /// Number of frames per second of audio.
    public var frame_rate: Float {
        Float(sample_rate) / Float(hop_length)
    }

    // MARK: - Initialization

    /// Creates an EnCodec configuration with default values.
    public init(
        sample_rate: Int = 32000,
        channels: Int = 1,
        num_codebooks: Int = 4,
        codebook_size: Int = 2048,
        codebook_dim: Int = 128,
        hidden_size: Int = 128,
        num_filters: Int = 32,
        num_residual_layers: Int = 1,
        ratios: [Int] = [8, 5, 4, 2],
        norm_type: String = "weight_norm",
        kernel_size: Int = 7,
        last_kernel_size: Int = 7,
        residual_kernel_size: Int = 3,
        dilation_base: Int = 2,
        causal: Bool = true,
        pad_mode: String = "constant",
        compress: Int = 2,
        lstm_layers: Int = 2,
        disable_norm_outer_blocks: Int = 0,
        trim_right_ratio: Float = 1.0
    ) {
        self.sample_rate = sample_rate
        self.channels = channels
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.hidden_size = hidden_size
        self.num_filters = num_filters
        self.num_residual_layers = num_residual_layers
        self.ratios = ratios
        self.norm_type = norm_type
        self.kernel_size = kernel_size
        self.last_kernel_size = last_kernel_size
        self.residual_kernel_size = residual_kernel_size
        self.dilation_base = dilation_base
        self.causal = causal
        self.pad_mode = pad_mode
        self.compress = compress
        self.lstm_layers = lstm_layers
        self.disable_norm_outer_blocks = disable_norm_outer_blocks
        self.trim_right_ratio = trim_right_ratio
    }

    // MARK: - Preset Configurations

    /// EnCodec 24kHz mono configuration (default for MusicGen).
    public static func encodec_24khz() -> EnCodecConfig {
        EnCodecConfig(
            sample_rate: 24000,
            channels: 1,
            num_codebooks: 8,
            codebook_size: 1024,
            codebook_dim: 128,
            hidden_size: 128,
            num_filters: 32,
            ratios: [8, 5, 4, 2],
            causal: true,
            lstm_layers: 2
        )
    }

    /// EnCodec 32kHz configuration for MusicGen.
    public static func encodec_32khz() -> EnCodecConfig {
        EnCodecConfig(
            sample_rate: 32000,
            channels: 1,
            num_codebooks: 4,
            codebook_size: 2048,
            codebook_dim: 128,
            hidden_size: 128,
            num_filters: 64,
            ratios: [8, 5, 4, 4],
            causal: true,
            lstm_layers: 2
        )
    }

    /// EnCodec 48kHz stereo configuration.
    public static func encodec_48khz_stereo() -> EnCodecConfig {
        EnCodecConfig(
            sample_rate: 48000,
            channels: 2,
            num_codebooks: 8,
            codebook_size: 1024,
            codebook_dim: 128,
            hidden_size: 128,
            num_filters: 32,
            ratios: [8, 5, 4, 2],
            causal: true,
            lstm_layers: 2
        )
    }

    /// Create config from model name.
    public static func fromName(_ name: String) throws -> EnCodecConfig {
        let normalizedName = name.lowercased().replacingOccurrences(of: "-", with: "_")

        switch normalizedName {
        case "encodec_24khz", "24khz":
            return encodec_24khz()
        case "encodec_32khz", "32khz":
            return encodec_32khz()
        case "encodec_48khz_stereo", "48khz":
            return encodec_48khz_stereo()
        default:
            throw EnCodecError.unknownModel(name)
        }
    }
}

// MARK: - Errors

public enum EnCodecError: Error, LocalizedError {
    case unknownModel(String)
    case configLoadFailed(String)
    case weightsLoadFailed(String)

    public var errorDescription: String? {
        switch self {
        case .unknownModel(let name):
            return "Unknown EnCodec model: \(name). Available: encodec_24khz, encodec_32khz, encodec_48khz_stereo"
        case .configLoadFailed(let reason):
            return "Failed to load EnCodec config: \(reason)"
        case .weightsLoadFailed(let reason):
            return "Failed to load EnCodec weights: \(reason)"
        }
    }
}
