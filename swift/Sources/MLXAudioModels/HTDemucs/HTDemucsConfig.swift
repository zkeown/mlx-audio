// HTDemucsConfig.swift
// Configuration for HTDemucs source separation model.

import Foundation

/// Configuration for the HTDemucs source separation model.
///
/// Mirrors the Python `HTDemucsConfig` exactly for weight compatibility.
public struct HTDemucsConfig: Codable, Sendable, Equatable {

    // MARK: - Audio Settings

    /// List of source names to separate.
    public var sources: [String]

    /// Number of audio channels (1=mono, 2=stereo).
    public var audio_channels: Int

    /// Expected sample rate in Hz.
    public var samplerate: Int

    /// Segment duration in seconds for chunked processing.
    public var segment: Float

    // MARK: - Encoder/Decoder Architecture

    /// Initial hidden channels.
    public var channels: Int

    /// Channel growth factor per layer.
    public var growth: Float

    /// Number of encoder/decoder layers.
    public var depth: Int

    /// Convolution kernel size.
    public var kernel_size: Int

    /// Convolution stride.
    public var stride: Int

    // MARK: - Spectrogram Settings

    /// FFT size for STFT.
    public var nfft: Int

    /// Hop length for STFT.
    public var hop_length: Int

    /// Frequency embedding scale.
    public var freq_emb: Float

    // MARK: - Transformer Settings

    /// Number of transformer layers.
    public var t_depth: Int

    /// Number of attention heads.
    public var t_heads: Int

    /// Transformer dropout rate.
    public var t_dropout: Float

    /// FFN hidden dimension multiplier.
    public var t_hidden_scale: Float

    /// Positional embedding type ("sin", "scaled", "cape").
    public var t_pos_embedding: String

    /// Transformer dimension (0 = use encoder output channels).
    public var bottom_channels: Int

    // MARK: - DConv Settings

    /// Depth of dilated conv residual blocks.
    public var dconv_depth: Int

    /// Compression factor for hidden channels.
    public var dconv_comp: Int

    /// Number of LSTM layers in DConv (0 = none).
    public var dconv_lstm: Int

    /// Number of attention heads in DConv (0 = none).
    public var dconv_attn: Int

    // MARK: - Output Mode

    /// Complex-as-channels mode.
    public var cac: Bool

    // MARK: - Computed Properties

    /// Number of sources to separate.
    public var num_sources: Int {
        sources.count
    }

    /// Number of frequency bins in STFT.
    public var freq_bins: Int {
        nfft / 2 + 1
    }

    // MARK: - Initialization

    /// Creates an HTDemucs configuration with default values.
    public init(
        sources: [String] = ["drums", "bass", "other", "vocals"],
        audio_channels: Int = 2,
        samplerate: Int = 44100,
        segment: Float = 6.0,
        channels: Int = 48,
        growth: Float = 2.0,
        depth: Int = 4,
        kernel_size: Int = 8,
        stride: Int = 4,
        nfft: Int = 4096,
        hop_length: Int = 1024,
        freq_emb: Float = 0.2,
        t_depth: Int = 5,
        t_heads: Int = 8,
        t_dropout: Float = 0.0,
        t_hidden_scale: Float = 4.0,
        t_pos_embedding: String = "sin",
        bottom_channels: Int = 512,
        dconv_depth: Int = 2,
        dconv_comp: Int = 8,
        dconv_lstm: Int = 0,
        dconv_attn: Int = 0,
        cac: Bool = true
    ) {
        self.sources = sources
        self.audio_channels = audio_channels
        self.samplerate = samplerate
        self.segment = segment
        self.channels = channels
        self.growth = growth
        self.depth = depth
        self.kernel_size = kernel_size
        self.stride = stride
        self.nfft = nfft
        self.hop_length = hop_length
        self.freq_emb = freq_emb
        self.t_depth = t_depth
        self.t_heads = t_heads
        self.t_dropout = t_dropout
        self.t_hidden_scale = t_hidden_scale
        self.t_pos_embedding = t_pos_embedding
        self.bottom_channels = bottom_channels
        self.dconv_depth = dconv_depth
        self.dconv_comp = dconv_comp
        self.dconv_lstm = dconv_lstm
        self.dconv_attn = dconv_attn
        self.cac = cac
    }

    // MARK: - Preset Configurations

    /// Fine-tuned HTDemucs configuration (default pretrained).
    public static func htdemucs_ft() -> HTDemucsConfig {
        HTDemucsConfig(
            sources: ["drums", "bass", "other", "vocals"],
            channels: 48,
            depth: 4,
            t_depth: 5,
            t_heads: 8
        )
    }

    /// 6-source HTDemucs configuration.
    public static func htdemucs_6s() -> HTDemucsConfig {
        HTDemucsConfig(
            sources: ["drums", "bass", "other", "vocals", "guitar", "piano"],
            channels: 48,
            depth: 4,
            t_depth: 5,
            t_heads: 8
        )
    }
}
