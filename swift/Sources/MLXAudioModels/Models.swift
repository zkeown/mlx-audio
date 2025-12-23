// Models.swift
// Placeholder for pre-built model implementations.
//
// This module will contain implementations of:
// - HTDemucs (source separation)
// - Whisper (speech transcription)
// - MusicGen (audio generation)
// - CLAP (audio-text embeddings)
// - Parler TTS (text-to-speech)
// - EnCodec (audio codec)

import Foundation
@preconcurrency import MLX
import MLXNN
import MLXAudioPrimitives

// MARK: - Model Protocol

/// Protocol that all audio models must conform to.
public protocol AudioModel: Sendable {
    /// The model name/identifier.
    static var modelName: String { get }

    /// Load model weights from a path.
    init(weightsPath: URL) throws

    /// Evaluate the model on input.
    func callAsFunction(_ input: MLXArray) -> MLXArray
}

// MARK: - Model Registry

/// Registry for available models.
public struct ModelRegistry: Sendable {
    /// Shared registry instance.
    public static let shared = ModelRegistry()

    private init() {}

    /// Get the default model for a task.
    public func defaultModel(for task: AudioTask) -> String {
        switch task {
        case .separation:
            return "htdemucs_ft"
        case .transcription:
            return "whisper-large-v3-turbo"
        case .generation:
            return "musicgen-medium"
        case .embedding:
            return "clap-htsat-fused"
        case .tts:
            return "parler-tts-mini"
        }
    }
}

/// Audio processing tasks.
public enum AudioTask: String, CaseIterable {
    case separation
    case transcription
    case generation
    case embedding
    case tts
}

// MARK: - Model Implementations
//
// HTDemucs: See HTDemucs/ subdirectory for source separation implementation.
// Whisper: See Whisper/ subdirectory for speech transcription implementation.

// MARK: - MusicGen
// The full MusicGen implementation is in the MusicGen/ subdirectory.
// Use: MusicGen, MusicGenConfig, etc.
// See MusicGen/MusicGen.swift for the main model class.

// MARK: - CLAP
// The full CLAP implementation is in the CLAP/ subdirectory.
// Use: CLAPModel, CLAPConfig, CLAPTokenizer, CLAPFeatureExtractor, etc.
// See CLAP/CLAP.swift for the main model class.
//
// Quick usage:
//   let model = CLAPModel()
//   let audioEmbed = try model.encodeAudio(audio)
//   let textEmbed = try model.encodeText("a dog barking")
//   let similarity = model.similarity(audioEmbeds: audioEmbed, textEmbeds: textEmbed)

// MARK: - EnCodec
// The full EnCodec implementation is in the EnCodec/ subdirectory.
// Use: EnCodec, EnCodecConfig, etc.
// See EnCodec/EnCodec.swift for the main model class.
//
// Quick usage:
//   let config = EnCodecConfig.encodec_24khz()
//   let model = EnCodec(config: config)
//   let codes = model.encode(audio)
//   let reconstructed = model.decode(codes)
