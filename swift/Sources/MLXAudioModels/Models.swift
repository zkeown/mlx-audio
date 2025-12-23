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

/// Placeholder for MusicGen audio generation model.
///
/// MusicGen is a transformer-based model for text-to-music generation.
public struct MusicGen {
    public static let modelName = "musicgen"

    /// Supported model sizes.
    public enum Size: String, CaseIterable {
        case small, medium, large
    }

    public init() {}

    /// Generate audio from text prompt.
    ///
    /// - Parameters:
    ///   - prompt: Text description of desired audio.
    ///   - duration: Duration in seconds.
    /// - Returns: Generated audio array.
    public func generate(prompt: String, duration: Float) throws -> MLXArray {
        // Placeholder - actual implementation requires model weights
        fatalError("MusicGen model not yet implemented")
    }
}

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

/// Placeholder for EnCodec audio codec.
///
/// EnCodec is a neural audio codec for high-quality audio compression.
public struct EnCodec {
    public static let modelName = "encodec"

    public init() {}

    /// Encode audio to tokens.
    public func encode(_ audio: MLXArray) throws -> MLXArray {
        fatalError("EnCodec model not yet implemented")
    }

    /// Decode tokens to audio.
    public func decode(_ tokens: MLXArray) throws -> MLXArray {
        fatalError("EnCodec model not yet implemented")
    }
}
