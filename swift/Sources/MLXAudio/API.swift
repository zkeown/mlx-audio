// API.swift
// High-level API for mlx-audio.
//
// Provides simple, one-liner functions for common audio ML tasks.

import Foundation
import MLX

// MARK: - Error Types

/// Errors that can occur during audio processing.
public enum MLXAudioError: Error, LocalizedError {
    case modelNotFound(String)
    case invalidInput(String)
    case processingFailed(String)
    case notImplemented(String)

    public var errorDescription: String? {
        switch self {
        case .modelNotFound(let model):
            return "Model not found: \(model)"
        case .invalidInput(let message):
            return "Invalid input: \(message)"
        case .processingFailed(let message):
            return "Processing failed: \(message)"
        case .notImplemented(let feature):
            return "Not yet implemented: \(feature)"
        }
    }
}

// MARK: - Source Separation

/// Separate audio sources from a mixed audio signal.
///
/// Separates audio into individual stems (e.g., vocals, drums, bass, other)
/// using neural source separation models.
///
/// - Parameters:
///   - audio: Input audio to separate.
///   - model: Model name or path. Default: "htdemucs_ft".
///   - stems: Specific stems to extract. If nil, extracts all available stems.
/// - Returns: SeparationResult containing separated stems.
/// - Throws: `MLXAudioError` if separation fails.
///
/// Example:
/// ```swift
/// let result = try await separate(audio: mixedAudio, model: "htdemucs_ft")
/// let vocals = result.stems["vocals"]
/// ```
///
/// Supported models:
/// - "htdemucs_ft": Hybrid Transformer Demucs (fine-tuned)
/// - "htdemucs_6s": 6-source variant
public func separate(
    audio: AudioData,
    model: String = "htdemucs_ft",
    stems: [String]? = nil
) async throws -> SeparationResult {
    // Placeholder implementation
    // Actual implementation will load model and perform separation
    throw MLXAudioError.notImplemented("Source separation requires HTDemucs model implementation")
}

// MARK: - Speech Transcription

/// Transcribe speech from audio to text.
///
/// Converts spoken audio to text with optional timestamps and language detection.
///
/// - Parameters:
///   - audio: Input audio to transcribe.
///   - model: Model name. Default: "whisper-large-v3-turbo".
///   - language: Language code (e.g., "en", "es"). If nil, auto-detects.
///   - task: Task to perform: "transcribe" or "translate". Default: "transcribe".
/// - Returns: TranscriptionResult containing text and timed segments.
/// - Throws: `MLXAudioError` if transcription fails.
///
/// Example:
/// ```swift
/// let result = try await transcribe(audio: speechAudio)
/// print(result.text)
/// let subtitles = result.toSRT()
/// ```
///
/// Supported models:
/// - "whisper-tiny", "whisper-base", "whisper-small", "whisper-medium", "whisper-large-v3"
/// - "whisper-large-v3-turbo" (default, fastest large model)
public func transcribe(
    audio: AudioData,
    model: String = "whisper-large-v3-turbo",
    language: String? = nil,
    task: String = "transcribe"
) async throws -> TranscriptionResult {
    // Placeholder implementation
    throw MLXAudioError.notImplemented("Speech transcription requires Whisper model implementation")
}

// MARK: - Audio Generation

/// Generate audio from a text prompt.
///
/// Creates music or audio based on a text description using generative models.
///
/// - Parameters:
///   - prompt: Text description of the desired audio.
///   - model: Model name. Default: "musicgen-medium".
///   - duration: Duration in seconds. Default: 10.0.
/// - Returns: GenerationResult containing the generated audio.
/// - Throws: `MLXAudioError` if generation fails.
///
/// Example:
/// ```swift
/// let result = try await generate(
///     prompt: "A calm acoustic guitar melody",
///     model: "musicgen-medium",
///     duration: 30.0
/// )
/// result.audio.save(to: outputURL)
/// ```
///
/// Supported models:
/// - "musicgen-small", "musicgen-medium", "musicgen-large"
public func generate(
    prompt: String,
    model: String = "musicgen-medium",
    duration: Float = 10.0
) async throws -> GenerationResult {
    // Placeholder implementation
    throw MLXAudioError.notImplemented("Audio generation requires MusicGen model implementation")
}

// MARK: - Audio Embedding

/// Compute embeddings for audio and/or text.
///
/// Creates dense vector representations that can be used for
/// similarity search, classification, and other downstream tasks.
///
/// - Parameters:
///   - audio: Input audio to embed. Optional if text is provided.
///   - text: Input text to embed. Optional if audio is provided.
///   - model: Model name. Default: "clap-htsat-fused".
/// - Returns: EmbeddingResult containing embedding vectors.
/// - Throws: `MLXAudioError` if embedding fails.
///
/// Example:
/// ```swift
/// let audioEmbed = try await embed(audio: audio)
/// let textEmbed = try await embed(text: "dog barking")
/// let similarity = audioEmbed.cosineSimilarity(with: textEmbed)
/// ```
///
/// Supported models:
/// - "clap-htsat-fused": CLAP model for audio-text embeddings
public func embed(
    audio: AudioData? = nil,
    text: String? = nil,
    model: String = "clap-htsat-fused"
) async throws -> EmbeddingResult {
    guard audio != nil || text != nil else {
        throw MLXAudioError.invalidInput("Either audio or text must be provided")
    }

    // Placeholder implementation
    throw MLXAudioError.notImplemented("Audio embedding requires CLAP model implementation")
}

// MARK: - Text-to-Speech

/// Synthesize speech from text.
///
/// Converts text to natural-sounding speech using neural TTS models.
///
/// - Parameters:
///   - text: Text to synthesize.
///   - model: Model name. Default: "parler-tts-mini".
///   - description: Optional voice description for style control.
/// - Returns: SpeechResult containing synthesized audio.
/// - Throws: `MLXAudioError` if synthesis fails.
///
/// Example:
/// ```swift
/// let result = try await speak(
///     text: "Hello, world!",
///     description: "A calm female voice"
/// )
/// result.audio.play()
/// ```
///
/// Supported models:
/// - "parler-tts-mini": Parler TTS (small, fast)
public func speak(
    text: String,
    model: String = "parler-tts-mini",
    description: String? = nil
) async throws -> SpeechResult {
    // Placeholder implementation
    throw MLXAudioError.notImplemented("Text-to-speech requires Parler TTS model implementation")
}

// MARK: - Audio Classification

/// Classify audio into predefined categories.
///
/// - Parameters:
///   - audio: Input audio to classify.
///   - model: Model name. Default: "clap-htsat-fused".
///   - labels: Optional list of class labels for zero-shot classification.
/// - Returns: ClassificationResult containing predictions.
/// - Throws: `MLXAudioError` if classification fails.
///
/// Example:
/// ```swift
/// let result = try await classify(
///     audio: audio,
///     labels: ["dog barking", "car horn", "music"]
/// )
/// print(result.predictedClass)
/// ```
public func classify(
    audio: AudioData,
    model: String = "clap-htsat-fused",
    labels: [String]? = nil
) async throws -> ClassificationResult {
    // Placeholder implementation
    throw MLXAudioError.notImplemented("Audio classification requires CLAP model implementation")
}

// MARK: - Audio Tagging

/// Tag audio with multiple labels (multi-label classification).
///
/// - Parameters:
///   - audio: Input audio to tag.
///   - model: Model name. Default: "clap-htsat-fused".
///   - tags: Optional list of tag labels.
///   - threshold: Probability threshold for tagging. Default: 0.5.
/// - Returns: TaggingResult containing active tags.
/// - Throws: `MLXAudioError` if tagging fails.
///
/// Example:
/// ```swift
/// let result = try await tag(
///     audio: audio,
///     tags: ["loud", "music", "speech", "nature"],
///     threshold: 0.3
/// )
/// print(result.tags)
/// ```
public func tag(
    audio: AudioData,
    model: String = "clap-htsat-fused",
    tags: [String]? = nil,
    threshold: Float = 0.5
) async throws -> TaggingResult {
    // Placeholder implementation
    throw MLXAudioError.notImplemented("Audio tagging requires CLAP model implementation")
}

// MARK: - Re-exports

// Re-export primitives for convenience
@_exported import MLXAudioPrimitives
