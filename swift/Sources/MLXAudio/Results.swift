// Results.swift
// Result types for mlx-audio tasks.
//
// Provides standardized result types for audio ML operations.

import Foundation
@preconcurrency import MLX

// MARK: - Separation Result

/// Result from audio source separation.
///
/// Contains separated audio stems (e.g., vocals, drums, bass, other)
/// with attribute-style access.
///
/// Example:
/// ```swift
/// let result = try await separate(audio: audio, model: "htdemucs_ft")
/// let vocals = result.stems["vocals"]
/// result.save(to: outputDirectory)
/// ```
public struct SeparationResult: @unchecked Sendable {
    /// Dictionary mapping stem names to separated audio.
    public let stems: [String: AudioData]

    /// Sample rate of all stems.
    public let sampleRate: Int

    /// Name of the model used for separation.
    public let modelName: String

    /// Additional metadata from separation.
    public let metadata: [String: Any]

    /// Creates a new separation result.
    public init(
        stems: [String: AudioData],
        sampleRate: Int,
        modelName: String = "",
        metadata: [String: Any] = [:]
    ) {
        self.stems = stems
        self.sampleRate = sampleRate
        self.modelName = modelName
        self.metadata = metadata
    }

    /// List of available stem names.
    public var availableStems: [String] {
        return Array(stems.keys).sorted()
    }

    /// Access a stem by name.
    ///
    /// - Parameter name: The stem name (e.g., "vocals", "drums").
    /// - Returns: The AudioData for the stem, or nil if not found.
    public subscript(name: String) -> AudioData? {
        return stems[name]
    }

    /// Save all stems to a directory.
    ///
    /// - Parameters:
    ///   - directory: Output directory URL.
    ///   - format: Audio format (wav, flac, mp3). Default: "wav".
    ///   - stemNames: Specific stems to save. If nil, saves all.
    /// - Returns: Dictionary mapping stem names to saved file URLs.
    public func save(
        to directory: URL,
        format: String = "wav",
        stemNames: [String]? = nil
    ) throws -> [String: URL] {
        // Note: Actual file saving requires AudioToolbox or similar
        // This is a placeholder that returns expected paths
        var savedFiles: [String: URL] = [:]

        for (name, _) in stems {
            if stemNames == nil || stemNames!.contains(name) {
                let fileURL = directory.appendingPathComponent("\(name).\(format)")
                savedFiles[name] = fileURL
            }
        }

        return savedFiles
    }
}

// MARK: - Transcription Result

/// A segment of transcribed text with timing.
public struct TranscriptionSegment: Sendable {
    /// The transcribed text for this segment.
    public let text: String

    /// Start time in seconds.
    public let start: Float

    /// End time in seconds.
    public let end: Float

    /// Confidence score (0-1).
    public let confidence: Float

    public init(text: String, start: Float, end: Float, confidence: Float = 0.0) {
        self.text = text
        self.start = start
        self.end = end
        self.confidence = confidence
    }

    /// Duration in seconds.
    public var duration: Float {
        return end - start
    }
}

/// Result from speech transcription.
///
/// Contains the full transcription and timed segments.
///
/// Example:
/// ```swift
/// let result = try await transcribe(audio: audio, model: "whisper-large-v3-turbo")
/// print(result.text)
/// let subtitles = result.toSRT()
/// ```
public struct TranscriptionResult: @unchecked Sendable {
    /// Full transcription text.
    public let text: String

    /// List of timed segments.
    public let segments: [TranscriptionSegment]

    /// Detected language code (e.g., "en", "es", "fr").
    public let language: String?

    /// Confidence in language detection (0-1).
    public let languageProbability: Float

    /// Name of the model used.
    public let modelName: String

    /// Additional metadata.
    public let metadata: [String: Any]

    public init(
        text: String,
        segments: [TranscriptionSegment] = [],
        language: String? = nil,
        languageProbability: Float = 0.0,
        modelName: String = "",
        metadata: [String: Any] = [:]
    ) {
        self.text = text
        self.segments = segments
        self.language = language
        self.languageProbability = languageProbability
        self.modelName = modelName
        self.metadata = metadata
    }

    /// Export as SRT subtitle format.
    public func toSRT() -> String {
        var lines: [String] = []

        for (index, segment) in segments.enumerated() {
            lines.append("\(index + 1)")
            lines.append("\(formatTimestamp(segment.start)) --> \(formatTimestamp(segment.end))")
            lines.append(segment.text.trimmingCharacters(in: .whitespaces))
            lines.append("")
        }

        return lines.joined(separator: "\n")
    }

    /// Export as WebVTT subtitle format.
    public func toVTT() -> String {
        var lines: [String] = ["WEBVTT", ""]

        for segment in segments {
            lines.append("\(formatTimestamp(segment.start, vtt: true)) --> \(formatTimestamp(segment.end, vtt: true))")
            lines.append(segment.text.trimmingCharacters(in: .whitespaces))
            lines.append("")
        }

        return lines.joined(separator: "\n")
    }

    /// Format timestamp for subtitle files.
    private func formatTimestamp(_ seconds: Float, vtt: Bool = false) -> String {
        let hours = Int(seconds / 3600)
        let minutes = Int((seconds.truncatingRemainder(dividingBy: 3600)) / 60)
        let secs = seconds.truncatingRemainder(dividingBy: 60)
        let separator = vtt ? "." : ","
        return String(format: "%02d:%02d:%06.3f", hours, minutes, secs).replacingOccurrences(of: ".", with: separator)
    }

    /// Save transcription to file.
    ///
    /// - Parameters:
    ///   - url: Output file URL.
    ///   - format: Output format (txt, srt, vtt, json). Default: "txt".
    public func save(to url: URL, format: String = "txt") throws {
        let content: String
        switch format.lowercased() {
        case "srt":
            content = toSRT()
        case "vtt":
            content = toVTT()
        case "json":
            let encoder = JSONEncoder()
            encoder.outputFormatting = .prettyPrinted
            let data = try encoder.encode(TranscriptionJSON(
                text: text,
                language: language,
                segments: segments.map { TranscriptionSegmentJSON(text: $0.text, start: $0.start, end: $0.end) }
            ))
            content = String(data: data, encoding: .utf8) ?? ""
        default:
            content = text
        }

        try content.write(to: url, atomically: true, encoding: .utf8)
    }
}

// JSON encoding helpers
private struct TranscriptionJSON: Encodable {
    let text: String
    let language: String?
    let segments: [TranscriptionSegmentJSON]
}

private struct TranscriptionSegmentJSON: Encodable {
    let text: String
    let start: Float
    let end: Float
}

// MARK: - Generation Result

/// Result from audio generation.
///
/// Contains generated audio with metadata about the generation.
///
/// Example:
/// ```swift
/// let result = try await generate(prompt: "A calm piano melody", model: "musicgen-medium")
/// result.audio.save(to: outputURL)
/// ```
public struct GenerationResult: @unchecked Sendable {
    /// The generated audio.
    public let audio: AudioData

    /// The text prompt used for generation.
    public let prompt: String

    /// Name of the model used.
    public let modelName: String

    /// Parameters used for generation.
    public let generationParams: [String: Any]

    public init(
        audio: AudioData,
        prompt: String = "",
        modelName: String = "",
        generationParams: [String: Any] = [:]
    ) {
        self.audio = audio
        self.prompt = prompt
        self.modelName = modelName
        self.generationParams = generationParams
    }

    /// Duration of the generated audio in seconds.
    public var duration: TimeInterval {
        return audio.duration
    }
}

// MARK: - Embedding Result

/// Result from audio/text embedding.
///
/// Supports both single and batch embeddings with similarity computation.
///
/// Example:
/// ```swift
/// let audioEmbed = try await embed(audio: audio, model: "clap-htsat-fused")
/// let textEmbed = try await embed(text: "dog barking", model: "clap-htsat-fused")
/// let similarity = audioEmbed.cosineSimilarity(with: textEmbed)
/// ```
public struct EmbeddingResult: @unchecked Sendable {
    /// Embedding vectors. Shape: [embeddingDim] or [batch, embeddingDim].
    public let vectors: MLXArray

    /// Name of the model used.
    public let modelName: String

    /// Additional metadata.
    public let metadata: [String: Any]

    public init(
        vectors: MLXArray,
        modelName: String = "",
        metadata: [String: Any] = [:]
    ) {
        self.vectors = vectors
        self.modelName = modelName
        self.metadata = metadata
    }

    /// Get single embedding vector (first if batched).
    public var vector: MLXArray {
        if vectors.ndim == 1 {
            return vectors
        }
        return vectors[0]
    }

    /// Embedding dimension.
    public var dimension: Int {
        return vectors.shape.last ?? 0
    }

    /// Compute cosine similarity with another embedding.
    ///
    /// - Parameter other: Another embedding result.
    /// - Returns: Cosine similarity (-1 to 1).
    public func cosineSimilarity(with other: EmbeddingResult) -> Float {
        // Compute L2 norms manually: sqrt(sum(x^2))
        let normA = MLX.sqrt(MLX.sum(vector * vector))
        let normB = MLX.sqrt(MLX.sum(other.vector * other.vector))
        let a = vector / normA
        let b = other.vector / normB
        return MLX.sum(a * b).item(Float.self)
    }

    /// Convert to Float array.
    public func toFloatArray() -> [Float] {
        return vector.asArray(Float.self)
    }
}

// MARK: - Speech Result

/// Result from text-to-speech synthesis.
///
/// Contains the synthesized audio and generation metadata.
public struct SpeechResult: @unchecked Sendable {
    /// The synthesized audio.
    public let audio: AudioData

    /// Original text input.
    public let text: String

    /// Voice description used (if any).
    public let description: String?

    /// Name of the model used.
    public let modelName: String

    /// Parameters used for generation.
    public let generationParams: [String: Any]

    public init(
        audio: AudioData,
        text: String = "",
        description: String? = nil,
        modelName: String = "",
        generationParams: [String: Any] = [:]
    ) {
        self.audio = audio
        self.text = text
        self.description = description
        self.modelName = modelName
        self.generationParams = generationParams
    }

    /// Duration of the synthesized speech in seconds.
    public var duration: TimeInterval {
        return audio.duration
    }
}

// MARK: - Classification Result

/// Result from audio classification.
///
/// Contains predicted class and probabilities.
public struct ClassificationResult: @unchecked Sendable {
    /// Predicted class index or name.
    public let predictedClass: String

    /// Class probabilities.
    public let probabilities: MLXArray

    /// Optional list of class names.
    public let classNames: [String]?

    /// Top-k predicted classes.
    public let topKClasses: [String]?

    /// Top-k probabilities.
    public let topKProbs: [Float]?

    /// Name of the model used.
    public let modelName: String

    /// Additional metadata.
    public let metadata: [String: Any]

    public init(
        predictedClass: String,
        probabilities: MLXArray,
        classNames: [String]? = nil,
        topKClasses: [String]? = nil,
        topKProbs: [Float]? = nil,
        modelName: String = "",
        metadata: [String: Any] = [:]
    ) {
        self.predictedClass = predictedClass
        self.probabilities = probabilities
        self.classNames = classNames
        self.topKClasses = topKClasses
        self.topKProbs = topKProbs
        self.modelName = modelName
        self.metadata = metadata
    }

    /// Confidence of the top prediction.
    public var confidence: Float {
        return MLX.max(probabilities).item(Float.self)
    }
}

// MARK: - Tagging Result

/// Result from audio tagging (multi-label classification).
///
/// Contains active tags above threshold and all probabilities.
public struct TaggingResult: @unchecked Sendable {
    /// List of active tag names/indices.
    public let tags: [String]

    /// Tag probabilities.
    public let probabilities: MLXArray

    /// Optional list of all tag names.
    public let tagNames: [String]?

    /// Threshold used for tagging.
    public let threshold: Float

    /// Name of the model used.
    public let modelName: String

    /// Additional metadata.
    public let metadata: [String: Any]

    public init(
        tags: [String],
        probabilities: MLXArray,
        tagNames: [String]? = nil,
        threshold: Float = 0.5,
        modelName: String = "",
        metadata: [String: Any] = [:]
    ) {
        self.tags = tags
        self.probabilities = probabilities
        self.tagNames = tagNames
        self.threshold = threshold
        self.modelName = modelName
        self.metadata = metadata
    }

    /// Get top-k tags by probability.
    ///
    /// - Parameter k: Number of top tags to return.
    /// - Returns: Array of (tag, probability) tuples.
    public func topK(_ k: Int = 5) -> [(String, Float)] {
        let probs = probabilities.asArray(Float.self)
        let indexed = probs.enumerated().map { ($0.offset, $0.element) }
        let sorted = indexed.sorted { $0.1 > $1.1 }.prefix(k)

        return sorted.map { index, prob in
            let name = tagNames?[index] ?? "\(index)"
            return (name, prob)
        }
    }
}
