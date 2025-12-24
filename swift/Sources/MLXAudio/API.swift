// API.swift
// High-level API for mlx-audio.
//
// Provides simple, one-liner functions for common audio ML tasks.

import Foundation
import MLX
import MLXAudioModels

// MARK: - Error Types

/// Errors that can occur during audio processing.
public enum MLXAudioError: Error, LocalizedError {
    case modelNotFound(String)
    case invalidInput(String)
    case processingFailed(String)
    case notImplemented(String)
    case modelLoadFailed(String)

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
        case .modelLoadFailed(let message):
            return "Failed to load model: \(message)"
        }
    }
}

// MARK: - Model Loading

/// Options for model loading and caching.
public struct ModelOptions: Sendable {
    /// Path to model directory (if pre-downloaded).
    public let modelPath: URL?

    /// Estimated memory usage in MB (for cache management).
    public let estimatedMemoryMB: UInt64

    public init(modelPath: URL? = nil, estimatedMemoryMB: UInt64 = 0) {
        self.modelPath = modelPath
        self.estimatedMemoryMB = estimatedMemoryMB
    }
}

/// Get the model cache for managing loaded models.
public func getModelCache() -> ModelCache {
    return ModelCache.shared
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
///   - options: Model loading options.
/// - Returns: SeparationResult containing separated stems.
/// - Throws: `MLXAudioError` if separation fails.
///
/// Example:
/// ```swift
/// let options = ModelOptions(modelPath: modelURL)
/// let result = try await separate(audio: mixedAudio, options: options)
/// let vocals = result.stems["vocals"]
/// ```
///
/// Supported models:
/// - "htdemucs_ft": Hybrid Transformer Demucs (fine-tuned)
/// - "htdemucs_6s": 6-source variant
public func separate(
    audio: AudioData,
    model: String = "htdemucs_ft",
    stems: [String]? = nil,
    options: ModelOptions = ModelOptions()
) async throws -> SeparationResult {
    // Get model path
    guard let modelPath = options.modelPath else {
        throw MLXAudioError.modelNotFound(
            "Model path must be provided via options.modelPath. " +
            "Download the model first using HuggingFaceDownloader."
        )
    }

    // Get model variant info for memory estimation
    let memoryMB = options.estimatedMemoryMB > 0
        ? options.estimatedMemoryMB
        : UInt64(ModelVariantRegistry.variant(id: model)?.estimatedMemoryMB ?? 2000)

    // Load model via cache
    let htdemucs: HTDemucs = try await ModelCache.shared.get(
        id: model,
        estimatedMemoryMB: memoryMB
    ) {
        try HTDemucs.fromPretrained(path: modelPath)
    }

    // Prepare input: ensure [B, C, T] format
    var inputArray = audio.array

    // Add batch dimension if needed
    if inputArray.ndim == 1 {
        // Mono: [T] -> [1, 1, T]
        inputArray = inputArray.reshaped([1, 1, -1])
    } else if inputArray.ndim == 2 {
        // Stereo [C, T] -> [1, C, T]
        inputArray = inputArray.expandedDimensions(axis: 0)
    }

    // Resample to model sample rate if needed
    let modelSampleRate = htdemucs.config.samplerate
    if audio.sampleRate != modelSampleRate {
        let resampled = audio.resample(to: modelSampleRate)
        inputArray = resampled.array
        if inputArray.ndim == 1 {
            inputArray = inputArray.reshaped([1, 1, -1])
        } else if inputArray.ndim == 2 {
            inputArray = inputArray.expandedDimensions(axis: 0)
        }
    }

    // Run separation
    let output = htdemucs(inputArray)  // [B, S, C, T]
    eval(output)

    // Get source names from config
    let sourceNames = htdemucs.config.sources

    // Build stems dictionary
    var stemDict: [String: AudioData] = [:]
    for (idx, name) in sourceNames.enumerated() {
        // Filter by requested stems if specified
        if let requestedStems = stems, !requestedStems.contains(name) {
            continue
        }

        // Extract stem: [B, S, C, T] -> [C, T]
        let stemArray = output[0, idx]  // [C, T]
        stemDict[name] = AudioData(array: stemArray, sampleRate: modelSampleRate)
    }

    return SeparationResult(
        stems: stemDict,
        sampleRate: modelSampleRate,
        modelName: model,
        metadata: ["sources": sourceNames]
    )
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
///   - options: Model loading options.
/// - Returns: TranscriptionResult containing text and timed segments.
/// - Throws: `MLXAudioError` if transcription fails.
///
/// Example:
/// ```swift
/// let options = ModelOptions(modelPath: modelURL)
/// let result = try await transcribe(audio: speechAudio, options: options)
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
    task: String = "transcribe",
    options: ModelOptions = ModelOptions()
) async throws -> TranscriptionResult {
    guard let modelPath = options.modelPath else {
        throw MLXAudioError.modelNotFound(
            "Model path must be provided via options.modelPath. " +
            "Download the model first using HuggingFaceDownloader."
        )
    }

    // Load Whisper model (not via cache due to Sendable requirement)
    let whisper = try WhisperModel.fromPretrained(path: modelPath)

    // Load tokenizer
    let tokenizerPath = modelPath.appendingPathComponent("tokenizer.json")
    let tokenizer = try WhisperTokenizer(tokenizerPath: tokenizerPath)

    // Resample to 16kHz if needed
    let targetSampleRate = 16000
    var processedAudio = audio
    if audio.sampleRate != targetSampleRate {
        processedAudio = audio.resample(to: targetSampleRate)
    }

    // Convert to mono if stereo
    if !processedAudio.isMono {
        processedAudio = processedAudio.toMono()
    }

    // Compute mel spectrogram using primitives
    let mel = try computeLogMelSpectrogram(
        audio: processedAudio.array,
        sampleRate: targetSampleRate,
        nMels: whisper.config.nMels,
        nFFT: 400,
        hopLength: 160
    )

    // Set up decoding options
    var decodingOptions = DecodingOptions()
    decodingOptions.language = language
    decodingOptions.task = task  // String: "transcribe" or "translate"

    // Run transcription using the MLXAudioModels transcribe function
    let result = MLXAudioModels.transcribe(
        model: whisper,
        mel: mel,
        tokenizer: tokenizer,
        options: decodingOptions
    )

    // Convert to our result type
    let segments = result.segments.map { segment in
        MLXAudio.TranscriptionSegment(
            text: segment.text,
            start: segment.start,
            end: segment.end,
            confidence: 0.0  // MLXAudioModels doesn't provide confidence per segment
        )
    }

    return TranscriptionResult(
        text: result.text,
        segments: segments,
        language: result.language,
        languageProbability: 0.0,  // Would need language detection to get this
        modelName: model,
        metadata: [:]
    )
}

/// Compute log-mel spectrogram for Whisper input.
private func computeLogMelSpectrogram(
    audio: MLXArray,
    sampleRate: Int,
    nMels: Int,
    nFFT: Int,
    hopLength: Int
) throws -> MLXArray {
    // Use primitives to compute mel spectrogram
    let stftConfig = STFTConfig(nFFT: nFFT, hopLength: hopLength)

    // Ensure audio is 1D
    var audioFlat = audio
    if audioFlat.ndim > 1 {
        audioFlat = audioFlat.squeezed()
    }

    // Compute STFT
    let spec = try stft(audioFlat, config: stftConfig)

    // Compute power spectrogram
    let power = spec.magnitude() * spec.magnitude()

    // Create mel filterbank with MelConfig
    let melConfig = MelConfig(sampleRate: sampleRate, nMels: nMels)
    let melFilter = try melFilterbank(nFFT: nFFT, config: melConfig)

    // Apply mel filterbank
    let melSpec = MLX.matmul(melFilter, power)

    // Convert to log scale
    let logMel = MLX.log(MLX.maximum(melSpec, MLXArray(Float(1e-10))))

    // Normalize
    let maxVal = MLX.max(logMel)
    let normalized = MLX.maximum(logMel, maxVal - MLXArray(Float(8.0)))
    let scaled = (normalized + MLXArray(Float(4.0))) / MLXArray(Float(4.0))

    return scaled
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
///   - options: Model loading options.
/// - Returns: GenerationResult containing the generated audio.
/// - Throws: `MLXAudioError` if generation fails.
///
/// Example:
/// ```swift
/// let options = ModelOptions(modelPath: modelURL)
/// let result = try await generate(
///     prompt: "A calm acoustic guitar melody",
///     duration: 30.0,
///     options: options
/// )
/// result.audio.save(to: outputURL)
/// ```
///
/// Supported models:
/// - "musicgen-small", "musicgen-medium", "musicgen-large"
public func generate(
    prompt: String,
    model: String = "musicgen-medium",
    duration: Float = 10.0,
    options: ModelOptions = ModelOptions()
) async throws -> GenerationResult {
    guard let modelPath = options.modelPath else {
        throw MLXAudioError.modelNotFound(
            "Model path must be provided via options.modelPath. " +
            "Download the model first using HuggingFaceDownloader."
        )
    }

    // Get config for model
    let config = musicGenConfig(for: model)

    // Load MusicGen model (not via cache due to Sendable requirement)
    let musicgen = try MusicGen.fromPretrained(path: modelPath.path, config: config)

    // Generate audio codes
    let result = try musicgen.generate(
        prompt: prompt,
        duration: duration
    )

    // Decode audio from codes using EnCodec
    // MusicGen uses EnCodec 32kHz by default
    let encodecConfig = EnCodecConfig.encodec_32khz()
    let encodec = EnCodec(config: encodecConfig)

    // Load EnCodec weights if available
    let encodecWeightsPath = modelPath.appendingPathComponent("encodec_model.safetensors")
    if FileManager.default.fileExists(atPath: encodecWeightsPath.path) {
        try encodec.loadWeights(from: encodecWeightsPath)
    }

    // Decode codes to audio waveform
    // result.codes shape: [B, K, T] where K = num_codebooks, T = sequence length
    let audio = encodec.decode(result.codes)  // Returns [B, C, T]
    eval(audio)

    // Extract first sample, squeeze to [T] or [C, T]
    var audioArray = audio[0]  // [C, T]
    if audioArray.dim(0) == 1 {
        audioArray = audioArray.squeezed(axis: 0)  // [T] for mono
    }

    let sampleRate = encodecConfig.sample_rate
    let audioData = AudioData(array: audioArray, sampleRate: sampleRate)

    return GenerationResult(
        audio: audioData,
        prompt: prompt,
        modelName: model,
        generationParams: ["duration": duration]
    )
}

/// Get MusicGen config for a model variant.
private func musicGenConfig(for model: String) -> MusicGenConfig {
    switch model {
    case "musicgen-small":
        return MusicGenConfig.small()
    case "musicgen-large":
        return MusicGenConfig.large()
    default:
        return MusicGenConfig.medium()
    }
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
///   - options: Model loading options.
/// - Returns: EmbeddingResult containing embedding vectors.
/// - Throws: `MLXAudioError` if embedding fails.
///
/// Example:
/// ```swift
/// let options = ModelOptions(modelPath: modelURL)
/// let audioEmbed = try await embed(audio: audio, options: options)
/// let textEmbed = try await embed(text: "dog barking", options: options)
/// let similarity = audioEmbed.cosineSimilarity(with: textEmbed)
/// ```
///
/// Supported models:
/// - "clap-htsat-fused": CLAP model for audio-text embeddings
public func embed(
    audio: AudioData? = nil,
    text: String? = nil,
    model: String = "clap-htsat-fused",
    options: ModelOptions = ModelOptions()
) async throws -> EmbeddingResult {
    guard audio != nil || text != nil else {
        throw MLXAudioError.invalidInput("Either audio or text must be provided")
    }

    guard let modelPath = options.modelPath else {
        throw MLXAudioError.modelNotFound(
            "Model path must be provided via options.modelPath. " +
            "Download the model first using HuggingFaceDownloader."
        )
    }

    // Get memory estimate
    let memoryMB = options.estimatedMemoryMB > 0
        ? options.estimatedMemoryMB
        : UInt64(ModelVariantRegistry.variant(id: model)?.estimatedMemoryMB ?? 800)

    // Load CLAP model
    let clap: CLAPModel = try await ModelCache.shared.get(
        id: model,
        estimatedMemoryMB: memoryMB
    ) {
        try CLAPModel.fromPretrained(path: modelPath)
    }

    var embeddings: MLXArray

    if let audio = audio {
        // Resample to CLAP sample rate if needed
        let targetSampleRate = clap.config.audio.sampleRate
        var processedAudio = audio
        if audio.sampleRate != targetSampleRate {
            processedAudio = audio.resample(to: targetSampleRate)
        }

        // Ensure batch dimension
        var inputArray = processedAudio.array
        if inputArray.ndim == 1 {
            inputArray = inputArray.expandedDimensions(axis: 0)
        }

        embeddings = try clap.encodeAudio(inputArray, normalize: true)
    } else if let text = text {
        embeddings = try clap.encodeText(text, normalize: true)
    } else {
        throw MLXAudioError.invalidInput("Either audio or text must be provided")
    }

    return EmbeddingResult(
        vectors: embeddings,
        modelName: model,
        metadata: [:]
    )
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
///   - options: Model loading options.
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
    description: String? = nil,
    options: ModelOptions = ModelOptions()
) async throws -> SpeechResult {
    // ParlerTTS not yet ported to Swift
    throw MLXAudioError.notImplemented("Text-to-speech requires ParlerTTS model (not yet ported to Swift)")
}

// MARK: - Audio Classification

/// Classify audio into predefined categories.
///
/// - Parameters:
///   - audio: Input audio to classify.
///   - model: Model name. Default: "clap-htsat-fused".
///   - labels: List of class labels for zero-shot classification.
///   - topK: Number of top predictions to return.
///   - options: Model loading options.
/// - Returns: ClassificationResult containing predictions.
/// - Throws: `MLXAudioError` if classification fails.
///
/// Example:
/// ```swift
/// let options = ModelOptions(modelPath: modelURL)
/// let result = try await classify(
///     audio: audio,
///     labels: ["dog barking", "car horn", "music"],
///     options: options
/// )
/// print(result.predictedClass)
/// ```
public func classify(
    audio: AudioData,
    model: String = "clap-htsat-fused",
    labels: [String],
    topK: Int = 1,
    options: ModelOptions = ModelOptions()
) async throws -> ClassificationResult {
    guard !labels.isEmpty else {
        throw MLXAudioError.invalidInput("Labels must be provided for zero-shot classification")
    }

    guard let modelPath = options.modelPath else {
        throw MLXAudioError.modelNotFound(
            "Model path must be provided via options.modelPath. " +
            "Download the model first using HuggingFaceDownloader."
        )
    }

    // Get memory estimate
    let memoryMB = options.estimatedMemoryMB > 0
        ? options.estimatedMemoryMB
        : UInt64(ModelVariantRegistry.variant(id: model)?.estimatedMemoryMB ?? 800)

    // Load CLAP model
    let clap: CLAPModel = try await ModelCache.shared.get(
        id: model,
        estimatedMemoryMB: memoryMB
    ) {
        try CLAPModel.fromPretrained(path: modelPath)
    }

    // Resample to CLAP sample rate if needed
    let targetSampleRate = clap.config.audio.sampleRate
    var processedAudio = audio
    if audio.sampleRate != targetSampleRate {
        processedAudio = audio.resample(to: targetSampleRate)
    }

    // Ensure batch dimension
    var inputArray = processedAudio.array
    if inputArray.ndim == 1 {
        inputArray = inputArray.expandedDimensions(axis: 0)
    }

    // Use CLAP's classify method
    let probs = try clap.classify(inputArray, labels: labels)

    // Get top-k predictions
    let probsArray = probs.asArray(Float.self)
    let indexed = probsArray.enumerated().map { ($0.offset, $0.element) }
    let sorted = indexed.sorted { $0.1 > $1.1 }

    let topKIndices = Array(sorted.prefix(topK))
    let topKClasses = topKIndices.map { labels[$0.0] }
    let topKProbs = topKIndices.map { $0.1 }

    let predictedClass = labels[sorted[0].0]

    return ClassificationResult(
        predictedClass: predictedClass,
        probabilities: probs,
        classNames: labels,
        topKClasses: topKClasses,
        topKProbs: topKProbs,
        modelName: model,
        metadata: ["method": "zero_shot"]
    )
}

// MARK: - Audio Tagging

/// Tag audio with multiple labels (multi-label classification).
///
/// - Parameters:
///   - audio: Input audio to tag.
///   - model: Model name. Default: "clap-htsat-fused".
///   - tags: List of tag labels.
///   - threshold: Probability threshold for tagging. Default: 0.5.
///   - options: Model loading options.
/// - Returns: TaggingResult containing active tags.
/// - Throws: `MLXAudioError` if tagging fails.
///
/// Example:
/// ```swift
/// let options = ModelOptions(modelPath: modelURL)
/// let result = try await tag(
///     audio: audio,
///     tags: ["loud", "music", "speech", "nature"],
///     threshold: 0.3,
///     options: options
/// )
/// print(result.tags)
/// ```
public func tag(
    audio: AudioData,
    model: String = "clap-htsat-fused",
    tags: [String],
    threshold: Float = 0.5,
    options: ModelOptions = ModelOptions()
) async throws -> TaggingResult {
    guard !tags.isEmpty else {
        throw MLXAudioError.invalidInput("Tags must be provided for zero-shot tagging")
    }

    guard let modelPath = options.modelPath else {
        throw MLXAudioError.modelNotFound(
            "Model path must be provided via options.modelPath. " +
            "Download the model first using HuggingFaceDownloader."
        )
    }

    // Get memory estimate
    let memoryMB = options.estimatedMemoryMB > 0
        ? options.estimatedMemoryMB
        : UInt64(ModelVariantRegistry.variant(id: model)?.estimatedMemoryMB ?? 800)

    // Load CLAP model
    let clap: CLAPModel = try await ModelCache.shared.get(
        id: model,
        estimatedMemoryMB: memoryMB
    ) {
        try CLAPModel.fromPretrained(path: modelPath)
    }

    // Resample to CLAP sample rate if needed
    let targetSampleRate = clap.config.audio.sampleRate
    var processedAudio = audio
    if audio.sampleRate != targetSampleRate {
        processedAudio = audio.resample(to: targetSampleRate)
    }

    // Ensure batch dimension
    var inputArray = processedAudio.array
    if inputArray.ndim == 1 {
        inputArray = inputArray.expandedDimensions(axis: 0)
    }

    // Encode audio
    let audioEmbed = try clap.encodeAudio(inputArray, normalize: true)

    // Encode all tags
    var textEmbeds: [MLXArray] = []
    for tagText in tags {
        let embed = try clap.encodeText(tagText, normalize: true)
        textEmbeds.append(embed)
    }
    let textEmbedsStacked = MLX.concatenated(textEmbeds, axis: 0)

    // Compute similarities
    let similarities = clap.similarity(audioEmbeds: audioEmbed, textEmbeds: textEmbedsStacked)

    // Apply sigmoid for multi-label (independent probabilities)
    let probs = sigmoid(similarities).squeezed(axis: 0)

    // Get active tags above threshold
    let probsArray = probs.asArray(Float.self)
    var activeTags: [String] = []
    for (idx, prob) in probsArray.enumerated() {
        if prob >= threshold {
            activeTags.append(tags[idx])
        }
    }

    return TaggingResult(
        tags: activeTags,
        probabilities: probs,
        tagNames: tags,
        threshold: threshold,
        modelName: model,
        metadata: ["method": "zero_shot"]
    )
}

// MARK: - Re-exports

// Re-export primitives for convenience
@_exported import MLXAudioPrimitives
