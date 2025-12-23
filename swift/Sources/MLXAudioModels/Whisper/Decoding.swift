// Decoding.swift
// Whisper decoding strategies (greedy and beam search).

import Foundation
import MLX
import MLXNN

// MARK: - Decoding Options

/// Configuration for Whisper transcription decoding.
public struct DecodingOptions: Sendable {
    /// Language code for transcription (nil for auto-detect).
    public var language: String?

    /// Task type ("transcribe" or "translate").
    public var task: String

    /// Sampling temperature (0.0 for greedy).
    public var temperature: Float

    /// Number of beams for beam search.
    public var beamSize: Int

    /// Number of candidates to consider for best-of sampling.
    public var bestOf: Int

    /// Patience factor for beam search.
    public var patience: Float

    /// Length penalty for beam search.
    public var lengthPenalty: Float

    /// Maximum tokens to generate.
    public var maxTokens: Int

    /// Whether to disable timestamp tokens.
    public var withoutTimestamps: Bool

    /// Whether to suppress blank tokens at the start.
    public var suppressBlank: Bool

    /// Token IDs to suppress during generation.
    public var suppressTokens: [Int]?

    /// Whether to use greedy decoding.
    public var isGreedy: Bool {
        temperature == 0.0 && beamSize == 1
    }

    /// Default decoding options.
    public init(
        language: String? = nil,
        task: String = "transcribe",
        temperature: Float = 0.0,
        beamSize: Int = 1,
        bestOf: Int = 1,
        patience: Float = 1.0,
        lengthPenalty: Float = 1.0,
        maxTokens: Int = 448,
        withoutTimestamps: Bool = false,
        suppressBlank: Bool = true,
        suppressTokens: [Int]? = nil
    ) {
        self.language = language
        self.task = task
        self.temperature = temperature
        self.beamSize = beamSize
        self.bestOf = bestOf
        self.patience = patience
        self.lengthPenalty = lengthPenalty
        self.maxTokens = maxTokens
        self.withoutTimestamps = withoutTimestamps
        self.suppressBlank = suppressBlank
        self.suppressTokens = suppressTokens
    }
}

// MARK: - Transcription Results

/// A segment of transcribed audio.
public struct TranscriptionSegment: Sendable {
    /// Transcribed text for this segment.
    public let text: String

    /// Start time in seconds.
    public let start: Float

    /// End time in seconds.
    public let end: Float

    /// Token IDs for this segment.
    public let tokens: [Int]

    public init(text: String, start: Float, end: Float, tokens: [Int]) {
        self.text = text
        self.start = start
        self.end = end
        self.tokens = tokens
    }
}

/// Result of transcription.
public struct TranscriptionResult: Sendable {
    /// Full transcribed text.
    public let text: String

    /// Individual segments with timestamps.
    public let segments: [TranscriptionSegment]

    /// Detected or specified language.
    public let language: String

    public init(text: String, segments: [TranscriptionSegment], language: String) {
        self.text = text
        self.segments = segments
        self.language = language
    }

    /// Export to SRT format.
    public func toSRT() -> String {
        var srt = ""
        for (i, segment) in segments.enumerated() {
            srt += "\(i + 1)\n"
            srt += "\(formatTimestamp(segment.start, srt: true)) --> \(formatTimestamp(segment.end, srt: true))\n"
            srt += "\(segment.text)\n\n"
        }
        return srt
    }

    /// Export to VTT format.
    public func toVTT() -> String {
        var vtt = "WEBVTT\n\n"
        for segment in segments {
            vtt += "\(formatTimestamp(segment.start)) --> \(formatTimestamp(segment.end))\n"
            vtt += "\(segment.text)\n\n"
        }
        return vtt
    }

    /// Format timestamp for subtitle export.
    private func formatTimestamp(_ seconds: Float, srt: Bool = false) -> String {
        let hours = Int(seconds) / 3600
        let minutes = (Int(seconds) % 3600) / 60
        let secs = Int(seconds) % 60
        let millis = Int((seconds - Float(Int(seconds))) * 1000)

        if srt {
            return String(format: "%02d:%02d:%02d,%03d", hours, minutes, secs, millis)
        } else {
            return String(format: "%02d:%02d:%02d.%03d", hours, minutes, secs, millis)
        }
    }
}

// MARK: - Greedy Decoding

/// Perform greedy decoding.
///
/// - Parameters:
///   - model: Whisper model
///   - audioFeatures: Encoded audio features [B, T, D]
///   - tokenizer: Whisper tokenizer
///   - options: Decoding options
/// - Returns: Generated token IDs
public func greedyDecode(
    model: WhisperModel,
    audioFeatures: MLXArray,
    tokenizer: WhisperTokenizer,
    options: DecodingOptions = DecodingOptions()
) -> [Int] {
    // Get initial tokens
    var tokens = tokenizer.getInitialTokens(
        language: options.language,
        task: options.task,
        timestamps: !options.withoutTimestamps
    )

    // Create token array
    var tokenArray = MLXArray(tokens.map { Int32($0) }).reshaped([1, -1])

    // KV cache for efficient decoding
    var kvCache: [(MLXArray, MLXArray)]?

    // Generate tokens
    for _ in 0..<options.maxTokens {
        // Get logits for last token(s)
        let inputTokens: MLXArray
        if let cache = kvCache, !cache.isEmpty {
            // Only feed the last token when using cache
            inputTokens = tokenArray[0..., (-1)...].reshaped([1, 1])
        } else {
            inputTokens = tokenArray
        }

        let (logits, newCache) = model.decode(
            tokens: inputTokens,
            audioFeatures: audioFeatures,
            kvCache: kvCache
        )
        kvCache = newCache

        // Get logits for last position
        var lastLogits = logits[0, -1]

        // Apply temperature if > 0
        if options.temperature > 0 {
            lastLogits = lastLogits / options.temperature
            let probs = softmax(lastLogits, axis: -1)
            // Sample from distribution
            let nextToken = categoricalSample(probs)
            tokens.append(nextToken)
        } else {
            // Greedy: take argmax
            let nextToken = Int(argMax(lastLogits).item(Int32.self))
            tokens.append(nextToken)
        }

        // Check for end of text
        if tokens.last == tokenizer.eot {
            break
        }

        // Update token array
        tokenArray = MLXArray(tokens.map { Int32($0) }).reshaped([1, -1])
    }

    return tokens
}

/// Sample from a categorical distribution.
private func categoricalSample(_ probs: MLXArray) -> Int {
    // Generate uniform random number
    let uniform = MLXRandom.uniform(low: 0.0, high: 1.0, [1])

    // Compute cumulative probabilities
    let cumProbs = MLX.cumsum(probs, axis: -1)

    // Find first index where cumsum > uniform
    let mask = cumProbs .> uniform
    let n = probs.shape[probs.ndim - 1]
    let indices = MLXArray(0..<n).asType(.int32)
    let maskedIndices = MLX.where(mask, indices, MLXArray(Int32(n)))
    return Int(MLX.min(maskedIndices, axis: -1).item(Int32.self))
}

// MARK: - Beam Search Decoding

/// Beam hypothesis for beam search.
private struct BeamHypothesis {
    var tokens: [Int]
    var score: Float
    var kvCache: [(MLXArray, MLXArray)]?
}

/// Perform beam search decoding.
///
/// - Parameters:
///   - model: Whisper model
///   - audioFeatures: Encoded audio features [B, T, D]
///   - tokenizer: Whisper tokenizer
///   - options: Decoding options
/// - Returns: Generated token IDs from best beam
public func beamSearchDecode(
    model: WhisperModel,
    audioFeatures: MLXArray,
    tokenizer: WhisperTokenizer,
    options: DecodingOptions = DecodingOptions()
) -> [Int] {
    let beamSize = max(1, options.beamSize)

    // Get initial tokens
    let initialTokens = tokenizer.getInitialTokens(
        language: options.language,
        task: options.task,
        timestamps: !options.withoutTimestamps
    )

    // Initialize beams
    var beams: [BeamHypothesis] = [
        BeamHypothesis(tokens: initialTokens, score: 0.0, kvCache: nil)
    ]

    // Replicate audio features for beam size
    let batchedAudioFeatures = tiled(audioFeatures, repetitions: [beamSize, 1, 1])

    // Generate tokens
    for step in 0..<options.maxTokens {
        var allCandidates: [(hypothesis: BeamHypothesis, token: Int, score: Float)] = []

        for beam in beams {
            // Skip completed beams
            if beam.tokens.last == tokenizer.eot {
                allCandidates.append((beam, tokenizer.eot, beam.score))
                continue
            }

            // Create token array
            let tokenArray: MLXArray
            if let cache = beam.kvCache, !cache.isEmpty {
                // Only feed the last token when using cache
                tokenArray = MLXArray([Int32(beam.tokens.last!)]).reshaped([1, 1])
            } else {
                tokenArray = MLXArray(beam.tokens.map { Int32($0) }).reshaped([1, -1])
            }

            // Get logits
            let (logits, newCache) = model.decode(
                tokens: tokenArray,
                audioFeatures: audioFeatures,
                kvCache: beam.kvCache
            )

            // Get log probabilities
            let logProbs = logSoftmax(logits[0, -1], axis: -1)

            // Get top-k candidates
            let k = beamSize * 2
            let (topValues, topIndices) = topK(logProbs, k: k)

            for i in 0..<k {
                let token = Int(topIndices[i].item(Int32.self))
                let logProb = topValues[i].item(Float.self)
                let newScore = beam.score + logProb

                var newHypothesis = BeamHypothesis(
                    tokens: beam.tokens + [token],
                    score: newScore,
                    kvCache: newCache
                )

                allCandidates.append((newHypothesis, token, newScore))
            }
        }

        // Sort by length-normalized score
        allCandidates.sort { a, b in
            let scoreA = a.score / pow(Float(a.hypothesis.tokens.count), options.lengthPenalty)
            let scoreB = b.score / pow(Float(b.hypothesis.tokens.count), options.lengthPenalty)
            return scoreA > scoreB
        }

        // Keep top beams
        beams = allCandidates.prefix(beamSize).map { $0.hypothesis }

        // Check if all beams are complete
        let allComplete = beams.allSatisfy { $0.tokens.last == tokenizer.eot }
        if allComplete {
            break
        }
    }

    // Return best beam
    let bestBeam = beams.max { a, b in
        let scoreA = a.score / pow(Float(a.tokens.count), options.lengthPenalty)
        let scoreB = b.score / pow(Float(b.tokens.count), options.lengthPenalty)
        return scoreA < scoreB
    }

    return bestBeam?.tokens ?? []
}

/// Get top-k values and indices.
private func topK(_ array: MLXArray, k: Int) -> (values: MLXArray, indices: MLXArray) {
    let sorted = argSort(array, axis: -1)
    let topIndices = sorted[(-k)...]
    let topValues = array[topIndices]
    return (topValues, topIndices)
}

// MARK: - High-Level Transcription

/// Transcribe audio using Whisper.
///
/// - Parameters:
///   - model: Whisper model
///   - mel: Log-mel spectrogram [B, nMels, T] or [nMels, T]
///   - tokenizer: Whisper tokenizer
///   - options: Decoding options
/// - Returns: Transcription result
public func transcribe(
    model: WhisperModel,
    mel: MLXArray,
    tokenizer: WhisperTokenizer,
    options: DecodingOptions = DecodingOptions()
) -> TranscriptionResult {
    // Encode audio
    let audioFeatures = model.encode(mel)

    // Detect language if needed
    var language = options.language ?? "en"
    if options.language == nil && tokenizer.isMultilingual {
        let (detectedLang, _) = model.detectLanguage(mel: mel, tokenizer: tokenizer)
        language = detectedLang
    }

    // Update options with detected language
    var opts = options
    opts.language = language

    // Decode
    let tokens: [Int]
    if options.isGreedy {
        tokens = greedyDecode(
            model: model,
            audioFeatures: audioFeatures,
            tokenizer: tokenizer,
            options: opts
        )
    } else {
        tokens = beamSearchDecode(
            model: model,
            audioFeatures: audioFeatures,
            tokenizer: tokenizer,
            options: opts
        )
    }

    // Decode tokens to text
    let text = tokenizer.decode(tokens, skipSpecialTokens: true)

    // Extract segments from timestamps (simplified)
    let segments = extractSegments(tokens: tokens, tokenizer: tokenizer)

    return TranscriptionResult(
        text: text,
        segments: segments,
        language: language
    )
}

/// Extract segments from tokens with timestamps.
private func extractSegments(tokens: [Int], tokenizer: WhisperTokenizer) -> [TranscriptionSegment] {
    var segments: [TranscriptionSegment] = []
    var currentTokens: [Int] = []
    var currentStart: Float = 0.0

    for token in tokens {
        if tokenizer.isTimestamp(token) {
            let time = (try? tokenizer.timestampToSeconds(token)) ?? 0.0

            if !currentTokens.isEmpty {
                // End current segment
                let text = tokenizer.decode(currentTokens, skipSpecialTokens: true)
                if !text.trimmingCharacters(in: .whitespaces).isEmpty {
                    segments.append(TranscriptionSegment(
                        text: text,
                        start: currentStart,
                        end: time,
                        tokens: currentTokens
                    ))
                }
                currentTokens = []
            }

            currentStart = time
        } else if token != tokenizer.eot && token != tokenizer.sot {
            // Skip special tokens but collect text tokens
            let specialIds = Set([
                tokenizer.translate,
                tokenizer.transcribe,
                tokenizer.noTimestamps,
                tokenizer.noSpeech
            ] + tokenizer.allLanguageTokens)

            if !specialIds.contains(token) {
                currentTokens.append(token)
            }
        }
    }

    // Add final segment if there are remaining tokens
    if !currentTokens.isEmpty {
        let text = tokenizer.decode(currentTokens, skipSpecialTokens: true)
        if !text.trimmingCharacters(in: .whitespaces).isEmpty {
            segments.append(TranscriptionSegment(
                text: text,
                start: currentStart,
                end: 30.0,  // Default to chunk end
                tokens: currentTokens
            ))
        }
    }

    return segments
}
