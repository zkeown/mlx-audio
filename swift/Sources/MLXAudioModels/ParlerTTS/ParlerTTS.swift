// ParlerTTS.swift
// Parler-TTS text-to-speech model.

import Foundation
@preconcurrency import MLX
import MLXNN

// MARK: - Parler-TTS Model

/// Parler-TTS text-to-speech model.
///
/// Parler-TTS generates natural speech from text descriptions using:
/// 1. T5 text encoder for text prompt conditioning
/// 2. T5 encoder for voice description conditioning
/// 3. Transformer decoder with delay pattern for multi-codebook generation
/// 4. DAC/EnCodec for audio tokenization/detokenization
///
/// Example:
/// ```swift
/// let config = ParlerTTSConfig.mini()
/// let model = ParlerTTS(config: config)
///
/// // Generate from text conditioning
/// let promptEmbeds = ... // T5 encoded text
/// let descEmbeds = ... // T5 encoded voice description
/// let codes = try model.generate(
///     promptHiddenStates: promptEmbeds,
///     descriptionHiddenStates: descEmbeds,
///     duration: 5.0
/// )
/// let audio = model.decodeAudio(codes)
/// ```
public class ParlerTTS: Module, @unchecked Sendable {
    /// Model configuration.
    public let config: ParlerTTSConfig

    /// Codebook embeddings.
    @ModuleInfo var embeddings: ParlerTTSCodebookEmbeddings

    /// Text conditioning projection.
    @ModuleInfo(key: "text_projection") var textProjection: Linear

    /// Description conditioning projection.
    @ModuleInfo(key: "description_projection") var descriptionProjection: Linear

    /// Transformer decoder.
    @ModuleInfo var decoder: ParlerTTSDecoder

    /// Language model head.
    @ModuleInfo(key: "lm_head") var lmHead: ParlerTTSLMHead

    /// Delay pattern scheduler.
    public let delayPattern: ParlerTTSDelayPattern

    /// Audio codec (EnCodec/DAC).
    private var _audioCodec: EnCodec?

    /// Creates a Parler-TTS model.
    public init(config: ParlerTTSConfig = .mini()) {
        self.config = config

        // Codebook embeddings
        self._embeddings.wrappedValue = ParlerTTSCodebookEmbeddings(config: config)

        // Text conditioning projection
        self._textProjection.wrappedValue = Linear(
            config.textHiddenSize,
            config.hiddenSize,
            bias: false
        )

        // Description conditioning projection
        self._descriptionProjection.wrappedValue = Linear(
            config.descriptionHiddenSize,
            config.hiddenSize,
            bias: false
        )

        // Transformer decoder
        self._decoder.wrappedValue = ParlerTTSDecoder(config: config)

        // Language model head
        self._lmHead.wrappedValue = ParlerTTSLMHead(config: config)

        // Delay pattern scheduler
        self.delayPattern = ParlerTTSDelayPattern(
            numCodebooks: config.numCodebooks,
            padTokenId: Int32(config.padTokenId)
        )

        super.init()
    }

    /// Get audio codec.
    public var audioCodec: EnCodec {
        get throws {
            guard let codec = _audioCodec else {
                throw ParlerTTSError.codecNotSet
            }
            return codec
        }
    }

    /// Set the audio codec.
    public func setAudioCodec(_ codec: EnCodec) {
        self._audioCodec = codec
    }

    /// Project text embeddings to decoder hidden size.
    public func projectTextEmbeddings(_ textEmbeddings: MLXArray) -> MLXArray {
        return textProjection(textEmbeddings)
    }

    /// Project description embeddings to decoder hidden size.
    public func projectDescriptionEmbeddings(_ descriptionEmbeddings: MLXArray) -> MLXArray {
        return descriptionProjection(descriptionEmbeddings)
    }

    /// Forward pass through the model.
    ///
    /// - Parameters:
    ///   - inputIds: Codebook token IDs [B, K, T]
    ///   - encoderHiddenStates: Conditioning [B, S, D]
    ///   - attentionMask: Causal self-attention mask
    ///   - encoderAttentionMask: Cross-attention mask
    ///   - kvCache: Cached key/values for incremental decoding
    ///   - positionOffset: Position offset for RoPE
    /// - Returns: Tuple of logits [B, K, T, V] and updated KV cache
    public func forward(
        inputIds: MLXArray,
        encoderHiddenStates: MLXArray? = nil,
        attentionMask: MLXArray? = nil,
        encoderAttentionMask: MLXArray? = nil,
        kvCache: [(MLXArray, MLXArray)]? = nil,
        positionOffset: Int = 0
    ) -> (MLXArray, [(MLXArray, MLXArray)]) {
        // Compute embeddings from codebook tokens
        let hiddenStates = embeddings(inputIds)

        // Create causal mask if not provided
        var mask = attentionMask
        if mask == nil {
            let seqLength = inputIds.dim(2)
            mask = decoder.createCausalMask(seqLength: seqLength, offset: positionOffset)
        }

        // Run decoder
        let (decoderOutput, newKVCache) = decoder(
            hiddenStates,
            encoderHiddenStates: encoderHiddenStates,
            attentionMask: mask,
            encoderAttentionMask: encoderAttentionMask,
            kvCache: kvCache,
            positionOffset: positionOffset
        )

        // Compute logits for all codebooks
        let logits = lmHead(decoderOutput)  // [B, K, T, V]

        return (logits, newKVCache)
    }

    /// Forward pass returning logits only.
    public func callAsFunction(
        _ inputIds: MLXArray,
        encoderHiddenStates: MLXArray? = nil
    ) -> MLXArray {
        let (logits, _) = forward(
            inputIds: inputIds,
            encoderHiddenStates: encoderHiddenStates
        )
        return logits
    }

    /// Generate audio tokens from text conditioning.
    ///
    /// - Parameters:
    ///   - promptHiddenStates: Text prompt conditioning [B, S, D]
    ///   - descriptionHiddenStates: Voice description conditioning [B, S, D]
    ///   - maxNewTokens: Maximum tokens to generate
    ///   - duration: Generation duration in seconds
    ///   - temperature: Sampling temperature
    ///   - topK: Top-k sampling parameter
    ///   - topP: Nucleus sampling threshold
    ///   - seed: Random seed
    ///   - progressCallback: Optional progress callback
    /// - Returns: Generated codes [B, K, T]
    public func generate(
        promptHiddenStates: MLXArray,
        descriptionHiddenStates: MLXArray? = nil,
        maxNewTokens: Int? = nil,
        duration: Float? = nil,
        temperature: Float = 1.0,
        topK: Int = 50,
        topP: Float = 0.0,
        seed: UInt64? = nil,
        progressCallback: ((Float) -> Void)? = nil
    ) throws -> MLXArray {
        // Determine max tokens
        let maxTokens: Int
        if let tokens = maxNewTokens {
            maxTokens = tokens
        } else if let dur = duration {
            maxTokens = Int(dur * Float(config.frameRate))
        } else {
            maxTokens = config.maxNewTokens
        }

        // Project and combine conditioning
        let promptProjected = projectTextEmbeddings(promptHiddenStates)

        let encoderHiddenStates: MLXArray
        if let descHidden = descriptionHiddenStates {
            let descProjected = projectDescriptionEmbeddings(descHidden)
            encoderHiddenStates = MLX.concatenated(
                [descProjected, promptProjected],
                axis: 1
            )
        } else {
            encoderHiddenStates = promptProjected
        }

        // Generate tokens
        let codes = try generateTokens(
            encoderHiddenStates: encoderHiddenStates,
            maxNewTokens: maxTokens,
            temperature: temperature,
            topK: topK,
            topP: topP,
            seed: seed,
            progressCallback: progressCallback
        )

        return codes
    }

    /// Decode audio tokens to waveform.
    public func decodeAudio(_ codes: MLXArray) throws -> MLXArray {
        return try audioCodec.decode(codes)
    }
}

// MARK: - Token Generation

extension ParlerTTS {
    /// Generate audio tokens autoregressively.
    private func generateTokens(
        encoderHiddenStates: MLXArray,
        maxNewTokens: Int,
        temperature: Float,
        topK: Int,
        topP: Float,
        seed: UInt64?,
        progressCallback: ((Float) -> Void)?
    ) throws -> MLXArray {
        let batchSize = encoderHiddenStates.dim(0)
        let numCodebooks = config.numCodebooks

        // Set random seed
        if let s = seed {
            MLXRandom.seed(s)
        }

        // Initialize with BOS tokens for all codebooks [B, K, 1]
        var currentTokens = MLXArray.full(
            [batchSize, numCodebooks, 1],
            values: MLXArray(Int32(config.bosTokenId))
        )

        // KV cache
        var kvCache: [(MLXArray, MLXArray)]? = nil

        // Total steps including delay pattern overhead
        let totalSteps = maxNewTokens + numCodebooks - 1

        // Collect all tokens
        var allTokens: [MLXArray] = [currentTokens]

        for step in 0..<totalSteps {
            // Report progress
            progressCallback?(Float(step) / Float(totalSteps))

            // Forward pass
            let (logits, newCache) = forward(
                inputIds: currentTokens,
                encoderHiddenStates: encoderHiddenStates,
                kvCache: kvCache,
                positionOffset: step
            )
            kvCache = newCache

            // Evaluate
            eval(logits)
            if let cache = kvCache {
                for (k, v) in cache {
                    eval(k, v)
                }
            }

            // Sample next tokens for each codebook
            // logits shape: [B, K, 1, V]
            var sampledTokens: [MLXArray] = []

            for k in 0..<numCodebooks {
                if step >= k {
                    // Codebook is active
                    let codebookLogits = logits[0..., k, 0, 0...]  // [B, V]
                    let sampled = sampleNextToken(
                        logits: codebookLogits,
                        temperature: temperature,
                        topK: topK,
                        topP: topP
                    )
                    sampledTokens.append(sampled.expandedDimensions(axes: [1, 2]))  // [B, 1, 1]
                } else {
                    // Use padding token
                    let padTokens = MLXArray.full(
                        [batchSize, 1, 1],
                        values: MLXArray(Int32(config.padTokenId))
                    )
                    sampledTokens.append(padTokens)
                }
            }

            // Stack along codebook dimension: [B, K, 1]
            let nextTokens = MLX.concatenated(sampledTokens, axis: 1)
            allTokens.append(nextTokens)
            currentTokens = nextTokens
        }

        // Concatenate all tokens [B, K, totalSteps + 1]
        let allCodes = MLX.concatenated(allTokens, axis: 2)

        // Remove BOS token
        let delayedCodes = allCodes[0..., 0..., 1...]

        // Revert delay pattern
        var codes = delayPattern.revertDelayPattern(delayedCodes)

        // Trim to requested length
        codes = codes[0..., 0..., 0..<maxNewTokens]

        // Final progress
        progressCallback?(1.0)

        return codes
    }

}

// MARK: - Weight Loading

extension ParlerTTS {
    /// Load weights from safetensors file.
    public func loadWeights(from url: URL) throws {
        let weights = try loadArrays(url: url)
        update(parameters: ModuleParameters.unflattened(weights))
    }

    /// Load pretrained Parler-TTS model.
    public static func fromPretrained(path: URL, config: ParlerTTSConfig? = nil) throws -> ParlerTTS {
        // Load config
        let modelConfig: ParlerTTSConfig
        let configPath = path.appendingPathComponent("config.json")
        if FileManager.default.fileExists(atPath: configPath.path) {
            modelConfig = try ParlerTTSConfig.load(from: configPath)
        } else {
            modelConfig = config ?? .mini()
        }

        // Create model
        let model = ParlerTTS(config: modelConfig)

        // Load weights
        let weightsPath = path.appendingPathComponent("model.safetensors")
        if FileManager.default.fileExists(atPath: weightsPath.path) {
            try model.loadWeights(from: weightsPath)
        }

        // Try to load audio codec
        let dacPath = path.appendingPathComponent("dac")
        if FileManager.default.fileExists(atPath: dacPath.path) {
            let codec = try EnCodec.fromPretrained(path: dacPath)
            model.setAudioCodec(codec)
        }

        return model
    }
}
