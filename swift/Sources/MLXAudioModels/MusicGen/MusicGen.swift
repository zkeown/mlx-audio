// MusicGen.swift
// Main MusicGen model for text-to-music generation.

import Foundation
import MLX
import MLXFast
import MLXNN

/// MusicGen model for generating music from text prompts.
/// Architecture: T5 encoder → projection → transformer decoder → LM head → codes
public class MusicGen: Module {

    // MARK: - Text Encoder Components

    public var t5Encoder: T5Encoder?
    public var t5Tokenizer: T5Tokenizer?
    public let t5Config: T5Config

    // MARK: - Audio Decoder Components

    @ModuleInfo(key: "embeddings") var embeddings: CodebookEmbeddings
    @ModuleInfo(key: "text_projection") var textProjection: Linear
    @ModuleInfo(key: "decoder") var decoder: MusicGenDecoder
    @ModuleInfo(key: "lm_head") var lmHead: MusicGenLMHead

    // MARK: - Configuration

    public let config: MusicGenConfig
    public let delayPattern: DelayPatternScheduler

    // MARK: - Initialization

    public init(config: MusicGenConfig, t5Config: T5Config = .base()) {
        self.config = config
        self.t5Config = t5Config
        self.delayPattern = DelayPatternScheduler(
            numCodebooks: config.numCodebooks,
            padTokenId: config.padTokenId
        )

        self._embeddings.wrappedValue = CodebookEmbeddings(config: config)
        self._textProjection.wrappedValue = Linear(
            config.textHiddenSize, config.hiddenSize, bias: true)
        self._decoder.wrappedValue = MusicGenDecoder(config: config)
        self._lmHead.wrappedValue = MusicGenLMHead(config: config)

        // T5 components are loaded separately
        self.t5Encoder = nil
        self.t5Tokenizer = nil

        super.init()
    }

    /// Initialize T5 encoder and tokenizer.
    public func loadT5(from path: String) throws {
        t5Encoder = try T5Encoder.fromPretrained(path: path, config: t5Config)
        t5Tokenizer = try T5Tokenizer.fromPretrained(path: path)
    }

    // MARK: - Forward Pass

    /// Forward pass through the decoder.
    /// - Parameters:
    ///   - inputIds: Audio token IDs [B, K, T]
    ///   - encoderHiddenStates: Text encoder output [B, S, D_text]
    ///   - encoderAttentionMask: Mask for encoder output [B, S]
    ///   - kvCache: KV cache for incremental decoding
    /// - Returns: Tuple of (logits [B, K, T, V], updatedKVCache)
    public func callAsFunction(
        inputIds: MLXArray,
        encoderHiddenStates: MLXArray,
        encoderAttentionMask: MLXArray? = nil,
        kvCache: MusicGenKVCache? = nil
    ) -> (MLXArray, MusicGenKVCache?) {
        // Embed audio tokens
        let hidden = embeddings(inputIds)  // [B, T, D]

        // Project text embeddings to decoder dimension
        let projectedEncoder = textProjection(encoderHiddenStates)  // [B, S, D]

        // Create attention masks
        let seqLength = inputIds.dim(2)
        let offset = kvCache?.currentLength ?? 0
        let totalLength = offset + seqLength
        let causalMask = createCausalMask(
            queryLength: seqLength,
            keyLength: totalLength,
            offset: offset
        )

        // Encoder attention mask (if provided)
        var encAttnMask: MLXArray? = nil
        if let mask = encoderAttentionMask {
            // Convert [B, S] to [B, 1, 1, S]
            let maskExpanded = mask.expandedDimensions(axes: [1, 2])
            let negInf = MLXArray(-Float.infinity)
            let zero = MLXArray(Float(0.0))
            encAttnMask = MLX.where(maskExpanded .== 0, negInf, zero)
        }

        // Run decoder
        let (decoderOutput, updatedCache) = decoder(
            hidden,
            encoderHiddenStates: projectedEncoder,
            attentionMask: causalMask,
            encoderAttentionMask: encAttnMask,
            kvCache: kvCache
        )

        // Compute logits
        let logits = lmHead(decoderOutput)

        return (logits, updatedCache)
    }

    // MARK: - Generation

    /// Generate audio codes from text encoder hidden states.
    /// - Parameters:
    ///   - encoderHiddenStates: Text encoder output [B, S, D_text]
    ///   - encoderAttentionMask: Mask for encoder output [B, S]
    ///   - maxNewTokens: Maximum number of tokens to generate
    ///   - samplingConfig: Sampling parameters
    ///   - progressCallback: Optional callback for progress updates
    /// - Returns: Generated audio codes [B, K, T]
    public func generate(
        encoderHiddenStates: MLXArray,
        encoderAttentionMask: MLXArray? = nil,
        maxNewTokens: Int? = nil,
        samplingConfig: MusicGenSamplingConfig = .default,
        progressCallback: MusicGenProgressCallback? = nil
    ) -> MusicGenGenerationResult {
        let batchSize = encoderHiddenStates.dim(0)
        let numTokens = maxNewTokens ?? config.maxNewTokens

        // Initialize with BOS tokens
        var generatedCodes = MLXArray.full(
            [batchSize, config.numCodebooks, 1],
            values: MLXArray(config.bosTokenId)
        )

        // Initialize KV cache
        var kvCache: MusicGenKVCache? = nil

        // Generate tokens autoregressively
        for step in 0 ..< numTokens {
            // Build input for this step using delay pattern
            let inputCodes: MLXArray
            if step == 0 {
                inputCodes = generatedCodes
            } else {
                inputCodes = delayPattern.buildInput(generatedCodes: generatedCodes, step: step)
            }

            // Forward pass
            let (logits, newCache) = self(
                inputIds: inputCodes,
                encoderHiddenStates: encoderHiddenStates,
                encoderAttentionMask: encoderAttentionMask,
                kvCache: kvCache
            )
            kvCache = newCache

            // Get logits for last position: [B, K, V]
            let lastLogits = logits[0..., 0..., -1, 0...]

            // Sample next token for each codebook
            var nextTokens = MLXArray.full(
                [batchSize, config.numCodebooks],
                values: MLXArray(config.padTokenId)
            )

            // Only sample for valid codebooks at this step
            let validCodebooks = delayPattern.getValidCodebooks(step: step + 1)

            for k in validCodebooks {
                let codebookLogits = lastLogits[0..., k, 0...]  // [B, V]

                let sampledToken = sampleNextToken(
                    logits: codebookLogits,
                    temperature: samplingConfig.temperature,
                    topK: samplingConfig.topK,
                    topP: samplingConfig.topP
                )

                nextTokens[0..., k] = sampledToken
            }

            // Append to generated codes
            let nextTokensExpanded = nextTokens.expandedDimensions(axis: 2)  // [B, K, 1]
            generatedCodes = concatenated([generatedCodes, nextTokensExpanded], axis: 2)

            // Progress callback
            progressCallback?(step + 1, numTokens)

            // Evaluate to ensure computation happens
            eval(generatedCodes)
        }

        // Revert delay pattern to get final codes
        let finalCodes = delayPattern.revertDelayPattern(generatedCodes)

        return MusicGenGenerationResult(codes: finalCodes)
    }

    // MARK: - Text-to-Music Generation

    /// Generate music from a text prompt.
    /// - Parameters:
    ///   - prompt: Text description of the desired music
    ///   - duration: Duration in seconds (default: 10)
    ///   - samplingConfig: Sampling parameters
    ///   - progressCallback: Optional callback for progress updates
    /// - Returns: Generated audio codes
    public func generate(
        prompt: String,
        duration: Float = 10.0,
        samplingConfig: MusicGenSamplingConfig = .default,
        progressCallback: MusicGenProgressCallback? = nil
    ) throws -> MusicGenGenerationResult {
        guard let tokenizer = t5Tokenizer else {
            throw MusicGenError.invalidInput("T5 tokenizer not loaded. Call loadT5(from:) first.")
        }
        guard let encoder = t5Encoder else {
            throw MusicGenError.invalidInput("T5 encoder not loaded. Call loadT5(from:) first.")
        }

        // Tokenize the prompt
        let (inputIds, attentionMask) = tokenizer.encode(prompt)

        // Add batch dimension
        let batchedInputIds = inputIds.expandedDimensions(axis: 0)
        let batchedMask = attentionMask.expandedDimensions(axis: 0)

        // Encode with T5
        let encoderHiddenStates = encoder(
            inputIds: batchedInputIds,
            attentionMask: batchedMask
        )

        // Calculate number of tokens for desired duration
        let maxNewTokens = Int(duration * Float(config.frameRate))

        // Generate audio codes
        return generate(
            encoderHiddenStates: encoderHiddenStates,
            encoderAttentionMask: batchedMask,
            maxNewTokens: maxNewTokens,
            samplingConfig: samplingConfig,
            progressCallback: progressCallback
        )
    }

    /// Generate music from multiple text prompts (batch).
    public func generate(
        prompts: [String],
        duration: Float = 10.0,
        samplingConfig: MusicGenSamplingConfig = .default,
        progressCallback: MusicGenProgressCallback? = nil
    ) throws -> MusicGenGenerationResult {
        guard let tokenizer = t5Tokenizer else {
            throw MusicGenError.invalidInput("T5 tokenizer not loaded. Call loadT5(from:) first.")
        }
        guard let encoder = t5Encoder else {
            throw MusicGenError.invalidInput("T5 encoder not loaded. Call loadT5(from:) first.")
        }

        // Batch tokenize
        let (inputIds, attentionMask) = tokenizer.encodeBatch(prompts)

        // Encode with T5
        let encoderHiddenStates = encoder(
            inputIds: inputIds,
            attentionMask: attentionMask
        )

        // Calculate number of tokens
        let maxNewTokens = Int(duration * Float(config.frameRate))

        return generate(
            encoderHiddenStates: encoderHiddenStates,
            encoderAttentionMask: attentionMask,
            maxNewTokens: maxNewTokens,
            samplingConfig: samplingConfig,
            progressCallback: progressCallback
        )
    }

    // MARK: - Weight Loading

    /// Load model weights from safetensors file.
    public static func fromPretrained(path: String, config: MusicGenConfig) throws -> MusicGen {
        let model = MusicGen(config: config)

        // Load weights
        let weightsPath = (path as NSString).appendingPathComponent("model.safetensors")
        let weights = try loadArrays(url: URL(fileURLWithPath: weightsPath))

        // Map and load weights
        let mappedWeights = mapWeightKeys(weights)
        let parameters = ModuleParameters.unflattened(mappedWeights)
        try model.update(parameters: parameters, verify: .noUnusedKeys)

        return model
    }

    /// Load complete model with T5 text encoder.
    /// - Parameters:
    ///   - musicgenPath: Path to MusicGen model weights
    ///   - t5Path: Path to T5 encoder weights
    ///   - config: MusicGen configuration
    ///   - t5Config: T5 configuration
    public static func fromPretrained(
        musicgenPath: String,
        t5Path: String,
        config: MusicGenConfig,
        t5Config: T5Config = .base()
    ) throws -> MusicGen {
        let model = try fromPretrained(path: musicgenPath, config: config)
        try model.loadT5(from: t5Path)
        return model
    }

    /// Map weight keys from Python naming to Swift naming.
    private static func mapWeightKeys(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var mapped: [String: MLXArray] = [:]

        for (key, value) in weights {
            var newKey = key

            // Common transformations
            newKey = newKey.replacingOccurrences(of: "self_attn", with: "selfAttn")
            newKey = newKey.replacingOccurrences(of: "encoder_attn", with: "crossAttn")
            newKey = newKey.replacingOccurrences(of: "self_attn_layer_norm", with: "selfAttnLayerNorm")
            newKey = newKey.replacingOccurrences(of: "encoder_attn_layer_norm", with: "crossAttnLayerNorm")
            newKey = newKey.replacingOccurrences(of: "final_layer_norm", with: "finalLayerNorm")
            newKey = newKey.replacingOccurrences(of: "layer_norm", with: "layerNorm")
            newKey = newKey.replacingOccurrences(of: "text_projection", with: "textProjection")
            newKey = newKey.replacingOccurrences(of: "lm_head", with: "lmHead")
            newKey = newKey.replacingOccurrences(of: "q_proj", with: "qProj")
            newKey = newKey.replacingOccurrences(of: "k_proj", with: "kProj")
            newKey = newKey.replacingOccurrences(of: "v_proj", with: "vProj")
            newKey = newKey.replacingOccurrences(of: "out_proj", with: "outProj")
            newKey = newKey.replacingOccurrences(of: "position_embedding", with: "positionEmbedding")

            mapped[newKey] = value
        }

        return mapped
    }
}
