// CLAPClassifier.swift
// CLAP-based audio classifier for fine-tuning.
//
// Uses frozen CLAP embeddings with a trainable classification head
// for few-shot audio classification.

import Foundation
import MLX
import MLXNN

// MARK: - CLAP Classifier Configuration

/// Configuration for CLAP classifier.
public struct CLAPClassifierConfig: Codable, Sendable {
    /// Number of output classes.
    public let numClasses: Int

    /// Embedding dimension (from CLAP model).
    public let embeddingDim: Int

    /// Hidden dimension for MLP head (nil = single linear layer).
    public let hiddenDim: Int?

    /// Dropout probability.
    public let dropout: Float

    /// Temperature for softmax scaling.
    public let temperature: Float

    /// Whether to use LoRA on CLAP audio encoder.
    public let useLoRA: Bool

    /// LoRA rank (if useLoRA is true).
    public let loraRank: Int

    /// Creates a classifier configuration.
    public init(
        numClasses: Int,
        embeddingDim: Int = 512,
        hiddenDim: Int? = nil,
        dropout: Float = 0.0,
        temperature: Float = 1.0,
        useLoRA: Bool = false,
        loraRank: Int = 8
    ) {
        self.numClasses = numClasses
        self.embeddingDim = embeddingDim
        self.hiddenDim = hiddenDim
        self.dropout = dropout
        self.temperature = temperature
        self.useLoRA = useLoRA
        self.loraRank = loraRank
    }
}

// MARK: - Classification Head

/// MLP classification head.
public class ClassificationHead: Module, @unchecked Sendable {
    @ModuleInfo var linear1: Linear
    @ModuleInfo var linear2: Linear?
    let dropout: Dropout?
    let hasHidden: Bool

    /// Creates a classification head.
    ///
    /// - Parameters:
    ///   - inputDim: Input dimension
    ///   - numClasses: Number of output classes
    ///   - hiddenDim: Optional hidden dimension
    ///   - dropout: Dropout probability
    public init(
        inputDim: Int,
        numClasses: Int,
        hiddenDim: Int? = nil,
        dropout: Float = 0.0
    ) {
        if let hidden = hiddenDim {
            self._linear1.wrappedValue = Linear(inputDim, hidden)
            self._linear2.wrappedValue = Linear(hidden, numClasses)
            self.hasHidden = true
        } else {
            self._linear1.wrappedValue = Linear(inputDim, numClasses)
            self._linear2.wrappedValue = nil
            self.hasHidden = false
        }

        self.dropout = dropout > 0 ? Dropout(p: dropout) : nil

        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = linear1(x)

        if hasHidden {
            out = gelu(out)
            if let dropout = dropout, training {
                out = dropout(out)
            }
            out = linear2!(out)
        }

        return out
    }
}

// MARK: - CLAP Classifier

/// CLAP-based few-shot audio classifier.
///
/// Uses frozen CLAP audio embeddings with a trainable classification head.
/// This enables efficient fine-tuning on small datasets.
///
/// Example:
/// ```swift
/// // Load pretrained CLAP
/// let clap = try CLAPModel.fromPretrained(path: clapPath)
///
/// // Create classifier for 10 classes
/// let classifier = CLAPClassifier(
///     clapEncoder: clap,
///     config: CLAPClassifierConfig(numClasses: 10)
/// )
///
/// // Train with few examples
/// let trainer = Trainer()
/// trainer.fit(module: classifier, trainData: fewShotData)
/// ```
public class CLAPClassifier: Module, TrainModule, @unchecked Sendable {
    /// Configuration.
    public let config: CLAPClassifierConfig

    /// CLAP audio encoder (frozen by default).
    private let clapEncoder: Module

    /// Whether encoder is frozen.
    private let encoderFrozen: Bool

    /// Classification head (trainable).
    @ModuleInfo(key: "head") var classificationHead: ClassificationHead

    /// Temperature for softmax.
    public var temperature: Float

    /// Feature extractor for preprocessing (optional).
    public var featureExtractor: Any?

    /// Learning rate for training.
    private let learningRate: Float

    /// Weight decay for training.
    private let weightDecay: Float

    /// Creates a CLAP classifier.
    ///
    /// - Parameters:
    ///   - clapEncoder: CLAP model or audio encoder
    ///   - config: Classifier configuration
    ///   - learningRate: Learning rate (default: 1e-3)
    ///   - weightDecay: Weight decay (default: 0.01)
    public init(
        clapEncoder: Module,
        config: CLAPClassifierConfig,
        learningRate: Float = 1e-3,
        weightDecay: Float = 0.01
    ) {
        self.config = config
        self.clapEncoder = clapEncoder
        self.temperature = config.temperature
        self.learningRate = learningRate
        self.weightDecay = weightDecay

        // Create classification head
        self._classificationHead.wrappedValue = ClassificationHead(
            inputDim: config.embeddingDim,
            numClasses: config.numClasses,
            hiddenDim: config.hiddenDim,
            dropout: config.dropout
        )

        // Initialize encoderFrozen before super.init() - will be updated after
        self.encoderFrozen = !config.useLoRA

        super.init()

        // Apply LoRA if configured
        if config.useLoRA {
            applyLoRA(
                to: clapEncoder,
                config: LoRAConfig(
                    rank: config.loraRank,
                    targetModules: ["query", "value"]
                )
            )
        } else {
            // Freeze encoder
            clapEncoder.freeze()
        }
    }

    /// Encode audio to embeddings.
    ///
    /// - Parameter audio: Audio input (waveform or mel spectrogram)
    /// - Returns: Audio embeddings [B, D]
    /// - Note: This is a placeholder. The actual implementation should call
    ///         the encoder's specific method (e.g., encodeAudio on CLAPModel).
    public func encodeAudio(_ audio: MLXArray) throws -> MLXArray {
        // This is a placeholder that returns the audio as-is
        // In actual use, users should subclass or provide a proper encoder
        // that implements embedding extraction

        // For a proper implementation with CLAPModel:
        // let embedding = clapModel.encodeAudio(audio)

        // Placeholder: assume input is already embeddings
        let embedding = audio

        // L2 normalize
        let squaredEmb = MLX.multiply(embedding, embedding)
        let norm = MLX.sqrt(MLX.sum(squaredEmb, axis: -1, keepDims: true))
        return embedding / (norm + 1e-8)
    }

    /// Forward pass.
    ///
    /// - Parameter audio: Audio input [B, ...]
    /// - Returns: Class logits [B, numClasses]
    public func callAsFunction(_ audio: MLXArray) -> MLXArray {
        do {
            let embeddings = try encodeAudio(audio)
            let logits = classificationHead(embeddings)
            return logits / temperature
        } catch {
            // Return zeros on error (shouldn't happen in practice)
            return MLXArray.zeros([audio.dim(0), config.numClasses])
        }
    }

    /// Predict class probabilities.
    ///
    /// - Parameter audio: Audio input
    /// - Returns: Probabilities [B, numClasses]
    public func predict(_ audio: MLXArray) -> MLXArray {
        let logits = self(audio)
        return softmax(logits, axis: -1)
    }

    /// Predict class labels.
    ///
    /// - Parameter audio: Audio input
    /// - Returns: Class indices [B]
    public func predictLabels(_ audio: MLXArray) -> MLXArray {
        let logits = self(audio)
        return argMax(logits, axis: -1)
    }

    // MARK: - TrainModule

    public func computeLoss(batch: [MLXArray]) -> (MLXArray, [String: MLXArray]) {
        guard batch.count >= 2 else {
            return (MLXArray(Float.infinity), [:])
        }

        let audio = batch[0]
        let labels = batch[1]

        // Forward pass
        let logits = self(audio)

        // Cross-entropy loss
        let loss = crossEntropyLoss(
            logits: logits,
            targets: labels,
            reduction: .mean
        )

        // Compute accuracy
        let predictions = argMax(logits, axis: -1)
        let correct = (predictions .== labels).asType(.float32)
        let accuracy = MLX.mean(correct)

        return (loss, ["accuracy": accuracy])
    }

    public func configureOptimizers() -> OptimizerConfig {
        // Use warmup cosine schedule
        let optimizer = AdamW(
            schedule: WarmupCosineSchedule(
                peakLR: learningRate,
                warmupSteps: 100,
                totalSteps: 1000,
                minLR: learningRate / 10
            ),
            weightDecay: weightDecay
        )

        return OptimizerConfig(
            optimizer: optimizer,
            schedulerName: "warmup_cosine"
        )
    }

    // MARK: - Save/Load

    /// Save classifier weights.
    public func saveWeights(to path: URL) throws {
        var weights: [String: MLXArray] = [:]

        // Save classification head
        for (key, value) in classificationHead.parameters().flattened() {
            weights["head.\(key)"] = value
        }

        // Save LoRA weights if using LoRA
        if config.useLoRA {
            for (key, value) in clapEncoder.trainableParameters().flattened() {
                if key.contains("lora") {
                    weights["encoder.\(key)"] = value
                }
            }
        }

        try save(arrays: weights, url: path)
    }

    /// Load classifier weights.
    public func loadWeights(from path: URL) throws {
        let weights = try loadArrays(url: path)

        var headUpdates: [(String, MLXArray)] = []
        var encoderUpdates: [(String, MLXArray)] = []

        for (key, value) in weights {
            if key.hasPrefix("head.") {
                let subKey = String(key.dropFirst(5))
                headUpdates.append((subKey, value))
            } else if key.hasPrefix("encoder.") {
                let subKey = String(key.dropFirst(8))
                encoderUpdates.append((subKey, value))
            }
        }

        if !headUpdates.isEmpty {
            try classificationHead.update(
                parameters: ModuleParameters.unflattened(headUpdates)
            )
        }

        if !encoderUpdates.isEmpty && config.useLoRA {
            try clapEncoder.update(
                parameters: ModuleParameters.unflattened(encoderUpdates)
            )
        }
    }
}

// MARK: - Zero-Shot Classifier

/// Zero-shot CLAP classifier using text prompts.
///
/// No training required - uses CLAP's audio-text alignment directly.
public class CLAPZeroShotClassifier: @unchecked Sendable {
    /// CLAP model for embeddings.
    private let clap: Module

    /// Text embeddings for each class.
    private var textEmbeddings: MLXArray?

    /// Class labels.
    private var classLabels: [String] = []

    /// Text encoder function.
    private let encodeText: (String) throws -> MLXArray

    /// Audio encoder function.
    private let encodeAudio: (MLXArray) throws -> MLXArray

    /// Creates a zero-shot classifier.
    ///
    /// - Parameters:
    ///   - clap: CLAP model
    ///   - encodeText: Function to encode text to embeddings
    ///   - encodeAudio: Function to encode audio to embeddings
    public init(
        clap: Module,
        encodeText: @escaping (String) throws -> MLXArray,
        encodeAudio: @escaping (MLXArray) throws -> MLXArray
    ) {
        self.clap = clap
        self.encodeText = encodeText
        self.encodeAudio = encodeAudio
    }

    /// Set classification labels.
    ///
    /// This computes text embeddings for all labels once.
    ///
    /// - Parameter labels: Array of class labels
    public func setLabels(_ labels: [String]) throws {
        classLabels = labels

        var embeddings: [MLXArray] = []
        for label in labels {
            let emb = try encodeText(label)
            embeddings.append(emb)
        }

        textEmbeddings = MLX.concatenated(embeddings, axis: 0)
        eval(textEmbeddings!)
    }

    /// Classify audio.
    ///
    /// - Parameter audio: Audio input
    /// - Returns: (label, probability, all probabilities)
    public func classify(_ audio: MLXArray) throws -> (
        label: String,
        probability: Float,
        allProbabilities: [String: Float]
    ) {
        guard let textEmb = textEmbeddings else {
            throw ClassifierError.labelsNotSet
        }

        // Encode audio
        let audioEmb = try encodeAudio(audio)

        // Compute similarities
        let similarity = MLX.matmul(audioEmb, textEmb.T)
        let probs = softmax(similarity, axis: -1).squeezed(axis: 0)

        eval(probs)

        // Get best prediction
        let maxIdx = Int(argMax(probs).item(Int32.self))
        let maxProb = probs[maxIdx].item(Float.self)

        // Build probability dictionary
        var allProbs: [String: Float] = [:]
        for (i, label) in classLabels.enumerated() {
            allProbs[label] = probs[i].item(Float.self)
        }

        return (classLabels[maxIdx], maxProb, allProbs)
    }

    /// Classify multiple audio samples.
    public func classifyBatch(_ audio: MLXArray) throws -> [(label: String, probability: Float)] {
        guard let textEmb = textEmbeddings else {
            throw ClassifierError.labelsNotSet
        }

        let audioEmb = try encodeAudio(audio)
        let similarity = MLX.matmul(audioEmb, textEmb.T)
        let probs = softmax(similarity, axis: -1)

        eval(probs)

        var results: [(String, Float)] = []
        let batchSize = probs.dim(0)

        for i in 0..<batchSize {
            let sampleProbs = probs[i]
            let maxIdx = Int(argMax(sampleProbs).item(Int32.self))
            let maxProb = sampleProbs[maxIdx].item(Float.self)
            results.append((classLabels[maxIdx], maxProb))
        }

        return results
    }
}

// MARK: - Errors

public enum ClassifierError: Error, LocalizedError {
    case labelsNotSet
    case invalidInput(String)

    public var errorDescription: String? {
        switch self {
        case .labelsNotSet:
            return "Class labels not set. Call setLabels() first."
        case .invalidInput(let message):
            return "Invalid input: \(message)"
        }
    }
}
