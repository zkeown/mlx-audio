// Trainer.swift
// Core training orchestration for MLX-Audio.
//
// Manages the training loop, gradient computation, optimization,
// and integration with callbacks for monitoring and checkpointing.

import Foundation
import MLX
import MLXNN

// MARK: - Trainer Configuration

/// Configuration for the Trainer.
public struct TrainerConfig: Sendable {
    /// Maximum number of epochs (nil for unlimited).
    public var maxEpochs: Int?

    /// Maximum number of training steps (nil for unlimited).
    public var maxSteps: Int?

    /// Validation check interval (1.0 = every epoch, 0.5 = twice per epoch).
    public var valCheckInterval: Float

    /// Maximum gradient norm for clipping (nil = no clipping).
    public var gradientClipVal: Float?

    /// Gradient clipping algorithm.
    public var gradientClipAlgorithm: GradientClipAlgorithm

    /// Random seed for reproducibility.
    public var seed: UInt64?

    /// Whether to enable checkpointing.
    public var enableCheckpointing: Bool

    /// Directory for checkpoints.
    public var checkpointDir: URL?

    /// Log metrics every N steps.
    public var logEveryNSteps: Int

    /// Creates a trainer configuration.
    public init(
        maxEpochs: Int? = nil,
        maxSteps: Int? = nil,
        valCheckInterval: Float = 1.0,
        gradientClipVal: Float? = nil,
        gradientClipAlgorithm: GradientClipAlgorithm = .norm,
        seed: UInt64? = nil,
        enableCheckpointing: Bool = true,
        checkpointDir: URL? = nil,
        logEveryNSteps: Int = 10
    ) {
        // Default to 1 epoch if neither is specified
        if maxEpochs == nil && maxSteps == nil {
            self.maxEpochs = 1
        } else {
            self.maxEpochs = maxEpochs
        }
        self.maxSteps = maxSteps
        self.valCheckInterval = valCheckInterval
        self.gradientClipVal = gradientClipVal
        self.gradientClipAlgorithm = gradientClipAlgorithm
        self.seed = seed
        self.enableCheckpointing = enableCheckpointing
        self.checkpointDir = checkpointDir
        self.logEveryNSteps = logEveryNSteps
    }
}

// MARK: - Training Progress

/// Progress information during training.
public struct TrainingProgress: Sendable {
    /// Current epoch.
    public let epoch: Int

    /// Current step within epoch.
    public let step: Int

    /// Global step across all epochs.
    public let globalStep: Int

    /// Current loss value.
    public let loss: Float

    /// Current learning rate.
    public let learningRate: Float

    /// Additional metrics.
    public let metrics: [String: Float]
}

/// Delegate for receiving training progress updates.
public protocol TrainerDelegate: AnyObject, Sendable {
    /// Called after each training step.
    func trainer(_ trainer: Trainer, didCompleteStep progress: TrainingProgress)

    /// Called at the end of each epoch.
    func trainer(_ trainer: Trainer, didCompleteEpoch epoch: Int, metrics: [String: Float])

    /// Called when training completes.
    func trainerDidFinishTraining(_ trainer: Trainer)

    /// Called when an error occurs.
    func trainer(_ trainer: Trainer, didEncounterError error: Error)
}

// Default implementations
public extension TrainerDelegate {
    func trainer(_ trainer: Trainer, didCompleteStep progress: TrainingProgress) {}
    func trainer(_ trainer: Trainer, didCompleteEpoch epoch: Int, metrics: [String: Float]) {}
    func trainerDidFinishTraining(_ trainer: Trainer) {}
    func trainer(_ trainer: Trainer, didEncounterError error: Error) {}
}

// MARK: - Trainer

/// Core training orchestrator.
///
/// The Trainer handles the training loop, gradient computation, optimization,
/// and coordination of callbacks. It supports:
/// - Automatic gradient computation via MLX's valueAndGrad
/// - Gradient clipping
/// - Learning rate scheduling (via optimizer)
/// - Validation during training
/// - Early stopping via callbacks
/// - Checkpointing
public final class Trainer: @unchecked Sendable {
    /// Trainer configuration.
    public let config: TrainerConfig

    /// Current training state.
    public private(set) var state: TrainingState = TrainingState()

    /// Registered callbacks.
    public private(set) var callbacks: [any TrainingCallback] = []

    /// Delegate for progress updates.
    public weak var delegate: (any TrainerDelegate)?

    /// Lock for thread-safe state access.
    private let lock = NSLock()

    /// Creates a trainer with the given configuration.
    public init(config: TrainerConfig = TrainerConfig()) {
        self.config = config

        if let seed = config.seed {
            MLXRandom.seed(seed)
        }
    }

    // MARK: - Callback Management

    /// Add a training callback.
    public func addCallback(_ callback: any TrainingCallback) {
        lock.lock()
        defer { lock.unlock() }
        callbacks.append(callback)
        callbacks.sort { $0.priority < $1.priority }
    }

    /// Remove all callbacks.
    public func removeAllCallbacks() {
        lock.lock()
        defer { lock.unlock() }
        callbacks.removeAll()
    }

    // MARK: - Training

    /// Train a model.
    ///
    /// - Parameters:
    ///   - module: The module to train (must conform to TrainModule)
    ///   - trainData: Training data as array of batches
    ///   - valData: Optional validation data
    ///   - checkpointPath: Optional path to resume from checkpoint
    public func fit<M: Module & TrainModule>(
        module: M,
        trainData: [[MLXArray]],
        valData: [[MLXArray]]? = nil,
        checkpointPath: URL? = nil
    ) throws {
        // Get optimizer configuration
        let optConfig = module.configureOptimizers()
        let optimizer = optConfig.optimizer

        // Load checkpoint if provided
        if let path = checkpointPath {
            try loadCheckpoint(from: path, module: module, optimizer: optimizer)
        }

        // Fire onFitStart callbacks
        for callback in callbacks {
            callback.onFitStart(trainer: self, module: module)
        }
        module.onTrainStart()

        do {
            try runTrainingLoop(
                module: module,
                optimizer: optimizer,
                trainData: trainData,
                valData: valData
            )
        } catch {
            // Fire exception callbacks
            for callback in callbacks {
                callback.onException(trainer: self, module: module, error: error)
            }
            delegate?.trainer(self, didEncounterError: error)
            throw error
        }

        // Fire completion callbacks
        module.onTrainEnd()
        for callback in callbacks {
            callback.onFitEnd(trainer: self, module: module)
        }
        delegate?.trainerDidFinishTraining(self)
    }

    /// Signal training to stop.
    public func stop() {
        lock.lock()
        defer { lock.unlock() }
        state.shouldStop = true
    }

    // MARK: - Training Loop

    private func runTrainingLoop<M: Module & TrainModule>(
        module: M,
        optimizer: any Optimizer,
        trainData: [[MLXArray]],
        valData: [[MLXArray]]?
    ) throws {
        while !state.shouldStop {
            // Check epoch limit
            if let maxEpochs = config.maxEpochs, state.currentEpoch >= maxEpochs {
                break
            }

            // Check step limit
            if let maxSteps = config.maxSteps, state.globalStep >= maxSteps {
                break
            }

            try runTrainingEpoch(
                module: module,
                optimizer: optimizer,
                trainData: trainData,
                valData: valData
            )

            state.currentEpoch += 1
        }
    }

    private func runTrainingEpoch<M: Module & TrainModule>(
        module: M,
        optimizer: any Optimizer,
        trainData: [[MLXArray]],
        valData: [[MLXArray]]?
    ) throws {
        // Fire epoch start callbacks
        module.onTrainEpochStart(epoch: state.currentEpoch)
        for callback in callbacks {
            callback.onEpochStart(trainer: self, module: module, epoch: state.currentEpoch)
        }

        var epochLoss: Float = 0
        var epochMetrics: [String: [Float]] = [:]
        var batchCount = 0

        for (batchIdx, batch) in trainData.enumerated() {
            // Check step limit
            if let maxSteps = config.maxSteps, state.globalStep >= maxSteps {
                break
            }

            // Check if stopped
            if state.shouldStop {
                break
            }

            // Compute loss and gradients
            let (loss, grads, metrics) = computeLossAndGradients(module: module, batch: batch)

            // Clip gradients if configured
            var clippedGrads = grads
            if let clipVal = config.gradientClipVal {
                clippedGrads = clipGradients(
                    grads,
                    maxNorm: clipVal,
                    algorithm: config.gradientClipAlgorithm
                )
            }

            // Optimizer step
            optimizer.update(model: module, gradients: clippedGrads)

            // Force evaluation (critical for MLX lazy evaluation)
            eval(loss)
            eval(module.parameters())

            // Track metrics
            let lossValue = loss.item(Float.self)
            epochLoss += lossValue
            batchCount += 1

            for (key, value) in metrics {
                let floatValue = value.item(Float.self)
                epochMetrics[key, default: []].append(floatValue)
            }

            state.globalStep += 1

            // Log progress
            if state.globalStep % config.logEveryNSteps == 0 {
                var stepMetrics: [String: Float] = ["loss": lossValue]
                for (key, values) in epochMetrics {
                    stepMetrics[key] = values.last ?? 0
                }

                let progress = TrainingProgress(
                    epoch: state.currentEpoch,
                    step: batchIdx,
                    globalStep: state.globalStep,
                    loss: lossValue,
                    learningRate: optimizer.learningRate,
                    metrics: stepMetrics
                )

                delegate?.trainer(self, didCompleteStep: progress)

                // Fire batch callbacks
                for callback in callbacks {
                    callback.onBatchEnd(
                        trainer: self,
                        module: module,
                        batch: batchIdx,
                        metrics: stepMetrics
                    )
                }
            }
        }

        // Compute epoch averages
        var avgMetrics: [String: Float] = [:]
        if batchCount > 0 {
            avgMetrics["train_loss"] = epochLoss / Float(batchCount)
            for (key, values) in epochMetrics {
                avgMetrics["train_\(key)"] = values.reduce(0, +) / Float(values.count)
            }
        }

        // Run validation if configured
        if let valData = valData, config.valCheckInterval > 0 {
            let valMetrics = try runValidation(module: module, valData: valData)
            for (key, value) in valMetrics {
                avgMetrics[key] = value
            }
        }

        state.epochMetrics = avgMetrics

        // Fire epoch end callbacks
        module.onTrainEpochEnd(epoch: state.currentEpoch, metrics: avgMetrics)
        for callback in callbacks {
            callback.onEpochEnd(
                trainer: self,
                module: module,
                epoch: state.currentEpoch,
                metrics: avgMetrics
            )
        }

        delegate?.trainer(self, didCompleteEpoch: state.currentEpoch, metrics: avgMetrics)
    }

    private func computeLossAndGradients<M: Module & TrainModule>(
        module: M,
        batch: [MLXArray]
    ) -> (MLXArray, [String: MLXArray], [String: MLXArray]) {
        // Create the loss function for valueAndGrad
        var capturedMetrics: [String: MLXArray] = [:]

        let lossFunction: ([MLXArray]) -> [MLXArray] = { params in
            // Note: In MLX-Swift, valueAndGrad works on the model parameters
            let (loss, metrics) = module.computeLoss(batch: batch)
            capturedMetrics = metrics
            return [loss]
        }

        // Get trainable parameters
        let trainableParams = module.trainableParameters().flattened()
        var paramArrays: [MLXArray] = []
        var paramKeys: [String] = []

        for (key, value) in trainableParams {
            paramKeys.append(key)
            paramArrays.append(value)
        }

        // Compute loss and gradients manually
        // Since MLX-Swift's valueAndGrad works differently, we compute manually
        let (loss, metrics) = module.computeLoss(batch: batch)
        capturedMetrics = metrics

        // Compute gradients via MLX's grad function
        // For now, use finite differences as a placeholder
        // In production, we'd use MLX's native autodiff
        var gradients: [String: MLXArray] = [:]

        // Use MLX's grad function
        let gradFn = grad { (params: [MLXArray]) -> MLXArray in
            // Update module with params temporarily
            var updates: [(String, MLXArray)] = []
            for (i, key) in paramKeys.enumerated() {
                updates.append((key, params[i]))
            }
            try? module.update(parameters: ModuleParameters.unflattened(updates))

            let (l, _) = module.computeLoss(batch: batch)
            return l
        }

        let grads = gradFn(paramArrays)

        for (i, key) in paramKeys.enumerated() {
            gradients[key] = grads[i]
        }

        return (loss, gradients, capturedMetrics)
    }

    // MARK: - Validation

    private func runValidation<M: Module & TrainModule>(
        module: M,
        valData: [[MLXArray]]
    ) throws -> [String: Float] {
        module.onValidationStart()

        var allMetrics: [String: [Float]] = [:]

        for batch in valData {
            let metrics = module.validationStep(batch: batch)

            // Evaluate and collect
            for (_, value) in metrics {
                eval(value)
            }

            for (key, value) in metrics {
                let floatValue = value.item(Float.self)
                allMetrics[key, default: []].append(floatValue)
            }
        }

        // Average metrics
        var avgMetrics: [String: Float] = [:]
        for (key, values) in allMetrics {
            avgMetrics[key] = values.reduce(0, +) / Float(values.count)
        }

        module.onValidationEnd(metrics: avgMetrics)

        return avgMetrics
    }

    // MARK: - Checkpointing

    /// Save a checkpoint.
    public func saveCheckpoint<M: Module>(
        to path: URL,
        module: M,
        optimizer: any Optimizer
    ) throws {
        try FileManager.default.createDirectory(at: path, withIntermediateDirectories: true)

        // Save model weights
        let weightsPath = path.appendingPathComponent("model.safetensors")
        let params = module.parameters().flattened()
        var paramDict: [String: MLXArray] = [:]
        for (key, value) in params {
            paramDict[key] = value
        }
        try save(arrays: paramDict, url: weightsPath)

        // Save optimizer state
        let optimizerPath = path.appendingPathComponent("optimizer.safetensors")
        let optimizerState = optimizer.stateDict()
        try save(arrays: optimizerState, url: optimizerPath)

        // Save training state
        let statePath = path.appendingPathComponent("trainer_state.json")
        let encoder = JSONEncoder()
        encoder.outputFormatting = .prettyPrinted
        let stateData = try encoder.encode(state)
        try stateData.write(to: statePath)
    }

    /// Load a checkpoint.
    public func loadCheckpoint<M: Module>(
        from path: URL,
        module: M,
        optimizer: any Optimizer
    ) throws {
        // Load model weights
        let weightsPath = path.appendingPathComponent("model.safetensors")
        if FileManager.default.fileExists(atPath: weightsPath.path) {
            let weights = try loadArrays(url: weightsPath)
            try module.update(parameters: ModuleParameters.unflattened(weights))
        }

        // Load optimizer state
        let optimizerPath = path.appendingPathComponent("optimizer.safetensors")
        if FileManager.default.fileExists(atPath: optimizerPath.path) {
            let optimizerState = try loadArrays(url: optimizerPath)
            optimizer.loadStateDict(optimizerState)
        }

        // Load training state
        let statePath = path.appendingPathComponent("trainer_state.json")
        if FileManager.default.fileExists(atPath: statePath.path) {
            let stateData = try Data(contentsOf: statePath)
            state = try JSONDecoder().decode(TrainingState.self, from: stateData)
        }
    }
}
