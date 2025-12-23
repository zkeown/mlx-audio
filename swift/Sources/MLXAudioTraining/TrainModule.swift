// TrainModule.swift
// Protocol for trainable modules in MLX-Audio.
//
// Inspired by PyTorch Lightning's LightningModule, this protocol defines
// the interface for modules that can be trained with the Trainer.

import Foundation
import MLX
import MLXNN

// MARK: - Optimizer Configuration

/// Configuration for optimizer and learning rate schedule.
public struct OptimizerConfig: Sendable {
    /// The optimizer to use for training.
    public let optimizer: any Optimizer

    /// Optional name of the learning rate schedule (for logging).
    public let schedulerName: String?

    /// Creates an optimizer configuration.
    ///
    /// - Parameters:
    ///   - optimizer: The optimizer instance
    ///   - schedulerName: Optional name for the schedule
    public init(optimizer: any Optimizer, schedulerName: String? = nil) {
        self.optimizer = optimizer
        self.schedulerName = schedulerName
    }
}

// MARK: - Train Module Protocol

/// Protocol for modules that can be trained with the Trainer.
///
/// Implementations must provide:
/// - `computeLoss(batch:)`: Core training logic returning loss and metrics
/// - `configureOptimizers()`: Optimizer and schedule configuration
///
/// Example:
/// ```swift
/// class MyClassifier: Module, TrainModule {
///     @ModuleInfo var linear: Linear
///
///     func computeLoss(batch: MLXArray...) -> (MLXArray, [String: MLXArray]) {
///         let (x, y) = (batch[0], batch[1])
///         let logits = linear(x)
///         let loss = crossEntropyLoss(logits: logits, targets: y)
///         let accuracy = mean((argMax(logits, axis: -1) .== y).asType(.float32))
///         return (loss, ["accuracy": accuracy])
///     }
///
///     func configureOptimizers() -> OptimizerConfig {
///         OptimizerConfig(optimizer: AdamW(learningRate: 1e-3))
///     }
/// }
/// ```
public protocol TrainModule: Module {
    /// Compute loss and metrics for a training batch.
    ///
    /// This method is called inside the gradient computation, so it must be
    /// a pure function of model parameters and the batch.
    ///
    /// - Parameter batch: Training batch as array of MLXArrays
    /// - Returns: Tuple of (loss scalar, additional metrics dictionary)
    func computeLoss(batch: [MLXArray]) -> (MLXArray, [String: MLXArray])

    /// Configure the optimizer for training.
    ///
    /// Called once at the start of training. The optimizer can use a learning
    /// rate schedule by passing a schedule to the optimizer's constructor.
    ///
    /// - Returns: Optimizer configuration
    func configureOptimizers() -> OptimizerConfig

    /// Validation step. Called during validation.
    ///
    /// Default implementation calls computeLoss and prefixes metrics with "val_".
    ///
    /// - Parameter batch: Validation batch
    /// - Returns: Dictionary of validation metrics
    func validationStep(batch: [MLXArray]) -> [String: MLXArray]

    /// Test step. Called during testing.
    ///
    /// Default implementation calls computeLoss and prefixes metrics with "test_".
    ///
    /// - Parameter batch: Test batch
    /// - Returns: Dictionary of test metrics
    func testStep(batch: [MLXArray]) -> [String: MLXArray]

    // MARK: - Lifecycle Hooks

    /// Called at the start of training.
    func onTrainStart()

    /// Called at the end of training.
    func onTrainEnd()

    /// Called at the start of each training epoch.
    func onTrainEpochStart(epoch: Int)

    /// Called at the end of each training epoch.
    func onTrainEpochEnd(epoch: Int, metrics: [String: Float])

    /// Called at the start of validation.
    func onValidationStart()

    /// Called at the end of validation.
    func onValidationEnd(metrics: [String: Float])
}

// MARK: - Default Implementations

public extension TrainModule {
    func validationStep(batch: [MLXArray]) -> [String: MLXArray] {
        let (loss, metrics) = computeLoss(batch: batch)
        var result: [String: MLXArray] = ["val_loss": loss]
        for (key, value) in metrics {
            result["val_\(key)"] = value
        }
        return result
    }

    func testStep(batch: [MLXArray]) -> [String: MLXArray] {
        let (loss, metrics) = computeLoss(batch: batch)
        var result: [String: MLXArray] = ["test_loss": loss]
        for (key, value) in metrics {
            result["test_\(key)"] = value
        }
        return result
    }

    func onTrainStart() {}
    func onTrainEnd() {}
    func onTrainEpochStart(epoch: Int) {}
    func onTrainEpochEnd(epoch: Int, metrics: [String: Float]) {}
    func onValidationStart() {}
    func onValidationEnd(metrics: [String: Float]) {}
}

// MARK: - Training State

/// State maintained during training.
public struct TrainingState: Codable, Sendable {
    /// Current epoch (0-indexed).
    public var currentEpoch: Int = 0

    /// Global step count across all epochs.
    public var globalStep: Int = 0

    /// Whether training should stop.
    public var shouldStop: Bool = false

    /// Best metric value seen (for checkpointing).
    public var bestMetricValue: Float?

    /// Name of the best metric being tracked.
    public var bestMetricName: String?

    /// Metrics from the last epoch.
    public var epochMetrics: [String: Float] = [:]

    public init() {}
}

// MARK: - Gradient Clipping

/// Algorithm for gradient clipping.
public enum GradientClipAlgorithm: String, Sendable {
    /// Clip by global L2 norm.
    case norm

    /// Clip by value (element-wise).
    case value
}

/// Clip gradients by norm or value.
///
/// - Parameters:
///   - gradients: Gradient dictionary
///   - maxNorm: Maximum norm or value
///   - algorithm: Clipping algorithm
/// - Returns: Clipped gradients
public func clipGradients(
    _ gradients: [String: MLXArray],
    maxNorm: Float,
    algorithm: GradientClipAlgorithm = .norm
) -> [String: MLXArray] {
    switch algorithm {
    case .norm:
        // Compute global L2 norm
        var totalNormSq = MLXArray(Float(0))
        for (_, g) in gradients {
            totalNormSq = totalNormSq + MLX.sum(g * g)
        }
        let totalNorm = MLX.sqrt(totalNormSq)
        let clipCoef = MLX.minimum(MLXArray(maxNorm) / (totalNorm + 1e-6), MLXArray(Float(1)))

        var clipped: [String: MLXArray] = [:]
        for (key, g) in gradients {
            clipped[key] = g * clipCoef
        }
        return clipped

    case .value:
        var clipped: [String: MLXArray] = [:]
        for (key, g) in gradients {
            clipped[key] = MLX.clip(g, min: -maxNorm, max: maxNorm)
        }
        return clipped
    }
}
