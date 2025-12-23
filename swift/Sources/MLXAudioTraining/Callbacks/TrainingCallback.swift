// TrainingCallback.swift
// Callback system for training hooks.

import Foundation
import MLX
import MLXNN

// MARK: - Callback Priority

/// Priority levels for callback execution.
public enum CallbackPriority: Int, Comparable {
    /// System-critical (gradient clipping).
    case highest = 0

    /// Monitoring/logging.
    case high = 25

    /// User callbacks (default).
    case normal = 50

    /// Post-processing.
    case low = 75

    /// Cleanup (checkpointing).
    case lowest = 100

    public static func < (lhs: CallbackPriority, rhs: CallbackPriority) -> Bool {
        lhs.rawValue < rhs.rawValue
    }
}

// MARK: - Training Callback Protocol

/// Protocol for training callbacks.
///
/// Callbacks provide hooks into the training loop at various points.
/// They can be used for logging, checkpointing, early stopping, etc.
public protocol TrainingCallback: AnyObject {
    /// Callback priority (lower = executes first).
    var priority: CallbackPriority { get }

    /// Called at the start of training.
    func onFitStart<M: Module>(trainer: Trainer, module: M)

    /// Called at the end of training.
    func onFitEnd<M: Module>(trainer: Trainer, module: M)

    /// Called at the start of each epoch.
    func onEpochStart<M: Module>(trainer: Trainer, module: M, epoch: Int)

    /// Called at the end of each epoch.
    func onEpochEnd<M: Module>(trainer: Trainer, module: M, epoch: Int, metrics: [String: Float])

    /// Called at the end of each training batch.
    func onBatchEnd<M: Module>(trainer: Trainer, module: M, batch: Int, metrics: [String: Float])

    /// Called when an exception occurs.
    func onException<M: Module>(trainer: Trainer, module: M, error: Error)

    /// Get callback state for checkpointing.
    func stateDict() -> [String: Any]

    /// Load callback state from checkpoint.
    func loadStateDict(_ state: [String: Any])
}

// Default implementations
public extension TrainingCallback {
    var priority: CallbackPriority { .normal }

    func onFitStart<M: Module>(trainer: Trainer, module: M) {}
    func onFitEnd<M: Module>(trainer: Trainer, module: M) {}
    func onEpochStart<M: Module>(trainer: Trainer, module: M, epoch: Int) {}
    func onEpochEnd<M: Module>(trainer: Trainer, module: M, epoch: Int, metrics: [String: Float]) {}
    func onBatchEnd<M: Module>(trainer: Trainer, module: M, batch: Int, metrics: [String: Float]) {}
    func onException<M: Module>(trainer: Trainer, module: M, error: Error) {}
    func stateDict() -> [String: Any] { [:] }
    func loadStateDict(_ state: [String: Any]) {}
}

// MARK: - Early Stopping

/// Early stopping callback.
///
/// Stops training when a monitored metric stops improving.
public class EarlyStopping: TrainingCallback {
    /// Metric to monitor.
    public let monitor: String

    /// Number of epochs with no improvement before stopping.
    public let patience: Int

    /// Minimum change to qualify as improvement.
    public let minDelta: Float

    /// Whether lower is better.
    public let mode: Mode

    /// Whether to print messages.
    public let verbose: Bool

    /// Best value seen.
    private var bestValue: Float?

    /// Epochs since last improvement.
    private var waitCount: Int = 0

    /// Whether stopping was triggered.
    public private(set) var stopped: Bool = false

    public enum Mode {
        case min
        case max
    }

    public init(
        monitor: String = "val_loss",
        patience: Int = 3,
        minDelta: Float = 0.0,
        mode: Mode = .min,
        verbose: Bool = true
    ) {
        self.monitor = monitor
        self.patience = patience
        self.minDelta = minDelta
        self.mode = mode
        self.verbose = verbose
    }

    public var priority: CallbackPriority { .normal }

    public func onEpochEnd<M: Module>(
        trainer: Trainer,
        module: M,
        epoch: Int,
        metrics: [String: Float]
    ) {
        guard let value = metrics[monitor] else {
            if verbose {
                print("EarlyStopping: metric '\(monitor)' not found")
            }
            return
        }

        let improved: Bool
        if let best = bestValue {
            switch mode {
            case .min:
                improved = value < best - minDelta
            case .max:
                improved = value > best + minDelta
            }
        } else {
            improved = true
        }

        if improved {
            bestValue = value
            waitCount = 0
            if verbose {
                print("EarlyStopping: \(monitor) improved to \(value)")
            }
        } else {
            waitCount += 1
            if verbose {
                print("EarlyStopping: no improvement, wait count: \(waitCount)/\(patience)")
            }

            if waitCount >= patience {
                stopped = true
                trainer.stop()
                if verbose {
                    print("EarlyStopping: stopping training")
                }
            }
        }
    }

    public func stateDict() -> [String: Any] {
        [
            "bestValue": bestValue as Any,
            "waitCount": waitCount,
            "stopped": stopped
        ]
    }

    public func loadStateDict(_ state: [String: Any]) {
        bestValue = state["bestValue"] as? Float
        waitCount = state["waitCount"] as? Int ?? 0
        stopped = state["stopped"] as? Bool ?? false
    }
}

// MARK: - Model Checkpoint

/// Model checkpoint callback.
///
/// Saves model checkpoints during training.
public class ModelCheckpoint: TrainingCallback {
    /// Directory for checkpoints.
    public let directory: URL

    /// Metric to monitor for best checkpoint.
    public let monitor: String

    /// Whether lower is better.
    public let mode: EarlyStopping.Mode

    /// Number of best checkpoints to keep.
    public let saveTopK: Int

    /// Whether to save on every epoch.
    public let saveEveryEpoch: Bool

    /// Whether to save on exception.
    public let saveOnException: Bool

    /// Whether to print messages.
    public let verbose: Bool

    /// Tracked checkpoints: (path, metric value).
    private var checkpoints: [(URL, Float)] = []

    /// Best checkpoint path.
    public private(set) var bestCheckpointPath: URL?

    public init(
        directory: URL,
        monitor: String = "val_loss",
        mode: EarlyStopping.Mode = .min,
        saveTopK: Int = 3,
        saveEveryEpoch: Bool = false,
        saveOnException: Bool = true,
        verbose: Bool = true
    ) {
        self.directory = directory
        self.monitor = monitor
        self.mode = mode
        self.saveTopK = saveTopK
        self.saveEveryEpoch = saveEveryEpoch
        self.saveOnException = saveOnException
        self.verbose = verbose
    }

    public var priority: CallbackPriority { .lowest }

    public func onFitStart<M: Module>(trainer: Trainer, module: M) {
        // Create directory
        try? FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true
        )
    }

    public func onEpochEnd<M: Module>(
        trainer: Trainer,
        module: M,
        epoch: Int,
        metrics: [String: Float]
    ) {
        guard let value = metrics[monitor] else {
            if saveEveryEpoch {
                saveCheckpoint(trainer: trainer, module: module, epoch: epoch, value: nil)
            }
            return
        }

        // Check if this is a top-k checkpoint
        let shouldSave = shouldSaveCheckpoint(value: value)

        if shouldSave || saveEveryEpoch {
            saveCheckpoint(trainer: trainer, module: module, epoch: epoch, value: value)
        }
    }

    public func onException<M: Module>(trainer: Trainer, module: M, error: Error) {
        if saveOnException {
            let path = directory.appendingPathComponent("emergency_checkpoint")
            do {
                try trainer.saveCheckpoint(
                    to: path,
                    module: module,
                    optimizer: AdamW(learningRate: 0)  // Placeholder
                )
                if verbose {
                    print("ModelCheckpoint: saved emergency checkpoint")
                }
            } catch {
                print("ModelCheckpoint: failed to save emergency checkpoint: \(error)")
            }
        }
    }

    private func shouldSaveCheckpoint(value: Float) -> Bool {
        if checkpoints.count < saveTopK {
            return true
        }

        // Check if value is better than worst in top-k
        let worst = checkpoints.last!.1
        switch mode {
        case .min:
            return value < worst
        case .max:
            return value > worst
        }
    }

    private func saveCheckpoint<M: Module>(
        trainer: Trainer,
        module: M,
        epoch: Int,
        value: Float?
    ) {
        let name = "epoch_\(epoch)"
        let path = directory.appendingPathComponent(name)

        do {
            try trainer.saveCheckpoint(
                to: path,
                module: module,
                optimizer: AdamW(learningRate: 0)  // Would need actual optimizer
            )

            if let value = value {
                checkpoints.append((path, value))

                // Sort by value
                checkpoints.sort { lhs, rhs in
                    switch mode {
                    case .min: return lhs.1 < rhs.1
                    case .max: return lhs.1 > rhs.1
                    }
                }

                // Update best
                bestCheckpointPath = checkpoints.first?.0

                // Remove excess checkpoints
                while checkpoints.count > saveTopK {
                    let removed = checkpoints.removeLast()
                    try? FileManager.default.removeItem(at: removed.0)
                }
            }

            if verbose {
                print("ModelCheckpoint: saved checkpoint to \(path.lastPathComponent)")
            }
        } catch {
            print("ModelCheckpoint: failed to save: \(error)")
        }
    }

    public func stateDict() -> [String: Any] {
        [
            "checkpoints": checkpoints.map { ($0.0.path, $0.1) },
            "bestPath": bestCheckpointPath?.path as Any
        ]
    }

    public func loadStateDict(_ state: [String: Any]) {
        if let paths = state["checkpoints"] as? [(String, Float)] {
            checkpoints = paths.map { (URL(fileURLWithPath: $0.0), $0.1) }
        }
        if let bestPath = state["bestPath"] as? String {
            bestCheckpointPath = URL(fileURLWithPath: bestPath)
        }
    }
}

// MARK: - Learning Rate Monitor

/// Learning rate monitor callback.
///
/// Logs the current learning rate during training.
public class LearningRateMonitor: TrainingCallback {
    /// Learning rate history.
    public private(set) var history: [Float] = []

    public var priority: CallbackPriority { .high }

    public func onBatchEnd<M: Module>(
        trainer: Trainer,
        module: M,
        batch: Int,
        metrics: [String: Float]
    ) {
        // Get LR from optimizer if available
        // This would need access to the optimizer
        if let lr = metrics["learning_rate"] {
            history.append(lr)
        }
    }
}

// MARK: - Progress Callback

/// Progress callback for UI updates.
public class ProgressCallback: TrainingCallback {
    /// Progress handler.
    public var onProgress: ((Float, [String: Float]) -> Void)?

    /// Total expected steps.
    public var totalSteps: Int?

    public var priority: CallbackPriority { .high }

    public init(
        totalSteps: Int? = nil,
        onProgress: ((Float, [String: Float]) -> Void)? = nil
    ) {
        self.totalSteps = totalSteps
        self.onProgress = onProgress
    }

    public func onBatchEnd<M: Module>(
        trainer: Trainer,
        module: M,
        batch: Int,
        metrics: [String: Float]
    ) {
        let progress: Float
        if let total = totalSteps, total > 0 {
            progress = Float(trainer.state.globalStep) / Float(total)
        } else {
            progress = Float(batch) / 100.0  // Approximate
        }

        onProgress?(min(1.0, progress), metrics)
    }
}
