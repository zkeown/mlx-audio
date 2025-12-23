// Optimizer.swift
// Base optimizer protocol for MLX-Audio training.
//
// Defines the interface for all optimizers used in training.

import Foundation
import MLX
import MLXNN

// MARK: - Optimizer Protocol

/// Protocol for all optimizers.
///
/// Optimizers update model parameters using gradients computed during
/// the backward pass. They maintain internal state (e.g., momentum, velocity)
/// and support learning rate schedules.
public protocol Optimizer: AnyObject, Sendable {
    /// Current step count.
    var step: Int { get set }

    /// Get the current learning rate.
    var learningRate: Float { get }

    /// Update model parameters with gradients.
    ///
    /// - Parameters:
    ///   - model: The model to update
    ///   - gradients: Gradient dictionary (key: parameter path, value: gradient)
    func update(model: Module, gradients: [String: MLXArray])

    /// Get optimizer state for checkpointing.
    ///
    /// - Returns: Dictionary of state tensors
    func stateDict() -> [String: MLXArray]

    /// Load optimizer state from checkpoint.
    ///
    /// - Parameter state: Dictionary of state tensors
    func loadStateDict(_ state: [String: MLXArray])

    /// Reset optimizer state.
    func reset()
}

// MARK: - Learning Rate Schedule Protocol

/// Protocol for learning rate schedules.
///
/// Schedules compute the learning rate as a function of the current step.
/// They can be passed to optimizers to control learning rate decay.
public protocol LRSchedule: Sendable {
    /// Get the learning rate for a given step.
    ///
    /// - Parameter step: Current training step
    /// - Returns: Learning rate value
    func getValue(step: Int) -> Float
}

// MARK: - Constant Learning Rate

/// Constant learning rate (no decay).
public struct ConstantLR: LRSchedule {
    /// The constant learning rate value.
    public let value: Float

    /// Creates a constant learning rate schedule.
    ///
    /// - Parameter value: The learning rate
    public init(_ value: Float) {
        self.value = value
    }

    public func getValue(step: Int) -> Float {
        value
    }
}

// MARK: - Optimizer Base

/// Base class for optimizers providing common functionality.
open class OptimizerBase: Optimizer, @unchecked Sendable {
    /// Learning rate schedule.
    public let schedule: any LRSchedule

    /// Current step count.
    public var step: Int = 0

    /// Internal state storage.
    internal var state: [String: MLXArray] = [:]

    /// Lock for thread-safe state access.
    private let lock = NSLock()

    /// Current learning rate.
    public var learningRate: Float {
        schedule.getValue(step: step)
    }

    /// Creates an optimizer with a learning rate schedule.
    ///
    /// - Parameter schedule: Learning rate schedule
    public init(schedule: any LRSchedule) {
        self.schedule = schedule
    }

    /// Creates an optimizer with a constant learning rate.
    ///
    /// - Parameter learningRate: Constant learning rate value
    public convenience init(learningRate: Float) {
        self.init(schedule: ConstantLR(learningRate))
    }

    /// Update model parameters. Subclasses must override.
    open func update(model: Module, gradients: [String: MLXArray]) {
        fatalError("Subclasses must implement update(model:gradients:)")
    }

    /// Get optimizer state.
    public func stateDict() -> [String: MLXArray] {
        lock.lock()
        defer { lock.unlock() }

        var result = state
        result["_step"] = MLXArray(Int32(step))
        return result
    }

    /// Load optimizer state.
    public func loadStateDict(_ stateDict: [String: MLXArray]) {
        lock.lock()
        defer { lock.unlock() }

        state = stateDict
        if let stepArray = stateDict["_step"] {
            step = Int(stepArray.item(Int32.self))
            state.removeValue(forKey: "_step")
        }
    }

    /// Reset optimizer state.
    public func reset() {
        lock.lock()
        defer { lock.unlock() }

        state.removeAll()
        step = 0
    }

    /// Thread-safe state access.
    internal func withState<T>(_ body: (inout [String: MLXArray]) -> T) -> T {
        lock.lock()
        defer { lock.unlock() }
        return body(&state)
    }
}
