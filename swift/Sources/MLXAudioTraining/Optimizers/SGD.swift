// SGD.swift
// Stochastic Gradient Descent optimizer with momentum.

import Foundation
import MLX
import MLXNN

/// Stochastic Gradient Descent optimizer with optional momentum and Nesterov acceleration.
///
/// Update rule (with momentum):
/// ```
/// v_t = momentum * v_{t-1} + (1 - dampening) * g_t
/// if nesterov:
///     theta_t = theta_{t-1} - lr * (g_t + momentum * v_t)
/// else:
///     theta_t = theta_{t-1} - lr * v_t
/// ```
public final class SGD: OptimizerBase, @unchecked Sendable {
    /// Momentum factor (default: 0.0).
    public let momentum: Float

    /// Dampening for momentum (default: 0.0).
    public let dampening: Float

    /// Weight decay (L2 penalty, default: 0.0).
    public let weightDecay: Float

    /// Whether to use Nesterov momentum (default: false).
    public let nesterov: Bool

    /// Creates an SGD optimizer.
    ///
    /// - Parameters:
    ///   - learningRate: Learning rate
    ///   - momentum: Momentum factor (default: 0.0)
    ///   - dampening: Dampening for momentum (default: 0.0)
    ///   - weightDecay: Weight decay (L2 penalty, default: 0.0)
    ///   - nesterov: Use Nesterov momentum (default: false)
    public init(
        learningRate: Float,
        momentum: Float = 0.0,
        dampening: Float = 0.0,
        weightDecay: Float = 0.0,
        nesterov: Bool = false
    ) {
        precondition(!(nesterov && (momentum <= 0 || dampening != 0)),
                     "Nesterov momentum requires a momentum > 0 and dampening = 0")

        self.momentum = momentum
        self.dampening = dampening
        self.weightDecay = weightDecay
        self.nesterov = nesterov
        super.init(schedule: ConstantLR(learningRate))
    }

    /// Creates an SGD optimizer with a learning rate schedule.
    ///
    /// - Parameters:
    ///   - schedule: Learning rate schedule
    ///   - momentum: Momentum factor (default: 0.0)
    ///   - dampening: Dampening for momentum (default: 0.0)
    ///   - weightDecay: Weight decay (L2 penalty, default: 0.0)
    ///   - nesterov: Use Nesterov momentum (default: false)
    public init(
        schedule: any LRSchedule,
        momentum: Float = 0.0,
        dampening: Float = 0.0,
        weightDecay: Float = 0.0,
        nesterov: Bool = false
    ) {
        precondition(!(nesterov && (momentum <= 0 || dampening != 0)),
                     "Nesterov momentum requires a momentum > 0 and dampening = 0")

        self.momentum = momentum
        self.dampening = dampening
        self.weightDecay = weightDecay
        self.nesterov = nesterov
        super.init(schedule: schedule)
    }

    public override func update(model: Module, gradients: [String: MLXArray]) {
        step += 1
        let lr = learningRate

        // Get trainable parameters
        let params = model.trainableParameters().flattened()
        var paramDict: [String: MLXArray] = [:]
        for (key, value) in params {
            paramDict[key] = value
        }

        // Compute updates
        var updates: [(String, MLXArray)] = []

        for (key, grad) in gradients {
            guard let param = paramDict[key] else { continue }

            var d = grad

            // Apply weight decay (L2 regularization)
            if weightDecay > 0 {
                d = d + weightDecay * param
            }

            // Apply momentum
            if momentum > 0 {
                let vKey = "v_\(key)"
                let v: MLXArray? = withState { state in
                    state[vKey]
                }

                let newV: MLXArray
                if let v = v {
                    newV = momentum * v + (1 - dampening) * d
                } else {
                    newV = d
                }

                // Store velocity
                withState { state in
                    state[vKey] = newV
                }

                if nesterov {
                    d = d + momentum * newV
                } else {
                    d = newV
                }
            }

            // Compute new parameter
            let newParam = param - lr * d
            updates.append((key, newParam))
        }

        // Apply all updates to model
        if !updates.isEmpty {
            try? model.update(parameters: ModuleParameters.unflattened(updates))
        }
    }
}
