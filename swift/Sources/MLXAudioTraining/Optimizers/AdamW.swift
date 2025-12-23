// AdamW.swift
// AdamW optimizer with decoupled weight decay.
//
// Based on "Decoupled Weight Decay Regularization" (Loshchilov & Hutter, 2019).

import Foundation
import MLX
import MLXNN

/// AdamW optimizer with decoupled weight decay.
///
/// AdamW is Adam with weight decay applied directly to the parameters,
/// rather than being incorporated into the gradient. This leads to better
/// generalization in many cases.
///
/// Update rule:
/// ```
/// m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
/// v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
/// m_hat = m_t / (1 - beta1^t)
/// v_hat = v_t / (1 - beta2^t)
/// theta_t = theta_{t-1} - lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * theta_{t-1})
/// ```
public final class AdamW: OptimizerBase, @unchecked Sendable {
    /// Exponential decay rate for first moment estimates.
    public let beta1: Float

    /// Exponential decay rate for second moment estimates.
    public let beta2: Float

    /// Small constant for numerical stability.
    public let eps: Float

    /// Weight decay coefficient (L2 penalty).
    public let weightDecay: Float

    /// Creates an AdamW optimizer.
    ///
    /// - Parameters:
    ///   - learningRate: Learning rate (default: 1e-3)
    ///   - beta1: First moment decay rate (default: 0.9)
    ///   - beta2: Second moment decay rate (default: 0.999)
    ///   - eps: Numerical stability constant (default: 1e-8)
    ///   - weightDecay: Weight decay coefficient (default: 0.01)
    public init(
        learningRate: Float = 1e-3,
        beta1: Float = 0.9,
        beta2: Float = 0.999,
        eps: Float = 1e-8,
        weightDecay: Float = 0.01
    ) {
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weightDecay = weightDecay
        super.init(schedule: ConstantLR(learningRate))
    }

    /// Creates an AdamW optimizer with a learning rate schedule.
    ///
    /// - Parameters:
    ///   - schedule: Learning rate schedule
    ///   - beta1: First moment decay rate (default: 0.9)
    ///   - beta2: Second moment decay rate (default: 0.999)
    ///   - eps: Numerical stability constant (default: 1e-8)
    ///   - weightDecay: Weight decay coefficient (default: 0.01)
    public init(
        schedule: any LRSchedule,
        beta1: Float = 0.9,
        beta2: Float = 0.999,
        eps: Float = 1e-8,
        weightDecay: Float = 0.01
    ) {
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weightDecay = weightDecay
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

            // Get or initialize first moment (m)
            let mKey = "m_\(key)"
            let m: MLXArray = withState { state in
                if let existing = state[mKey] {
                    return existing
                } else {
                    let zeros = MLXArray.zeros(like: param)
                    state[mKey] = zeros
                    return zeros
                }
            }

            // Get or initialize second moment (v)
            let vKey = "v_\(key)"
            let v: MLXArray = withState { state in
                if let existing = state[vKey] {
                    return existing
                } else {
                    let zeros = MLXArray.zeros(like: param)
                    state[vKey] = zeros
                    return zeros
                }
            }

            // Update biased first moment estimate
            let newM = beta1 * m + (1 - beta1) * grad

            // Update biased second moment estimate
            let newV = beta2 * v + (1 - beta2) * (grad * grad)

            // Store updated moments
            withState { state in
                state[mKey] = newM
                state[vKey] = newV
            }

            // Bias correction
            let biasCorrection1 = 1 - pow(beta1, Float(step))
            let biasCorrection2 = 1 - pow(beta2, Float(step))

            let mHat = newM / biasCorrection1
            let vHat = newV / biasCorrection2

            // Compute update (AdamW: weight decay applied to params, not gradients)
            let update = lr * mHat / (MLX.sqrt(vHat) + eps)
            var newParam = param - update

            // Apply decoupled weight decay
            if weightDecay > 0 {
                newParam = newParam - lr * weightDecay * param
            }

            updates.append((key, newParam))
        }

        // Apply all updates to model
        if !updates.isEmpty {
            try? model.update(parameters: ModuleParameters.unflattened(updates))
        }
    }
}
