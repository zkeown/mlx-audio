// LRSchedule.swift
// Learning rate schedulers for training.

import Foundation

// MARK: - Warmup Cosine Schedule

/// Linear warmup followed by cosine decay.
///
/// This is the most commonly used schedule for transformer training.
/// The learning rate increases linearly during warmup, then decays
/// following a cosine curve.
public struct WarmupCosineSchedule: LRSchedule {
    /// Peak learning rate (reached at end of warmup).
    public let peakLR: Float

    /// Number of warmup steps.
    public let warmupSteps: Int

    /// Total number of training steps.
    public let totalSteps: Int

    /// Minimum learning rate (at end of training).
    public let minLR: Float

    /// Creates a warmup cosine schedule.
    ///
    /// - Parameters:
    ///   - peakLR: Peak learning rate
    ///   - warmupSteps: Number of warmup steps
    ///   - totalSteps: Total training steps
    ///   - minLR: Minimum learning rate (default: 0.0)
    public init(
        peakLR: Float,
        warmupSteps: Int,
        totalSteps: Int,
        minLR: Float = 0.0
    ) {
        self.peakLR = peakLR
        self.warmupSteps = warmupSteps
        self.totalSteps = totalSteps
        self.minLR = minLR
    }

    public func getValue(step: Int) -> Float {
        if step < warmupSteps {
            // Linear warmup
            return peakLR * Float(step) / Float(max(1, warmupSteps))
        } else {
            // Cosine decay
            let decaySteps = totalSteps - warmupSteps
            let progress = Float(step - warmupSteps) / Float(max(1, decaySteps))
            let clampedProgress = min(1.0, progress)
            let cosineDecay = 0.5 * (1 + cos(Float.pi * clampedProgress))
            return max(minLR, minLR + (peakLR - minLR) * cosineDecay)
        }
    }
}

// MARK: - Warmup Linear Schedule

/// Linear warmup followed by linear decay.
public struct WarmupLinearSchedule: LRSchedule {
    /// Peak learning rate.
    public let peakLR: Float

    /// Number of warmup steps.
    public let warmupSteps: Int

    /// Total training steps.
    public let totalSteps: Int

    /// Minimum learning rate.
    public let minLR: Float

    /// Creates a warmup linear schedule.
    public init(
        peakLR: Float,
        warmupSteps: Int,
        totalSteps: Int,
        minLR: Float = 0.0
    ) {
        self.peakLR = peakLR
        self.warmupSteps = warmupSteps
        self.totalSteps = totalSteps
        self.minLR = minLR
    }

    public func getValue(step: Int) -> Float {
        if step < warmupSteps {
            // Linear warmup
            return peakLR * Float(step) / Float(max(1, warmupSteps))
        } else {
            // Linear decay
            let decaySteps = totalSteps - warmupSteps
            let progress = Float(step - warmupSteps) / Float(max(1, decaySteps))
            let clampedProgress = min(1.0, progress)
            return max(minLR, peakLR - (peakLR - minLR) * clampedProgress)
        }
    }
}

// MARK: - Step Learning Rate

/// Step decay: multiply learning rate by gamma every stepSize steps.
public struct StepLR: LRSchedule {
    /// Initial learning rate.
    public let initialLR: Float

    /// Number of steps between each decay.
    public let stepSize: Int

    /// Multiplicative decay factor.
    public let gamma: Float

    /// Creates a step learning rate schedule.
    ///
    /// - Parameters:
    ///   - initialLR: Initial learning rate
    ///   - stepSize: Steps between decays
    ///   - gamma: Decay factor (default: 0.1)
    public init(initialLR: Float, stepSize: Int, gamma: Float = 0.1) {
        self.initialLR = initialLR
        self.stepSize = stepSize
        self.gamma = gamma
    }

    public func getValue(step: Int) -> Float {
        let numDecays = step / stepSize
        return initialLR * pow(gamma, Float(numDecays))
    }
}

// MARK: - Exponential Learning Rate

/// Exponential decay: lr = initialLR * decayRate^step
public struct ExponentialLR: LRSchedule {
    /// Initial learning rate.
    public let initialLR: Float

    /// Per-step decay rate.
    public let decayRate: Float

    /// Creates an exponential learning rate schedule.
    ///
    /// - Parameters:
    ///   - initialLR: Initial learning rate
    ///   - decayRate: Per-step decay rate (e.g., 0.9999)
    public init(initialLR: Float, decayRate: Float) {
        self.initialLR = initialLR
        self.decayRate = decayRate
    }

    public func getValue(step: Int) -> Float {
        initialLR * pow(decayRate, Float(step))
    }
}

// MARK: - Polynomial Learning Rate

/// Polynomial decay schedule.
public struct PolynomialLR: LRSchedule {
    /// Initial learning rate.
    public let initialLR: Float

    /// End learning rate.
    public let endLR: Float

    /// Total training steps.
    public let totalSteps: Int

    /// Polynomial power (default: 1.0 for linear decay).
    public let power: Float

    /// Creates a polynomial learning rate schedule.
    public init(
        initialLR: Float,
        endLR: Float = 0.0,
        totalSteps: Int,
        power: Float = 1.0
    ) {
        self.initialLR = initialLR
        self.endLR = endLR
        self.totalSteps = totalSteps
        self.power = power
    }

    public func getValue(step: Int) -> Float {
        let progress = Float(min(step, totalSteps)) / Float(max(1, totalSteps))
        return (initialLR - endLR) * pow(1 - progress, power) + endLR
    }
}

// MARK: - One Cycle Learning Rate

/// One cycle learning rate policy.
///
/// Follows the one-cycle policy from "Super-Convergence" (Smith & Topin, 2017).
public struct OneCycleLR: LRSchedule {
    /// Maximum learning rate.
    public let maxLR: Float

    /// Total training steps.
    public let totalSteps: Int

    /// Fraction of steps for the increasing phase.
    public let pctStart: Float

    /// Initial learning rate divisor.
    public let divFactor: Float

    /// Final learning rate divisor.
    public let finalDivFactor: Float

    /// Creates a one cycle learning rate schedule.
    public init(
        maxLR: Float,
        totalSteps: Int,
        pctStart: Float = 0.3,
        divFactor: Float = 25.0,
        finalDivFactor: Float = 10000.0
    ) {
        self.maxLR = maxLR
        self.totalSteps = totalSteps
        self.pctStart = pctStart
        self.divFactor = divFactor
        self.finalDivFactor = finalDivFactor
    }

    public func getValue(step: Int) -> Float {
        let initialLR = maxLR / divFactor
        let minLR = maxLR / finalDivFactor
        let stepUp = Int(Float(totalSteps) * pctStart)

        if step < stepUp {
            // Increasing phase
            let progress = Float(step) / Float(max(1, stepUp))
            return initialLR + (maxLR - initialLR) * progress
        } else {
            // Decreasing phase (cosine annealing)
            let stepDown = totalSteps - stepUp
            let progress = Float(step - stepUp) / Float(max(1, stepDown))
            let clampedProgress = min(1.0, progress)
            let cosineDecay = 0.5 * (1 + cos(Float.pi * clampedProgress))
            return minLR + (maxLR - minLR) * cosineDecay
        }
    }
}
