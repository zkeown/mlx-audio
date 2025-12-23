// LossTests.swift
// Tests for loss functions.

import XCTest
import MLX
import MLXNN
@testable import MLXAudioTraining

final class LossTests: XCTestCase {

    // MARK: - Cross Entropy Tests

    func testCrossEntropyLossShape() {
        let logits = MLXRandom.uniform(low: 0.0, high: 1.0, [4, 10])  // 4 samples, 10 classes
        let targets = MLXArray([0, 1, 2, 3])  // Class indices

        let loss = crossEntropyLoss(logits: logits, targets: targets, reduction: MLXAudioTraining.LossReduction.mean)
        eval(loss)

        XCTAssertEqual(loss.ndim, 0)  // Scalar
    }

    func testCrossEntropyLossNoReduction() {
        let logits = MLXRandom.uniform(low: 0.0, high: 1.0, [4, 10])
        let targets = MLXArray([0, 1, 2, 3])

        let loss = crossEntropyLoss(logits: logits, targets: targets, reduction: MLXAudioTraining.LossReduction.none)
        eval(loss)

        XCTAssertEqual(loss.shape, [4])  // Per-sample losses
    }

    func testCrossEntropyLossValue() {
        // Perfect prediction: one-hot logits
        let logits = MLXArray(
            [10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0],
            [3, 3]
        )
        let targets = MLXArray([0, 1, 2])

        let loss = crossEntropyLoss(logits: logits, targets: targets, reduction: MLXAudioTraining.LossReduction.mean)
        eval(loss)

        // Should be close to 0 for perfect predictions
        XCTAssertLessThan(loss.item(Float.self), 0.1)
    }

    func testCrossEntropyWithLabelSmoothing() {
        let logits = MLXRandom.uniform(low: 0.0, high: 1.0, [4, 10])
        let targets = MLXArray([0, 1, 2, 3])

        let lossNoSmooth = crossEntropyLoss(
            logits: logits,
            targets: targets,
            labelSmoothing: 0.0,
            reduction: MLXAudioTraining.LossReduction.mean
        )
        let lossSmooth = crossEntropyLoss(
            logits: logits,
            targets: targets,
            labelSmoothing: 0.1,
            reduction: MLXAudioTraining.LossReduction.mean
        )

        eval(lossNoSmooth, lossSmooth)

        // Smoothed loss should be different (usually higher)
        XCTAssertNotEqual(lossNoSmooth.item(Float.self), lossSmooth.item(Float.self))
    }

    // MARK: - Contrastive Loss Tests

    func testContrastiveLossShape() {
        let audioEmb = MLXRandom.uniform(low: 0.0, high: 1.0, [8, 512])
        let textEmb = MLXRandom.uniform(low: 0.0, high: 1.0, [8, 512])

        let loss = contrastiveLoss(
            audioEmbeddings: audioEmb,
            textEmbeddings: textEmb,
            temperature: 0.07
        )
        eval(loss)

        XCTAssertEqual(loss.ndim, 0)  // Scalar
    }

    func testContrastiveLossMatching() {
        // Create embeddings that match on diagonal
        let emb = MLXRandom.uniform(low: 0.0, high: 1.0, [4, 128])
        let norm = MLX.sqrt(MLX.sum(emb * emb, axis: -1, keepDims: true))
        let normEmb = emb / (norm + 1e-8)

        let loss = contrastiveLoss(
            audioEmbeddings: normEmb,
            textEmbeddings: normEmb,
            temperature: 0.07
        )
        eval(loss)

        // Perfect matching should have low loss
        XCTAssertLessThan(loss.item(Float.self), 1.0)
    }

    // MARK: - SDR Loss Tests

    func testSDRLossShape() {
        let pred = MLXRandom.uniform(low: 0.0, high: 1.0, [2, 4, 2, 1000])  // [B, S, C, T]
        let target = MLXRandom.uniform(low: 0.0, high: 1.0, [2, 4, 2, 1000])

        let loss = sdrLoss(predictions: pred, targets: target, reduction: MLXAudioTraining.LossReduction.mean)
        eval(loss)

        XCTAssertEqual(loss.ndim, 0)  // Scalar
    }

    func testSDRLossPerfectPrediction() {
        let signal = MLXRandom.uniform(low: 0.0, high: 1.0, [2, 4, 2, 1000])

        // Perfect prediction = target itself
        let loss = sdrLoss(predictions: signal, targets: signal, reduction: MLXAudioTraining.LossReduction.mean)
        eval(loss)

        // Loss should be very negative (high SDR = low negative loss)
        XCTAssertLessThan(loss.item(Float.self), -20)  // SDR > 20dB
    }

    func testSDRLossWorseWithNoise() {
        let signal = MLXRandom.uniform(low: 0.0, high: 1.0, [2, 4, 2, 1000])
        let noise = MLXRandom.uniform(low: 0.0, high: 1.0, [2, 4, 2, 1000]) * 0.1
        let noisyPred = signal + noise

        let perfectLoss = sdrLoss(predictions: signal, targets: signal, reduction: MLXAudioTraining.LossReduction.mean)
        let noisyLoss = sdrLoss(predictions: noisyPred, targets: signal, reduction: MLXAudioTraining.LossReduction.mean)

        eval(perfectLoss, noisyLoss)

        // Noisy prediction should have higher (worse) loss
        XCTAssertGreaterThan(noisyLoss.item(Float.self), perfectLoss.item(Float.self))
    }

    // MARK: - SI-SDR Loss Tests

    func testSISDRLossShape() {
        let pred = MLXRandom.uniform(low: 0.0, high: 1.0, [2, 1000])
        let target = MLXRandom.uniform(low: 0.0, high: 1.0, [2, 1000])

        let loss = siSdrLoss(predictions: pred, targets: target, reduction: MLXAudioTraining.LossReduction.mean)
        eval(loss)

        XCTAssertEqual(loss.ndim, 0)
    }

    func testSISDRScaleInvariance() {
        let target = MLXRandom.uniform(low: 0.0, high: 1.0, [2, 1000])
        let pred = target * 0.5  // Scaled version

        let loss = siSdrLoss(predictions: pred, targets: target, reduction: MLXAudioTraining.LossReduction.mean)
        eval(loss)

        // SI-SDR should still be good for scaled prediction
        // (perfect up to scale, so loss should be very negative)
        XCTAssertLessThan(loss.item(Float.self), 0)
    }

    // MARK: - MSE Loss Tests

    func testMSELossShape() {
        let pred = MLXRandom.uniform(low: 0.0, high: 1.0, [4, 100])
        let target = MLXRandom.uniform(low: 0.0, high: 1.0, [4, 100])

        let loss = mseLoss(predictions: pred, targets: target, reduction: MLXAudioTraining.LossReduction.mean)
        eval(loss)

        XCTAssertEqual(loss.ndim, 0)
    }

    func testMSELossZeroForIdentical() {
        let x = MLXRandom.uniform(low: 0.0, high: 1.0, [4, 100])

        let loss = mseLoss(predictions: x, targets: x, reduction: MLXAudioTraining.LossReduction.mean)
        eval(loss)

        XCTAssertLessThan(loss.item(Float.self), 1e-6)
    }

    // MARK: - L1 Loss Tests

    func testL1LossShape() {
        let pred = MLXRandom.uniform(low: 0.0, high: 1.0, [4, 100])
        let target = MLXRandom.uniform(low: 0.0, high: 1.0, [4, 100])

        let loss = l1Loss(predictions: pred, targets: target, reduction: MLXAudioTraining.LossReduction.mean)
        eval(loss)

        XCTAssertEqual(loss.ndim, 0)
    }

    // MARK: - Huber Loss Tests

    func testHuberLossShape() {
        let pred = MLXRandom.uniform(low: 0.0, high: 1.0, [4, 100])
        let target = MLXRandom.uniform(low: 0.0, high: 1.0, [4, 100])

        let loss = huberLoss(predictions: pred, targets: target, delta: 1.0, reduction: MLXAudioTraining.LossReduction.mean)
        eval(loss)

        XCTAssertEqual(loss.ndim, 0)
    }

    func testHuberLossBetweenL1AndL2() {
        let pred = MLXRandom.uniform(low: 0.0, high: 1.0, [4, 100])
        let target = MLXRandom.uniform(low: 0.0, high: 1.0, [4, 100])

        let l1 = l1Loss(predictions: pred, targets: target, reduction: MLXAudioTraining.LossReduction.mean)
        let l2 = mseLoss(predictions: pred, targets: target, reduction: MLXAudioTraining.LossReduction.mean)
        let huber = huberLoss(predictions: pred, targets: target, delta: 1.0, reduction: MLXAudioTraining.LossReduction.mean)

        eval(l1, l2, huber)

        // Huber should be between L1 and L2 in behavior
        // This is a weak test, but checks they're all reasonable
        XCTAssertGreaterThan(huber.item(Float.self), 0)
    }

    // MARK: - Reduction Tests

    func testReductionNone() {
        let loss = MLXRandom.uniform(low: 0.0, high: 1.0, [4, 10])
        let reduced = applyReduction(loss, reduction: MLXAudioTraining.LossReduction.none)

        XCTAssertEqual(reduced.shape, loss.shape)
    }

    func testReductionMean() {
        let loss = MLXArray([1.0, 2.0, 3.0, 4.0], [2, 2])
        let reduced = applyReduction(loss, reduction: MLXAudioTraining.LossReduction.mean)
        eval(reduced)

        XCTAssertEqual(reduced.item(Float.self), 2.5, accuracy: 1e-6)
    }

    func testReductionSum() {
        let loss = MLXArray([1.0, 2.0, 3.0, 4.0], [2, 2])
        let reduced = applyReduction(loss, reduction: MLXAudioTraining.LossReduction.sum)
        eval(reduced)

        XCTAssertEqual(reduced.item(Float.self), 10.0, accuracy: 1e-6)
    }
}
