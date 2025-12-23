// OptimizerTests.swift
// Tests for optimizers.

import XCTest
import MLX
import MLXNN
@testable import MLXAudioTraining

final class OptimizerTests: XCTestCase {

    // MARK: - AdamW Tests

    func testAdamWCreation() {
        let optimizer = AdamW(learningRate: 1e-3)
        XCTAssertEqual(optimizer.learningRate, 1e-3)
        XCTAssertEqual(optimizer.step, 0)
    }

    func testAdamWWithSchedule() {
        let schedule = WarmupCosineSchedule(
            peakLR: 1e-3,
            warmupSteps: 100,
            totalSteps: 1000
        )
        let optimizer = AdamW(schedule: schedule)

        // At step 0, LR should be 0 (start of warmup)
        XCTAssertEqual(optimizer.learningRate, 0, accuracy: 1e-6)

        // At step 50, LR should be half of peak (linear warmup)
        optimizer.step = 50
        XCTAssertEqual(optimizer.learningRate, 0.5e-3, accuracy: 1e-6)

        // At step 100, LR should be peak
        optimizer.step = 100
        XCTAssertEqual(optimizer.learningRate, 1e-3, accuracy: 1e-6)
    }

    func testAdamWStateDict() {
        let optimizer = AdamW(learningRate: 1e-3)
        optimizer.step = 100

        let stateDict = optimizer.stateDict()
        XCTAssertNotNil(stateDict["_step"])

        // Reset and load
        optimizer.reset()
        XCTAssertEqual(optimizer.step, 0)

        optimizer.loadStateDict(stateDict)
        XCTAssertEqual(optimizer.step, 100)
    }

    // MARK: - SGD Tests

    func testSGDCreation() {
        let optimizer = SGD(learningRate: 0.01)
        XCTAssertEqual(optimizer.learningRate, 0.01)
        XCTAssertEqual(optimizer.step, 0)
    }

    func testSGDWithMomentum() {
        let optimizer = SGD(learningRate: 0.01, momentum: 0.9)
        XCTAssertEqual(optimizer.learningRate, 0.01)
    }

    func testSGDNesterovRequiresMomentum() {
        // This should not crash with proper momentum
        _ = SGD(learningRate: 0.01, momentum: 0.9, nesterov: true)
    }

    // MARK: - Learning Rate Schedule Tests

    func testConstantLR() {
        let schedule = ConstantLR(1e-3)
        XCTAssertEqual(schedule.getValue(step: 0), 1e-3)
        XCTAssertEqual(schedule.getValue(step: 100), 1e-3)
        XCTAssertEqual(schedule.getValue(step: 10000), 1e-3)
    }

    func testWarmupCosineSchedule() {
        let schedule = WarmupCosineSchedule(
            peakLR: 1.0,
            warmupSteps: 100,
            totalSteps: 1000,
            minLR: 0.0
        )

        // Start at 0
        XCTAssertEqual(schedule.getValue(step: 0), 0.0, accuracy: 1e-6)

        // Linear warmup
        XCTAssertEqual(schedule.getValue(step: 50), 0.5, accuracy: 1e-6)

        // Peak at end of warmup
        XCTAssertEqual(schedule.getValue(step: 100), 1.0, accuracy: 1e-6)

        // End near minLR
        XCTAssertEqual(schedule.getValue(step: 1000), 0.0, accuracy: 1e-5)
    }

    func testStepLR() {
        let schedule = StepLR(initialLR: 1.0, stepSize: 100, gamma: 0.1)

        XCTAssertEqual(schedule.getValue(step: 0), 1.0)
        XCTAssertEqual(schedule.getValue(step: 99), 1.0)
        XCTAssertEqual(schedule.getValue(step: 100), 0.1, accuracy: 1e-6)
        XCTAssertEqual(schedule.getValue(step: 200), 0.01, accuracy: 1e-6)
    }

    func testExponentialLR() {
        let schedule = ExponentialLR(initialLR: 1.0, decayRate: 0.99)

        XCTAssertEqual(schedule.getValue(step: 0), 1.0)
        XCTAssertEqual(schedule.getValue(step: 1), 0.99, accuracy: 1e-6)
        XCTAssertEqual(schedule.getValue(step: 100), pow(0.99, 100), accuracy: 1e-6)
    }
}
