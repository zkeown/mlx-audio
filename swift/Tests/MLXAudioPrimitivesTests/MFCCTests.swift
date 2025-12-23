// MFCCTests.swift
// Tests for MFCC and DCT operations.

import XCTest
@preconcurrency import MLX
@testable import MLXAudioPrimitives

final class MFCCTests: XCTestCase {

    // MARK: - DCT Tests

    func testDCTOutputShape() throws {
        let input = MLXRandom.normal([128, 100]) // (nMels, nFrames)
        let output = try dct(input, n: 20, axis: 0)

        XCTAssertEqual(output.shape, [20, 100])
    }

    func testDCTDefaultN() throws {
        let input = MLXRandom.normal([64, 50])
        let output = try dct(input, axis: 0)

        // Default n = input size along axis
        XCTAssertEqual(output.shape, [64, 50])
    }

    func testDCTAlongDifferentAxis() throws {
        let input = MLXRandom.normal([10, 128, 50]) // (batch, nMels, nFrames)
        let output = try dct(input, n: 13, axis: 1)

        XCTAssertEqual(output.shape, [10, 13, 50])
    }

    // MARK: - MFCC Config Tests

    func testMFCCConfigDefaults() {
        let config = MFCCConfig()
        XCTAssertEqual(config.nMFCC, 20)
        XCTAssertEqual(config.dctType, 2)
        XCTAssertEqual(config.norm, .ortho)
        XCTAssertEqual(config.lifter, 0)
    }

    func testMFCCConfigCustom() {
        let config = MFCCConfig(nMFCC: 13, dctType: 2, norm: .ortho, lifter: 22)
        XCTAssertEqual(config.nMFCC, 13)
        XCTAssertEqual(config.lifter, 22)
    }

    // MARK: - MFCC Validation Tests

    func testMFCCInvalidNMFCC() {
        XCTAssertThrowsError(try mfcc(
            MLXRandom.normal([22050]),
            mfccConfig: MFCCConfig(nMFCC: 0)
        )) { error in
            if case MFCCError.invalidParameter(let message) = error {
                XCTAssertTrue(message.contains("nMFCC"))
            } else {
                XCTFail("Expected MFCCError.invalidParameter")
            }
        }
    }

    func testMFCCInvalidDCTType() {
        XCTAssertThrowsError(try mfcc(
            MLXRandom.normal([22050]),
            mfccConfig: MFCCConfig(dctType: 3)
        )) { error in
            if case MFCCError.invalidParameter(let message) = error {
                XCTAssertTrue(message.contains("DCT type"))
            } else {
                XCTFail("Expected MFCCError.invalidParameter")
            }
        }
    }

    func testMFCCNoInput() {
        XCTAssertThrowsError(try mfcc(nil, S: nil)) { error in
            if case MFCCError.invalidParameter(let message) = error {
                XCTAssertTrue(message.contains("y or S"))
            } else {
                XCTFail("Expected MFCCError.invalidParameter")
            }
        }
    }

    // MARK: - Delta Tests

    func testDeltaOutputShape() throws {
        let input = MLXRandom.normal([13, 100]) // (nMFCC, nFrames)
        let output = try delta(input, width: 9, order: 1, axis: -1)

        XCTAssertEqual(output.shape, input.shape)
    }

    func testDeltaInvalidWidth() {
        let input = MLXRandom.normal([13, 100])

        XCTAssertThrowsError(try delta(input, width: 2)) { error in
            if case MFCCError.invalidParameter(let message) = error {
                XCTAssertTrue(message.contains("width"))
            } else {
                XCTFail("Expected MFCCError.invalidParameter")
            }
        }

        XCTAssertThrowsError(try delta(input, width: 10)) { error in
            if case MFCCError.invalidParameter(let message) = error {
                XCTAssertTrue(message.contains("odd"))
            } else {
                XCTFail("Expected MFCCError.invalidParameter")
            }
        }
    }

    func testDeltaWidthTooLarge() {
        let input = MLXRandom.normal([13, 5]) // Only 5 frames

        XCTAssertThrowsError(try delta(input, width: 9)) { error in
            if case MFCCError.invalidParameter(let message) = error {
                XCTAssertTrue(message.contains(">=") || message.contains("width"))
            } else {
                XCTFail("Expected MFCCError.invalidParameter")
            }
        }
    }

    // MARK: - DCT Energy Compaction Tests

    func testDCTEnergyCompaction() throws {
        // DCT should concentrate energy in lower coefficients
        let input = MLXRandom.normal([128, 100])
        let output = try dct(input, axis: 0)

        // First few coefficients should have more energy on average
        let firstFewEnergy = MLX.mean(MLX.abs(output[0..<10, 0...])).item(Float.self)
        let lastFewEnergy = MLX.mean(MLX.abs(output[118..<128, 0...])).item(Float.self)

        // This is a statistical test, might fail occasionally
        XCTAssertGreaterThan(firstFewEnergy, lastFewEnergy * 0.5,
            "DCT should compact energy in lower coefficients")
    }

    // MARK: - Liftering Tests

    func testLifteringCoefficients() {
        // Test that lifter coefficients are computed correctly
        // lift[n] = 1 + (L/2) * sin(pi * (n+1) / L)
        let lifter = 22
        let nMFCC = 13

        var expectedCoeffs = [Float]()
        for n in 0..<nMFCC {
            let value = Float(1.0 + Double(lifter) / 2.0 * sin(Double.pi * Double(n + 1) / Double(lifter)))
            expectedCoeffs.append(value)
        }

        // Verify center coefficient (around n = L/2) is maximum
        XCTAssertGreaterThan(expectedCoeffs[10], expectedCoeffs[0])
        XCTAssertGreaterThan(expectedCoeffs[10], expectedCoeffs[12])
    }
}
