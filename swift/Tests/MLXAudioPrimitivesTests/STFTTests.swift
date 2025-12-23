// STFTTests.swift
// Tests for STFT and ISTFT operations.

import XCTest
@preconcurrency import MLX
@testable import MLXAudioPrimitives

final class STFTTests: XCTestCase {

    // MARK: - STFT Shape Tests

    func testSTFTOutputShape() throws {
        // 1 second of audio at 22050 Hz
        let signal = MLXRandom.normal([22050])
        let config = STFTConfig(nFFT: 2048, hopLength: 512)

        let result = try stft(signal, config: config)

        // Expected shape: (nFFT/2 + 1, nFrames)
        // nFrames = 1 + (22050 + 2*1024 - 2048) / 512 = 1 + (22050) / 512 ≈ 44
        XCTAssertEqual(result.shape[0], 1025) // nFFT/2 + 1
        XCTAssertGreaterThan(result.shape[1], 40)
    }

    func testSTFTBatchedShape() throws {
        // Batch of 3 signals
        let signals = MLXRandom.normal([3, 22050])
        let config = STFTConfig(nFFT: 2048, hopLength: 512)

        let result = try stft(signals, config: config)

        // Expected shape: (batch, nFFT/2 + 1, nFrames)
        XCTAssertEqual(result.shape[0], 3)
        XCTAssertEqual(result.shape[1], 1025)
    }

    // MARK: - Configuration Tests

    func testSTFTDefaultConfig() {
        let config = STFTConfig()
        XCTAssertEqual(config.nFFT, 2048)
        XCTAssertEqual(config.resolvedHopLength, 512)
        XCTAssertEqual(config.resolvedWinLength, 2048)
        XCTAssertEqual(config.window, .hann)
        XCTAssertTrue(config.center)
        XCTAssertEqual(config.padMode, .constant)
    }

    func testSTFTCustomConfig() {
        let config = STFTConfig(
            nFFT: 1024,
            hopLength: 256,
            winLength: 512,
            window: .hamming,
            center: false,
            padMode: .edge
        )
        XCTAssertEqual(config.nFFT, 1024)
        XCTAssertEqual(config.resolvedHopLength, 256)
        XCTAssertEqual(config.resolvedWinLength, 512)
        XCTAssertEqual(config.window, .hamming)
        XCTAssertFalse(config.center)
        XCTAssertEqual(config.padMode, .edge)
    }

    // MARK: - Validation Tests

    func testSTFTInvalidHopLength() {
        let signal = MLXRandom.normal([1000])
        let config = STFTConfig(nFFT: 2048, hopLength: 0)

        XCTAssertThrowsError(try stft(signal, config: config)) { error in
            if case STFTError.invalidParameter(let message) = error {
                XCTAssertTrue(message.contains("hopLength"))
            } else {
                XCTFail("Expected STFTError.invalidParameter")
            }
        }
    }

    func testSTFTWinLengthTooLarge() {
        let signal = MLXRandom.normal([1000])
        let config = STFTConfig(nFFT: 1024, winLength: 2048)

        XCTAssertThrowsError(try stft(signal, config: config)) { error in
            if case STFTError.invalidParameter(let message) = error {
                XCTAssertTrue(message.contains("winLength"))
            } else {
                XCTFail("Expected STFTError.invalidParameter")
            }
        }
    }

    // MARK: - ComplexArray Tests

    func testComplexArrayMagnitude() {
        let real = MLXArray([3.0, 0.0, 4.0] as [Float])
        let imag = MLXArray([4.0, 5.0, 0.0] as [Float])
        let complex = ComplexArray(real: real, imag: imag)

        let mag = complex.magnitude().asArray(Float.self)

        XCTAssertEqual(mag[0], 5.0, accuracy: 1e-5) // sqrt(9 + 16)
        XCTAssertEqual(mag[1], 5.0, accuracy: 1e-5) // sqrt(0 + 25)
        XCTAssertEqual(mag[2], 4.0, accuracy: 1e-5) // sqrt(16 + 0)
    }

    func testComplexArrayPhase() {
        let real = MLXArray([1.0, 0.0, -1.0] as [Float])
        let imag = MLXArray([0.0, 1.0, 0.0] as [Float])
        let complex = ComplexArray(real: real, imag: imag)

        let phase = complex.phase().asArray(Float.self)

        XCTAssertEqual(phase[0], 0.0, accuracy: 1e-5) // atan2(0, 1)
        XCTAssertEqual(phase[1], Float.pi / 2, accuracy: 1e-5) // atan2(1, 0)
        XCTAssertEqual(phase[2], Float.pi, accuracy: 1e-5) // atan2(0, -1)
    }

    func testComplexArrayConjugate() {
        let real = MLXArray([1.0, 2.0, 3.0] as [Float])
        let imag = MLXArray([4.0, 5.0, 6.0] as [Float])
        let complex = ComplexArray(real: real, imag: imag)

        let conj = complex.conjugate()

        let realValues = conj.real.asArray(Float.self)
        let imagValues = conj.imag.asArray(Float.self)

        XCTAssertEqual(realValues, [1.0, 2.0, 3.0])
        XCTAssertEqual(imagValues, [-4.0, -5.0, -6.0])
    }

    func testComplexArrayMultiplication() {
        // (1 + 2i) * (3 + 4i) = 3 + 4i + 6i + 8i^2 = 3 + 10i - 8 = -5 + 10i
        let a = ComplexArray(real: MLXArray([1.0] as [Float]), imag: MLXArray([2.0] as [Float]))
        let b = ComplexArray(real: MLXArray([3.0] as [Float]), imag: MLXArray([4.0] as [Float]))

        let result = a * b

        let real = result.real.asArray(Float.self)[0]
        let imag = result.imag.asArray(Float.self)[0]

        XCTAssertEqual(real, -5.0, accuracy: 1e-5)
        XCTAssertEqual(imag, 10.0, accuracy: 1e-5)
    }

    // MARK: - Magnitude and Phase Functions

    func testMagnitudeFunction() throws {
        let signal = MLXArray([1.0, 0.0, 1.0, 0.0] as [Float])
        let config = STFTConfig(nFFT: 4, hopLength: 2, center: false)

        let S = try stft(signal, config: config)
        let mag = magnitude(S)

        // Magnitude should be non-negative
        let magValues = mag.asArray(Float.self)
        for value in magValues {
            XCTAssertGreaterThanOrEqual(value, 0.0)
        }
    }

    func testPhaseFunction() throws {
        let signal = MLXRandom.normal([100])
        let config = STFTConfig(nFFT: 32, hopLength: 8, center: false)

        let S = try stft(signal, config: config)
        let ph = phase(S)

        // Phase should be in [-pi, pi]
        let phaseValues = ph.asArray(Float.self)
        for value in phaseValues {
            XCTAssertGreaterThanOrEqual(value, -Float.pi - 0.01)
            XCTAssertLessThanOrEqual(value, Float.pi + 0.01)
        }
    }

    // MARK: - ComplexArray Factory Methods

    func testComplexArrayZeros() {
        let zeros = ComplexArray.zeros([2, 3])
        XCTAssertEqual(zeros.shape, [2, 3])

        let realValues = zeros.real.asArray(Float.self)
        let imagValues = zeros.imag.asArray(Float.self)

        for value in realValues {
            XCTAssertEqual(value, 0.0, accuracy: 1e-6)
        }
        for value in imagValues {
            XCTAssertEqual(value, 0.0, accuracy: 1e-6)
        }
    }

    func testComplexArrayFromPolar() {
        let magnitude = MLXArray([1.0, 2.0] as [Float])
        let phase = MLXArray([0.0, Float.pi / 2] as [Float])

        let complex = ComplexArray.fromPolar(magnitude: magnitude, phase: phase)

        let real = complex.real.asArray(Float.self)
        let imag = complex.imag.asArray(Float.self)

        // (1, 0): mag=1, phase=0 -> real=1, imag=0
        XCTAssertEqual(real[0], 1.0, accuracy: 1e-5)
        XCTAssertEqual(imag[0], 0.0, accuracy: 1e-5)

        // (0, 2): mag=2, phase=pi/2 -> real=0, imag=2
        XCTAssertEqual(real[1], 0.0, accuracy: 1e-5)
        XCTAssertEqual(imag[1], 2.0, accuracy: 1e-5)
    }
}
