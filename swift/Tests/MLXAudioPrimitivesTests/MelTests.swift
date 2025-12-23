// MelTests.swift
// Tests for mel filterbank and spectrogram.

import XCTest
@preconcurrency import MLX
@testable import MLXAudioPrimitives

final class MelTests: XCTestCase {

    // MARK: - Mel Scale Conversion Tests

    func testHzToMelSlaney() {
        // Test Slaney formula (librosa default)
        let hz: [Double] = [0, 500, 1000, 2000, 4000]
        let mels = hzToMel(hz, htk: false)

        // Below 1000 Hz: linear
        XCTAssertEqual(mels[0], 0.0, accuracy: 1e-5)
        XCTAssertEqual(mels[1], 500.0 / SlaneyConstants.fSp, accuracy: 1e-5)
        XCTAssertEqual(mels[2], 1000.0 / SlaneyConstants.fSp, accuracy: 1e-5)

        // Above 1000 Hz: logarithmic
        XCTAssertGreaterThan(mels[3], mels[2])
        XCTAssertGreaterThan(mels[4], mels[3])
    }

    func testHzToMelHTK() {
        // Test HTK formula
        let hz: [Double] = [0, 700, 1000, 2000]
        let mels = hzToMel(hz, htk: true)

        // HTK: mel = 2595 * log10(1 + hz/700)
        XCTAssertEqual(mels[0], 0.0, accuracy: 1e-5)
        XCTAssertEqual(mels[1], 2595.0 * log10(2.0), accuracy: 1e-3) // log10(1 + 700/700) = log10(2)
    }

    func testMelToHzRoundtrip() {
        let originalHz: [Double] = [100, 500, 1000, 2000, 5000, 8000]

        // Slaney roundtrip
        let melsSlaney = hzToMel(originalHz, htk: false)
        let recoveredSlaney = melToHz(melsSlaney, htk: false)

        for (original, recovered) in zip(originalHz, recoveredSlaney) {
            XCTAssertEqual(original, recovered, accuracy: 1e-5,
                "Slaney roundtrip failed for \(original) Hz")
        }

        // HTK roundtrip
        let melsHTK = hzToMel(originalHz, htk: true)
        let recoveredHTK = melToHz(melsHTK, htk: true)

        for (original, recovered) in zip(originalHz, recoveredHTK) {
            XCTAssertEqual(original, recovered, accuracy: 1e-5,
                "HTK roundtrip failed for \(original) Hz")
        }
    }

    // MARK: - Mel Filterbank Shape Tests

    func testMelFilterbankShape() throws {
        let fb = try melFilterbank(nFFT: 2048, config: MelConfig(sampleRate: 22050, nMels: 128))

        // Expected shape: (nMels, nFFT/2 + 1)
        XCTAssertEqual(fb.shape, [128, 1025])
    }

    func testMelFilterbankDifferentSizes() throws {
        let fb80 = try melFilterbank(nFFT: 2048, config: MelConfig(nMels: 80))
        let fb64 = try melFilterbank(nFFT: 2048, config: MelConfig(nMels: 64))

        XCTAssertEqual(fb80.shape, [80, 1025])
        XCTAssertEqual(fb64.shape, [64, 1025])
    }

    // MARK: - Mel Filterbank Value Tests

    func testMelFilterbankNonNegative() throws {
        let fb = try melFilterbank(nFFT: 2048, config: MelConfig(nMels: 128))
        let values = fb.asArray(Float.self)

        for (i, value) in values.enumerated() {
            XCTAssertGreaterThanOrEqual(value, 0.0, "Negative value at index \(i): \(value)")
        }
    }

    func testMelFilterbankRowSum() throws {
        // With Slaney normalization, filters should be normalized by bandwidth
        let fb = try melFilterbank(nFFT: 2048, config: MelConfig(nMels: 128, norm: .slaney))

        // Each row should have some non-zero values (triangular filter)
        for mel in 0..<128 {
            let row = fb[mel]
            let rowSum = MLX.sum(row).item(Float.self)
            XCTAssertGreaterThan(rowSum, 0.0, "Mel band \(mel) has zero sum")
        }
    }

    // MARK: - Mel Config Tests

    func testMelConfigDefaults() {
        let config = MelConfig()
        XCTAssertEqual(config.sampleRate, 22050)
        XCTAssertEqual(config.nMels, 128)
        XCTAssertEqual(config.fMin, 0.0)
        XCTAssertNil(config.fMax)
        XCTAssertFalse(config.htk)
        XCTAssertEqual(config.norm, .slaney)
    }

    func testMelConfigResolvedFMax() {
        var config = MelConfig(sampleRate: 16000)
        XCTAssertEqual(config.resolvedFMax, 8000.0)

        config.fMax = 4000.0
        XCTAssertEqual(config.resolvedFMax, 4000.0)
    }

    // MARK: - Mel Filterbank Validation Tests

    func testMelFilterbankInvalidNMels() {
        XCTAssertThrowsError(try melFilterbank(nFFT: 2048, config: MelConfig(nMels: 0))) { error in
            if case MelError.invalidParameter(let message) = error {
                XCTAssertTrue(message.contains("nMels"))
            } else {
                XCTFail("Expected MelError.invalidParameter")
            }
        }
    }

    func testMelFilterbankInvalidFMin() {
        XCTAssertThrowsError(try melFilterbank(nFFT: 2048, config: MelConfig(fMin: -100))) { error in
            if case MelError.invalidParameter(let message) = error {
                XCTAssertTrue(message.contains("fMin"))
            } else {
                XCTFail("Expected MelError.invalidParameter")
            }
        }
    }

    func testMelFilterbankFMaxExceedsNyquist() {
        XCTAssertThrowsError(try melFilterbank(nFFT: 2048, config: MelConfig(sampleRate: 16000, fMax: 10000))) { error in
            if case MelError.invalidParameter(let message) = error {
                XCTAssertTrue(message.contains("fMax") || message.contains("Nyquist"))
            } else {
                XCTFail("Expected MelError.invalidParameter")
            }
        }
    }

    // MARK: - Decibel Conversion Tests

    func testPowerToDb() {
        let power = MLXArray([1.0, 10.0, 100.0] as [Float])
        let db = powerToDb(power, ref: 1.0, amin: 1e-10, topDb: 80.0)

        let values = db.asArray(Float.self)

        // 10 * log10(1) = 0
        XCTAssertEqual(values[0], 0.0, accuracy: 1e-5)
        // 10 * log10(10) = 10
        XCTAssertEqual(values[1], 10.0, accuracy: 1e-5)
        // 10 * log10(100) = 20
        XCTAssertEqual(values[2], 20.0, accuracy: 1e-5)
    }

    func testAmplitudeToDb() {
        let amplitude = MLXArray([1.0, 10.0, 100.0] as [Float])
        let db = amplitudeToDb(amplitude, ref: 1.0, amin: 1e-5, topDb: 80.0)

        let values = db.asArray(Float.self)

        // 20 * log10(1) = 0
        XCTAssertEqual(values[0], 0.0, accuracy: 1e-5)
        // 20 * log10(10) = 20
        XCTAssertEqual(values[1], 20.0, accuracy: 1e-5)
        // 20 * log10(100) = 40
        XCTAssertEqual(values[2], 40.0, accuracy: 1e-5)
    }

    func testPowerToDbClipping() {
        let power = MLXArray([1e-20, 1.0, 100.0] as [Float])
        let db = powerToDb(power, ref: 1.0, amin: 1e-10, topDb: 80.0)

        let values = db.asArray(Float.self)

        // Very small power should be clipped to topDb below max
        // Max is 20 dB (from 100), so min should be 20 - 80 = -60 dB
        XCTAssertGreaterThan(values[0], -80.0)
    }

    // MARK: - Slaney Constants Tests

    func testSlaneyConstantsValues() {
        XCTAssertEqual(SlaneyConstants.fMin, 0.0)
        XCTAssertEqual(SlaneyConstants.fSp, 200.0 / 3.0, accuracy: 1e-10)
        XCTAssertEqual(SlaneyConstants.minLogHz, 1000.0)

        // logstep = log(6.4) / 27
        let expectedLogstep = log(6.4) / 27.0
        XCTAssertEqual(SlaneyConstants.logstep, expectedLogstep, accuracy: 1e-10)
    }

    // MARK: - HTK Constants Tests

    func testHTKConstantsValues() {
        XCTAssertEqual(HTKConstants.melFactor, 2595.0)
        XCTAssertEqual(HTKConstants.melBase, 700.0)
    }
}
