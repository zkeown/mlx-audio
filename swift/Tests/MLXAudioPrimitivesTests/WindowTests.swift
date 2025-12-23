// WindowTests.swift
// Tests for window functions.

import XCTest
@preconcurrency import MLX
@testable import MLXAudioPrimitives

final class WindowTests: XCTestCase {

    // MARK: - Basic Window Tests

    func testHannWindowShape() {
        let window = getWindow(.hann, length: 2048)
        XCTAssertEqual(window.shape, [2048])
    }

    func testHammingWindowShape() {
        let window = getWindow(.hamming, length: 1024)
        XCTAssertEqual(window.shape, [1024])
    }

    func testBlackmanWindowShape() {
        let window = getWindow(.blackman, length: 512)
        XCTAssertEqual(window.shape, [512])
    }

    func testBartlettWindowShape() {
        let window = getWindow(.bartlett, length: 256)
        XCTAssertEqual(window.shape, [256])
    }

    func testRectangularWindowShape() {
        let window = getWindow(.rectangular, length: 128)
        XCTAssertEqual(window.shape, [128])
    }

    // MARK: - Window Value Tests

    func testHannWindowValues() {
        let window = getWindow(.hann, length: 5, periodic: false)
        let values = window.asArray(Float.self)

        // Hann window: endpoints should be 0, center should be 1
        XCTAssertEqual(values[0], 0.0, accuracy: 1e-6)
        XCTAssertEqual(values[4], 0.0, accuracy: 1e-6)
        XCTAssertEqual(values[2], 1.0, accuracy: 1e-6)
    }

    func testRectangularWindowValues() {
        let window = getWindow(.rectangular, length: 10)
        let values = window.asArray(Float.self)

        // Rectangular window: all ones
        for value in values {
            XCTAssertEqual(value, 1.0, accuracy: 1e-6)
        }
    }

    func testBartlettWindowValues() {
        let window = getWindow(.bartlett, length: 5, periodic: false)
        let values = window.asArray(Float.self)

        // Bartlett window: triangular shape, endpoints = 0, center = 1
        XCTAssertEqual(values[0], 0.0, accuracy: 1e-6)
        XCTAssertEqual(values[2], 1.0, accuracy: 1e-6)
        XCTAssertEqual(values[4], 0.0, accuracy: 1e-6)
    }

    // MARK: - Window Symmetry Tests

    func testHannWindowSymmetry() {
        let window = getWindow(.hann, length: 101, periodic: false)
        let values = window.asArray(Float.self)

        // Symmetric window: w[i] should equal w[n-1-i]
        for i in 0..<50 {
            XCTAssertEqual(values[i], values[100 - i], accuracy: 1e-5,
                "Window not symmetric at index \(i)")
        }
    }

    func testHammingWindowSymmetry() {
        let window = getWindow(.hamming, length: 100, periodic: false)
        let values = window.asArray(Float.self)

        for i in 0..<50 {
            XCTAssertEqual(values[i], values[99 - i], accuracy: 1e-5,
                "Window not symmetric at index \(i)")
        }
    }

    // MARK: - Periodic vs Symmetric Tests

    func testPeriodicWindowLength() {
        // Periodic window should have the requested length
        let periodicWindow = getWindow(.hann, length: 100, periodic: true)
        XCTAssertEqual(periodicWindow.shape[0], 100)

        // Symmetric window should also have the requested length
        let symmetricWindow = getWindow(.hann, length: 100, periodic: false)
        XCTAssertEqual(symmetricWindow.shape[0], 100)
    }

    // MARK: - Window Aliases

    func testWindowAliases() {
        // hanning should be the same as hann
        let hann = getWindow(.hann, length: 100)
        let hanning = getWindow(.hanning, length: 100)

        let hannValues = hann.asArray(Float.self)
        let hanningValues = hanning.asArray(Float.self)

        for i in 0..<100 {
            XCTAssertEqual(hannValues[i], hanningValues[i], accuracy: 1e-6)
        }
    }

    // MARK: - Edge Cases

    func testWindowLengthOne() {
        let window = getWindow(.hann, length: 1)
        XCTAssertEqual(window.shape, [1])
        XCTAssertEqual(window.asArray(Float.self)[0], 1.0, accuracy: 1e-6)
    }

    func testWindowLengthTwo() {
        let window = getWindow(.hann, length: 2, periodic: false)
        XCTAssertEqual(window.shape, [2])
        // For length 2 symmetric Hann: both endpoints are 0
        let values = window.asArray(Float.self)
        XCTAssertEqual(values[0], 0.0, accuracy: 1e-6)
        XCTAssertEqual(values[1], 0.0, accuracy: 1e-6)
    }

    // MARK: - Non-Negative Tests

    func testBlackmanWindowNonNegative() {
        let window = getWindow(.blackman, length: 1000)
        let values = window.asArray(Float.self)

        for (i, value) in values.enumerated() {
            XCTAssertGreaterThanOrEqual(value, 0.0, "Negative value at index \(i): \(value)")
        }
    }

    // MARK: - NOLA Constraint Tests

    func testNOLAConstraintHann() {
        // Hann window with 50% overlap should satisfy NOLA
        let isValid = checkNOLA(.hann, hopLength: 512, nFFT: 1024)
        XCTAssertTrue(isValid, "Hann window with 50% overlap should satisfy NOLA")
    }

    func testNOLAConstraintHann75Overlap() {
        // Hann window with 75% overlap should satisfy NOLA
        let isValid = checkNOLA(.hann, hopLength: 256, nFFT: 1024)
        XCTAssertTrue(isValid, "Hann window with 75% overlap should satisfy NOLA")
    }

    // MARK: - Caching Tests

    func testWindowCaching() {
        // First call should compute and cache
        let window1 = getWindow(.hann, length: 2048)

        // Second call should return cached result
        let window2 = getWindow(.hann, length: 2048)

        // Both should have the same values
        let values1 = window1.asArray(Float.self)
        let values2 = window2.asArray(Float.self)

        for i in 0..<values1.count {
            XCTAssertEqual(values1[i], values2[i], accuracy: 1e-6)
        }
    }
}
