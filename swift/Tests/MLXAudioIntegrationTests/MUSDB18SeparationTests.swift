// MUSDB18SeparationTests.swift
// Integration tests for HTDemucs source separation on MUSDB18-HQ fixtures.
//
// Run with:
//   INTEGRATION_FIXTURES=/path/to/fixtures swift test --filter MUSDB18SeparationTests
//
// Generate fixtures first:
//   python python/tests/integration/generate_musdb18_fixtures.py \
//     --musdb18-root /path/to/MUSDB18HQ \
//     --output-dir swift/Tests/Fixtures/Integration/musdb18

import XCTest
@testable import MLXAudioModels
@testable import MLXAudioPrimitives
import MLX
import MLXNN

/// Integration tests for HTDemucs on MUSDB18-HQ fixtures.
final class MUSDB18SeparationTests: XCTestCase {

    // MARK: - Quality Targets

    /// SDR targets based on published HTDemucs results.
    static let sdrTargets: [String: Float] = [
        "drums": 8.5,
        "bass": 7.0,
        "other": 4.5,
        "vocals": 8.0,
    ]

    /// Minimum acceptable SDR (below this indicates a problem).
    static let sdrMinimum: [String: Float] = [
        "drums": 5.0,
        "bass": 4.0,
        "other": 2.0,
        "vocals": 5.0,
    ]

    /// Tolerance for Python parity (max absolute difference in SDR).
    static let parityToleranceSDR: Float = 0.5  // dB

    // MARK: - Properties

    var fixtureLoader: IntegrationFixtureLoader?
    var manifest: IntegrationManifest?
    var model: HTDemucs?

    // MARK: - Setup

    override func setUpWithError() throws {
        try super.setUpWithError()

        // Skip if fixtures not available
        guard ProcessInfo.processInfo.environment[integrationFixturesEnvVar] != nil else {
            throw XCTSkip(
                "Integration fixtures not available. Set \(integrationFixturesEnvVar) environment variable."
            )
        }

        fixtureLoader = try IntegrationFixtureLoader()
        manifest = try fixtureLoader?.loadManifest()
    }

    /// Load HTDemucs model (lazy, shared across tests).
    func loadModel() async throws -> HTDemucs {
        if let model = self.model {
            return model
        }

        // Load htdemucs_ft model
        let model = try await HTDemucs.fromPretrained(modelId: "htdemucs_ft")
        self.model = model
        return model
    }

    // MARK: - Quick Smoke Tests

    /// Verify model can separate a fixture track without errors.
    func testSingleTrackSeparates() async throws {
        guard let loader = fixtureLoader else {
            throw XCTSkip("Fixture loader not available")
        }

        let model = try await loadModel()
        let fixture = try loader.loadTrack(index: 0)

        print("Testing track: \(fixture.metadata.trackName)")
        print("  Duration: \(fixture.metadata.durationSeconds)s")
        print("  Samples: \(fixture.metadata.samples)")

        // Run separation
        let separated = try separateWithHTDemucs(model, audio: fixture.mixture)

        // Verify output shapes
        XCTAssertEqual(separated.drums.ndim, 2, "Drums should be 2D [channels, samples]")
        XCTAssertEqual(separated.bass.ndim, 2, "Bass should be 2D [channels, samples]")
        XCTAssertEqual(separated.other.ndim, 2, "Other should be 2D [channels, samples]")
        XCTAssertEqual(separated.vocals.ndim, 2, "Vocals should be 2D [channels, samples]")

        // Verify stereo output
        XCTAssertEqual(separated.drums.shape[0], 2, "Drums should be stereo")
        XCTAssertEqual(separated.vocals.shape[0], 2, "Vocals should be stereo")

        print("  Separation completed successfully")
    }

    /// Verify separated stems approximately sum to mixture.
    func testOutputSumsToMixture() async throws {
        guard let loader = fixtureLoader else {
            throw XCTSkip("Fixture loader not available")
        }

        let model = try await loadModel()
        let fixture = try loader.loadTrack(index: 0)

        let separated = try separateWithHTDemucs(model, audio: fixture.mixture)

        // Sum separated stems
        let reconstructed = separated.drums + separated.bass + separated.other + separated.vocals

        // Compare to mixture (should be close but not exact due to processing)
        let minLen = min(fixture.mixture.shape[1], reconstructed.shape[1])
        let mixture = fixture.mixture[.ellipsis, ..<minLen]
        let recon = reconstructed[.ellipsis, ..<minLen]

        // Compute correlation coefficient
        let mixFlat = mixture.flattened()
        let reconFlat = recon.flattened()

        let mixMean = mixFlat.mean().item(Float.self)
        let reconMean = reconFlat.mean().item(Float.self)

        let mixCentered = mixFlat - mixMean
        let reconCentered = reconFlat - reconMean

        let numerator = (mixCentered * reconCentered).sum().item(Float.self)
        let denomMix = sqrt((mixCentered * mixCentered).sum().item(Float.self))
        let denomRecon = sqrt((reconCentered * reconCentered).sum().item(Float.self))

        let correlation = numerator / (denomMix * denomRecon + 1e-8)

        print("  Reconstruction correlation: \(correlation)")
        XCTAssertGreaterThan(correlation, 0.9, "Reconstruction correlation too low")
    }

    /// Verify SDR meets minimum thresholds.
    func testSDRAboveMinimum() async throws {
        guard let loader = fixtureLoader,
              let manifest = manifest else {
            throw XCTSkip("Fixtures not available")
        }

        let model = try await loadModel()

        // Test first track
        let fixture = try loader.loadTrack(index: 0)
        print("Testing SDR on: \(fixture.metadata.trackName)")

        let separated = try separateWithHTDemucs(model, audio: fixture.mixture)
        let sdrs = computeAllSDR(reference: fixture.groundTruth, estimate: separated)

        for (stem, sdr) in sdrs {
            print("  \(stem): \(String(format: "%.2f", sdr)) dB")

            guard let minimum = Self.sdrMinimum[stem] else { continue }
            XCTAssertGreaterThan(
                sdr,
                minimum,
                "\(stem) SDR \(sdr) below minimum \(minimum)"
            )
        }
    }

    // MARK: - Python Parity Tests

    /// Verify Swift separation matches Python within tolerance.
    func testPythonParity() async throws {
        guard let loader = fixtureLoader else {
            throw XCTSkip("Fixtures not available")
        }

        let fixture = try loader.loadTrack(index: 0)

        guard let pythonOutput = fixture.pythonOutput else {
            throw XCTSkip("Python output not available in fixtures")
        }

        let model = try await loadModel()
        let swiftOutput = try separateWithHTDemucs(model, audio: fixture.mixture)

        print("Comparing Swift vs Python on: \(fixture.metadata.trackName)")

        // Compute SDR for both against ground truth
        let pythonSDR = computeAllSDR(reference: fixture.groundTruth, estimate: pythonOutput)
        let swiftSDR = computeAllSDR(reference: fixture.groundTruth, estimate: swiftOutput)

        for stem in StemData.stemNames {
            guard let pSDR = pythonSDR[stem], let sSDR = swiftSDR[stem] else { continue }

            let diff = abs(pSDR - sSDR)
            print("  \(stem): Python=\(String(format: "%.2f", pSDR))dB, "
                  + "Swift=\(String(format: "%.2f", sSDR))dB, diff=\(String(format: "%.2f", diff))dB")

            XCTAssertLessThan(
                diff,
                Self.parityToleranceSDR,
                "\(stem) SDR differs too much: Python=\(pSDR), Swift=\(sSDR)"
            )
        }
    }

    /// Compare raw output values between Swift and Python.
    func testOutputValuesParity() async throws {
        guard let loader = fixtureLoader else {
            throw XCTSkip("Fixtures not available")
        }

        let fixture = try loader.loadTrack(index: 0)

        guard let pythonOutput = fixture.pythonOutput else {
            throw XCTSkip("Python output not available in fixtures")
        }

        let model = try await loadModel()
        let swiftOutput = try separateWithHTDemucs(model, audio: fixture.mixture)

        // Compare max absolute difference for each stem
        let tolerance: Float = 1e-2  // Allow some numerical difference

        for (stemName, pythonStem, swiftStem) in [
            ("drums", pythonOutput.drums, swiftOutput.drums),
            ("bass", pythonOutput.bass, swiftOutput.bass),
            ("other", pythonOutput.other, swiftOutput.other),
            ("vocals", pythonOutput.vocals, swiftOutput.vocals),
        ] {
            let minLen = min(pythonStem.shape[1], swiftStem.shape[1])
            let pyStem = pythonStem[.ellipsis, ..<minLen]
            let swStem = swiftStem[.ellipsis, ..<minLen]

            let diff = abs(pyStem - swStem)
            let maxDiff = diff.max().item(Float.self)
            let meanDiff = diff.mean().item(Float.self)

            print("  \(stemName): maxDiff=\(String(format: "%.6f", maxDiff)), "
                  + "meanDiff=\(String(format: "%.6f", meanDiff))")

            // Mean should be very small, max can be slightly larger
            XCTAssertLessThan(
                meanDiff,
                tolerance,
                "\(stemName) mean difference too large"
            )
        }
    }

    // MARK: - Full Evaluation Tests

    /// Evaluate on all fixture tracks and compute aggregate metrics.
    func testFullEvaluation() async throws {
        guard let loader = fixtureLoader,
              let manifest = manifest else {
            throw XCTSkip("Fixtures not available")
        }

        let model = try await loadModel()

        var allSDRs: [String: [Float]] = [
            "drums": [],
            "bass": [],
            "other": [],
            "vocals": [],
        ]

        print("\nFull MUSDB18 Evaluation")
        print("=" + String(repeating: "=", count: 59))

        for i in 0..<manifest.numTracks {
            let fixture = try loader.loadTrack(index: i)
            print("\n[\(i + 1)/\(manifest.numTracks)] \(fixture.metadata.trackName)")

            let separated = try separateWithHTDemucs(model, audio: fixture.mixture)
            let sdrs = computeAllSDR(reference: fixture.groundTruth, estimate: separated)

            for (stem, sdr) in sdrs {
                if !sdr.isNaN && !sdr.isInfinite {
                    allSDRs[stem]?.append(sdr)
                }
                print("  \(stem): \(String(format: "%.2f", sdr)) dB")
            }
        }

        // Compute and print aggregate statistics
        print("\n" + String(repeating: "=", count: 60))
        print("Aggregate Results")
        print(String(repeating: "=", count: 60))

        for stem in StemData.stemNames {
            guard let sdrs = allSDRs[stem], !sdrs.isEmpty else { continue }

            let mean = sdrs.reduce(0, +) / Float(sdrs.count)
            let target = Self.sdrTargets[stem] ?? 0

            print("  \(stem): mean=\(String(format: "%.2f", mean)) dB (target: \(target) dB)")

            // Assert meets target (with 1 dB tolerance)
            XCTAssertGreaterThan(
                mean,
                target - 1.0,
                "\(stem) mean SDR \(mean) below target \(target - 1.0)"
            )
        }
    }

    // MARK: - Determinism Tests

    /// Verify running twice produces identical results.
    func testDeterminism() async throws {
        guard let loader = fixtureLoader else {
            throw XCTSkip("Fixtures not available")
        }

        let model = try await loadModel()
        let fixture = try loader.loadTrack(index: 0)

        // Run twice
        let result1 = try separateWithHTDemucs(model, audio: fixture.mixture)
        let result2 = try separateWithHTDemucs(model, audio: fixture.mixture)

        // Should be identical
        for (name, stem1, stem2) in [
            ("drums", result1.drums, result2.drums),
            ("bass", result1.bass, result2.bass),
            ("other", result1.other, result2.other),
            ("vocals", result1.vocals, result2.vocals),
        ] {
            let diff = abs(stem1 - stem2).max().item(Float.self)
            XCTAssertEqual(
                diff,
                0,
                "Non-deterministic output for \(name): max diff = \(diff)"
            )
        }

        print("  Determinism verified: identical outputs on repeated runs")
    }
}
