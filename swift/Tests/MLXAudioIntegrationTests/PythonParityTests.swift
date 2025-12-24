// PythonParityTests.swift
// Cross-platform parity tests comparing Swift outputs against Python on real audio.
//
// These tests verify that Swift implementations produce numerically equivalent
// results to Python implementations when given the same input audio.
//
// Run with:
//   INTEGRATION_FIXTURES=/path/to/fixtures swift test --filter PythonParityTests

import XCTest
@testable import MLXAudioModels
@testable import MLXAudioPrimitives
import MLX
import MLXNN

/// Parity tests between Swift and Python implementations on real audio.
final class PythonParityTests: XCTestCase {

    // MARK: - Tolerances

    /// Maximum acceptable SDR difference between Swift and Python.
    static let sdrToleranceDB: Float = 0.5

    /// Maximum acceptable mean absolute difference for waveforms.
    static let waveformMeanTolerance: Float = 1e-3

    /// Maximum acceptable max absolute difference for waveforms.
    static let waveformMaxTolerance: Float = 1e-2

    // MARK: - Properties

    var fixtureLoader: IntegrationFixtureLoader?
    var manifest: IntegrationManifest?

    // MARK: - Setup

    override func setUpWithError() throws {
        try super.setUpWithError()

        guard ProcessInfo.processInfo.environment[integrationFixturesEnvVar] != nil else {
            throw XCTSkip(
                "Integration fixtures not available. Set \(integrationFixturesEnvVar) environment variable."
            )
        }

        fixtureLoader = try IntegrationFixtureLoader()
        manifest = try fixtureLoader?.loadManifest()
    }

    // MARK: - HTDemucs Parity Tests

    /// Verify HTDemucs Swift output matches Python output on real audio.
    func testHTDemucsOutputParity() async throws {
        guard let loader = fixtureLoader,
              let manifest = manifest else {
            throw XCTSkip("Fixtures not available")
        }

        // Load model
        let model = try await HTDemucs.fromPretrained(modelId: "htdemucs_ft")

        print("\nHTDemucs Python↔Swift Parity Test")
        print("=" + String(repeating: "=", count: 59))

        var allPassed = true

        for i in 0..<manifest.numTracks {
            let fixture = try loader.loadTrack(index: i)

            guard let pythonOutput = fixture.pythonOutput else {
                print("[\(i + 1)] \(fixture.metadata.trackName): SKIPPED (no Python output)")
                continue
            }

            print("\n[\(i + 1)/\(manifest.numTracks)] \(fixture.metadata.trackName)")

            // Run Swift separation
            let swiftOutput = try separateWithHTDemucs(model, audio: fixture.mixture)

            // Compare each stem
            for (stemName, pythonStem, swiftStem) in [
                ("drums", pythonOutput.drums, swiftOutput.drums),
                ("bass", pythonOutput.bass, swiftOutput.bass),
                ("other", pythonOutput.other, swiftOutput.other),
                ("vocals", pythonOutput.vocals, swiftOutput.vocals),
            ] {
                let result = compareWaveforms(
                    python: pythonStem,
                    swift: swiftStem,
                    name: stemName
                )

                if !result.passed {
                    allPassed = false
                }
            }
        }

        XCTAssertTrue(allPassed, "Some parity tests failed")
    }

    /// Compare SDR metrics between Swift and Python.
    func testSDRMetricsParity() async throws {
        guard let loader = fixtureLoader else {
            throw XCTSkip("Fixtures not available")
        }

        let fixture = try loader.loadTrack(index: 0)

        guard let pythonOutput = fixture.pythonOutput else {
            throw XCTSkip("No Python output available")
        }

        let model = try await HTDemucs.fromPretrained(modelId: "htdemucs_ft")
        let swiftOutput = try separateWithHTDemucs(model, audio: fixture.mixture)

        print("\nSDR Metrics Comparison: \(fixture.metadata.trackName)")
        print("=" + String(repeating: "=", count: 59))

        let pythonSDR = computeAllSDR(reference: fixture.groundTruth, estimate: pythonOutput)
        let swiftSDR = computeAllSDR(reference: fixture.groundTruth, estimate: swiftOutput)

        print("\nStem         Python (dB)   Swift (dB)    Diff (dB)")
        print(String(repeating: "-", count: 55))

        var maxDiff: Float = 0

        for stem in StemData.stemNames {
            guard let pSDR = pythonSDR[stem],
                  let sSDR = swiftSDR[stem] else { continue }

            let diff = abs(pSDR - sSDR)
            maxDiff = max(maxDiff, diff)

            let status = diff < Self.sdrToleranceDB ? "OK" : "FAIL"
            print(String(format: "%-12s %8.2f      %8.2f      %+6.2f  [%@]",
                        stem, pSDR, sSDR, sSDR - pSDR, status))

            XCTAssertLessThan(
                diff,
                Self.sdrToleranceDB,
                "\(stem) SDR differs by \(diff) dB, exceeds tolerance \(Self.sdrToleranceDB) dB"
            )
        }

        print(String(repeating: "-", count: 55))
        print(String(format: "Max difference: %.2f dB (tolerance: %.1f dB)",
                    maxDiff, Self.sdrToleranceDB))
    }

    /// Test that Swift and Python produce same stem ordering.
    func testStemOrderingParity() async throws {
        guard let loader = fixtureLoader else {
            throw XCTSkip("Fixtures not available")
        }

        let fixture = try loader.loadTrack(index: 0)

        guard let pythonOutput = fixture.pythonOutput else {
            throw XCTSkip("No Python output available")
        }

        let model = try await HTDemucs.fromPretrained(modelId: "htdemucs_ft")
        let swiftOutput = try separateWithHTDemucs(model, audio: fixture.mixture)

        print("\nStem Ordering Verification")

        // Verify by checking which stem has highest correlation with ground truth
        for (i, stemName) in StemData.stemNames.enumerated() {
            let groundTruth: MLXArray
            let pythonStem: MLXArray
            let swiftStem: MLXArray

            switch i {
            case 0:
                groundTruth = fixture.groundTruth.drums
                pythonStem = pythonOutput.drums
                swiftStem = swiftOutput.drums
            case 1:
                groundTruth = fixture.groundTruth.bass
                pythonStem = pythonOutput.bass
                swiftStem = swiftOutput.bass
            case 2:
                groundTruth = fixture.groundTruth.other
                pythonStem = pythonOutput.other
                swiftStem = swiftOutput.other
            case 3:
                groundTruth = fixture.groundTruth.vocals
                pythonStem = pythonOutput.vocals
                swiftStem = swiftOutput.vocals
            default:
                continue
            }

            // Compute correlation between Python/Swift outputs
            let minLen = min(pythonStem.shape[1], swiftStem.shape[1])
            let pFlat = pythonStem[.ellipsis, ..<minLen].flattened()
            let sFlat = swiftStem[.ellipsis, ..<minLen].flattened()

            let corr = computeCorrelation(pFlat, sFlat)
            print("  \(stemName): Python↔Swift correlation = \(String(format: "%.4f", corr))")

            // Stems should be highly correlated
            XCTAssertGreaterThan(corr, 0.95, "\(stemName) correlation too low")
        }
    }

    // MARK: - CLAP Parity Tests

    /// Verify CLAP embeddings match between Swift and Python.
    func testCLAPEmbeddingParity() async throws {
        guard let fixturesPath = ProcessInfo.processInfo.environment[integrationFixturesEnvVar] else {
            throw XCTSkip("Integration fixtures not available")
        }

        let clapFixturesPath = URL(fileURLWithPath: fixturesPath)
            .appendingPathComponent("clap")

        guard FileManager.default.fileExists(atPath: clapFixturesPath.path) else {
            throw XCTSkip("CLAP parity fixtures not available")
        }

        // Load Python embeddings
        let embeddingsURL = clapFixturesPath.appendingPathComponent("embeddings.safetensors")
        guard FileManager.default.fileExists(atPath: embeddingsURL.path) else {
            throw XCTSkip("CLAP embedding fixtures not found")
        }

        let fixtures = try MLX.loadArrays(url: embeddingsURL)

        guard let audio = fixtures["audio"],
              let pythonAudioEmb = fixtures["audio_embedding"],
              let pythonTextEmb = fixtures["text_embeddings"] else {
            throw XCTSkip("Missing CLAP fixture arrays")
        }

        // Load Swift model
        let model = try await CLAP.fromPretrained(modelId: "clap-htsat-fused")

        // Compute Swift embeddings
        let swiftAudioEmb = model.encodeAudio(audio)
        eval(swiftAudioEmb)

        // Compare
        let audioDiff = abs(pythonAudioEmb - swiftAudioEmb)
        let maxAudioDiff = audioDiff.max().item(Float.self)
        let meanAudioDiff = audioDiff.mean().item(Float.self)

        print("\nCLAP Audio Embedding Parity")
        print("  Max diff: \(String(format: "%.6f", maxAudioDiff))")
        print("  Mean diff: \(String(format: "%.6f", meanAudioDiff))")

        XCTAssertLessThan(
            meanAudioDiff,
            1e-3,
            "CLAP audio embedding mean diff too large"
        )
    }

    // MARK: - Helper Methods

    struct WaveformComparisonResult {
        let meanDiff: Float
        let maxDiff: Float
        let passed: Bool
    }

    func compareWaveforms(python: MLXArray, swift: MLXArray, name: String) -> WaveformComparisonResult {
        let minLen = min(python.shape.last!, swift.shape.last!)
        let pWav = python[.ellipsis, ..<minLen]
        let sWav = swift[.ellipsis, ..<minLen]

        let diff = abs(pWav - sWav)
        let meanDiff = diff.mean().item(Float.self)
        let maxDiff = diff.max().item(Float.self)

        let meanPass = meanDiff < Self.waveformMeanTolerance
        let maxPass = maxDiff < Self.waveformMaxTolerance
        let passed = meanPass && maxPass

        let status = passed ? "OK" : "FAIL"
        print("  \(name): mean=\(String(format: "%.6f", meanDiff)), "
              + "max=\(String(format: "%.6f", maxDiff)) [\(status)]")

        return WaveformComparisonResult(
            meanDiff: meanDiff,
            maxDiff: maxDiff,
            passed: passed
        )
    }

    func computeCorrelation(_ a: MLXArray, _ b: MLXArray) -> Float {
        let aMean = a.mean().item(Float.self)
        let bMean = b.mean().item(Float.self)

        let aCentered = a - aMean
        let bCentered = b - bMean

        let numerator = (aCentered * bCentered).sum().item(Float.self)
        let denomA = sqrt((aCentered * aCentered).sum().item(Float.self))
        let denomB = sqrt((bCentered * bCentered).sum().item(Float.self))

        return numerator / (denomA * denomB + 1e-8)
    }
}
