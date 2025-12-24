// IntegrationTestHelpers.swift
// Shared utilities for integration tests with real audio datasets.

import Foundation
import MLX
@testable import MLXAudioModels
@testable import MLXAudioPrimitives

// MARK: - Configuration

/// Environment variable for integration test fixtures path.
let integrationFixturesEnvVar = "INTEGRATION_FIXTURES"

/// Environment variable for MUSDB18-HQ dataset root.
let musdb18RootEnvVar = "MUSDB18_ROOT"

/// Environment variable for ESC-50 dataset root.
let esc50RootEnvVar = "ESC50_ROOT"

// MARK: - Fixture Loading

/// Errors for integration testing.
enum IntegrationTestError: Error, LocalizedError {
    case fixtureNotFound(String)
    case datasetNotFound(String)
    case invalidManifest
    case trackNotFound(String)

    var errorDescription: String? {
        switch self {
        case .fixtureNotFound(let path):
            return "Integration fixture not found: \(path)"
        case .datasetNotFound(let dataset):
            return "Dataset not found: \(dataset). Set environment variable."
        case .invalidManifest:
            return "Invalid or missing manifest.json"
        case .trackNotFound(let name):
            return "Track not found: \(name)"
        }
    }
}

/// Manifest for integration test fixtures.
struct IntegrationManifest: Codable {
    let dataset: String
    let split: String
    let numTracks: Int
    let durationPerTrack: Double
    let model: String
    let tracks: [TrackMetadata]

    enum CodingKeys: String, CodingKey {
        case dataset
        case split
        case numTracks = "num_tracks"
        case durationPerTrack = "duration_per_track"
        case model
        case tracks
    }
}

/// Metadata for a single track fixture.
struct TrackMetadata: Codable {
    let trackName: String
    let trackIndex: Int
    let durationSeconds: Double
    let sampleRate: Int
    let channels: Int
    let samples: Int
    let hasPythonOutput: Bool

    enum CodingKeys: String, CodingKey {
        case trackName = "track_name"
        case trackIndex = "track_index"
        case durationSeconds = "duration_seconds"
        case sampleRate = "sample_rate"
        case channels
        case samples
        case hasPythonOutput = "has_python_output"
    }
}

/// Fixture data for a single track.
struct TrackFixture {
    let metadata: TrackMetadata
    let mixture: MLXArray      // [channels, samples]
    let groundTruth: StemData  // Ground truth stems
    let pythonOutput: StemData? // Python separation output (for parity)
}

/// Separated stem data.
struct StemData {
    let drums: MLXArray    // [channels, samples]
    let bass: MLXArray     // [channels, samples]
    let other: MLXArray    // [channels, samples]
    let vocals: MLXArray   // [channels, samples]

    var all: [MLXArray] { [drums, bass, other, vocals] }

    static let stemNames = ["drums", "bass", "other", "vocals"]
}

// MARK: - Integration Test Fixture Loader

/// Loader for MUSDB18 integration test fixtures.
class IntegrationFixtureLoader {

    let fixturesPath: URL

    init() throws {
        guard let path = ProcessInfo.processInfo.environment[integrationFixturesEnvVar] else {
            throw IntegrationTestError.datasetNotFound(
                "INTEGRATION_FIXTURES environment variable not set"
            )
        }
        self.fixturesPath = URL(fileURLWithPath: path)

        guard FileManager.default.fileExists(atPath: fixturesPath.path) else {
            throw IntegrationTestError.fixtureNotFound(path)
        }
    }

    /// Load the manifest file.
    func loadManifest() throws -> IntegrationManifest {
        let manifestURL = fixturesPath.appendingPathComponent("manifest.json")
        guard FileManager.default.fileExists(atPath: manifestURL.path) else {
            throw IntegrationTestError.invalidManifest
        }

        let data = try Data(contentsOf: manifestURL)
        return try JSONDecoder().decode(IntegrationManifest.self, from: data)
    }

    /// Load fixture data for a specific track.
    func loadTrack(index: Int) throws -> TrackFixture {
        let trackDir = fixturesPath.appendingPathComponent("track_\(index)")

        // Load metadata
        let metadataURL = trackDir.appendingPathComponent("metadata.json")
        let metadataData = try Data(contentsOf: metadataURL)
        let metadata = try JSONDecoder().decode(TrackMetadata.self, from: metadataData)

        // Load mixture
        let mixtureURL = trackDir.appendingPathComponent("mixture.safetensors")
        let mixtureArrays = try MLX.loadArrays(url: mixtureURL)
        guard let mixture = mixtureArrays["mixture"] else {
            throw IntegrationTestError.fixtureNotFound("mixture in track_\(index)")
        }

        // Load ground truth stems
        let stemsURL = trackDir.appendingPathComponent("stems.safetensors")
        let stemsArrays = try MLX.loadArrays(url: stemsURL)
        let groundTruth = try extractStems(from: stemsArrays, name: "stems")

        // Load Python output if available
        var pythonOutput: StemData?
        if metadata.hasPythonOutput {
            let pythonURL = trackDir.appendingPathComponent("python_output.safetensors")
            if FileManager.default.fileExists(atPath: pythonURL.path) {
                let pythonArrays = try MLX.loadArrays(url: pythonURL)
                pythonOutput = try extractStems(from: pythonArrays, name: "python_output")
            }
        }

        return TrackFixture(
            metadata: metadata,
            mixture: mixture,
            groundTruth: groundTruth,
            pythonOutput: pythonOutput
        )
    }

    private func extractStems(from arrays: [String: MLXArray], name: String) throws -> StemData {
        guard let drums = arrays["drums"],
              let bass = arrays["bass"],
              let other = arrays["other"],
              let vocals = arrays["vocals"] else {
            throw IntegrationTestError.fixtureNotFound("Missing stem in \(name)")
        }
        return StemData(drums: drums, bass: bass, other: other, vocals: vocals)
    }
}

// MARK: - Quality Metrics

/// Signal-to-Distortion Ratio (SDR) computation.
///
/// SDR measures how well an estimated source matches the reference.
/// Higher is better; typical good values are 8-12 dB.
func computeSDR(reference: MLXArray, estimate: MLXArray) -> Float {
    // Ensure same length
    let minLen = min(reference.shape[reference.ndim - 1], estimate.shape[estimate.ndim - 1])
    let ref = reference[.ellipsis, ..<minLen]
    let est = estimate[.ellipsis, ..<minLen]

    // Convert to mono for SDR computation
    let refMono = ref.mean(axis: 0)
    let estMono = est.mean(axis: 0)

    // SDR = 10 * log10(||ref||^2 / ||ref - est||^2)
    let refPower = (refMono * refMono).sum()
    let noisePower = ((refMono - estMono) * (refMono - estMono)).sum()

    // Avoid division by zero
    let eps: Float = 1e-8
    let sdr = 10 * log10((refPower / (noisePower + eps)).item(Float.self) + eps)

    return sdr
}

/// Compute SDR for all stems.
func computeAllSDR(reference: StemData, estimate: StemData) -> [String: Float] {
    var results: [String: Float] = [:]

    results["drums"] = computeSDR(reference: reference.drums, estimate: estimate.drums)
    results["bass"] = computeSDR(reference: reference.bass, estimate: estimate.bass)
    results["other"] = computeSDR(reference: reference.other, estimate: estimate.other)
    results["vocals"] = computeSDR(reference: reference.vocals, estimate: estimate.vocals)

    return results
}

/// Compute mean SDR across all stems.
func computeMeanSDR(reference: StemData, estimate: StemData) -> Float {
    let sdrs = computeAllSDR(reference: reference, estimate: estimate)
    let values = sdrs.values.filter { !$0.isNaN && !$0.isInfinite }
    guard !values.isEmpty else { return Float.nan }
    return values.reduce(0, +) / Float(values.count)
}

// MARK: - Separation Wrapper

/// Run HTDemucs separation on audio.
func separateWithHTDemucs(
    _ model: HTDemucs,
    audio: MLXArray,
    segment: Float = 6.0,
    overlap: Float = 0.25
) throws -> StemData {
    // Run inference
    let sources = applyModel(model, audio: audio, segment: segment, overlap: overlap)
    eval(sources)

    // sources shape: [4, channels, samples]
    return StemData(
        drums: sources[0],
        bass: sources[1],
        other: sources[2],
        vocals: sources[3]
    )
}
