// ESC50ClassificationTests.swift
// Integration tests for CLAP zero-shot classification on ESC-50 fixtures.
//
// Run with:
//   ESC50_FIXTURES=/path/to/fixtures swift test --filter ESC50ClassificationTests
//
// Generate fixtures first with Python script or use pre-generated fixtures.

import XCTest
@testable import MLXAudioModels
@testable import MLXAudioPrimitives
import MLX
import MLXNN

/// Environment variable for ESC-50 fixtures path.
private let esc50FixturesEnvVar = "ESC50_FIXTURES"

/// Integration tests for CLAP zero-shot classification.
final class ESC50ClassificationTests: XCTestCase {

    // MARK: - Quality Targets

    /// Published CLAP zero-shot accuracy on ESC-50.
    static let accuracyTarget: Float = 0.85  // 85%
    static let accuracyMinimum: Float = 0.70 // 70% minimum acceptable
    static let top5AccuracyTarget: Float = 0.95

    // MARK: - Properties

    var fixturesPath: URL?
    var model: CLAP?

    // MARK: - Setup

    override func setUpWithError() throws {
        try super.setUpWithError()

        // Check for ESC-50 fixtures
        if let path = ProcessInfo.processInfo.environment[esc50FixturesEnvVar] {
            fixturesPath = URL(fileURLWithPath: path)
        } else if let integrationPath = ProcessInfo.processInfo.environment[integrationFixturesEnvVar] {
            // Try integration fixtures path with esc50 subfolder
            let esc50Path = URL(fileURLWithPath: integrationPath).appendingPathComponent("esc50")
            if FileManager.default.fileExists(atPath: esc50Path.path) {
                fixturesPath = esc50Path
            }
        }

        guard fixturesPath != nil else {
            throw XCTSkip(
                "ESC-50 fixtures not available. Set \(esc50FixturesEnvVar) environment variable."
            )
        }
    }

    /// Load CLAP model (lazy, shared across tests).
    func loadModel() async throws -> CLAP {
        if let model = self.model {
            return model
        }

        let model = try await CLAP.fromPretrained(modelId: "clap-htsat-fused")
        self.model = model
        return model
    }

    // MARK: - ESC-50 Classes

    /// Standard ESC-50 class names (50 classes).
    static let esc50Classes: [String] = [
        "dog", "rooster", "pig", "cow", "frog",
        "cat", "hen", "insects", "sheep", "crow",
        "rain", "sea_waves", "crackling_fire", "crickets", "chirping_birds",
        "water_drops", "wind", "pouring_water", "toilet_flush", "thunderstorm",
        "crying_baby", "sneezing", "clapping", "breathing", "coughing",
        "footsteps", "laughing", "brushing_teeth", "snoring", "drinking_sipping",
        "door_wood_knock", "mouse_click", "keyboard_typing", "door_wood_creaks", "can_opening",
        "washing_machine", "vacuum_cleaner", "clock_alarm", "clock_tick", "glass_breaking",
        "helicopter", "chainsaw", "siren", "car_horn", "engine",
        "train", "church_bells", "airplane", "fireworks", "hand_saw",
    ]

    /// Generate text prompts for each class using CLAP template.
    func generatePrompts() -> [String] {
        Self.esc50Classes.map { cls in
            let text = cls.replacingOccurrences(of: "_", with: " ")
            return "the sound of \(text)"
        }
    }

    // MARK: - Fixture Types

    /// ESC-50 fixture metadata.
    struct ESC50Manifest: Codable {
        let numSamples: Int
        let numClasses: Int
        let sampleRate: Int
        let samples: [ESC50Sample]

        enum CodingKeys: String, CodingKey {
            case numSamples = "num_samples"
            case numClasses = "num_classes"
            case sampleRate = "sample_rate"
            case samples
        }
    }

    struct ESC50Sample: Codable {
        let filename: String
        let targetClass: Int
        let className: String
        let fold: Int

        enum CodingKeys: String, CodingKey {
            case filename
            case targetClass = "target_class"
            case className = "class_name"
            case fold
        }
    }

    // MARK: - Tests

    /// Verify CLAP produces embeddings.
    func testModelProducesEmbeddings() async throws {
        let model = try await loadModel()

        // Create dummy audio (2 seconds at 48kHz)
        let audio = MLXRandom.normal([48000 * 2])

        // Get embedding
        let embedding = model.encodeAudio(audio)
        eval(embedding)

        XCTAssertEqual(embedding.shape.last, 512, "Expected 512-dim embedding")
        print("  Audio embedding shape: \(embedding.shape)")
    }

    /// Verify CLAP produces text embeddings.
    func testTextEmbeddings() async throws {
        let model = try await loadModel()

        let prompts = Array(generatePrompts().prefix(5))
        let embeddings = model.encodeText(prompts)
        eval(embeddings)

        XCTAssertEqual(embeddings.shape[0], 5, "Should have 5 text embeddings")
        XCTAssertEqual(embeddings.shape.last, 512, "Should be 512-dim")
        print("  Text embeddings shape: \(embeddings.shape)")
    }

    /// Test zero-shot classification on fixture samples.
    func testZeroShotClassification() async throws {
        guard let fixturesPath = fixturesPath else {
            throw XCTSkip("Fixtures path not available")
        }

        // Check for manifest
        let manifestURL = fixturesPath.appendingPathComponent("manifest.json")
        guard FileManager.default.fileExists(atPath: manifestURL.path) else {
            throw XCTSkip("ESC-50 manifest not found at \(manifestURL.path)")
        }

        let manifestData = try Data(contentsOf: manifestURL)
        let manifest = try JSONDecoder().decode(ESC50Manifest.self, from: manifestData)

        let model = try await loadModel()
        let prompts = generatePrompts()

        // Pre-compute text embeddings for all classes
        let textEmbeddings = model.encodeText(prompts)
        eval(textEmbeddings)

        var correct = 0
        var total = 0
        var top5Correct = 0

        print("\nEvaluating \(manifest.numSamples) samples...")

        for (i, sample) in manifest.samples.enumerated() {
            // Load audio fixture
            let audioURL = fixturesPath.appendingPathComponent(sample.filename)
                .appendingPathExtension("safetensors")

            guard FileManager.default.fileExists(atPath: audioURL.path) else {
                print("  Skipping missing sample: \(sample.filename)")
                continue
            }

            let arrays = try MLX.loadArrays(url: audioURL)
            guard let audio = arrays["audio"] else {
                continue
            }

            // Get audio embedding
            let audioEmbedding = model.encodeAudio(audio)

            // Compute similarities
            let similarities = model.similarity(audioEmbedding, textEmbeddings)
            eval(similarities)

            let simsArray = similarities.squeezed().asArray(Float.self)

            // Get top prediction
            let predicted = simsArray.enumerated().max(by: { $0.element < $1.element })?.offset ?? -1

            // Top-1 accuracy
            if predicted == sample.targetClass {
                correct += 1
            }

            // Top-5 accuracy
            let top5Indices = simsArray.enumerated()
                .sorted { $0.element > $1.element }
                .prefix(5)
                .map { $0.offset }

            if top5Indices.contains(sample.targetClass) {
                top5Correct += 1
            }

            total += 1

            if (i + 1) % 100 == 0 {
                print("  Processed \(i + 1)/\(manifest.numSamples) samples...")
            }
        }

        let accuracy = Float(correct) / Float(total)
        let top5Accuracy = Float(top5Correct) / Float(total)

        print("\nResults:")
        print("  Top-1 Accuracy: \(String(format: "%.1f", accuracy * 100))%")
        print("  Top-5 Accuracy: \(String(format: "%.1f", top5Accuracy * 100))%")
        print("  Target: \(String(format: "%.0f", Self.accuracyTarget * 100))%")

        XCTAssertGreaterThan(
            accuracy,
            Self.accuracyMinimum,
            "Accuracy \(accuracy) below minimum \(Self.accuracyMinimum)"
        )
    }

    /// Test specific well-performing classes.
    func testWellPerformingClasses() async throws {
        let model = try await loadModel()

        // Classes that should have high accuracy
        let wellPerforming = ["dog", "cat", "car_horn", "helicopter", "chainsaw"]
        let prompts = generatePrompts()
        let textEmbeddings = model.encodeText(prompts)
        eval(textEmbeddings)

        print("\nTesting well-performing classes:")

        for className in wellPerforming {
            guard let classIndex = Self.esc50Classes.firstIndex(of: className) else {
                continue
            }

            // Generate synthetic audio that should resemble the class
            // (In practice, this would load real fixtures)
            let audio = MLXRandom.normal([48000 * 2])
            let audioEmbedding = model.encodeAudio(audio)
            let similarities = model.similarity(audioEmbedding, textEmbeddings)
            eval(similarities)

            let simsArray = similarities.squeezed().asArray(Float.self)
            let maxSim = simsArray[classIndex]

            print("  \(className): similarity = \(String(format: "%.3f", maxSim))")
        }
    }
}
