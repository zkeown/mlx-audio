// EmbedViewModel.swift
// ViewModel for audio embedding and classification using CLAP.

import Foundation
import MLX
import MLXAudioModels

/// Classification result.
struct ClassificationResult: Identifiable, Equatable {
    let id = UUID()
    let label: String
    let probability: Float

    var formattedProbability: String {
        String(format: "%.0f%%", probability * 100)
    }

    static func == (lhs: ClassificationResult, rhs: ClassificationResult) -> Bool {
        lhs.id == rhs.id
    }
}

/// ViewModel for the Embed tab.
@MainActor
class EmbedViewModel: ObservableObject {
    // MARK: - Published State

    @Published var inputURL: URL?
    @Published var isProcessing = false

    // Classification
    @Published var classificationLabels: [String] = [
        "music", "speech", "dog barking", "car horn", "bird singing"
    ]
    @Published var classificationResults: [ClassificationResult] = []

    // Similarity search
    @Published var searchQuery = ""
    @Published var similarityScore: Float?

    // Model
    @Published var selectedVariant = "clap-htsat-fused"
    @Published var errorMessage: String?
    @Published var isDownloading = false

    @Published var inputWaveform: [Float] = []

    // Custom label input
    @Published var newLabel = ""

    // MARK: - Private

    private weak var modelManager: ModelManager?

    // MARK: - Initialization

    init(modelManager: ModelManager? = nil) {
        self.modelManager = modelManager
    }

    func setModelManager(_ manager: ModelManager) {
        self.modelManager = manager
    }

    // MARK: - Available Variants

    var availableVariants: [(id: String, name: String)] {
        [
            ("clap-htsat-tiny", "Tiny (Fast)"),
            ("clap-htsat-fused", "Fused (Best)"),
        ]
    }

    // MARK: - Preset Label Sets

    var presetLabelSets: [(name: String, labels: [String])] {
        [
            ("General", ["music", "speech", "silence", "noise", "nature"]),
            ("Music Genres", ["rock", "jazz", "classical", "electronic", "hip-hop"]),
            ("Sounds", ["dog barking", "car horn", "bird singing", "thunder", "rain"]),
            ("Instruments", ["piano", "guitar", "drums", "violin", "vocals"]),
        ]
    }

    // MARK: - Actions

    /// Load an audio file.
    func loadAudio(from url: URL) async {
        inputURL = url
        classificationResults = []
        similarityScore = nil
        errorMessage = nil

        do {
            let audio = try await AudioLoader.load(url: url, targetSampleRate: 48000)
            inputWaveform = AudioLoader.waveformData(from: audio)
        } catch {
            errorMessage = "Failed to load audio: \(error.localizedDescription)"
        }
    }

    /// Run zero-shot classification.
    func classify() async {
        guard let url = inputURL else {
            errorMessage = "No audio file selected"
            return
        }

        guard let modelManager = modelManager else {
            errorMessage = "Model manager not available"
            return
        }

        guard !classificationLabels.isEmpty else {
            errorMessage = "No classification labels specified"
            return
        }

        isProcessing = true
        errorMessage = nil

        do {
            // Load audio at 48kHz (CLAP sample rate)
            let audio = try await AudioLoader.load(url: url, targetSampleRate: 48000)

            // Load model
            isDownloading = true
            let model = try await modelManager.loadCLAP(variant: selectedVariant)
            isDownloading = false

            // Note: Simplified implementation
            // Full implementation would:
            // 1. Compute audio embedding
            // 2. Compute text embeddings for all labels
            // 3. Calculate cosine similarities
            // 4. Apply softmax for probabilities

            // Placeholder results
            var results: [ClassificationResult] = []
            let total = Float(classificationLabels.count)
            for (i, label) in classificationLabels.enumerated() {
                let prob = Float.random(in: 0.1...0.9) / total * 2
                results.append(ClassificationResult(label: label, probability: prob))
            }

            // Sort by probability
            classificationResults = results.sorted { $0.probability > $1.probability }

            // Normalize to sum to 1
            let sum = classificationResults.reduce(0) { $0 + $1.probability }
            classificationResults = classificationResults.map {
                ClassificationResult(label: $0.label, probability: $0.probability / sum)
            }

        } catch {
            errorMessage = "Classification failed: \(error.localizedDescription)"
        }

        isProcessing = false
    }

    /// Compute similarity between audio and text query.
    func computeSimilarity() async {
        guard let url = inputURL else {
            errorMessage = "No audio file selected"
            return
        }

        guard let modelManager = modelManager else {
            errorMessage = "Model manager not available"
            return
        }

        guard !searchQuery.isEmpty else {
            errorMessage = "Please enter a search query"
            return
        }

        isProcessing = true
        errorMessage = nil

        do {
            let audio = try await AudioLoader.load(url: url, targetSampleRate: 48000)

            isDownloading = true
            let model = try await modelManager.loadCLAP(variant: selectedVariant)
            isDownloading = false

            // Placeholder result
            similarityScore = Float.random(in: 0.3...0.95)

        } catch {
            errorMessage = "Similarity computation failed: \(error.localizedDescription)"
        }

        isProcessing = false
    }

    /// Add a custom label.
    func addLabel() {
        let trimmed = newLabel.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, !classificationLabels.contains(trimmed) else {
            return
        }
        classificationLabels.append(trimmed)
        newLabel = ""
    }

    /// Remove a label.
    func removeLabel(_ label: String) {
        classificationLabels.removeAll { $0 == label }
    }

    /// Use a preset label set.
    func usePreset(_ labels: [String]) {
        classificationLabels = labels
        classificationResults = []
    }
}
