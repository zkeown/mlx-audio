// GenerateViewModel.swift
// ViewModel for music generation using MusicGen.

import Foundation
import MLX
import MLXAudioModels

/// ViewModel for the Generate tab.
@MainActor
class GenerateViewModel: ObservableObject {
    // MARK: - Published State

    @Published var prompt = ""
    @Published var duration: Float = 10.0
    @Published var isGenerating = false
    @Published var generationProgress: Float = 0
    @Published var generatedAudio: MLXArray?
    @Published var generatedWaveform: [Float] = []
    @Published var selectedVariant = "musicgen-medium"
    @Published var errorMessage: String?
    @Published var isDownloading = false

    // Generation settings
    @Published var temperature: Float = 1.0
    @Published var topK: Int = 250
    @Published var topP: Float = 0.0
    @Published var cfgCoeff: Float = 3.0

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
            ("musicgen-small", "Small (300M)"),
            ("musicgen-medium", "Medium (1.5B)"),
            ("musicgen-large", "Large (3.3B)"),
        ]
    }

    // MARK: - Duration Range

    var durationRange: ClosedRange<Float> {
        5...30
    }

    // MARK: - Sample Prompts

    var samplePrompts: [String] {
        [
            "jazz piano, upbeat mood, smooth",
            "electronic dance music, energetic beat",
            "acoustic guitar, relaxing, mellow",
            "orchestral, epic, cinematic",
            "lo-fi hip hop, chill, study music",
            "rock guitar riff, powerful drums",
        ]
    }

    // MARK: - Actions

    /// Generate music from the prompt.
    func generate() async {
        guard !prompt.isEmpty else {
            errorMessage = "Please enter a prompt"
            return
        }

        guard let modelManager = modelManager else {
            errorMessage = "Model manager not available"
            return
        }

        isGenerating = true
        generationProgress = 0
        generatedAudio = nil
        generatedWaveform = []
        errorMessage = nil

        do {
            // Load model
            isDownloading = true
            let model = try await modelManager.loadMusicGen(variant: selectedVariant)
            isDownloading = false

            // Calculate number of tokens for the duration
            // MusicGen generates at ~50 tokens/second
            let tokensPerSecond = 50
            let maxNewTokens = Int(duration) * tokensPerSecond

            // Note: This is a simplified implementation
            // Full implementation would:
            // 1. Encode the prompt with T5
            // 2. Generate audio codes
            // 3. Decode with EnCodec

            // Simulate generation progress
            for i in 1...10 {
                try await Task.sleep(nanoseconds: 200_000_000)
                generationProgress = Float(i) / 10.0
            }

            // Create placeholder audio
            // In real implementation, this would be the actual generated audio
            let sampleCount = Int(duration) * 32000  // 32kHz sample rate
            generatedAudio = MLXArray.zeros([1, sampleCount])

            // Generate waveform for display
            generatedWaveform = (0..<500).map { i in
                let t = Float(i) / 500.0 * .pi * 20
                return abs(sin(t) * cos(t * 0.3)) * Float.random(in: 0.3...0.8)
            }

        } catch {
            errorMessage = "Generation failed: \(error.localizedDescription)"
        }

        isGenerating = false
    }

    /// Use a sample prompt.
    func useSamplePrompt(_ sample: String) {
        prompt = sample
    }

    /// Save generated audio to file.
    func save(to url: URL) async throws {
        guard let audio = generatedAudio else {
            throw AudioLoadError.emptyFile
        }

        try AudioLoader.save(audio: audio, to: url, sampleRate: 32000)
    }

    /// Clear the generated audio.
    func clear() {
        generatedAudio = nil
        generatedWaveform = []
        generationProgress = 0
    }
}
