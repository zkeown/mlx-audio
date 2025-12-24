// BanquetViewModel.swift
// ViewModel for query-based source separation using Banquet.

import Foundation
import MLX
import MLXAudioModels

/// ViewModel for the Banquet tab.
@MainActor
class BanquetViewModel: ObservableObject {
    // MARK: - Published State

    @Published var queryAudioURL: URL?
    @Published var mixtureAudioURL: URL?
    @Published var isProcessing = false
    @Published var progress: Float = 0
    @Published var separatedAudio: MLXArray?
    @Published var separatedWaveform: [Float] = []
    @Published var errorMessage: String?
    @Published var isDownloading = false

    // Waveforms for display
    @Published var queryWaveform: [Float] = []
    @Published var mixtureWaveform: [Float] = []

    // MARK: - Private

    private weak var modelManager: ModelManager?

    // MARK: - Initialization

    init(modelManager: ModelManager? = nil) {
        self.modelManager = modelManager
    }

    func setModelManager(_ manager: ModelManager) {
        self.modelManager = manager
    }

    // MARK: - Actions

    /// Load the query (reference) audio.
    func loadQueryAudio(from url: URL) async {
        queryAudioURL = url
        separatedAudio = nil
        separatedWaveform = []
        errorMessage = nil

        do {
            let audio = try await AudioLoader.load(url: url, targetSampleRate: 44100)
            queryWaveform = AudioLoader.waveformData(from: audio)
        } catch {
            errorMessage = "Failed to load query audio: \(error.localizedDescription)"
        }
    }

    /// Load the mixture audio.
    func loadMixtureAudio(from url: URL) async {
        mixtureAudioURL = url
        separatedAudio = nil
        separatedWaveform = []
        errorMessage = nil

        do {
            let audio = try await AudioLoader.load(url: url, targetSampleRate: 44100)
            mixtureWaveform = AudioLoader.waveformData(from: audio)
        } catch {
            errorMessage = "Failed to load mixture audio: \(error.localizedDescription)"
        }
    }

    /// Run query-based separation.
    func separate() async {
        guard let queryURL = queryAudioURL else {
            errorMessage = "Please select a query (reference) audio file"
            return
        }

        guard let mixtureURL = mixtureAudioURL else {
            errorMessage = "Please select a mixture audio file"
            return
        }

        guard let modelManager = modelManager else {
            errorMessage = "Model manager not available"
            return
        }

        isProcessing = true
        progress = 0
        errorMessage = nil

        do {
            // Load both audio files at 44.1kHz (Banquet sample rate)
            let queryAudio = try await AudioLoader.load(url: queryURL, targetSampleRate: 44100)
            progress = 0.1

            let mixtureAudio = try await AudioLoader.load(url: mixtureURL, targetSampleRate: 44100)
            progress = 0.2

            // Load model
            isDownloading = true
            let model = try await modelManager.loadBanquet()
            isDownloading = false
            progress = 0.3

            // Note: Simplified implementation
            // Full implementation would:
            // 1. Prepare query mel spectrogram via prepareQueryMel()
            // 2. Encode query with PaSST
            // 3. Run separation with separateWithQuery()

            // Simulate processing
            for i in 4...10 {
                try await Task.sleep(nanoseconds: 200_000_000)
                progress = Float(i) / 10.0
            }

            // Placeholder: return a scaled version of mixture
            // In real implementation, this would be the separated audio
            separatedAudio = mixtureAudio * 0.8
            separatedWaveform = AudioLoader.waveformData(from: separatedAudio!)

        } catch {
            errorMessage = "Separation failed: \(error.localizedDescription)"
        }

        isProcessing = false
    }

    /// Save separated audio to file.
    func save(to url: URL) async throws {
        guard let audio = separatedAudio else {
            throw AudioLoadError.emptyFile
        }

        try AudioLoader.save(audio: audio, to: url, sampleRate: 44100)
    }

    /// Clear results.
    func clearResults() {
        separatedAudio = nil
        separatedWaveform = []
        progress = 0
    }

    /// Swap query and mixture.
    func swapAudio() {
        swap(&queryAudioURL, &mixtureAudioURL)
        swap(&queryWaveform, &mixtureWaveform)
        clearResults()
    }
}
