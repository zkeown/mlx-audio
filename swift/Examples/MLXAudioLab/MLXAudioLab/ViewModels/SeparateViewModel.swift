// SeparateViewModel.swift
// ViewModel for source separation using HTDemucs.

import Foundation
import MLX
import MLXAudioModels

/// Separation result containing individual stems.
struct SeparationResult {
    let drums: [Float]
    let bass: [Float]
    let other: [Float]
    let vocals: [Float]

    /// Full audio arrays for export.
    let drumsAudio: MLXArray
    let bassAudio: MLXArray
    let otherAudio: MLXArray
    let vocalsAudio: MLXArray
}

/// ViewModel for the Separate tab.
@MainActor
class SeparateViewModel: ObservableObject {
    // MARK: - Published State

    @Published var inputURL: URL?
    @Published var isProcessing = false
    @Published var progress: Float = 0
    @Published var separationResult: SeparationResult?
    @Published var selectedVariant = "htdemucs_ft"
    @Published var errorMessage: String?
    @Published var isDownloading = false

    // Waveform data for display
    @Published var inputWaveform: [Float] = []

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

    var availableVariants: [String] {
        ["htdemucs", "htdemucs_ft", "htdemucs_6s"]
    }

    // MARK: - Actions

    /// Load an audio file.
    func loadAudio(from url: URL) async {
        inputURL = url
        separationResult = nil
        errorMessage = nil

        do {
            // Load audio and generate waveform
            let audio = try await AudioLoader.load(url: url, targetSampleRate: 44100)
            inputWaveform = AudioLoader.waveformData(from: audio)
        } catch {
            errorMessage = "Failed to load audio: \(error.localizedDescription)"
        }
    }

    /// Run source separation.
    func separate() async {
        guard let url = inputURL else {
            errorMessage = "No audio file selected"
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
            // Load audio at 44.1kHz (HTDemucs sample rate)
            let audio = try await AudioLoader.load(url: url, targetSampleRate: 44100)

            // Load model (this may trigger download)
            isDownloading = true
            let model = try await modelManager.loadHTDemucs(variant: selectedVariant)
            isDownloading = false

            // Run separation with progress callback
            let stems = applyModel(
                model,
                mix: audio,
                progressCallback: { [weak self] p in
                    Task { @MainActor in
                        self?.progress = p
                    }
                }
            )

            // Extract individual stems
            // Shape: [S, C, T] where S=4 (drums, bass, other, vocals)
            let drumsArray = stems[0, 0..., 0...]
            let bassArray = stems[1, 0..., 0...]
            let otherArray = stems[2, 0..., 0...]
            let vocalsArray = stems[3, 0..., 0...]

            // Generate waveform data for display
            let drumsWaveform = AudioLoader.waveformData(from: drumsArray)
            let bassWaveform = AudioLoader.waveformData(from: bassArray)
            let otherWaveform = AudioLoader.waveformData(from: otherArray)
            let vocalsWaveform = AudioLoader.waveformData(from: vocalsArray)

            separationResult = SeparationResult(
                drums: drumsWaveform,
                bass: bassWaveform,
                other: otherWaveform,
                vocals: vocalsWaveform,
                drumsAudio: drumsArray,
                bassAudio: bassArray,
                otherAudio: otherArray,
                vocalsAudio: vocalsArray
            )

        } catch {
            errorMessage = "Separation failed: \(error.localizedDescription)"
        }

        isProcessing = false
    }

    /// Export a stem to a file.
    func exportStem(_ stem: String, to url: URL) async throws {
        guard let result = separationResult else {
            throw AudioLoadError.emptyFile
        }

        let audio: MLXArray
        switch stem.lowercased() {
        case "drums":
            audio = result.drumsAudio
        case "bass":
            audio = result.bassAudio
        case "other":
            audio = result.otherAudio
        case "vocals":
            audio = result.vocalsAudio
        default:
            throw AudioLoadError.unsupportedFormat("Unknown stem: \(stem)")
        }

        try AudioLoader.save(audio: audio, to: url, sampleRate: 44100)
    }

    /// Export all stems to a directory.
    func exportAllStems(to directory: URL) async throws {
        try await exportStem("drums", to: directory.appendingPathComponent("drums.wav"))
        try await exportStem("bass", to: directory.appendingPathComponent("bass.wav"))
        try await exportStem("other", to: directory.appendingPathComponent("other.wav"))
        try await exportStem("vocals", to: directory.appendingPathComponent("vocals.wav"))
    }
}
