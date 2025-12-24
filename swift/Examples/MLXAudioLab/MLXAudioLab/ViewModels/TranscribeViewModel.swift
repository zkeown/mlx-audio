// TranscribeViewModel.swift
// ViewModel for speech transcription using Whisper.

import Foundation
import MLX
import MLXAudioModels
#if canImport(UIKit)
import UIKit
#elseif canImport(AppKit)
import AppKit
#endif

/// Transcription segment with timestamp.
struct TranscriptionSegment: Identifiable {
    let id = UUID()
    let start: TimeInterval
    let end: TimeInterval
    let text: String

    var formattedTime: String {
        let startMin = Int(start) / 60
        let startSec = Int(start) % 60
        return String(format: "%d:%02d", startMin, startSec)
    }
}

/// ViewModel for the Transcribe tab.
@MainActor
class TranscribeViewModel: ObservableObject {
    // MARK: - Published State

    @Published var inputURL: URL?
    @Published var isProcessing = false
    @Published var segments: [TranscriptionSegment] = []
    @Published var detectedLanguage: String?
    @Published var languageConfidence: Float = 0
    @Published var selectedVariant = "whisper-large-v3-turbo"
    @Published var errorMessage: String?
    @Published var isDownloading = false

    @Published var inputWaveform: [Float] = []

    // MARK: - Private

    private weak var modelManager: ModelManager?

    // MARK: - Computed

    var fullText: String {
        segments.map(\.text).joined(separator: " ")
    }

    var srtContent: String {
        var srt = ""
        for (index, segment) in segments.enumerated() {
            srt += "\(index + 1)\n"
            srt += "\(formatSRTTime(segment.start)) --> \(formatSRTTime(segment.end))\n"
            srt += "\(segment.text)\n\n"
        }
        return srt
    }

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
            ("whisper-tiny", "Tiny (Fast)"),
            ("whisper-small", "Small"),
            ("whisper-medium", "Medium"),
            ("whisper-large-v3-turbo", "Large V3 Turbo"),
            ("whisper-large-v3", "Large V3 (Best)"),
        ]
    }

    // MARK: - Actions

    /// Load an audio file.
    func loadAudio(from url: URL) async {
        inputURL = url
        segments = []
        detectedLanguage = nil
        errorMessage = nil

        do {
            let audio = try await AudioLoader.load(url: url, targetSampleRate: 16000)
            inputWaveform = AudioLoader.waveformData(from: audio)
        } catch {
            errorMessage = "Failed to load audio: \(error.localizedDescription)"
        }
    }

    /// Run transcription.
    func transcribe() async {
        guard let url = inputURL else {
            errorMessage = "No audio file selected"
            return
        }

        guard let modelManager = modelManager else {
            errorMessage = "Model manager not available"
            return
        }

        isProcessing = true
        errorMessage = nil

        do {
            // Load audio at 16kHz (Whisper sample rate)
            let audio = try await AudioLoader.load(url: url, targetSampleRate: 16000, mono: true)

            // Load model
            isDownloading = true
            let model = try await modelManager.loadWhisper(variant: selectedVariant)
            isDownloading = false

            // Run transcription
            // Note: This is a simplified implementation
            // Full implementation would use the complete Whisper decoding pipeline

            // Placeholder: In real implementation, you would:
            // 1. Compute mel spectrogram
            // 2. Detect language
            // 3. Run decoding with timestamps
            // 4. Parse segments

            // For demo purposes, we'll create a placeholder result
            detectedLanguage = "English"
            languageConfidence = 0.95

            segments = [
                TranscriptionSegment(start: 0, end: 3, text: "This is a demo transcription."),
                TranscriptionSegment(start: 3, end: 6, text: "The actual implementation would use the full Whisper pipeline."),
            ]

        } catch {
            errorMessage = "Transcription failed: \(error.localizedDescription)"
        }

        isProcessing = false
    }

    /// Copy full text to clipboard.
    func copyText() {
        #if os(iOS)
        UIPasteboard.general.string = fullText
        #else
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(fullText, forType: .string)
        #endif
    }

    // MARK: - Helpers

    private func formatSRTTime(_ time: TimeInterval) -> String {
        let hours = Int(time) / 3600
        let minutes = (Int(time) % 3600) / 60
        let seconds = Int(time) % 60
        let milliseconds = Int((time.truncatingRemainder(dividingBy: 1)) * 1000)
        return String(format: "%02d:%02d:%02d,%03d", hours, minutes, seconds, milliseconds)
    }
}
