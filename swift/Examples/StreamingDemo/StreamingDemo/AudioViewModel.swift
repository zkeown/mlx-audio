// AudioViewModel.swift
// View model for managing audio streaming state.

import Foundation
import SwiftUI
import MLXAudioStreaming
import MLX

/// Streaming mode selection.
enum StreamingMode: String, CaseIterable {
    case passthrough = "Passthrough"
    case spectrogram = "Spectrogram"
    case transcribe = "Transcribe"
}

/// Main view model for audio streaming demo.
@MainActor
class AudioViewModel: ObservableObject {
    // MARK: - Published State

    @Published var isRunning = false
    @Published var mode: StreamingMode = .passthrough
    @Published var statusText = "Ready"

    @Published var waveformData: [Float] = []
    @Published var spectrogramData: [[Float]] = []
    @Published var transcriptionText = ""

    @Published var latency: TimeInterval = 0
    @Published var bufferLevel: Double = 0
    @Published var realtimeFactor: Double = 1.0

    var showTranscription: Bool {
        mode == .transcribe
    }

    // MARK: - Private Properties

    private var microphoneSource: MicrophoneSource?
    private var speakerSink: SpeakerSink?
    private var stftProcessor: StreamingSTFT?
    private var transcriber: StreamingTranscriber?
    private var pipeline: StreamingPipeline?

    private var processingTask: Task<Void, Never>?
    private var updateTask: Task<Void, Never>?

    // Visualization buffers
    private let maxWaveformSamples = 2048
    private let maxSpectrogramFrames = 100

    // MARK: - Control Methods

    /// Start audio streaming with current mode.
    func start() async {
        guard !isRunning else { return }

        statusText = "Starting..."

        do {
            switch mode {
            case .passthrough:
                try await startPassthrough()
            case .spectrogram:
                try await startSpectrogram()
            case .transcribe:
                try await startTranscription()
            }

            isRunning = true
            statusText = "Running"
            startUpdateLoop()

        } catch {
            statusText = "Error: \(error.localizedDescription)"
        }
    }

    /// Stop audio streaming.
    func stop() async {
        guard isRunning else { return }

        statusText = "Stopping..."

        // Cancel tasks
        processingTask?.cancel()
        processingTask = nil
        updateTask?.cancel()
        updateTask = nil

        // Stop pipeline
        if let pipeline = pipeline {
            await pipeline.stop()
        }

        // Stop individual components
        if let mic = microphoneSource {
            try? await mic.stop()
        }
        if let speaker = speakerSink {
            try? await speaker.stop()
        }

        // Clear references
        microphoneSource = nil
        speakerSink = nil
        stftProcessor = nil
        transcriber = nil
        pipeline = nil

        isRunning = false
        statusText = "Stopped"
    }

    // MARK: - Mode-Specific Setup

    private func startPassthrough() async throws {
        // Create microphone source (mono, 44.1kHz)
        let mic = MicrophoneSource(configuration: .mono)
        microphoneSource = mic

        // Create speaker sink (stereo, 44.1kHz)
        let speaker = SpeakerSink(configuration: .stereo)
        speakerSink = speaker

        // Start components
        try await mic.start()
        try await speaker.start()

        // Start processing loop
        processingTask = Task {
            await runPassthroughLoop()
        }
    }

    private func startSpectrogram() async throws {
        // Create microphone source (mono, 44.1kHz)
        let mic = MicrophoneSource(configuration: .mono)
        microphoneSource = mic

        // Create STFT processor
        let stft = StreamingSTFT(configuration: .visualization)
        stftProcessor = stft

        // Start microphone
        try await mic.start()

        // Start processing loop
        processingTask = Task {
            await runSpectrogramLoop()
        }
    }

    private func startTranscription() async throws {
        // Create microphone source (16kHz for Whisper)
        let mic = MicrophoneSource(configuration: .whisper)
        microphoneSource = mic

        // Create transcriber
        let trans = StreamingTranscriber(configuration: .realtime)
        transcriber = trans

        // Set up delegate (self as receiver)
        await trans.setDelegate(TranscriptionHandler(viewModel: self))

        // Start microphone
        try await mic.start()

        // Start processing loop
        processingTask = Task {
            await runTranscriptionLoop()
        }
    }

    // MARK: - Processing Loops

    private func runPassthroughLoop() async {
        guard let mic = microphoneSource,
              let speaker = speakerSink
        else { return }

        while !Task.isCancelled {
            do {
                // Read from microphone
                if let audio = try await mic.read(count: 1024) {
                    // Update waveform visualization
                    await updateWaveform(audio)

                    // Convert mono to stereo for playback
                    let stereo = MLX.stacked([audio.squeezed(), audio.squeezed()], axis: 0)

                    // Write to speaker
                    try await speaker.write(stereo)
                }
            } catch {
                if !Task.isCancelled {
                    await MainActor.run {
                        statusText = "Error: \(error.localizedDescription)"
                    }
                }
                break
            }
        }
    }

    private func runSpectrogramLoop() async {
        guard let mic = microphoneSource,
              let stft = stftProcessor
        else { return }

        while !Task.isCancelled {
            do {
                // Read from microphone
                if let audio = try await mic.read(count: 512) {
                    // Update waveform
                    await updateWaveform(audio)

                    // Compute STFT
                    let spectrogram = try await stft.process(audio)

                    // Update spectrogram visualization
                    if spectrogram.shape[1] > 0 {
                        await updateSpectrogram(spectrogram)
                    }
                }
            } catch {
                if !Task.isCancelled {
                    await MainActor.run {
                        statusText = "Error: \(error.localizedDescription)"
                    }
                }
                break
            }
        }
    }

    private func runTranscriptionLoop() async {
        guard let mic = microphoneSource,
              let trans = transcriber
        else { return }

        while !Task.isCancelled {
            do {
                // Read from microphone
                if let audio = try await mic.read(count: 1024) {
                    // Update waveform
                    await updateWaveform(audio)

                    // Process for transcription (results come via delegate)
                    let _ = try await trans.process(audio)
                }
            } catch {
                if !Task.isCancelled {
                    await MainActor.run {
                        statusText = "Error: \(error.localizedDescription)"
                    }
                }
                break
            }
        }
    }

    // MARK: - Visualization Updates

    private func updateWaveform(_ audio: MLXArray) async {
        // Get samples (take first channel if multi-channel)
        let samples: [Float]
        if audio.ndim == 1 {
            samples = audio.asArray(Float.self)
        } else {
            samples = audio[0].asArray(Float.self)
        }

        await MainActor.run {
            // Append new samples
            waveformData.append(contentsOf: samples)

            // Trim to max size
            if waveformData.count > maxWaveformSamples {
                waveformData.removeFirst(waveformData.count - maxWaveformSamples)
            }
        }
    }

    private func updateSpectrogram(_ spectrogram: MLXArray) async {
        // Shape: [freqBins, numFrames]
        let numFrames = spectrogram.shape[1]
        let numBins = spectrogram.shape[0]

        let data = spectrogram.asArray(Float.self)

        await MainActor.run {
            // Convert to [[Float]] for visualization (frame-major)
            for f in 0..<numFrames {
                var frame = [Float]()
                for b in 0..<min(numBins, 128) {  // Limit bins for visualization
                    frame.append(data[b * numFrames + f])
                }
                spectrogramData.append(frame)
            }

            // Trim to max frames
            while spectrogramData.count > maxSpectrogramFrames {
                spectrogramData.removeFirst()
            }
        }
    }

    func updateTranscription(_ text: String) {
        Task { @MainActor in
            transcriptionText = text
        }
    }

    // MARK: - Statistics Update

    private func startUpdateLoop() {
        updateTask = Task {
            while !Task.isCancelled {
                await updateStatistics()
                try? await Task.sleep(nanoseconds: 100_000_000)  // 100ms
            }
        }
    }

    private func updateStatistics() async {
        await MainActor.run {
            // Update latency estimate
            latency = 0.023  // ~23ms typical for audio streaming

            // Update buffer level
            if let mic = microphoneSource {
                Task {
                    let stats = await mic.bufferStatistics
                    await MainActor.run {
                        bufferLevel = stats?.fillLevel ?? 0
                    }
                }
            }

            // Update realtime factor
            realtimeFactor = 1.0
        }
    }
}

// MARK: - Transcription Delegate Handler

/// Handler for transcription results (bridges actor to view model).
final class TranscriptionHandler: StreamingTranscriberDelegate, @unchecked Sendable {
    private weak var viewModel: AudioViewModel?

    init(viewModel: AudioViewModel) {
        self.viewModel = viewModel
    }

    func transcriber(_ transcriber: StreamingTranscriber, didReceive result: StreamingTranscriptionResult) async {
        await MainActor.run {
            viewModel?.updateTranscription(result.text)
        }
    }

    func transcriber(_ transcriber: StreamingTranscriber, didEncounterError error: Error) async {
        await MainActor.run {
            viewModel?.updateTranscription("Error: \(error.localizedDescription)")
        }
    }
}
