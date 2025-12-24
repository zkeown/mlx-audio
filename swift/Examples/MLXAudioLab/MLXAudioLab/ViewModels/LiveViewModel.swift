// LiveViewModel.swift
// ViewModel for real-time audio processing.

import Foundation
import MLX
import MLXAudioModels
import MLXAudioStreaming

/// Stem type for live separation.
enum StemType: String, CaseIterable, Identifiable {
    case drums = "Drums"
    case bass = "Bass"
    case other = "Other"
    case vocals = "Vocals"

    var id: String { rawValue }

    var index: Int {
        switch self {
        case .drums: return 0
        case .bass: return 1
        case .other: return 2
        case .vocals: return 3
        }
    }

    var color: Color {
        switch self {
        case .drums: return .orange
        case .bass: return .purple
        case .other: return .green
        case .vocals: return .blue
        }
    }
}

import SwiftUI

/// ViewModel for the Live tab.
@MainActor
class LiveViewModel: ObservableObject {
    // MARK: - Published State

    @Published var isRunning = false
    @Published var selectedStem: StemType = .vocals
    @Published var latencyMs: Double = 0
    @Published var waveformData: [Float] = []
    @Published var outputWaveformData: [Float] = []
    @Published var errorMessage: String?
    @Published var isDownloading = false

    // Statistics
    @Published var bufferLevel: Double = 0
    @Published var processingTimeMs: Double = 0

    // MARK: - Private

    private weak var modelManager: ModelManager?
    private var micSource: MicrophoneSource?
    private var speakerSink: SpeakerSink?
    private var htdemucs: HTDemucs?
    private var processingTask: Task<Void, Never>?

    // Processing settings
    private let sampleRate = 44100
    private let chunkSize = 4096

    // MARK: - Initialization

    init(modelManager: ModelManager? = nil) {
        self.modelManager = modelManager
    }

    func setModelManager(_ manager: ModelManager) {
        self.modelManager = manager
    }

    // MARK: - Actions

    /// Start live processing.
    func start() async {
        guard let modelManager = modelManager else {
            errorMessage = "Model manager not available"
            return
        }

        errorMessage = nil

        do {
            // Load lightweight HTDemucs model
            isDownloading = true
            htdemucs = try await modelManager.loadHTDemucs(variant: "htdemucs")
            isDownloading = false

            // Setup audio I/O
            let micConfig = MicrophoneConfiguration(
                sampleRate: sampleRate,
                channels: 2,
                bufferSize: chunkSize,
                bufferDurationSeconds: 2.0
            )
            micSource = MicrophoneSource(configuration: micConfig)

            let speakerConfig = SpeakerConfiguration(
                sampleRate: sampleRate,
                channels: 2,
                scheduledBufferCount: 3,
                bufferFrameCount: chunkSize,
                bufferDurationSeconds: 2.0
            )
            speakerSink = SpeakerSink(configuration: speakerConfig)

            // Start audio I/O
            try await micSource?.start()
            try await speakerSink?.start()

            isRunning = true

            // Start processing loop
            processingTask = Task {
                await runProcessingLoop()
            }

        } catch {
            errorMessage = "Failed to start: \(error.localizedDescription)"
            isRunning = false
        }
    }

    /// Stop live processing.
    func stop() async {
        processingTask?.cancel()
        processingTask = nil

        do {
            try await micSource?.stop()
            try await speakerSink?.stop()
        } catch {
            // Ignore stop errors
        }

        micSource = nil
        speakerSink = nil
        isRunning = false
        waveformData = []
        outputWaveformData = []
    }

    // MARK: - Processing Loop

    private func runProcessingLoop() async {
        guard micSource != nil,
              speakerSink != nil,
              htdemucs != nil else {
            return
        }

        // Simulate processing loop for demo
        // Note: Real implementation would use proper MLXArray processing
        // with appropriate actor isolation and Sendable considerations
        while !Task.isCancelled {
            do {
                try await Task.sleep(nanoseconds: 100_000_000)  // 100ms

                let startTime = CFAbsoluteTimeGetCurrent()

                // Simulate waveform data
                let samples = (0..<1024).map { _ in Float.random(in: -0.5...0.5) }
                waveformData = samples
                outputWaveformData = samples.map { $0 * 0.8 }

                // Simulate processing time
                let processingTime = (CFAbsoluteTimeGetCurrent() - startTime) * 1000
                processingTimeMs = processingTime
                latencyMs = Double(chunkSize) / Double(sampleRate) * 1000 + processingTime
                bufferLevel = Double.random(in: 0.3...0.7)

            } catch {
                if !Task.isCancelled {
                    errorMessage = "Processing error: \(error.localizedDescription)"
                }
                break
            }
        }
    }
}
