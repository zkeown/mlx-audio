// ContentView.swift
// Main view for the streaming demo app.

import SwiftUI
import MLXAudioStreaming
import MLX

struct ContentView: View {
    @StateObject private var viewModel = AudioViewModel()

    var body: some View {
        NavigationStack {
            VStack(spacing: 20) {
                // Status indicator
                HStack {
                    Circle()
                        .fill(viewModel.isRunning ? Color.green : Color.gray)
                        .frame(width: 12, height: 12)
                    Text(viewModel.statusText)
                        .font(.headline)
                }
                .padding()

                // Waveform visualization
                WaveformView(samples: viewModel.waveformData)
                    .frame(height: 100)
                    .padding(.horizontal)

                // Spectrogram visualization
                SpectrogramView(data: viewModel.spectrogramData)
                    .frame(height: 150)
                    .padding(.horizontal)

                // Transcription display
                if viewModel.showTranscription {
                    TranscriptionView(text: viewModel.transcriptionText)
                        .frame(height: 100)
                        .padding(.horizontal)
                }

                // Statistics
                StatisticsView(
                    latency: viewModel.latency,
                    bufferLevel: viewModel.bufferLevel,
                    realtimeFactor: viewModel.realtimeFactor
                )
                .padding(.horizontal)

                Spacer()

                // Controls
                ControlsView(
                    isRunning: viewModel.isRunning,
                    mode: $viewModel.mode,
                    onStart: { await viewModel.start() },
                    onStop: { await viewModel.stop() }
                )
                .padding()
            }
            .navigationTitle("MLX Audio Streaming")
            .navigationBarTitleDisplayMode(.inline)
        }
        .onDisappear {
            Task {
                await viewModel.stop()
            }
        }
    }
}

// MARK: - Waveform View

struct WaveformView: View {
    let samples: [Float]

    var body: some View {
        GeometryReader { geometry in
            Canvas { context, size in
                guard !samples.isEmpty else { return }

                let midY = size.height / 2
                let stepX = size.width / CGFloat(samples.count)

                var path = Path()
                path.move(to: CGPoint(x: 0, y: midY))

                for (index, sample) in samples.enumerated() {
                    let x = CGFloat(index) * stepX
                    let y = midY - CGFloat(sample) * midY * 0.9
                    path.addLine(to: CGPoint(x: x, y: y))
                }

                context.stroke(path, with: .color(.blue), lineWidth: 1)
            }
        }
        .background(Color.black.opacity(0.1))
        .cornerRadius(8)
    }
}

// MARK: - Spectrogram View

struct SpectrogramView: View {
    let data: [[Float]]

    var body: some View {
        GeometryReader { geometry in
            Canvas { context, size in
                guard !data.isEmpty && !data[0].isEmpty else { return }

                let numFrames = data.count
                let numBins = data[0].count
                let cellWidth = size.width / CGFloat(numFrames)
                let cellHeight = size.height / CGFloat(numBins)

                for (frameIndex, frame) in data.enumerated() {
                    for (binIndex, value) in frame.enumerated() {
                        let x = CGFloat(frameIndex) * cellWidth
                        let y = size.height - CGFloat(binIndex + 1) * cellHeight

                        // Map value to color (assuming dB scale, -80 to 0)
                        let normalized = max(0, min(1, (value + 80) / 80))
                        let color = spectrogramColor(normalized)

                        let rect = CGRect(x: x, y: y, width: cellWidth + 1, height: cellHeight + 1)
                        context.fill(Path(rect), with: .color(color))
                    }
                }
            }
        }
        .background(Color.black)
        .cornerRadius(8)
    }

    private func spectrogramColor(_ value: Float) -> Color {
        // Viridis-like colormap
        let v = Double(value)
        let r = min(1, max(0, 0.267 + v * 0.329 + v * v * 0.404))
        let g = min(1, max(0, 0.004 + v * 0.873 - v * v * 0.377))
        let b = min(1, max(0, 0.329 + v * 0.294 - v * v * 0.623))
        return Color(red: r, green: g, blue: b)
    }
}

// MARK: - Transcription View

struct TranscriptionView: View {
    let text: String

    var body: some View {
        ScrollView {
            Text(text.isEmpty ? "Listening..." : text)
                .font(.body)
                .foregroundColor(text.isEmpty ? .secondary : .primary)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding()
        }
        .background(Color(.systemBackground))
        .cornerRadius(8)
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(Color.gray.opacity(0.3), lineWidth: 1)
        )
    }
}

// MARK: - Statistics View

struct StatisticsView: View {
    let latency: TimeInterval
    let bufferLevel: Double
    let realtimeFactor: Double

    var body: some View {
        HStack(spacing: 20) {
            StatItem(label: "Latency", value: String(format: "%.1f ms", latency * 1000))
            StatItem(label: "Buffer", value: String(format: "%.0f%%", bufferLevel * 100))
            StatItem(label: "RT Factor", value: String(format: "%.1fx", realtimeFactor))
        }
        .font(.caption)
    }
}

struct StatItem: View {
    let label: String
    let value: String

    var body: some View {
        VStack(spacing: 2) {
            Text(label)
                .foregroundColor(.secondary)
            Text(value)
                .fontWeight(.medium)
        }
    }
}

// MARK: - Controls View

struct ControlsView: View {
    let isRunning: Bool
    @Binding var mode: StreamingMode
    let onStart: () async -> Void
    let onStop: () async -> Void

    var body: some View {
        VStack(spacing: 16) {
            // Mode picker
            Picker("Mode", selection: $mode) {
                Text("Passthrough").tag(StreamingMode.passthrough)
                Text("Spectrogram").tag(StreamingMode.spectrogram)
                Text("Transcribe").tag(StreamingMode.transcribe)
            }
            .pickerStyle(.segmented)
            .disabled(isRunning)

            // Start/Stop button
            Button(action: {
                Task {
                    if isRunning {
                        await onStop()
                    } else {
                        await onStart()
                    }
                }
            }) {
                HStack {
                    Image(systemName: isRunning ? "stop.fill" : "play.fill")
                    Text(isRunning ? "Stop" : "Start")
                }
                .font(.headline)
                .foregroundColor(.white)
                .frame(maxWidth: .infinity)
                .padding()
                .background(isRunning ? Color.red : Color.blue)
                .cornerRadius(12)
            }
        }
    }
}

// MARK: - Preview

#Preview {
    ContentView()
}
