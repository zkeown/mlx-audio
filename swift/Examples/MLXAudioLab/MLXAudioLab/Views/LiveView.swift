// LiveView.swift
// Real-time audio processing UI.

import SwiftUI

struct LiveView: View {
    @EnvironmentObject var modelManager: ModelManager
    @StateObject private var viewModel = LiveViewModel()

    var body: some View {
        NavigationStack {
            VStack(spacing: 20) {
                // Header
                headerSection

                Spacer()

                // Input waveform
                waveformSection("Input", samples: viewModel.waveformData, color: .blue)

                // Output waveform
                waveformSection("Output (\(viewModel.selectedStem.rawValue))", samples: viewModel.outputWaveformData, color: viewModel.selectedStem.color)

                Spacer()

                // Stem selector
                stemSelector

                // Statistics
                statisticsSection

                // Controls
                controlsSection
            }
            .padding()
            .navigationTitle("Live Processing")
            .onAppear {
                viewModel.setModelManager(modelManager)
            }
            .onDisappear {
                Task {
                    await viewModel.stop()
                }
            }
            .downloadOverlay(
                isDownloading: viewModel.isDownloading,
                modelName: "HTDemucs",
                progress: modelManager.downloadProgress["htdemucs"]
            )
        }
    }

    // MARK: - Sections

    private var headerSection: some View {
        VStack(spacing: 8) {
            HStack {
                Circle()
                    .fill(viewModel.isRunning ? Color.green : Color.gray)
                    .frame(width: 12, height: 12)

                Text(viewModel.isRunning ? "Live" : "Stopped")
                    .font(.headline)
            }

            Text("Real-time stem isolation")
                .font(.caption)
                .foregroundColor(.secondary)
        }
    }

    private func waveformSection(_ title: String, samples: [Float], color: Color) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
                .font(.caption)
                .foregroundColor(.secondary)

            WaveformView(
                samples: samples.isEmpty ? Array(repeating: 0, count: 100) : samples,
                color: color
            )
            .frame(height: 80)
        }
    }

    private var stemSelector: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Isolate Stem")
                .font(.headline)

            HStack(spacing: 12) {
                ForEach(StemType.allCases) { stem in
                    Button {
                        viewModel.selectedStem = stem
                    } label: {
                        Text(stem.rawValue)
                            .font(.caption)
                            .fontWeight(viewModel.selectedStem == stem ? .bold : .regular)
                            .padding(.horizontal, 16)
                            .padding(.vertical, 10)
                            .background(viewModel.selectedStem == stem ? stem.color : stem.color.opacity(0.2))
                            .foregroundColor(viewModel.selectedStem == stem ? .white : stem.color)
                            .cornerRadius(20)
                    }
                }
            }
        }
    }

    private var statisticsSection: some View {
        HStack(spacing: 30) {
            StatItem(label: "Latency", value: String(format: "%.0f ms", viewModel.latencyMs))
            StatItem(label: "Processing", value: String(format: "%.1f ms", viewModel.processingTimeMs))
            StatItem(label: "Buffer", value: String(format: "%.0f%%", viewModel.bufferLevel * 100))
        }
        .padding()
        .background(Color.systemGray6)
        .cornerRadius(12)
    }

    private var controlsSection: some View {
        VStack(spacing: 12) {
            // Error display
            if let error = viewModel.errorMessage {
                Text(error)
                    .font(.caption)
                    .foregroundColor(.red)
            }

            // Start/Stop button
            Button {
                Task {
                    if viewModel.isRunning {
                        await viewModel.stop()
                    } else {
                        await viewModel.start()
                    }
                }
            } label: {
                HStack {
                    Image(systemName: viewModel.isRunning ? "stop.fill" : "play.fill")
                    Text(viewModel.isRunning ? "Stop" : "Start")
                }
                .font(.headline)
                .frame(maxWidth: .infinity)
                .padding()
                .background(viewModel.isRunning ? Color.red : Color.accentColor)
                .foregroundColor(.white)
                .cornerRadius(12)
            }
        }
    }
}

/// Statistics display item.
private struct StatItem: View {
    let label: String
    let value: String

    var body: some View {
        VStack(spacing: 2) {
            Text(label)
                .font(.caption)
                .foregroundColor(.secondary)

            Text(value)
                .font(.callout)
                .fontWeight(.medium)
                .monospacedDigit()
        }
    }
}

#Preview {
    LiveView()
        .environmentObject(ModelManager())
}
