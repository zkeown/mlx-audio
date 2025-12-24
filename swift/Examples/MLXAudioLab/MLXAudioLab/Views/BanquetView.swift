// BanquetView.swift
// Query-based source separation UI using Banquet.

import SwiftUI

struct BanquetView: View {
    @EnvironmentObject var modelManager: ModelManager
    @StateObject private var viewModel = BanquetViewModel()

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 20) {
                    // Info header
                    infoSection

                    // Query audio section
                    audioSection(
                        title: "Query (Reference)",
                        subtitle: "Select an audio clip containing the sound you want to extract",
                        url: viewModel.queryAudioURL,
                        waveform: viewModel.queryWaveform,
                        onSelect: { url in
                            Task {
                                await viewModel.loadQueryAudio(from: url)
                            }
                        }
                    )

                    // Swap button
                    Button {
                        viewModel.swapAudio()
                    } label: {
                        Image(systemName: "arrow.up.arrow.down")
                            .font(.title2)
                            .foregroundColor(.accentColor)
                    }

                    // Mixture audio section
                    audioSection(
                        title: "Mixture",
                        subtitle: "Select the audio file to separate",
                        url: viewModel.mixtureAudioURL,
                        waveform: viewModel.mixtureWaveform,
                        onSelect: { url in
                            Task {
                                await viewModel.loadMixtureAudio(from: url)
                            }
                        }
                    )

                    // Separate button
                    separateButton

                    // Progress
                    if viewModel.isProcessing {
                        progressSection
                    }

                    // Results
                    if !viewModel.separatedWaveform.isEmpty {
                        resultsSection
                    }

                    // Error display
                    if let error = viewModel.errorMessage {
                        errorSection(error)
                    }
                }
                .padding()
            }
            .navigationTitle("Banquet")
            .onAppear {
                viewModel.setModelManager(modelManager)
            }
            .downloadOverlay(
                isDownloading: viewModel.isDownloading,
                modelName: "Banquet",
                progress: modelManager.downloadProgress["banquet"]
            )
        }
    }

    // MARK: - Sections

    private var infoSection: some View {
        VStack(spacing: 8) {
            Image(systemName: "arrow.triangle.branch")
                .font(.largeTitle)
                .foregroundColor(.accentColor)

            Text("Query-Based Separation")
                .font(.headline)

            Text("Provide a reference audio clip and Banquet will extract similar sounds from the mixture.")
                .font(.caption)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
        }
        .padding()
        .frame(maxWidth: .infinity)
        .background(Color.systemGray6)
        .cornerRadius(12)
    }

    private func audioSection(
        title: String,
        subtitle: String,
        url: URL?,
        waveform: [Float],
        onSelect: @escaping (URL) -> Void
    ) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            VStack(alignment: .leading, spacing: 4) {
                Text(title)
                    .font(.headline)

                Text(subtitle)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            AudioFilePicker(
                selectedURL: Binding(
                    get: { url },
                    set: { newURL in
                        if let newURL = newURL {
                            onSelect(newURL)
                        }
                    }
                )
            )

            if !waveform.isEmpty {
                WaveformView(samples: waveform, color: .accentColor)
                    .frame(height: 50)
            }
        }
        .padding()
        .background(Color.systemGray6)
        .cornerRadius(12)
    }

    private var separateButton: some View {
        Button {
            Task {
                await viewModel.separate()
            }
        } label: {
            HStack {
                Image(systemName: "waveform.path.ecg")
                Text("Separate")
            }
            .font(.headline)
            .frame(maxWidth: .infinity)
            .padding()
            .background(canSeparate ? Color.accentColor : Color.gray)
            .foregroundColor(.white)
            .cornerRadius(12)
        }
        .disabled(!canSeparate || viewModel.isProcessing)
    }

    private var canSeparate: Bool {
        viewModel.queryAudioURL != nil && viewModel.mixtureAudioURL != nil
    }

    private var progressSection: some View {
        VStack(spacing: 12) {
            ProgressView(value: Double(viewModel.progress))
                .progressViewStyle(.linear)

            Text("Separating... \(Int(viewModel.progress * 100))%")
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .padding()
        .background(Color.systemGray6)
        .cornerRadius(12)
    }

    private var resultsSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Extracted Audio")
                .font(.headline)

            WaveformView(samples: viewModel.separatedWaveform, color: .green)
                .frame(height: 80)

            // Playback controls
            AudioPlayerView(url: nil)

            // Action buttons
            HStack(spacing: 12) {
                Button {
                    // Save functionality
                } label: {
                    Label("Save", systemImage: "square.and.arrow.down")
                }
                .buttonStyle(.bordered)

                Button {
                    viewModel.clearResults()
                } label: {
                    Label("Clear", systemImage: "trash")
                }
                .buttonStyle(.bordered)
                .tint(.red)
            }
        }
        .padding()
        .background(Color.systemGray6)
        .cornerRadius(12)
    }

    private func errorSection(_ error: String) -> some View {
        HStack {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundColor(.red)

            Text(error)
                .font(.callout)
                .foregroundColor(.red)
        }
        .padding()
        .background(Color.red.opacity(0.1))
        .cornerRadius(12)
    }
}

#Preview {
    BanquetView()
        .environmentObject(ModelManager())
}
