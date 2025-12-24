// SeparateView.swift
// Source separation UI using HTDemucs.

import SwiftUI

struct SeparateView: View {
    @EnvironmentObject var modelManager: ModelManager
    @StateObject private var viewModel = SeparateViewModel()
    @State private var selectedStem: String?

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 20) {
                    // File picker section
                    filePickerSection

                    // Variant selector
                    variantSelector

                    // Input waveform
                    if !viewModel.inputWaveform.isEmpty {
                        inputWaveformSection
                    }

                    // Separate button
                    separateButton

                    // Progress indicator
                    if viewModel.isProcessing {
                        progressSection
                    }

                    // Results
                    if let result = viewModel.separationResult {
                        resultsSection(result)
                    }

                    // Error display
                    if let error = viewModel.errorMessage {
                        errorSection(error)
                    }
                }
                .padding()
            }
            .navigationTitle("Separate")
            .onAppear {
                viewModel.setModelManager(modelManager)
            }
            .downloadOverlay(
                isDownloading: viewModel.isDownloading,
                modelName: viewModel.selectedVariant,
                progress: modelManager.downloadProgress[viewModel.selectedVariant]
            )
        }
    }

    // MARK: - Sections

    private var filePickerSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Input Audio")
                .font(.headline)

            AudioFilePicker(
                selectedURL: Binding(
                    get: { viewModel.inputURL },
                    set: { url in
                        if let url = url {
                            Task {
                                await viewModel.loadAudio(from: url)
                            }
                        }
                    }
                ),
                label: "Select audio file to separate"
            )
        }
    }

    private var variantSelector: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Model Variant")
                .font(.headline)

            Picker("Variant", selection: $viewModel.selectedVariant) {
                Text("HTDemucs").tag("htdemucs")
                Text("HTDemucs Fine-tuned").tag("htdemucs_ft")
                Text("HTDemucs 6-Stem").tag("htdemucs_6s")
            }
            .pickerStyle(.segmented)
            .disabled(viewModel.isProcessing)
        }
    }

    private var inputWaveformSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Input Waveform")
                .font(.headline)

            WaveformView(samples: viewModel.inputWaveform)
                .frame(height: 80)
        }
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
            .background(viewModel.inputURL != nil ? Color.accentColor : Color.gray)
            .foregroundColor(.white)
            .cornerRadius(12)
        }
        .disabled(viewModel.inputURL == nil || viewModel.isProcessing)
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

    private func resultsSection(_ result: SeparationResult) -> some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Separated Stems")
                .font(.headline)

            // Stem waveforms
            StemWaveformsView(
                stems: [
                    ("Drums", result.drums, .orange),
                    ("Bass", result.bass, .purple),
                    ("Other", result.other, .green),
                    ("Vocals", result.vocals, .blue),
                ],
                selectedStem: selectedStem,
                onStemTap: { stem in
                    selectedStem = stem
                }
            )

            // Export button
            Button {
                // Export functionality
            } label: {
                HStack {
                    Image(systemName: "square.and.arrow.up")
                    Text("Export All Stems")
                }
                .frame(maxWidth: .infinity)
                .padding()
                .background(Color.systemGray5)
                .cornerRadius(12)
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
    SeparateView()
        .environmentObject(ModelManager())
}
