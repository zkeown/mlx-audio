// TranscribeView.swift
// Speech transcription UI using Whisper.

import SwiftUI

struct TranscribeView: View {
    @EnvironmentObject var modelManager: ModelManager
    @StateObject private var viewModel = TranscribeViewModel()

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 20) {
                    // File picker
                    filePickerSection

                    // Variant selector
                    variantSelector

                    // Input waveform
                    if !viewModel.inputWaveform.isEmpty {
                        inputWaveformSection
                    }

                    // Transcribe button
                    transcribeButton

                    // Processing indicator
                    if viewModel.isProcessing {
                        ProgressView("Transcribing...")
                            .padding()
                    }

                    // Language detection
                    if let language = viewModel.detectedLanguage {
                        languageSection(language)
                    }

                    // Transcription results
                    if !viewModel.segments.isEmpty {
                        transcriptionSection
                    }

                    // Error display
                    if let error = viewModel.errorMessage {
                        errorSection(error)
                    }
                }
                .padding()
            }
            .navigationTitle("Transcribe")
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

            HStack(spacing: 12) {
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
                    label: "Select file"
                )

                Button {
                    // Record functionality would go here
                } label: {
                    HStack {
                        Image(systemName: "mic.fill")
                        Text("Record")
                    }
                    .padding()
                    .background(Color.systemGray5)
                    .cornerRadius(10)
                }
            }
        }
    }

    private var variantSelector: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Model")
                .font(.headline)

            Picker("Variant", selection: $viewModel.selectedVariant) {
                ForEach(viewModel.availableVariants, id: \.id) { variant in
                    Text(variant.name).tag(variant.id)
                }
            }
            .pickerStyle(.menu)
            .disabled(viewModel.isProcessing)
        }
    }

    private var inputWaveformSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Audio Preview")
                .font(.headline)

            WaveformView(samples: viewModel.inputWaveform)
                .frame(height: 60)
        }
    }

    private var transcribeButton: some View {
        Button {
            Task {
                await viewModel.transcribe()
            }
        } label: {
            HStack {
                Image(systemName: "text.bubble")
                Text("Transcribe")
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

    private func languageSection(_ language: String) -> some View {
        HStack {
            Text("Detected Language:")
                .foregroundColor(.secondary)

            Text(language)
                .fontWeight(.medium)

            Text(String(format: "(%.0f%%)", viewModel.languageConfidence * 100))
                .foregroundColor(.secondary)
        }
        .padding()
        .frame(maxWidth: .infinity)
        .background(Color.systemGray6)
        .cornerRadius(12)
    }

    private var transcriptionSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Transcription")
                .font(.headline)

            // Timestamped segments
            VStack(alignment: .leading, spacing: 8) {
                ForEach(viewModel.segments) { segment in
                    HStack(alignment: .top, spacing: 12) {
                        Text(segment.formattedTime)
                            .font(.caption)
                            .foregroundColor(.secondary)
                            .frame(width: 40, alignment: .leading)

                        Text(segment.text)
                            .font(.body)
                    }
                }
            }
            .padding()
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Color.systemBackground)
            .cornerRadius(12)
            .overlay(
                RoundedRectangle(cornerRadius: 12)
                    .stroke(Color.gray.opacity(0.3), lineWidth: 1)
            )

            // Action buttons
            HStack(spacing: 12) {
                Button {
                    viewModel.copyText()
                } label: {
                    Label("Copy Text", systemImage: "doc.on.doc")
                }
                .buttonStyle(.bordered)

                Button {
                    // Export SRT
                } label: {
                    Label("Export SRT", systemImage: "square.and.arrow.up")
                }
                .buttonStyle(.bordered)
            }
        }
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
    TranscribeView()
        .environmentObject(ModelManager())
}
