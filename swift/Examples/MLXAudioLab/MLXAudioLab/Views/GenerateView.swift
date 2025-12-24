// GenerateView.swift
// Music generation UI using MusicGen.

import SwiftUI

struct GenerateView: View {
    @EnvironmentObject var modelManager: ModelManager
    @StateObject private var viewModel = GenerateViewModel()

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 20) {
                    // Prompt input
                    promptSection

                    // Sample prompts
                    samplePromptsSection

                    // Duration slider
                    durationSection

                    // Model selector
                    modelSelector

                    // Generate button
                    generateButton

                    // Progress
                    if viewModel.isGenerating {
                        progressSection
                    }

                    // Results
                    if !viewModel.generatedWaveform.isEmpty {
                        resultsSection
                    }

                    // Error display
                    if let error = viewModel.errorMessage {
                        errorSection(error)
                    }
                }
                .padding()
            }
            .navigationTitle("Generate")
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

    private var promptSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Prompt")
                .font(.headline)

            TextField("Describe the music you want to generate...", text: $viewModel.prompt, axis: .vertical)
                .textFieldStyle(.roundedBorder)
                .lineLimit(3...6)
                .disabled(viewModel.isGenerating)
        }
    }

    private var samplePromptsSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Sample Prompts")
                .font(.subheadline)
                .foregroundColor(.secondary)

            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 8) {
                    ForEach(viewModel.samplePrompts, id: \.self) { sample in
                        Button {
                            viewModel.useSamplePrompt(sample)
                        } label: {
                            Text(sample)
                                .font(.caption)
                                .padding(.horizontal, 12)
                                .padding(.vertical, 6)
                                .background(Color.systemGray5)
                                .cornerRadius(16)
                        }
                        .buttonStyle(.plain)
                    }
                }
            }
        }
    }

    private var durationSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Duration")
                    .font(.headline)

                Spacer()

                Text("\(Int(viewModel.duration))s")
                    .font(.headline)
                    .foregroundColor(.accentColor)
            }

            Slider(
                value: $viewModel.duration,
                in: viewModel.durationRange,
                step: 1
            )
            .disabled(viewModel.isGenerating)

            HStack {
                Text("5s")
                    .font(.caption)
                    .foregroundColor(.secondary)

                Spacer()

                Text("30s")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
    }

    private var modelSelector: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Model")
                .font(.headline)

            Picker("Model", selection: $viewModel.selectedVariant) {
                ForEach(viewModel.availableVariants, id: \.id) { variant in
                    Text(variant.name).tag(variant.id)
                }
            }
            .pickerStyle(.segmented)
            .disabled(viewModel.isGenerating)
        }
    }

    private var generateButton: some View {
        Button {
            Task {
                await viewModel.generate()
            }
        } label: {
            HStack {
                Image(systemName: "music.note")
                Text("Generate")
            }
            .font(.headline)
            .frame(maxWidth: .infinity)
            .padding()
            .background(!viewModel.prompt.isEmpty ? Color.accentColor : Color.gray)
            .foregroundColor(.white)
            .cornerRadius(12)
        }
        .disabled(viewModel.prompt.isEmpty || viewModel.isGenerating)
    }

    private var progressSection: some View {
        VStack(spacing: 12) {
            ProgressView(value: Double(viewModel.generationProgress))
                .progressViewStyle(.linear)

            Text("Generating... \(Int(viewModel.generationProgress * 100))%")
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .padding()
        .background(Color.systemGray6)
        .cornerRadius(12)
    }

    private var resultsSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Generated Music")
                .font(.headline)

            WaveformView(samples: viewModel.generatedWaveform, color: .green)
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
                    viewModel.clear()
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
    GenerateView()
        .environmentObject(ModelManager())
}
