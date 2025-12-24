// EmbedView.swift
// Audio embedding and classification UI using CLAP.

import SwiftUI

struct EmbedView: View {
    @EnvironmentObject var modelManager: ModelManager
    @StateObject private var viewModel = EmbedViewModel()

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 20) {
                    // File picker
                    filePickerSection

                    // Waveform
                    if !viewModel.inputWaveform.isEmpty {
                        WaveformView(samples: viewModel.inputWaveform)
                            .frame(height: 60)
                    }

                    // Similarity search
                    similaritySection

                    Divider()

                    // Classification
                    classificationSection

                    // Error display
                    if let error = viewModel.errorMessage {
                        errorSection(error)
                    }
                }
                .padding()
            }
            .navigationTitle("Embed & Classify")
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
                )
            )
        }
    }

    private var similaritySection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Text Search")
                .font(.headline)

            HStack {
                TextField("Describe the sound...", text: $viewModel.searchQuery)
                    .textFieldStyle(.roundedBorder)

                Button {
                    Task {
                        await viewModel.computeSimilarity()
                    }
                } label: {
                    Image(systemName: "magnifyingglass")
                        .padding(10)
                        .background(Color.accentColor)
                        .foregroundColor(.white)
                        .cornerRadius(8)
                }
                .disabled(viewModel.inputURL == nil || viewModel.searchQuery.isEmpty || viewModel.isProcessing)
            }

            // Similarity result
            if let score = viewModel.similarityScore {
                HStack {
                    Text("Similarity:")
                        .foregroundColor(.secondary)

                    ProgressView(value: Double(score))
                        .progressViewStyle(.linear)

                    Text(String(format: "%.0f%%", score * 100))
                        .fontWeight(.bold)
                        .foregroundColor(score > 0.7 ? .green : (score > 0.4 ? .orange : .red))
                }
                .padding()
                .background(Color.systemGray6)
                .cornerRadius(12)
            }
        }
    }

    private var classificationSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Zero-Shot Classification")
                .font(.headline)

            // Label presets
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 8) {
                    ForEach(viewModel.presetLabelSets, id: \.name) { preset in
                        Button {
                            viewModel.usePreset(preset.labels)
                        } label: {
                            Text(preset.name)
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

            // Current labels
            FlowLayout(spacing: 8) {
                ForEach(viewModel.classificationLabels, id: \.self) { label in
                    HStack(spacing: 4) {
                        Text(label)
                            .font(.caption)

                        Button {
                            viewModel.removeLabel(label)
                        } label: {
                            Image(systemName: "xmark.circle.fill")
                                .font(.caption)
                        }
                    }
                    .padding(.horizontal, 10)
                    .padding(.vertical, 6)
                    .background(Color.accentColor.opacity(0.2))
                    .cornerRadius(16)
                }
            }

            // Add custom label
            HStack {
                TextField("Add label...", text: $viewModel.newLabel)
                    .textFieldStyle(.roundedBorder)

                Button {
                    viewModel.addLabel()
                } label: {
                    Image(systemName: "plus.circle.fill")
                }
                .disabled(viewModel.newLabel.isEmpty)
            }

            // Classify button
            Button {
                Task {
                    await viewModel.classify()
                }
            } label: {
                HStack {
                    if viewModel.isProcessing {
                        ProgressView()
                            .progressViewStyle(.circular)
                            .scaleEffect(0.8)
                    } else {
                        Image(systemName: "tag")
                    }
                    Text("Classify")
                }
                .frame(maxWidth: .infinity)
                .padding()
                .background(viewModel.inputURL != nil ? Color.accentColor : Color.gray)
                .foregroundColor(.white)
                .cornerRadius(12)
            }
            .disabled(viewModel.inputURL == nil || viewModel.classificationLabels.isEmpty || viewModel.isProcessing)

            // Results
            if !viewModel.classificationResults.isEmpty {
                VStack(alignment: .leading, spacing: 8) {
                    ForEach(viewModel.classificationResults) { result in
                        HStack {
                            Text(result.label)
                                .fontWeight(result == viewModel.classificationResults.first ? .bold : .regular)

                            Spacer()

                            ProgressView(value: Double(result.probability))
                                .progressViewStyle(.linear)
                                .frame(width: 100)

                            Text(result.formattedProbability)
                                .font(.caption)
                                .foregroundColor(.secondary)
                                .frame(width: 40, alignment: .trailing)
                        }
                    }
                }
                .padding()
                .background(Color.systemGray6)
                .cornerRadius(12)
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

/// Simple flow layout for labels.
struct FlowLayout: Layout {
    var spacing: CGFloat = 8

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let sizes = subviews.map { $0.sizeThatFits(.unspecified) }
        return layout(sizes: sizes, containerWidth: proposal.width ?? .infinity).size
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        let sizes = subviews.map { $0.sizeThatFits(.unspecified) }
        let offsets = layout(sizes: sizes, containerWidth: bounds.width).offsets

        for (subview, offset) in zip(subviews, offsets) {
            subview.place(at: CGPoint(x: bounds.minX + offset.x, y: bounds.minY + offset.y), proposal: .unspecified)
        }
    }

    private func layout(sizes: [CGSize], containerWidth: CGFloat) -> (size: CGSize, offsets: [CGPoint]) {
        var offsets: [CGPoint] = []
        var currentX: CGFloat = 0
        var currentY: CGFloat = 0
        var lineHeight: CGFloat = 0
        var maxWidth: CGFloat = 0

        for size in sizes {
            if currentX + size.width > containerWidth && currentX > 0 {
                currentX = 0
                currentY += lineHeight + spacing
                lineHeight = 0
            }

            offsets.append(CGPoint(x: currentX, y: currentY))
            currentX += size.width + spacing
            lineHeight = max(lineHeight, size.height)
            maxWidth = max(maxWidth, currentX)
        }

        return (CGSize(width: maxWidth, height: currentY + lineHeight), offsets)
    }
}

#Preview {
    EmbedView()
        .environmentObject(ModelManager())
}
