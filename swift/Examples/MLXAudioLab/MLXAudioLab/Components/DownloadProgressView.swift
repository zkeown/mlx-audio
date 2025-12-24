// DownloadProgressView.swift
// Model download progress indicator.

import SwiftUI

/// Displays model download progress.
struct DownloadProgressView: View {
    let modelName: String
    let progress: DownloadProgress?
    var onCancel: (() -> Void)?

    var body: some View {
        VStack(spacing: 16) {
            // Icon
            Image(systemName: "arrow.down.circle")
                .font(.system(size: 40))
                .foregroundColor(.accentColor)
                .symbolEffect(.pulse, options: .repeating)

            // Title
            Text("Downloading \(modelName)")
                .font(.headline)

            if let progress = progress {
                // Progress bar
                ProgressView(value: progress.progress)
                    .progressViewStyle(.linear)

                // Details
                HStack {
                    Text(progress.currentFile)
                        .font(.caption)
                        .foregroundColor(.secondary)

                    Spacer()

                    Text(progress.formattedBytes)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }

                // Percentage
                Text(progress.formattedProgress)
                    .font(.title2)
                    .fontWeight(.semibold)
                    .monospacedDigit()
            } else {
                ProgressView()
                    .progressViewStyle(.circular)

                Text("Preparing download...")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            // Cancel button
            if let onCancel = onCancel {
                Button("Cancel", role: .cancel, action: onCancel)
                    .buttonStyle(.bordered)
            }
        }
        .padding(24)
        .background(Color.systemBackground)
        .cornerRadius(16)
        .shadow(radius: 10)
    }
}

/// Overlay for showing download progress.
struct DownloadOverlay: ViewModifier {
    let isDownloading: Bool
    let modelName: String
    let progress: DownloadProgress?
    var onCancel: (() -> Void)?

    func body(content: Content) -> some View {
        ZStack {
            content
                .disabled(isDownloading)
                .blur(radius: isDownloading ? 3 : 0)

            if isDownloading {
                Color.black.opacity(0.4)
                    .ignoresSafeArea()

                DownloadProgressView(
                    modelName: modelName,
                    progress: progress,
                    onCancel: onCancel
                )
                .transition(.scale.combined(with: .opacity))
            }
        }
        .animation(.easeInOut, value: isDownloading)
    }
}

extension View {
    /// Add a download progress overlay.
    func downloadOverlay(
        isDownloading: Bool,
        modelName: String,
        progress: DownloadProgress?,
        onCancel: (() -> Void)? = nil
    ) -> some View {
        modifier(DownloadOverlay(
            isDownloading: isDownloading,
            modelName: modelName,
            progress: progress,
            onCancel: onCancel
        ))
    }
}

/// Loading indicator for model operations.
struct ModelLoadingView: View {
    let message: String

    var body: some View {
        VStack(spacing: 16) {
            ProgressView()
                .progressViewStyle(.circular)
                .scaleEffect(1.5)

            Text(message)
                .font(.headline)
                .foregroundColor(.secondary)
        }
        .padding(32)
        .background(Color.systemBackground)
        .cornerRadius(16)
        .shadow(radius: 10)
    }
}

/// Error display view.
struct ErrorView: View {
    let message: String
    var onRetry: (() -> Void)?
    var onDismiss: (() -> Void)?

    var body: some View {
        VStack(spacing: 16) {
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.system(size: 40))
                .foregroundColor(.red)

            Text("Error")
                .font(.headline)

            Text(message)
                .font(.body)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)

            HStack(spacing: 12) {
                if let onDismiss = onDismiss {
                    Button("Dismiss", action: onDismiss)
                        .buttonStyle(.bordered)
                }

                if let onRetry = onRetry {
                    Button("Retry", action: onRetry)
                        .buttonStyle(.borderedProminent)
                }
            }
        }
        .padding(24)
        .background(Color.systemBackground)
        .cornerRadius(16)
        .shadow(radius: 10)
    }
}

// MARK: - Preview

#Preview("Download Progress") {
    DownloadProgressView(
        modelName: "Whisper Large V3",
        progress: DownloadProgress(
            modelId: "whisper-large-v3",
            bytesReceived: 1_500_000_000,
            totalBytes: 3_000_000_000,
            currentFile: "model.safetensors"
        )
    )
}

#Preview("Loading") {
    ModelLoadingView(message: "Loading model...")
}

#Preview("Error") {
    ErrorView(
        message: "Failed to download model. Check your internet connection.",
        onRetry: {},
        onDismiss: {}
    )
}
