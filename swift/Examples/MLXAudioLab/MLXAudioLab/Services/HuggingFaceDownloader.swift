// HuggingFaceDownloader.swift
// Downloads model weights from HuggingFace Hub with progress tracking.

import Foundation

/// Progress information for model downloads.
public struct DownloadProgress: Sendable {
    public let modelId: String
    public let bytesReceived: Int64
    public let totalBytes: Int64
    public let currentFile: String

    public var progress: Double {
        totalBytes > 0 ? Double(bytesReceived) / Double(totalBytes) : 0
    }

    public var formattedProgress: String {
        String(format: "%.0f%%", progress * 100)
    }

    public var formattedBytes: String {
        let formatter = ByteCountFormatter()
        formatter.allowedUnits = [.useMB, .useGB]
        formatter.countStyle = .file
        let received = formatter.string(fromByteCount: bytesReceived)
        let total = formatter.string(fromByteCount: totalBytes)
        return "\(received) / \(total)"
    }
}

/// Errors that can occur during download.
public enum DownloadError: Error, LocalizedError {
    case invalidURL(String)
    case networkError(Error)
    case fileSystemError(Error)
    case noData
    case httpError(statusCode: Int)

    public var errorDescription: String? {
        switch self {
        case .invalidURL(let url):
            return "Invalid URL: \(url)"
        case .networkError(let error):
            return "Network error: \(error.localizedDescription)"
        case .fileSystemError(let error):
            return "File system error: \(error.localizedDescription)"
        case .noData:
            return "No data received"
        case .httpError(let statusCode):
            return "HTTP error: \(statusCode)"
        }
    }
}

/// Downloads model files from HuggingFace Hub.
public actor HuggingFaceDownloader {

    /// Base URL for HuggingFace Hub.
    private let baseURL = "https://huggingface.co"

    /// URL session for downloads.
    private let session: URLSession

    /// Base directory for downloaded models.
    private let modelsDirectory: URL

    public init() {
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 300
        config.timeoutIntervalForResource = 3600  // 1 hour for large models
        self.session = URLSession(configuration: config)

        // Store models in Documents/MLXAudioLab/Models/
        let documentsDir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        self.modelsDirectory = documentsDir.appendingPathComponent("MLXAudioLab/Models")
    }

    /// Check if a model is already downloaded.
    public func isModelDownloaded(modelId: String) -> Bool {
        let modelDir = modelsDirectory.appendingPathComponent(modelId)
        let weightsPath = modelDir.appendingPathComponent("model.safetensors")
        return FileManager.default.fileExists(atPath: weightsPath.path)
    }

    /// Get the local path for a downloaded model.
    public func modelPath(for modelId: String) -> URL {
        modelsDirectory.appendingPathComponent(modelId)
    }

    /// Download a model from HuggingFace Hub.
    ///
    /// - Parameters:
    ///   - modelId: Local identifier for the model.
    ///   - repoId: HuggingFace repository ID (e.g., "facebook/htdemucs").
    ///   - files: List of files to download (defaults to config.json and model.safetensors).
    ///   - progressCallback: Callback for download progress updates.
    /// - Returns: Local directory containing the downloaded model.
    public func download(
        modelId: String,
        repoId: String,
        files: [String] = ["config.json", "model.safetensors"],
        progressCallback: @escaping @Sendable (DownloadProgress) -> Void
    ) async throws -> URL {
        let modelDir = modelsDirectory.appendingPathComponent(modelId)

        // Check if already downloaded
        let weightsPath = modelDir.appendingPathComponent("model.safetensors")
        if FileManager.default.fileExists(atPath: weightsPath.path) {
            return modelDir
        }

        // Create directory
        try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)

        // Calculate total size first
        var totalBytes: Int64 = 0
        for file in files {
            let fileSize = try await getFileSize(repoId: repoId, file: file)
            totalBytes += fileSize
        }

        // Download each file
        var downloadedBytes: Int64 = 0
        let finalTotalBytes = totalBytes
        for file in files {
            let fileURL = URL(string: "\(baseURL)/\(repoId)/resolve/main/\(file)")!
            let destination = modelDir.appendingPathComponent(file)

            // Capture current state for Sendable closure
            let baseBytes = downloadedBytes
            try await downloadFile(
                from: fileURL,
                to: destination
            ) { received, _ in
                let currentBytes = baseBytes + received
                progressCallback(DownloadProgress(
                    modelId: modelId,
                    bytesReceived: currentBytes,
                    totalBytes: finalTotalBytes,
                    currentFile: file
                ))
            }

            // Update running total
            let fileSize = try FileManager.default.attributesOfItem(atPath: destination.path)[.size] as? Int64 ?? 0
            downloadedBytes += fileSize
        }

        return modelDir
    }

    /// Get the size of a file on HuggingFace Hub.
    private func getFileSize(repoId: String, file: String) async throws -> Int64 {
        guard let url = URL(string: "\(baseURL)/\(repoId)/resolve/main/\(file)") else {
            throw DownloadError.invalidURL("\(baseURL)/\(repoId)/resolve/main/\(file)")
        }

        var request = URLRequest(url: url)
        request.httpMethod = "HEAD"

        let (_, response) = try await session.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse else {
            return 0
        }

        return httpResponse.expectedContentLength
    }

    /// Download a single file with progress tracking.
    private func downloadFile(
        from url: URL,
        to destination: URL,
        progress: @escaping @Sendable (Int64, Int64) -> Void
    ) async throws {
        let request = URLRequest(url: url)

        // Use delegate-based download for progress
        let delegate = DownloadDelegate(progressHandler: progress)
        let delegateSession = URLSession(configuration: .default, delegate: delegate, delegateQueue: nil)

        defer {
            delegateSession.invalidateAndCancel()
        }

        let (tempURL, response) = try await delegateSession.download(for: request)

        guard let httpResponse = response as? HTTPURLResponse else {
            throw DownloadError.networkError(NSError(domain: "Invalid response", code: -1))
        }

        guard (200...299).contains(httpResponse.statusCode) else {
            throw DownloadError.httpError(statusCode: httpResponse.statusCode)
        }

        // Move to final destination
        try? FileManager.default.removeItem(at: destination)
        try FileManager.default.moveItem(at: tempURL, to: destination)
    }

    /// Delete a downloaded model.
    public func deleteModel(modelId: String) throws {
        let modelDir = modelsDirectory.appendingPathComponent(modelId)
        if FileManager.default.fileExists(atPath: modelDir.path) {
            try FileManager.default.removeItem(at: modelDir)
        }
    }

    /// Get total size of all downloaded models.
    public func totalDownloadedSize() throws -> Int64 {
        guard FileManager.default.fileExists(atPath: modelsDirectory.path) else {
            return 0
        }

        var total: Int64 = 0
        let enumerator = FileManager.default.enumerator(at: modelsDirectory, includingPropertiesForKeys: [.fileSizeKey])

        while let fileURL = enumerator?.nextObject() as? URL {
            let resourceValues = try fileURL.resourceValues(forKeys: [.fileSizeKey])
            total += Int64(resourceValues.fileSize ?? 0)
        }

        return total
    }
}

/// URLSession delegate for tracking download progress.
private final class DownloadDelegate: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
    let progressHandler: @Sendable (Int64, Int64) -> Void

    init(progressHandler: @escaping @Sendable (Int64, Int64) -> Void) {
        self.progressHandler = progressHandler
    }

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didWriteData bytesWritten: Int64,
        totalBytesWritten: Int64,
        totalBytesExpectedToWrite: Int64
    ) {
        progressHandler(totalBytesWritten, totalBytesExpectedToWrite)
    }

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didFinishDownloadingTo location: URL
    ) {
        // Handled in the async download method
    }
}
