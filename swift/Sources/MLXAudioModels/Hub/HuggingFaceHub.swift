// HuggingFaceHub.swift
// HuggingFace Hub integration for downloading and caching models.
//
// Provides automatic model downloading from HuggingFace Hub with local caching,
// similar to the Python huggingface_hub library.

import Foundation

// MARK: - Hub Errors

/// Errors that can occur during Hub operations.
public enum HubError: Error, LocalizedError, Sendable {
    /// Model not found on HuggingFace Hub
    case modelNotFound(String)
    /// Network request failed
    case networkError(String)
    /// File download failed
    case downloadFailed(String, statusCode: Int)
    /// Invalid response from Hub
    case invalidResponse(String)
    /// File not found in repository
    case fileNotFound(repo: String, file: String)
    /// Cache directory creation failed
    case cacheDirectoryError(String)
    /// File integrity check failed
    case integrityError(String)
    /// Rate limited by Hub
    case rateLimited(retryAfter: Int?)

    public var errorDescription: String? {
        switch self {
        case .modelNotFound(let repo):
            return "Model not found on HuggingFace Hub: \(repo)"
        case .networkError(let message):
            return "Network error: \(message)"
        case .downloadFailed(let url, let code):
            return "Download failed for \(url) with status code \(code)"
        case .invalidResponse(let message):
            return "Invalid Hub response: \(message)"
        case .fileNotFound(let repo, let file):
            return "File '\(file)' not found in repository '\(repo)'"
        case .cacheDirectoryError(let message):
            return "Cache directory error: \(message)"
        case .integrityError(let message):
            return "File integrity check failed: \(message)"
        case .rateLimited(let retryAfter):
            if let seconds = retryAfter {
                return "Rate limited by HuggingFace Hub. Retry after \(seconds) seconds."
            }
            return "Rate limited by HuggingFace Hub"
        }
    }
}

// MARK: - Hub Configuration

/// Configuration for HuggingFace Hub operations.
public struct HubConfiguration: Sendable {
    /// Base URL for HuggingFace Hub API
    public var hubURL: String

    /// Base URL for model file downloads
    public var downloadURL: String

    /// Cache directory for downloaded models
    public var cacheDirectory: URL

    /// Whether to use authentication token
    public var token: String?

    /// Request timeout in seconds
    public var timeout: TimeInterval

    /// Maximum number of retries for failed downloads
    public var maxRetries: Int

    /// Default configuration using standard paths
    public static var `default`: HubConfiguration {
        let cacheDir: URL
        if let customPath = ProcessInfo.processInfo.environment["HF_HOME"] {
            cacheDir = URL(fileURLWithPath: customPath)
        } else if let customPath = ProcessInfo.processInfo.environment["HUGGINGFACE_HUB_CACHE"] {
            cacheDir = URL(fileURLWithPath: customPath)
        } else {
            cacheDir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
                .appendingPathComponent("huggingface")
                .appendingPathComponent("hub")
        }

        return HubConfiguration(
            hubURL: "https://huggingface.co",
            downloadURL: "https://huggingface.co",
            cacheDirectory: cacheDir,
            token: ProcessInfo.processInfo.environment["HF_TOKEN"],
            timeout: 300,
            maxRetries: 3
        )
    }

    public init(
        hubURL: String = "https://huggingface.co",
        downloadURL: String = "https://huggingface.co",
        cacheDirectory: URL,
        token: String? = nil,
        timeout: TimeInterval = 300,
        maxRetries: Int = 3
    ) {
        self.hubURL = hubURL
        self.downloadURL = downloadURL
        self.cacheDirectory = cacheDirectory
        self.token = token
        self.timeout = timeout
        self.maxRetries = maxRetries
    }
}

// MARK: - Download Progress

/// Progress information for file downloads.
public struct DownloadProgress: Sendable {
    /// File being downloaded
    public let filename: String

    /// Bytes downloaded so far
    public let bytesDownloaded: Int64

    /// Total bytes to download (if known)
    public let totalBytes: Int64?

    /// Progress as a fraction (0.0 to 1.0)
    public var fraction: Double {
        guard let total = totalBytes, total > 0 else { return 0 }
        return Double(bytesDownloaded) / Double(total)
    }

    /// Progress as a percentage string
    public var percentageString: String {
        String(format: "%.1f%%", fraction * 100)
    }
}

/// Callback for download progress updates.
public typealias ProgressCallback = @Sendable (DownloadProgress) -> Void

// MARK: - HuggingFace Hub

/// Client for downloading models from HuggingFace Hub.
///
/// Provides automatic downloading and caching of model files from HuggingFace Hub,
/// with support for authentication, progress tracking, and resumable downloads.
///
/// ## Usage
///
/// ```swift
/// let hub = HuggingFaceHub()
///
/// // Download a model repository
/// let modelPath = try await hub.download(
///     repo: "mlx-community/htdemucs_ft-mlx",
///     files: ["model.safetensors", "config.json"]
/// )
///
/// // Or download the entire repo
/// let fullPath = try await hub.download(repo: "mlx-community/whisper-large-v3-turbo")
/// ```
///
/// ## Caching
///
/// Downloaded files are cached locally at `~/.cache/huggingface/hub/` (or the path
/// specified by `HF_HOME` or `HUGGINGFACE_HUB_CACHE` environment variables).
///
/// Files are organized as:
/// ```
/// ~/.cache/huggingface/hub/
///   models--mlx-community--htdemucs_ft-mlx/
///     snapshots/
///       abc123/
///         model.safetensors
///         config.json
/// ```
public actor HuggingFaceHub {

    /// Shared Hub instance with default configuration.
    public static let shared = HuggingFaceHub()

    /// Configuration for Hub operations.
    public let configuration: HubConfiguration

    /// URL session for network requests.
    private let session: URLSession

    /// Initialize with configuration.
    ///
    /// - Parameter configuration: Hub configuration (defaults to standard paths)
    public init(configuration: HubConfiguration = .default) {
        self.configuration = configuration

        let sessionConfig = URLSessionConfiguration.default
        sessionConfig.timeoutIntervalForRequest = configuration.timeout
        sessionConfig.timeoutIntervalForResource = configuration.timeout * 2
        self.session = URLSession(configuration: sessionConfig)
    }

    // MARK: - Public API

    /// Download a model repository from HuggingFace Hub.
    ///
    /// Downloads specified files (or all files) from a HuggingFace repository
    /// and returns the local path to the cached model directory.
    ///
    /// - Parameters:
    ///   - repo: Repository ID (e.g., "mlx-community/htdemucs_ft-mlx")
    ///   - revision: Git revision (branch, tag, or commit). Defaults to "main".
    ///   - files: Specific files to download. If nil, downloads common model files.
    ///   - progress: Optional callback for download progress updates.
    /// - Returns: Local URL to the cached model directory.
    /// - Throws: `HubError` if download fails.
    public func download(
        repo: String,
        revision: String = "main",
        files: [String]? = nil,
        progress: ProgressCallback? = nil
    ) async throws -> URL {
        // Create cache directory structure
        let repoDir = try ensureCacheDirectory(for: repo)
        let snapshotDir = repoDir.appendingPathComponent("snapshots").appendingPathComponent(revision)

        try FileManager.default.createDirectory(at: snapshotDir, withIntermediateDirectories: true)

        // Determine files to download
        let filesToDownload: [String]
        if let specified = files {
            filesToDownload = specified
        } else {
            // Default model files
            filesToDownload = [
                "model.safetensors",
                "config.json",
                "tokenizer.json",
                "vocab.json",
                "preprocessor_config.json"
            ]
        }

        // Download each file
        for filename in filesToDownload {
            let localPath = snapshotDir.appendingPathComponent(filename)

            // Skip if already cached
            if FileManager.default.fileExists(atPath: localPath.path) {
                continue
            }

            // Download file
            do {
                try await downloadFile(
                    repo: repo,
                    filename: filename,
                    revision: revision,
                    destination: localPath,
                    progress: progress
                )
            } catch HubError.fileNotFound {
                // Skip optional files that don't exist
                if !isRequiredFile(filename) {
                    continue
                }
                throw HubError.fileNotFound(repo: repo, file: filename)
            }
        }

        return snapshotDir
    }

    /// Download a specific file from a repository.
    ///
    /// - Parameters:
    ///   - repo: Repository ID
    ///   - filename: File path within the repository
    ///   - revision: Git revision
    ///   - progress: Optional progress callback
    /// - Returns: Local URL to the downloaded file
    public func downloadFile(
        repo: String,
        filename: String,
        revision: String = "main",
        progress: ProgressCallback? = nil
    ) async throws -> URL {
        let repoDir = try ensureCacheDirectory(for: repo)
        let snapshotDir = repoDir.appendingPathComponent("snapshots").appendingPathComponent(revision)
        try FileManager.default.createDirectory(at: snapshotDir, withIntermediateDirectories: true)

        let localPath = snapshotDir.appendingPathComponent(filename)

        // Return cached file if exists
        if FileManager.default.fileExists(atPath: localPath.path) {
            return localPath
        }

        try await downloadFile(
            repo: repo,
            filename: filename,
            revision: revision,
            destination: localPath,
            progress: progress
        )

        return localPath
    }

    /// Check if a model is cached locally.
    ///
    /// - Parameters:
    ///   - repo: Repository ID
    ///   - revision: Git revision
    /// - Returns: Local URL if cached, nil otherwise
    public func cachedPath(repo: String, revision: String = "main") -> URL? {
        let repoDir = cacheDirectory(for: repo)
        let snapshotDir = repoDir.appendingPathComponent("snapshots").appendingPathComponent(revision)

        // Check if config.json or model.safetensors exists
        let configPath = snapshotDir.appendingPathComponent("config.json")
        let modelPath = snapshotDir.appendingPathComponent("model.safetensors")

        if FileManager.default.fileExists(atPath: configPath.path) ||
           FileManager.default.fileExists(atPath: modelPath.path) {
            return snapshotDir
        }

        return nil
    }

    /// Clear cached files for a repository.
    ///
    /// - Parameter repo: Repository ID to clear
    public func clearCache(repo: String) throws {
        let repoDir = cacheDirectory(for: repo)
        if FileManager.default.fileExists(atPath: repoDir.path) {
            try FileManager.default.removeItem(at: repoDir)
        }
    }

    /// Clear all cached models.
    public func clearAllCache() throws {
        let modelsDir = configuration.cacheDirectory
        if FileManager.default.fileExists(atPath: modelsDir.path) {
            let contents = try FileManager.default.contentsOfDirectory(atPath: modelsDir.path)
            for item in contents where item.hasPrefix("models--") {
                let itemPath = modelsDir.appendingPathComponent(item)
                try FileManager.default.removeItem(at: itemPath)
            }
        }
    }

    // MARK: - Private Methods

    private func cacheDirectory(for repo: String) -> URL {
        let sanitized = "models--" + repo.replacingOccurrences(of: "/", with: "--")
        return configuration.cacheDirectory.appendingPathComponent(sanitized)
    }

    private func ensureCacheDirectory(for repo: String) throws -> URL {
        let dir = cacheDirectory(for: repo)
        do {
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        } catch {
            throw HubError.cacheDirectoryError("Failed to create cache directory: \(error.localizedDescription)")
        }
        return dir
    }

    private func downloadFile(
        repo: String,
        filename: String,
        revision: String,
        destination: URL,
        progress: ProgressCallback?
    ) async throws {
        // Build download URL
        // Format: https://huggingface.co/{repo}/resolve/{revision}/{filename}
        let downloadURL = "\(configuration.downloadURL)/\(repo)/resolve/\(revision)/\(filename)"

        guard let url = URL(string: downloadURL) else {
            throw HubError.invalidResponse("Invalid URL: \(downloadURL)")
        }

        var request = URLRequest(url: url)
        request.httpMethod = "GET"

        // Add auth token if available
        if let token = configuration.token {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }

        // Add user agent
        request.setValue("mlx-audio-swift/1.0", forHTTPHeaderField: "User-Agent")

        // Perform download with retries
        var lastError: Error?
        for attempt in 0..<configuration.maxRetries {
            do {
                try await performDownload(request: request, destination: destination, filename: filename, progress: progress)
                return
            } catch let error as HubError {
                lastError = error

                // Don't retry on certain errors
                switch error {
                case .fileNotFound, .modelNotFound:
                    throw error
                case .rateLimited(let retryAfter):
                    if let wait = retryAfter {
                        try await Task.sleep(nanoseconds: UInt64(wait) * 1_000_000_000)
                    } else {
                        try await Task.sleep(nanoseconds: UInt64(attempt + 1) * 5_000_000_000)
                    }
                default:
                    // Exponential backoff
                    let delay = UInt64(pow(2.0, Double(attempt))) * 1_000_000_000
                    try await Task.sleep(nanoseconds: delay)
                }
            } catch {
                lastError = error
                let delay = UInt64(pow(2.0, Double(attempt))) * 1_000_000_000
                try await Task.sleep(nanoseconds: delay)
            }
        }

        throw lastError ?? HubError.downloadFailed(downloadURL, statusCode: 0)
    }

    private func performDownload(
        request: URLRequest,
        destination: URL,
        filename: String,
        progress: ProgressCallback?
    ) async throws {
        // Create parent directories if needed
        let parentDir = destination.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: parentDir, withIntermediateDirectories: true)

        // Use delegate-based download for progress tracking
        let (tempURL, response) = try await session.download(for: request)

        guard let httpResponse = response as? HTTPURLResponse else {
            throw HubError.invalidResponse("Non-HTTP response received")
        }

        switch httpResponse.statusCode {
        case 200:
            // Success - move temp file to destination
            if FileManager.default.fileExists(atPath: destination.path) {
                try FileManager.default.removeItem(at: destination)
            }
            try FileManager.default.moveItem(at: tempURL, to: destination)

            // Report completion
            if let contentLength = httpResponse.value(forHTTPHeaderField: "Content-Length"),
               let totalBytes = Int64(contentLength) {
                let finalProgress = DownloadProgress(
                    filename: filename,
                    bytesDownloaded: totalBytes,
                    totalBytes: totalBytes
                )
                progress?(finalProgress)
            }

        case 401:
            throw HubError.networkError("Unauthorized. Check your HF_TOKEN.")

        case 404:
            throw HubError.fileNotFound(repo: request.url?.path ?? "", file: filename)

        case 429:
            let retryAfter = httpResponse.value(forHTTPHeaderField: "Retry-After").flatMap { Int($0) }
            throw HubError.rateLimited(retryAfter: retryAfter)

        default:
            throw HubError.downloadFailed(request.url?.absoluteString ?? "", statusCode: httpResponse.statusCode)
        }
    }

    private func isRequiredFile(_ filename: String) -> Bool {
        // These files are required for a model to function
        let required = ["model.safetensors", "config.json"]
        return required.contains(filename)
    }
}

// MARK: - Convenience Extensions

extension HuggingFaceHub {

    /// Download and return path to model weights file.
    ///
    /// Convenience method that downloads a model and returns the path
    /// to the weights file directly.
    ///
    /// - Parameters:
    ///   - repo: Repository ID
    ///   - weightsFile: Name of weights file (defaults to "model.safetensors")
    ///   - progress: Optional progress callback
    /// - Returns: URL to the weights file
    public func downloadWeights(
        repo: String,
        weightsFile: String = "model.safetensors",
        progress: ProgressCallback? = nil
    ) async throws -> URL {
        let modelDir = try await download(
            repo: repo,
            files: [weightsFile, "config.json"],
            progress: progress
        )
        return modelDir.appendingPathComponent(weightsFile)
    }

    /// Load model configuration from a repository.
    ///
    /// Downloads and parses the config.json file from a repository.
    ///
    /// - Parameters:
    ///   - repo: Repository ID
    ///   - configFile: Name of config file (defaults to "config.json")
    /// - Returns: Parsed configuration dictionary
    public func loadConfig(
        repo: String,
        configFile: String = "config.json"
    ) async throws -> [String: Any] {
        let configURL = try await downloadFile(repo: repo, filename: configFile)
        let data = try Data(contentsOf: configURL)

        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw HubError.invalidResponse("Config file is not a valid JSON object")
        }

        return json
    }
}

// MARK: - Model ID Helpers

/// Well-known MLX audio model repositories.
public enum MLXAudioModels {
    /// HTDemucs source separation models
    public enum HTDemucs: String, CaseIterable, Sendable {
        case htdemucs = "mlx-community/htdemucs-mlx"
        case htdemucsFt = "mlx-community/htdemucs_ft-mlx"
        case htdemucs6s = "mlx-community/htdemucs_6s-mlx"
    }

    /// Whisper transcription models
    public enum Whisper: String, CaseIterable, Sendable {
        case tiny = "mlx-community/whisper-tiny-mlx"
        case base = "mlx-community/whisper-base-mlx"
        case small = "mlx-community/whisper-small-mlx"
        case medium = "mlx-community/whisper-medium-mlx"
        case large = "mlx-community/whisper-large-mlx"
        case largeV3Turbo = "mlx-community/whisper-large-v3-turbo-mlx"
    }

    /// CLAP audio embedding models
    public enum CLAP: String, CaseIterable, Sendable {
        case htsatFused = "mlx-community/clap-htsat-fused-mlx"
    }

    /// MusicGen audio generation models
    public enum MusicGen: String, CaseIterable, Sendable {
        case small = "mlx-community/musicgen-small-mlx"
        case medium = "mlx-community/musicgen-medium-mlx"
    }

    /// EnCodec audio codec models
    public enum EnCodec: String, CaseIterable, Sendable {
        case encodec24khz = "mlx-community/encodec-24khz-mlx"
    }
}
