// ModelManager.swift
// Central model management service for loading and caching models.

import Foundation
import MLX
import MLXAudioModels

/// States a model can be in.
public enum ModelState: Sendable {
    case notDownloaded
    case downloading(progress: Double)
    case downloaded
    case loading
    case loaded
    case error(String)
}

/// Central manager for model loading, caching, and memory management.
@MainActor
public class ModelManager: ObservableObject {

    // MARK: - Published State

    /// Current state for each model type.
    @Published public var modelStates: [AudioTask: ModelState] = [:]

    /// Download progress for models being downloaded.
    @Published public var downloadProgress: [String: DownloadProgress] = [:]

    /// Currently loading model (if any).
    @Published public var loadingModel: String?

    /// Error message (if any).
    @Published public var errorMessage: String?

    // MARK: - Internal State

    /// Model cache for loaded models.
    private let cache = ModelCache.shared

    /// Downloader for fetching weights.
    private let downloader = HuggingFaceDownloader()

    /// Current device profile.
    public let deviceProfile: DeviceProfile

    /// Currently active tab.
    private var currentTab: AppTab?

    // MARK: - Loaded Models

    private var htdemucs: HTDemucs?
    private var whisperModel: WhisperModel?
    private var musicGen: MusicGen?
    private var clapModel: CLAPModel?
    private var banquet: Banquet?
    private var encodec: EnCodec?

    // MARK: - Initialization

    public init() {
        self.deviceProfile = DeviceProfile.current

        // Initialize model states
        for task in AudioTask.allCases {
            modelStates[task] = .notDownloaded
        }

        // Start memory pressure monitoring
        Task {
            await cache.startMemoryPressureMonitoring()
        }
    }

    // MARK: - Tab Management

    /// Called when the user switches tabs.
    func onTabChange(to tab: AppTab) async {
        currentTab = tab

        // On memory-constrained devices, evict models not needed for current tab
        if deviceProfile == .phone || deviceProfile == .tablet {
            // Evict all but keep some headroom
            let modelsToKeep = modelsForTab(tab)
            await evictModelsExcept(modelsToKeep)
        }
    }

    /// Get the models relevant for a tab.
    private func modelsForTab(_ tab: AppTab) -> Set<String> {
        switch tab {
        case .separate:
            return ["htdemucs", "htdemucs_ft", "htdemucs_6s"]
        case .transcribe:
            return ["whisper-tiny", "whisper-small", "whisper-medium", "whisper-large-v3-turbo", "whisper-large-v3"]
        case .generate:
            return ["musicgen-small", "musicgen-medium", "musicgen-large"]
        case .embed:
            return ["clap-htsat-tiny", "clap-htsat-fused"]
        case .live:
            return ["htdemucs"]  // Use lighter model for real-time
        case .banquet:
            return ["banquet"]
        }
    }

    /// Evict all models except the specified set.
    private func evictModelsExcept(_ keepModelIds: Set<String>) async {
        let loadedIds = await cache.loadedModelIds
        for id in loadedIds where !keepModelIds.contains(id) {
            await cache.evict(id)
        }
        GPU.clearCache()
    }

    // MARK: - Model Loading

    /// Load HTDemucs model.
    public func loadHTDemucs(variant: String = "htdemucs_ft") async throws -> HTDemucs {
        // Check cache first
        if let cached = htdemucs {
            return cached
        }

        let task = AudioTask.separation
        modelStates[task] = .loading
        loadingModel = variant

        do {
            // Get variant info
            guard let variantInfo = ModelVariantRegistry.variant(id: variant) else {
                throw ModelLoadError.unknownModel(variant)
            }

            // Ensure downloaded
            let modelPath = try await ensureDownloaded(
                modelId: variant,
                repoId: variantInfo.repoId
            )

            // Load model using fromPretrained
            let model = try HTDemucs.fromPretrained(path: modelPath)

            // Cache and return
            htdemucs = model
            modelStates[task] = .loaded
            loadingModel = nil

            return model

        } catch {
            modelStates[task] = .error(error.localizedDescription)
            loadingModel = nil
            throw error
        }
    }

    /// Load Whisper model.
    public func loadWhisper(variant: String = "whisper-large-v3-turbo") async throws -> WhisperModel {
        if let cached = whisperModel {
            return cached
        }

        let task = AudioTask.transcription
        modelStates[task] = .loading
        loadingModel = variant

        do {
            guard let variantInfo = ModelVariantRegistry.variant(id: variant) else {
                throw ModelLoadError.unknownModel(variant)
            }

            let modelPath = try await ensureDownloaded(
                modelId: variant,
                repoId: variantInfo.repoId
            )

            // Load model using fromPretrained
            let model = try WhisperModel.fromPretrained(path: modelPath)

            whisperModel = model
            modelStates[task] = .loaded
            loadingModel = nil

            return model

        } catch {
            modelStates[task] = .error(error.localizedDescription)
            loadingModel = nil
            throw error
        }
    }

    /// Load MusicGen model.
    public func loadMusicGen(variant: String = "musicgen-medium") async throws -> MusicGen {
        if let cached = musicGen {
            return cached
        }

        let task = AudioTask.generation
        modelStates[task] = .loading
        loadingModel = variant

        do {
            guard let variantInfo = ModelVariantRegistry.variant(id: variant) else {
                throw ModelLoadError.unknownModel(variant)
            }

            let modelPath = try await ensureDownloaded(
                modelId: variant,
                repoId: variantInfo.repoId
            )

            // Load model using fromPretrained
            let config = musicGenConfig(for: variant)
            let model = try MusicGen.fromPretrained(path: modelPath.path, config: config)

            musicGen = model
            modelStates[task] = .loaded
            loadingModel = nil

            return model

        } catch {
            modelStates[task] = .error(error.localizedDescription)
            loadingModel = nil
            throw error
        }
    }

    /// Load CLAP model.
    public func loadCLAP(variant: String = "clap-htsat-fused") async throws -> CLAPModel {
        if let cached = clapModel {
            return cached
        }

        let task = AudioTask.embedding
        modelStates[task] = .loading
        loadingModel = variant

        do {
            guard let variantInfo = ModelVariantRegistry.variant(id: variant) else {
                throw ModelLoadError.unknownModel(variant)
            }

            let modelPath = try await ensureDownloaded(
                modelId: variant,
                repoId: variantInfo.repoId
            )

            // Load model using fromPretrained
            let model = try CLAPModel.fromPretrained(path: modelPath)

            clapModel = model
            modelStates[task] = .loaded
            loadingModel = nil

            return model

        } catch {
            modelStates[task] = .error(error.localizedDescription)
            loadingModel = nil
            throw error
        }
    }

    /// Load Banquet model.
    public func loadBanquet() async throws -> Banquet {
        if let cached = banquet {
            return cached
        }

        modelStates[.separation] = .loading
        loadingModel = "banquet"

        do {
            // Banquet uses a custom repo
            _ = try await ensureDownloaded(
                modelId: "banquet",
                repoId: "mlx-community/banquet-mlx"
            )

            let config = BanquetConfig()
            let model = Banquet(config: config)

            // Note: Weight loading would happen here in production
            // For now, returning uninitialized model for demo

            banquet = model
            modelStates[.separation] = .loaded
            loadingModel = nil

            return model

        } catch {
            modelStates[.separation] = .error(error.localizedDescription)
            loadingModel = nil
            throw error
        }
    }

    /// Load EnCodec model.
    public func loadEnCodec(variant: String = "encodec-24khz") async throws -> EnCodec {
        if let cached = encodec {
            return cached
        }

        let task = AudioTask.codec
        modelStates[task] = .loading
        loadingModel = variant

        do {
            guard let variantInfo = ModelVariantRegistry.variant(id: variant) else {
                throw ModelLoadError.unknownModel(variant)
            }

            _ = try await ensureDownloaded(
                modelId: variant,
                repoId: variantInfo.repoId
            )

            let config = EnCodecConfig.encodec_24khz()
            let model = EnCodec(config: config)

            // Note: Weight loading would happen here in production
            // For now, returning uninitialized model for demo

            encodec = model
            modelStates[task] = .loaded
            loadingModel = nil

            return model

        } catch {
            modelStates[task] = .error(error.localizedDescription)
            loadingModel = nil
            throw error
        }
    }

    // MARK: - Download Management

    /// Ensure a model is downloaded, downloading if necessary.
    private func ensureDownloaded(modelId: String, repoId: String) async throws -> URL {
        // Check if already downloaded
        if await downloader.isModelDownloaded(modelId: modelId) {
            return await downloader.modelPath(for: modelId)
        }

        // Download with progress
        return try await downloader.download(
            modelId: modelId,
            repoId: repoId
        ) { [weak self] progress in
            Task { @MainActor in
                self?.downloadProgress[modelId] = progress
            }
        }
    }

    /// Check if a model is downloaded.
    public func isModelDownloaded(_ modelId: String) async -> Bool {
        await downloader.isModelDownloaded(modelId: modelId)
    }

    // MARK: - Configuration Helpers

    private func musicGenConfig(for variant: String) -> MusicGenConfig {
        switch variant {
        case "musicgen-small":
            return MusicGenConfig.small()
        case "musicgen-large":
            return MusicGenConfig.large()
        default:
            return MusicGenConfig.medium()
        }
    }

    // MARK: - Recommended Models

    /// Get the recommended model variant for a task on this device.
    public func recommendedModel(for task: AudioTask) -> ModelVariant? {
        ModelVariantRegistry.recommendedVariant(for: task, profile: deviceProfile)
    }

    /// Get all compatible model variants for a task.
    public func compatibleModels(for task: AudioTask) -> [ModelVariant] {
        ModelVariantRegistry.variants(for: task).filter { $0.isCompatible(with: deviceProfile) }
    }
}

// MARK: - Errors

public enum ModelLoadError: Error, LocalizedError {
    case unknownModel(String)
    case downloadFailed(String)
    case loadFailed(String)

    public var errorDescription: String? {
        switch self {
        case .unknownModel(let id):
            return "Unknown model: \(id)"
        case .downloadFailed(let message):
            return "Download failed: \(message)"
        case .loadFailed(let message):
            return "Load failed: \(message)"
        }
    }
}
