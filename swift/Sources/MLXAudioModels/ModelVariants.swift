// ModelVariants.swift
// Model variant registry for device-appropriate model selection.
//
// Provides metadata about available model variants and recommendations
// for different device profiles.

import Foundation

// AudioTask is defined in Models.swift

/// Information about a model variant.
public struct ModelVariant: Codable, Sendable, Identifiable {

    /// Unique model identifier (e.g., "whisper-large-v3").
    public let id: String

    /// Human-readable display name.
    public let displayName: String

    /// Task this model performs.
    public let task: AudioTask

    /// Approximate parameter count.
    public let parameterCount: Int

    /// Estimated memory usage in MB (full precision).
    public let estimatedMemoryMB: Int

    /// Estimated memory usage in MB (quantized, if applicable).
    public let quantizedMemoryMB: Int?

    /// HuggingFace repository ID.
    public let repoId: String

    /// Whether the model supports quantization.
    public let supportsQuantization: Bool

    /// Minimum device profile for this model.
    public let minimumProfile: DeviceProfile

    /// Quality tier (1 = smallest/fastest, 5 = largest/best).
    public let qualityTier: Int

    /// Sample rate the model expects/produces.
    public let sampleRate: Int?

    /// Additional notes about the model.
    public let notes: String?

    public init(
        id: String,
        displayName: String,
        task: AudioTask,
        parameterCount: Int,
        estimatedMemoryMB: Int,
        quantizedMemoryMB: Int? = nil,
        repoId: String,
        supportsQuantization: Bool = true,
        minimumProfile: DeviceProfile = .phone,
        qualityTier: Int = 3,
        sampleRate: Int? = nil,
        notes: String? = nil
    ) {
        self.id = id
        self.displayName = displayName
        self.task = task
        self.parameterCount = parameterCount
        self.estimatedMemoryMB = estimatedMemoryMB
        self.quantizedMemoryMB = quantizedMemoryMB
        self.repoId = repoId
        self.supportsQuantization = supportsQuantization
        self.minimumProfile = minimumProfile
        self.qualityTier = qualityTier
        self.sampleRate = sampleRate
        self.notes = notes
    }

    /// Check if this variant is compatible with a device profile.
    public func isCompatible(with profile: DeviceProfile) -> Bool {
        let profileOrder: [DeviceProfile] = [.phone, .tablet, .mac, .macPro]
        guard let minIndex = profileOrder.firstIndex(of: minimumProfile),
              let profileIndex = profileOrder.firstIndex(of: profile) else {
            return false
        }
        return profileIndex >= minIndex
    }

    /// Get effective memory for a device profile (considering quantization).
    public func effectiveMemoryMB(for profile: DeviceProfile) -> Int {
        if profile.preferQuantized, let qMem = quantizedMemoryMB {
            return qMem
        }
        return estimatedMemoryMB
    }
}

// MARK: - Model Registry

/// Registry of available model variants.
public struct ModelVariantRegistry {

    // MARK: - Whisper Variants

    public static let whisperVariants: [ModelVariant] = [
        ModelVariant(
            id: "whisper-tiny",
            displayName: "Whisper Tiny",
            task: .transcription,
            parameterCount: 39_000_000,
            estimatedMemoryMB: 150,
            quantizedMemoryMB: 50,
            repoId: "mlx-community/whisper-tiny-mlx",
            minimumProfile: .phone,
            qualityTier: 1,
            sampleRate: 16000,
            notes: "Fastest, lowest quality"
        ),
        ModelVariant(
            id: "whisper-small",
            displayName: "Whisper Small",
            task: .transcription,
            parameterCount: 244_000_000,
            estimatedMemoryMB: 500,
            quantizedMemoryMB: 150,
            repoId: "mlx-community/whisper-small-mlx",
            minimumProfile: .phone,
            qualityTier: 2,
            sampleRate: 16000,
            notes: "Good balance for mobile"
        ),
        ModelVariant(
            id: "whisper-medium",
            displayName: "Whisper Medium",
            task: .transcription,
            parameterCount: 769_000_000,
            estimatedMemoryMB: 1500,
            quantizedMemoryMB: 450,
            repoId: "mlx-community/whisper-medium-mlx",
            minimumProfile: .tablet,
            qualityTier: 3,
            sampleRate: 16000
        ),
        ModelVariant(
            id: "whisper-large-v3-turbo",
            displayName: "Whisper Large V3 Turbo",
            task: .transcription,
            parameterCount: 809_000_000,
            estimatedMemoryMB: 1600,
            quantizedMemoryMB: 500,
            repoId: "mlx-community/whisper-large-v3-turbo",
            minimumProfile: .tablet,
            qualityTier: 4,
            sampleRate: 16000,
            notes: "Fast, high quality"
        ),
        ModelVariant(
            id: "whisper-large-v3",
            displayName: "Whisper Large V3",
            task: .transcription,
            parameterCount: 1_550_000_000,
            estimatedMemoryMB: 3000,
            quantizedMemoryMB: 900,
            repoId: "mlx-community/whisper-large-v3-mlx",
            minimumProfile: .mac,
            qualityTier: 5,
            sampleRate: 16000,
            notes: "Highest quality"
        ),
    ]

    // MARK: - HTDemucs Variants

    public static let htdemucsVariants: [ModelVariant] = [
        ModelVariant(
            id: "htdemucs",
            displayName: "HTDemucs",
            task: .separation,
            parameterCount: 42_000_000,
            estimatedMemoryMB: 2000,
            quantizedMemoryMB: 600,
            repoId: "facebook/htdemucs",
            minimumProfile: .phone,
            qualityTier: 3,
            sampleRate: 44100,
            notes: "Standard 4-stem separation"
        ),
        ModelVariant(
            id: "htdemucs_ft",
            displayName: "HTDemucs Fine-tuned",
            task: .separation,
            parameterCount: 42_000_000,
            estimatedMemoryMB: 2000,
            quantizedMemoryMB: 600,
            repoId: "facebook/htdemucs_ft",
            minimumProfile: .tablet,
            qualityTier: 4,
            sampleRate: 44100,
            notes: "Fine-tuned for better quality"
        ),
        ModelVariant(
            id: "htdemucs_6s",
            displayName: "HTDemucs 6-Stem",
            task: .separation,
            parameterCount: 42_000_000,
            estimatedMemoryMB: 2500,
            quantizedMemoryMB: 750,
            repoId: "facebook/htdemucs_6s",
            minimumProfile: .mac,
            qualityTier: 5,
            sampleRate: 44100,
            notes: "6 sources: drums, bass, other, vocals, guitar, piano"
        ),
    ]

    // MARK: - MusicGen Variants

    public static let musicgenVariants: [ModelVariant] = [
        ModelVariant(
            id: "musicgen-small",
            displayName: "MusicGen Small",
            task: .generation,
            parameterCount: 300_000_000,
            estimatedMemoryMB: 1200,
            quantizedMemoryMB: 400,
            repoId: "facebook/musicgen-small",
            minimumProfile: .phone,
            qualityTier: 2,
            sampleRate: 32000
        ),
        ModelVariant(
            id: "musicgen-medium",
            displayName: "MusicGen Medium",
            task: .generation,
            parameterCount: 1_500_000_000,
            estimatedMemoryMB: 3500,
            quantizedMemoryMB: 1000,
            repoId: "facebook/musicgen-medium",
            minimumProfile: .tablet,
            qualityTier: 3,
            sampleRate: 32000
        ),
        ModelVariant(
            id: "musicgen-large",
            displayName: "MusicGen Large",
            task: .generation,
            parameterCount: 3_300_000_000,
            estimatedMemoryMB: 7000,
            quantizedMemoryMB: 2000,
            repoId: "facebook/musicgen-large",
            minimumProfile: .mac,
            qualityTier: 5,
            sampleRate: 32000
        ),
    ]

    // MARK: - CLAP Variants

    public static let clapVariants: [ModelVariant] = [
        ModelVariant(
            id: "clap-htsat-tiny",
            displayName: "CLAP Tiny",
            task: .embedding,
            parameterCount: 30_000_000,
            estimatedMemoryMB: 150,
            quantizedMemoryMB: 50,
            repoId: "laion/clap-htsat-tiny",
            minimumProfile: .phone,
            qualityTier: 2,
            sampleRate: 48000
        ),
        ModelVariant(
            id: "clap-htsat-fused",
            displayName: "CLAP Fused",
            task: .embedding,
            parameterCount: 200_000_000,
            estimatedMemoryMB: 800,
            quantizedMemoryMB: 250,
            repoId: "laion/clap-htsat-fused",
            minimumProfile: .tablet,
            qualityTier: 4,
            sampleRate: 48000
        ),
    ]

    // MARK: - EnCodec Variants

    public static let encodecVariants: [ModelVariant] = [
        ModelVariant(
            id: "encodec-24khz",
            displayName: "EnCodec 24kHz",
            task: .codec,
            parameterCount: 15_000_000,
            estimatedMemoryMB: 100,
            quantizedMemoryMB: nil,
            repoId: "facebook/encodec_24khz",
            supportsQuantization: false,  // Codec quality sensitive
            minimumProfile: .phone,
            qualityTier: 3,
            sampleRate: 24000
        ),
        ModelVariant(
            id: "encodec-48khz",
            displayName: "EnCodec 48kHz",
            task: .codec,
            parameterCount: 15_000_000,
            estimatedMemoryMB: 120,
            quantizedMemoryMB: nil,
            repoId: "facebook/encodec_48khz",
            supportsQuantization: false,
            minimumProfile: .phone,
            qualityTier: 4,
            sampleRate: 48000,
            notes: "Higher quality, larger bandwidth"
        ),
    ]

    // MARK: - Registry Access

    /// All available model variants.
    public static var allVariants: [ModelVariant] {
        whisperVariants + htdemucsVariants + musicgenVariants + clapVariants + encodecVariants
    }

    /// Get variants for a specific task.
    public static func variants(for task: AudioTask) -> [ModelVariant] {
        allVariants.filter { $0.task == task }
    }

    /// Get a specific variant by ID.
    public static func variant(id: String) -> ModelVariant? {
        allVariants.first { $0.id == id }
    }

    /// Get variants compatible with a device profile.
    public static func compatibleVariants(for profile: DeviceProfile) -> [ModelVariant] {
        allVariants.filter { $0.isCompatible(with: profile) }
    }

    /// Get recommended variant for a task on a device.
    public static func recommendedVariant(
        for task: AudioTask,
        profile: DeviceProfile
    ) -> ModelVariant? {
        let compatible = variants(for: task)
            .filter { $0.isCompatible(with: profile) }
            .sorted { $0.qualityTier > $1.qualityTier }

        // For constrained devices, also consider memory
        if profile == .phone || profile == .tablet {
            return compatible.first {
                $0.effectiveMemoryMB(for: profile) <= Int(profile.perModelBudgetMB)
            }
        }

        return compatible.first
    }

    /// Get all recommended variants for a device profile.
    public static func recommendedVariants(for profile: DeviceProfile) -> [ModelVariant] {
        AudioTask.allCases.compactMap { task in
            recommendedVariant(for: task, profile: profile)
        }
    }
}


// MARK: - Model Combinations

/// Recommended model combinations for different use cases.
public struct ModelCombination: Sendable {
    /// Descriptive name for the combination.
    public let name: String

    /// Model IDs in this combination.
    public let modelIds: [String]

    /// Total estimated memory in MB.
    public let totalMemoryMB: Int

    /// Minimum device profile required.
    public let minimumProfile: DeviceProfile

    /// Use case description.
    public let useCase: String

    /// Get variants in this combination.
    public var variants: [ModelVariant] {
        modelIds.compactMap { ModelVariantRegistry.variant(id: $0) }
    }
}

/// Predefined model combinations.
public struct ModelCombinations {

    /// Minimal setup for iPhone (transcription + codec).
    public static let phoneMinimal = ModelCombination(
        name: "Phone Minimal",
        modelIds: ["whisper-small", "encodec-24khz"],
        totalMemoryMB: 600,
        minimumProfile: .phone,
        useCase: "Basic transcription and audio codec"
    )

    /// Full audio suite for iPhone (with quantization).
    public static let phoneFull = ModelCombination(
        name: "Phone Full (Quantized)",
        modelIds: ["whisper-small", "htdemucs", "clap-htsat-tiny", "encodec-24khz"],
        totalMemoryMB: 900,  // With int4 quantization
        minimumProfile: .phone,
        useCase: "Full audio processing with quantized models"
    )

    /// Balanced setup for iPad.
    public static let tabletBalanced = ModelCombination(
        name: "Tablet Balanced",
        modelIds: ["whisper-medium", "htdemucs_ft", "clap-htsat-fused", "encodec-24khz"],
        totalMemoryMB: 3500,
        minimumProfile: .tablet,
        useCase: "High quality audio processing"
    )

    /// Full suite for Mac.
    public static let macFull = ModelCombination(
        name: "Mac Full",
        modelIds: ["whisper-large-v3-turbo", "htdemucs_ft", "musicgen-medium", "clap-htsat-fused", "encodec-48khz"],
        totalMemoryMB: 8000,
        minimumProfile: .mac,
        useCase: "Complete audio processing and generation"
    )

    /// Pro setup for Mac Studio/Pro.
    public static let macProComplete = ModelCombination(
        name: "Mac Pro Complete",
        modelIds: ["whisper-large-v3", "htdemucs_6s", "musicgen-large", "clap-htsat-fused", "encodec-48khz"],
        totalMemoryMB: 14000,
        minimumProfile: .macPro,
        useCase: "Maximum quality for professional use"
    )

    /// Get recommended combination for a device profile.
    public static func recommended(for profile: DeviceProfile) -> ModelCombination {
        switch profile {
        case .phone: return phoneFull
        case .tablet: return tabletBalanced
        case .mac: return macFull
        case .macPro: return macProComplete
        }
    }
}
