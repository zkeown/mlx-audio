// MLXAudioTraining.swift
// On-device training and fine-tuning for MLX-Audio models.
//
// This module provides infrastructure for fine-tuning audio models
// on Apple Silicon devices with:
// - Memory-efficient training via LoRA
// - Automatic gradient computation
// - Device-aware memory management
// - Pre-built adapters for CLAP, Whisper, and HTDemucs

import Foundation
import MLX
import MLXNN

// MARK: - Version

/// MLXAudioTraining version.
public let mlxAudioTrainingVersion = "0.1.0"

// MARK: - Quick Start

/// Quick start example for CLAP classification.
///
/// ```swift
/// import MLXAudioTraining
/// import MLXAudioModels
///
/// // Load pretrained CLAP
/// let clap = try CLAPModel.fromPretrained(path: clapPath)
///
/// // Create classifier for 5 sound classes
/// let classifier = CLAPClassifier(
///     clapEncoder: clap,
///     config: CLAPClassifierConfig(numClasses: 5)
/// )
///
/// // Prepare data
/// let dataset = FileAudioDataset(
///     urls: audioFiles,
///     labels: labels,
///     sampleRate: 16000
/// )
/// let loader = AudioDataLoader(dataset: dataset, batchSize: 4)
///
/// // Train
/// let trainer = Trainer(config: TrainerConfig(maxEpochs: 10))
/// trainer.addCallback(EarlyStopping(monitor: "val_loss", patience: 3))
///
/// // Convert loader to array format for training
/// var batches: [[MLXArray]] = []
/// for try await batch in loader {
///     batches.append(batch.toArrays())
/// }
///
/// try trainer.fit(module: classifier, trainData: batches)
///
/// // Save trained classifier
/// try classifier.saveWeights(to: outputPath)
/// ```
