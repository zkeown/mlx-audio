# ``MLXAudioTraining``

On-device training for audio models.

## Overview

MLXAudioTraining provides a Lightning-like training framework for fine-tuning audio models on Apple Silicon devices.

### Quick Start

```swift
import MLXAudioTraining
import MLXAudioModels

// Define your training module
class AudioClassifier: TrainModule {
    let clap: CLAP
    let classifier: Linear

    init(numClasses: Int) async throws {
        self.clap = try await CLAP.fromPretrained("clap-htsat-fused")
        self.classifier = Linear(inputDim: 512, outputDim: numClasses)
    }

    func forward(_ audio: MLXArray) -> MLXArray {
        let embeddings = clap.encodeAudio(audio)
        return classifier(embeddings)
    }

    func trainingStep(batch: Batch) -> Loss {
        let logits = forward(batch.audio)
        return crossEntropyLoss(logits, batch.labels)
    }
}

// Create trainer
let trainer = Trainer(
    config: TrainerConfig(
        maxEpochs: 10,
        learningRate: 1e-4,
        batchSize: 32
    )
)

// Train
let model = try await AudioClassifier(numClasses: 10)
try await trainer.fit(model, dataLoader: trainLoader)
```

### LoRA Fine-Tuning

For memory-efficient fine-tuning:

```swift
import MLXAudioTraining

// Apply LoRA to CLAP
let clap = try await CLAP.fromPretrained("clap-htsat-fused")
let loraClap = LoRAUtils.apply(clap, rank: 8, alpha: 16)

// Only LoRA parameters are trainable
let trainableParams = loraClap.trainableParameters
print("Trainable: \(trainableParams.count)")
```

## Topics

### Training

- ``Trainer``
- ``TrainerConfig``
- ``TrainModule``

### Optimization

- ``OptimizerConfig``
- ``LRScheduler``

### LoRA

- ``LoRALinear``
- ``LoRAUtils``

### Data Loading

- ``AudioDataLoader``
- ``Batch``

### Callbacks

- ``TrainerCallback``
- ``EarlyStopping``
- ``ModelCheckpoint``

### Logging

- ``TrainingLogger``
- ``TensorBoardLogger``
