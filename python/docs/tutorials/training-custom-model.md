# Training Custom Models

This tutorial walks through training custom models using mlx-audio's Lightning-like training framework. We'll cover two examples: a simple CNN classifier and fine-tuning a pre-trained CLAP model.

## Overview

The mlx-audio training framework provides:

- **TrainModule** — Base class for defining models with `compute_loss` and `configure_optimizers`
- **Trainer** — Handles the training loop, callbacks, and logging
- **DataLoader** — PyTorch-compatible data loading with MLX optimizations
- **Callbacks** — Modular training behaviors (checkpointing, early stopping, etc.)
- **Loggers** — Track metrics with Weights & Biases, TensorBoard, or MLflow

## Example 1: Simple CNN Classifier

Let's build a CNN classifier for MNIST-like data.

### Step 1: Define the Dataset

```python
import numpy as np
from mlx_audio.data.data.dataset import Dataset

class MyDataset(Dataset):
    """Custom dataset for image classification."""

    def __init__(self, size: int = 1000, train: bool = True):
        self.size = size
        np.random.seed(42 if train else 123)

        # Generate synthetic data (replace with real data loading)
        self._images = np.random.randn(size, 28, 28).astype(np.float32)
        self._labels = np.random.randint(0, 10, size=size)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> tuple:
        return self._images[idx], int(self._labels[idx])
```

### Step 2: Define the TrainModule

```python
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_audio import TrainModule, OptimizerConfig
from mlx_audio.train.schedulers import WarmupCosineScheduler

class MNISTClassifier(TrainModule):
    """CNN classifier for MNIST."""

    def __init__(self):
        super().__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def __call__(self, x: mx.array) -> mx.array:
        # Add channel dimension: [B, 28, 28] -> [B, 28, 28, 1]
        if x.ndim == 3:
            x = x[:, :, :, None]

        # Conv blocks
        x = nn.relu(self.conv1(x))
        x = self.pool(x)
        x = nn.relu(self.conv2(x))
        x = self.pool(x)

        # Flatten and FC
        x = x.reshape(x.shape[0], -1)
        x = nn.relu(self.fc1(x))
        return self.fc2(x)

    def compute_loss(self, batch: tuple) -> tuple[mx.array, dict[str, mx.array]]:
        """Compute cross-entropy loss and accuracy."""
        images, labels = batch

        logits = self(images)
        loss = mx.mean(nn.losses.cross_entropy(logits, labels))

        predictions = mx.argmax(logits, axis=-1)
        accuracy = mx.mean(predictions == labels)

        return loss, {"accuracy": accuracy}

    def configure_optimizers(self) -> OptimizerConfig:
        """Configure AdamW with warmup + cosine decay."""
        total_steps = self.trainer.max_steps or 1000
        warmup_steps = min(100, total_steps // 10)

        schedule = WarmupCosineScheduler(
            peak_lr=1e-3,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr=1e-5,
        )

        return OptimizerConfig(
            optimizer=optim.AdamW(learning_rate=schedule, weight_decay=0.01),
            lr_schedule_name="warmup_cosine",
        )
```

### Step 3: Set Up DataLoader

```python
import numpy as np
import mlx.core as mx
from mlx_audio import DataLoader

def to_mlx_batch(batch: list) -> tuple[mx.array, mx.array]:
    """Convert batch to MLX arrays."""
    images = np.stack([item[0] for item in batch])
    labels = np.array([item[1] for item in batch])
    return mx.array(images), mx.array(labels)

# Create datasets
train_dataset = MyDataset(size=5000, train=True)
val_dataset = MyDataset(size=1000, train=False)

# Create dataloaders
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    collate_fn=lambda x: x,  # Return list
    mlx_transforms=to_mlx_batch,  # Convert to MLX in main thread
)

val_loader = DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False,
    collate_fn=lambda x: x,
    mlx_transforms=to_mlx_batch,
)
```

### Step 4: Configure Trainer and Callbacks

```python
from mlx_audio import Trainer
from mlx_audio.train.callbacks import (
    ProgressBar,
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)

# Create callbacks
callbacks = [
    ProgressBar(refresh_rate=10),
    LearningRateMonitor(logging_interval="step"),
    ModelCheckpoint(
        dirpath="./checkpoints",
        monitor="val_accuracy",
        mode="max",
        save_top_k=2,
    ),
    EarlyStopping(
        monitor="val_accuracy",
        mode="max",
        patience=5,
        min_delta=0.001,
    ),
]

# Create trainer
trainer = Trainer(
    max_epochs=10,
    gradient_clip_val=1.0,
    callbacks=callbacks,
    seed=42,
)
```

### Step 5: Train

```python
model = MNISTClassifier()
print(f"Parameters: {sum(p.size for p in model.parameters().values()):,}")

trainer.fit(model, train_loader, val_loader)
```

## Example 2: Fine-tuning CLAP

This example shows how to fine-tune a pre-trained CLAP model for audio-text alignment.

### Step 1: Create Audio-Text Dataset

```python
import numpy as np
from mlx_audio.data.data.dataset import Dataset

class AudioTextDataset(Dataset):
    """Dataset of audio-text pairs."""

    def __init__(self, size: int = 500, train: bool = True):
        self.size = size
        np.random.seed(42 if train else 123)

        # Generate synthetic mel spectrograms and token IDs
        # In practice, load real audio and tokenize captions
        self._mels = [
            np.random.randn(1, 64, 256).astype(np.float32)
            for _ in range(size)
        ]
        self._input_ids = [
            np.random.randint(0, 50265, size=64).astype(np.int32)
            for _ in range(size)
        ]
        self._attention_masks = [
            np.ones(64, dtype=np.int32)
            for _ in range(size)
        ]

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> dict:
        return {
            "mel": self._mels[idx],
            "input_ids": self._input_ids[idx],
            "attention_mask": self._attention_masks[idx],
        }
```

### Step 2: Define CLAP TrainModule

```python
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_audio import TrainModule, OptimizerConfig
from mlx_audio.train.schedulers import WarmupLinearScheduler

class CLAPTrainModule(TrainModule):
    """Fine-tuning wrapper for CLAP."""

    def __init__(self, freeze_audio: bool = False, freeze_text: bool = True):
        super().__init__()

        # Load pre-trained CLAP
        from mlx_audio.models.clap import CLAP
        self.clap = CLAP()

        # Freeze encoders for efficient fine-tuning
        if freeze_audio:
            self.clap.audio_encoder.freeze()
            self.clap.audio_projection.freeze()
        if freeze_text:
            self.clap.text_encoder.freeze()

    def __call__(self, audio=None, input_ids=None, attention_mask=None):
        return self.clap(audio, input_ids, attention_mask)

    def compute_loss(self, batch: tuple) -> tuple[mx.array, dict[str, mx.array]]:
        """Compute contrastive loss (InfoNCE)."""
        mel, input_ids, attention_mask = batch

        outputs = self(audio=mel, input_ids=input_ids, attention_mask=attention_mask)

        logits_audio = outputs["logits_per_audio"]
        logits_text = outputs["logits_per_text"]
        batch_size = mel.shape[0]

        # Labels: diagonal is the correct match
        labels = mx.arange(batch_size)

        # Symmetric cross-entropy loss
        loss_audio = mx.mean(nn.losses.cross_entropy(logits_audio, labels))
        loss_text = mx.mean(nn.losses.cross_entropy(logits_text, labels))
        loss = (loss_audio + loss_text) / 2

        # Compute retrieval accuracy
        audio_preds = mx.argmax(logits_audio, axis=-1)
        text_preds = mx.argmax(logits_text, axis=-1)

        return loss, {
            "audio_acc": mx.mean(audio_preds == labels),
            "text_acc": mx.mean(text_preds == labels),
        }

    def configure_optimizers(self) -> OptimizerConfig:
        """Configure optimizer for fine-tuning."""
        total_steps = self.trainer.max_steps or 1000
        warmup_steps = min(200, total_steps // 5)

        schedule = WarmupLinearScheduler(
            peak_lr=5e-5,  # Lower LR for fine-tuning
            warmup_steps=warmup_steps,
            total_steps=total_steps,
        )

        return OptimizerConfig(
            optimizer=optim.AdamW(learning_rate=schedule, weight_decay=0.01),
            lr_schedule_name="warmup_linear",
        )
```

### Step 3: Set Up DataLoader with Custom Collate

```python
import numpy as np
import mlx.core as mx
from mlx_audio import DataLoader

def collate_fn(batch: list) -> tuple:
    """Collate audio-text pairs."""
    mels = np.stack([item["mel"] for item in batch])
    input_ids = np.stack([item["input_ids"] for item in batch])
    attention_masks = np.stack([item["attention_mask"] for item in batch])
    return mels, input_ids, attention_masks

def to_mlx_batch(batch: tuple) -> tuple:
    """Convert to MLX arrays."""
    mels, input_ids, attention_masks = batch
    return mx.array(mels), mx.array(input_ids), mx.array(attention_masks)

train_dataset = AudioTextDataset(size=500, train=True)
val_dataset = AudioTextDataset(size=100, train=False)

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=collate_fn,
    mlx_transforms=to_mlx_batch,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False,
    collate_fn=collate_fn,
    mlx_transforms=to_mlx_batch,
)
```

### Step 4: Train with Gradient Clipping

```python
from mlx_audio import Trainer
from mlx_audio.train.callbacks import (
    ProgressBar,
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    GradientClipper,
)
from mlx_audio.train.loggers import WandbLogger

model = CLAPTrainModule(freeze_audio=False, freeze_text=True)

# Check trainable parameters
total = sum(p.size for p in model.parameters().values())
trainable = sum(p.size for p in model.trainable_parameters().values())
print(f"Total: {total:,}, Trainable: {trainable:,}")

callbacks = [
    ProgressBar(refresh_rate=10),
    LearningRateMonitor(logging_interval="step"),
    GradientClipper(max_norm=1.0),  # Important for contrastive learning
    ModelCheckpoint(
        dirpath="./checkpoints/clap",
        monitor="val_loss",
        mode="min",
        save_top_k=2,
    ),
    EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=5,
    ),
]

# Optional: Add logging
logger = WandbLogger(project="clap-finetune", name="experiment-1")

trainer = Trainer(
    max_epochs=10,
    callbacks=callbacks,
    logger=logger,
    seed=42,
)

trainer.fit(model, train_loader, val_loader)
```

## Resuming from Checkpoint

Resume training from a saved checkpoint:

```python
trainer.fit(model, train_loader, val_loader, ckpt_path="./checkpoints/last")
```

## Tips and Best Practices

### 1. Use Appropriate Learning Rates

- **Training from scratch**: 1e-3 to 1e-4
- **Fine-tuning**: 1e-5 to 5e-5
- **Always use warmup** for stable training

### 2. Gradient Clipping

Essential for training stability, especially with:
- Contrastive losses
- Transformer models
- High learning rates

```python
trainer = Trainer(gradient_clip_val=1.0, ...)
# or
callbacks = [GradientClipper(max_norm=1.0)]
```

### 3. Freeze Layers for Efficient Fine-tuning

```python
# Freeze specific layers
model.some_layer.freeze()

# Unfreeze for fine-tuning
model.some_layer.unfreeze()
```

### 4. Use mlx_transforms for MLX Conversion

Keep data as numpy in workers, convert to MLX in the main thread:

```python
train_loader = DataLoader(
    dataset,
    cpu_transforms=augment_fn,    # Runs in workers (numpy)
    mlx_transforms=to_mlx_batch,  # Runs in main thread (mx.array)
)
```

### 5. Monitor Multiple Metrics

```python
def compute_loss(self, batch):
    loss = ...
    return loss, {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
    }
```

All metrics are automatically logged and available for monitoring in callbacks.

## Next Steps

- [API Reference: Training](../api/training.md) — Complete API documentation
- [API Reference: Models](../api/models.md) — Pre-trained models available
- [Contributing](../contributing.md) — Add your own models
