# Training

A Lightning-like training framework for MLX, providing a high-level API for training custom models.

```python
from mlx_audio import TrainModule, Trainer, OptimizerConfig
from mlx_audio.train.callbacks import ModelCheckpoint, EarlyStopping
from mlx_audio.train.schedulers import WarmupCosineScheduler
from mlx_audio.train.loggers import WandbLogger
```

## Quick Example

```python
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_audio import TrainModule, Trainer, OptimizerConfig

class MyModel(TrainModule):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(784, 10)

    def __call__(self, x):
        return self.linear(x)

    def compute_loss(self, batch):
        x, y = batch
        logits = self(x)
        loss = mx.mean(nn.losses.cross_entropy(logits, y))
        accuracy = mx.mean(mx.argmax(logits, axis=-1) == y)
        return loss, {"accuracy": accuracy}

    def configure_optimizers(self):
        return OptimizerConfig(optimizer=optim.AdamW(learning_rate=1e-4))

# Train
model = MyModel()
trainer = Trainer(max_epochs=10)
trainer.fit(model, train_loader, val_loader)
```

## TrainModule

Base class for all training modules. Subclass this to define your model.

### Required Methods

#### compute_loss

```python
def compute_loss(self, batch) -> tuple[mx.array, dict[str, mx.array]]:
    """Compute loss and metrics for a batch.

    Args:
        batch: Data from the dataloader (structure is user-defined)

    Returns:
        Tuple of (loss, metrics_dict) where both values are scalar mx.arrays
    """
    x, y = batch
    logits = self(x)
    loss = mx.mean(nn.losses.cross_entropy(logits, y))
    accuracy = mx.mean(mx.argmax(logits, axis=-1) == y)
    return loss, {"accuracy": accuracy}
```

#### configure_optimizers

```python
def configure_optimizers(self) -> OptimizerConfig:
    """Configure the optimizer for training.

    Returns:
        OptimizerConfig with optimizer and optional schedule name
    """
    return OptimizerConfig(
        optimizer=optim.AdamW(learning_rate=1e-4, weight_decay=0.01)
    )
```

### Optional Methods

| Method | Description |
|--------|-------------|
| `validation_step(batch)` | Custom validation logic (default: uses compute_loss with "val_" prefix) |
| `test_step(batch)` | Custom test logic (default: uses compute_loss with "test_" prefix) |

### Lifecycle Hooks

| Hook | When Called |
|------|-------------|
| `on_train_start()` | Beginning of training |
| `on_train_end()` | End of training |
| `on_train_epoch_start(epoch)` | Start of each epoch |
| `on_train_epoch_end(epoch, metrics)` | End of each epoch |
| `on_validation_start()` | Beginning of validation |
| `on_validation_end(metrics)` | End of validation |

### Properties

| Property | Description |
|----------|-------------|
| `trainer` | Reference to the Trainer instance |
| `current_epoch` | Current epoch number (0-indexed) |
| `global_step` | Total optimizer steps taken |

### Logging

```python
# Log single metric
self.log("train_loss", loss)

# Log multiple metrics
self.log_dict({"loss": loss, "accuracy": acc})
```

## Trainer

Orchestrates the training loop with proper MLX patterns.

### Configuration

```python
trainer = Trainer(
    # Duration (one required)
    max_epochs=10,              # Maximum epochs
    max_steps=10000,            # Maximum steps (takes precedence)

    # Validation
    val_check_interval=1.0,     # float: fraction of epoch, int: every N steps

    # Gradient clipping
    gradient_clip_val=1.0,      # Max gradient norm/value (None to disable)
    gradient_clip_algorithm="norm",  # "norm" or "value"

    # Checkpointing
    default_root_dir="./logs",  # Directory for checkpoints
    enable_checkpointing=True,  # Auto-save checkpoints

    # Performance
    compile=True,               # Use mx.compile for training step
    seed=42,                    # Random seed

    # Callbacks and logging
    callbacks=[...],            # List of Callback instances
    logger=WandbLogger(...),    # Logger or list of loggers

    # Debugging
    debug_lazy_eval=False,      # Debug lazy evaluation issues
)
```

### Methods

```python
# Train the model
trainer.fit(model, train_loader, val_loader, ckpt_path=None)

# Run validation only
trainer.validate(model, val_loader, ckpt_path=None)

# Run testing
trainer.test(model, test_loader, ckpt_path=None)

# Manual checkpoint management
trainer.save_checkpoint("path/to/checkpoint")
trainer.load_checkpoint("path/to/checkpoint")
```

## Callbacks

Callbacks customize training behavior at specific lifecycle points.

### Built-in Callbacks

#### ModelCheckpoint

Save model checkpoints during training.

```python
from mlx_audio.train.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint(
    dirpath="./checkpoints",
    filename="model-{epoch:02d}-{val_loss:.2f}",
    monitor="val_loss",
    mode="min",              # "min" or "max"
    save_last=True,          # Save latest checkpoint
    save_top_k=3,            # Keep top 3 checkpoints
    every_n_epochs=1,        # Save frequency
)
```

#### EarlyStopping

Stop training when a metric stops improving.

```python
from mlx_audio.train.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,              # Epochs without improvement
    min_delta=0.001,         # Minimum change to qualify as improvement
    mode="min",              # "min" or "max"
    verbose=True,
)
```

#### ProgressBar

Display training progress.

```python
from mlx_audio.train.callbacks import ProgressBar

progress = ProgressBar(refresh_rate=10)
```

#### LearningRateMonitor

Log learning rate at each step.

```python
from mlx_audio.train.callbacks import LearningRateMonitor

lr_monitor = LearningRateMonitor(logging_interval="step")  # "step" or "epoch"
```

#### GradientClipper

Clip gradients during training.

```python
from mlx_audio.train.callbacks import GradientClipper

clipper = GradientClipper(max_norm=1.0)
```

### Custom Callbacks

```python
from mlx_audio.train.callbacks import Callback

class MyCallback(Callback):
    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        if batch_idx % 100 == 0:
            print(f"Step {trainer.global_step}: loss = {outputs['loss']:.4f}")
```

## Loggers

Track metrics during training.

### WandbLogger

```python
from mlx_audio.train.loggers import WandbLogger

logger = WandbLogger(
    project="my-project",
    name="experiment-1",
    config={"lr": 1e-4, "batch_size": 32},
)

trainer = Trainer(logger=logger, ...)
```

### TensorBoardLogger

```python
from mlx_audio.train.loggers import TensorBoardLogger

logger = TensorBoardLogger(
    log_dir="./tb_logs",
    name="experiment-1",
)
```

### MLflowLogger

```python
from mlx_audio.train.loggers import MLflowLogger

logger = MLflowLogger(
    experiment_name="my-experiment",
    run_name="run-1",
)
```

### Multiple Loggers

```python
trainer = Trainer(
    logger=[WandbLogger(...), TensorBoardLogger(...)],
    ...
)
```

## Learning Rate Schedulers

MLX optimizers support callable learning rates for scheduling.

### WarmupCosineScheduler

```python
from mlx_audio.train.schedulers import WarmupCosineScheduler

schedule = WarmupCosineScheduler(
    peak_lr=1e-3,
    warmup_steps=500,
    total_steps=10000,
    min_lr=1e-6,
)

optimizer = optim.AdamW(learning_rate=schedule)
```

### WarmupLinearScheduler

```python
from mlx_audio.train.schedulers import WarmupLinearScheduler

schedule = WarmupLinearScheduler(
    peak_lr=1e-3,
    warmup_steps=500,
    total_steps=10000,
)
```

### StepLRScheduler

```python
from mlx_audio.train.schedulers import StepLRScheduler

schedule = StepLRScheduler(
    initial_lr=1e-3,
    step_size=1000,     # Decay every N steps
    gamma=0.9,          # Multiply by gamma
)
```

### ExponentialLRScheduler

```python
from mlx_audio.train.schedulers import ExponentialLRScheduler

schedule = ExponentialLRScheduler(
    initial_lr=1e-3,
    decay_rate=0.9999,
)
```

## DataLoader

PyTorch-compatible DataLoader with MLX optimizations.

```python
from mlx_audio import DataLoader, Dataset

class MyDataset(Dataset):
    def __len__(self):
        return 1000

    def __getitem__(self, idx):
        return {"x": np.random.randn(784), "y": np.random.randint(10)}

dataset = MyDataset()
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,              # 0 for main process
    collate_fn=None,            # Custom batch collation
    drop_last=False,            # Drop incomplete last batch
    prefetch_factor=2,          # Batches per worker to prefetch
    worker_type="thread",       # "thread" or "process"
    persistent_workers=False,   # Keep workers alive between epochs
    cpu_transforms=None,        # Per-item transforms in workers
    mlx_transforms=None,        # Per-batch transforms in main thread
)
```

### Transforms

```python
# CPU transforms run in workers (numpy arrays)
def cpu_transform(item):
    item["x"] = item["x"] / 255.0
    return item

# MLX transforms run in main thread (mx.array)
def mlx_transform(batch):
    batch["x"] = mx.array(batch["x"])
    batch["y"] = mx.array(batch["y"])
    return batch

loader = DataLoader(
    dataset,
    cpu_transforms=cpu_transform,
    mlx_transforms=mlx_transform,
)
```

## Full Example

```python
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_audio import TrainModule, Trainer, OptimizerConfig, DataLoader
from mlx_audio.train.callbacks import ModelCheckpoint, EarlyStopping, ProgressBar
from mlx_audio.train.schedulers import WarmupCosineScheduler
from mlx_audio.train.loggers import WandbLogger

class AudioClassifier(TrainModule):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc = nn.Linear(64 * 6 * 6, num_classes)

    def __call__(self, x):
        x = nn.relu(self.conv1(x))
        x = nn.max_pool2d(x, 2)
        x = nn.relu(self.conv2(x))
        x = nn.max_pool2d(x, 2)
        x = x.reshape(x.shape[0], -1)
        return self.fc(x)

    def compute_loss(self, batch):
        x, y = batch["mel"], batch["label"]
        logits = self(x)
        loss = mx.mean(nn.losses.cross_entropy(logits, y))
        accuracy = mx.mean(mx.argmax(logits, axis=-1) == y)
        return loss, {"accuracy": accuracy}

    def configure_optimizers(self):
        schedule = WarmupCosineScheduler(
            peak_lr=1e-3,
            warmup_steps=500,
            total_steps=self.trainer.max_steps or 10000,
        )
        return OptimizerConfig(
            optimizer=optim.AdamW(learning_rate=schedule, weight_decay=0.01),
            lr_schedule_name="warmup_cosine",
        )

# Create model and trainer
model = AudioClassifier(num_classes=10)

trainer = Trainer(
    max_epochs=50,
    gradient_clip_val=1.0,
    callbacks=[
        ProgressBar(),
        ModelCheckpoint(monitor="val_loss", save_top_k=3),
        EarlyStopping(monitor="val_loss", patience=5),
    ],
    logger=WandbLogger(project="audio-classifier"),
)

# Train
trainer.fit(model, train_loader, val_loader)
```
