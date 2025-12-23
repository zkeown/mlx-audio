# mlx-audio Training Examples

This directory contains example scripts demonstrating how to use the mlx-audio training framework.

## Quick Start

### Installation

```bash
# Install mlx-audio with training dependencies
cd python
pip install -e ".[train]"

# Optional: Install additional loggers
pip install -e ".[train-tensorboard]"  # TensorBoard support
pip install -e ".[train-mlflow]"       # MLflow support
pip install -e ".[train-all]"          # All training extras
```

### MNIST Classifier (Simple Example)

Demonstrates core framework features with a simple CNN:

```bash
# Basic training (uses synthetic data - no downloads needed)
python examples/train_mnist.py --epochs 5 --batch-size 64

# With TensorBoard logging
python examples/train_mnist.py --epochs 5 --tensorboard
tensorboard --logdir ./runs  # View at http://localhost:6006

# With W&B logging
python examples/train_mnist.py --epochs 5 --wandb

# With MLflow logging
python examples/train_mnist.py --epochs 5 --mlflow
mlflow ui  # View at http://localhost:5000

# Resume from checkpoint
python examples/train_mnist.py --epochs 10 --resume checkpoints/mnist/last
```

### CLAP Fine-tuning (Advanced Example)

Demonstrates fine-tuning a real audio-text model:

```bash
# Fine-tune with synthetic data (demo mode)
python examples/train_clap_finetune.py --epochs 3 --batch-size 8

# Fine-tune with frozen text encoder (recommended)
python examples/train_clap_finetune.py --epochs 10 --freeze-text

# Fine-tune entire model (requires more memory)
python examples/train_clap_finetune.py --epochs 10 --no-freeze-text

# With gradient clipping and logging
python examples/train_clap_finetune.py \
    --epochs 10 \
    --gradient-clip 1.0 \
    --tensorboard
```

## Features Demonstrated

### TrainModule API

Both examples show how to subclass `TrainModule`:

```python
from mlx_audio.train import TrainModule, OptimizerConfig

class MyModel(TrainModule):
    def __init__(self):
        super().__init__()
        # Define model architecture

    def compute_loss(self, batch):
        # Return (loss, metrics_dict)
        loss = ...
        return loss, {"accuracy": accuracy}

    def configure_optimizers(self):
        # Return optimizer configuration
        return OptimizerConfig(
            optimizer=optim.AdamW(learning_rate=1e-3),
            lr_schedule_name="constant"
        )
```

### Learning Rate Schedulers

```python
from mlx_audio.train.schedulers import WarmupCosineScheduler, WarmupLinearScheduler

# Warmup + cosine decay (good for training from scratch)
schedule = WarmupCosineScheduler(
    peak_lr=1e-3,
    warmup_steps=500,
    total_steps=10000,
    min_lr=1e-5
)

# Warmup + linear decay (good for fine-tuning)
schedule = WarmupLinearScheduler(
    peak_lr=5e-5,
    warmup_steps=200,
    total_steps=5000
)
```

### Callbacks

```python
from mlx_audio.train.callbacks import (
    ProgressBar,
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    GradientClipper,
)

callbacks = [
    ProgressBar(refresh_rate=10),
    LearningRateMonitor(logging_interval="step"),
    ModelCheckpoint(
        dirpath="checkpoints",
        monitor="val_loss",
        mode="min",
        save_top_k=2
    ),
    EarlyStopping(
        monitor="val_loss",
        patience=5,
        min_delta=0.001
    ),
    GradientClipper(max_norm=1.0),
]
```

### Loggers

```python
from mlx_audio.train.loggers import WandbLogger, TensorBoardLogger, MLflowLogger

# Weights & Biases
logger = WandbLogger(project="my-project", name="run-1")

# TensorBoard
logger = TensorBoardLogger(log_dir="./runs", name="run-1")

# MLflow
logger = MLflowLogger(experiment_name="my-experiment", run_name="run-1")

# Multiple loggers
trainer = Trainer(logger=[wandb_logger, tensorboard_logger])
```

### DataLoader

```python
from mlx_audio.data import DataLoader

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    collate_fn=my_collate,
    mlx_transforms=to_mlx_batch,  # Runs in main thread
)
```

## Command Line Options

### train_mnist.py

| Option | Default | Description |
|--------|---------|-------------|
| `--epochs` | 5 | Number of training epochs |
| `--batch-size` | 64 | Batch size |
| `--train-size` | 5000 | Training set size (synthetic) |
| `--val-size` | 1000 | Validation set size (synthetic) |
| `--checkpoint-dir` | checkpoints/mnist | Checkpoint directory |
| `--resume` | None | Checkpoint to resume from |
| `--wandb` | False | Enable W&B logging |
| `--tensorboard` | False | Enable TensorBoard logging |
| `--mlflow` | False | Enable MLflow logging |
| `--seed` | 42 | Random seed |

### train_clap_finetune.py

| Option | Default | Description |
|--------|---------|-------------|
| `--epochs` | 3 | Number of training epochs |
| `--batch-size` | 8 | Batch size |
| `--freeze-audio` | False | Freeze audio encoder |
| `--freeze-text` | True | Freeze text encoder |
| `--gradient-clip` | 1.0 | Gradient clipping value |
| `--checkpoint-dir` | checkpoints/clap | Checkpoint directory |
| `--resume` | None | Checkpoint to resume from |
| `--wandb` | False | Enable W&B logging |
| `--tensorboard` | False | Enable TensorBoard logging |
| `--mlflow` | False | Enable MLflow logging |
| `--seed` | 42 | Random seed |

## Troubleshooting

### Import Errors

If you see import errors for loggers:
```bash
pip install tensorboardX  # For TensorBoard
pip install mlflow        # For MLflow
pip install wandb         # For W&B
```

### Memory Issues

For CLAP fine-tuning, if you run out of memory:
1. Reduce batch size: `--batch-size 4`
2. Freeze more layers: `--freeze-text --freeze-audio`
3. Use gradient checkpointing (not yet implemented)

### Checkpoint Resume Issues

Make sure the checkpoint directory exists and contains valid checkpoints:
```bash
ls -la checkpoints/mnist/
# Should show: best/, last/, model.safetensors, etc.
```
