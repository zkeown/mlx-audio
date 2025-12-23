# mlx-audio

Complete audio ML toolkit for Apple Silicon using MLX.

```python
import mlx_audio

# Source separation
stems = mlx_audio.separate("song.mp3")
stems.vocals  # Isolated vocal track
stems.drums   # Isolated drum track
stems.save("output/")

# Speech transcription (coming soon)
text = mlx_audio.transcribe("speech.wav")

# Audio generation (coming soon)
audio = mlx_audio.generate("jazz piano, upbeat mood")
```

## Installation

```bash
pip install mlx-audio
```

For development:

```bash
git clone https://github.com/zkeown/mlx-audio
cd mlx-audio/python
pip install -e ".[dev]"
```

## Features

### Audio Primitives

Production-grade DSP operations optimized for Apple Silicon:

```python
from mlx_audio.primitives import stft, istft, melspectrogram, mfcc

# Compute spectrogram
spec = stft(audio, n_fft=2048, hop_length=512)

# Mel spectrogram
mel = melspectrogram(audio, n_fft=2048, n_mels=128)

# MFCC features
mfcc_features = mfcc(audio, n_mfcc=13)
```

### Data Loading

PyTorch-compatible DataLoader with async prefetching:

```python
from mlx_audio.data import DataLoader, Dataset

loader = DataLoader(dataset, batch_size=32, num_workers=4)
for batch in loader:
    # Process batch
    pass
```

### Training

Lightning-like training framework:

```python
from mlx_audio.train import TrainModule, Trainer

class MyModel(TrainModule):
    def compute_loss(self, batch):
        x, y = batch
        pred = self(x)
        loss = mx.mean((pred - y) ** 2)
        return loss, {"mse": loss}

    def configure_optimizers(self):
        return OptimizerConfig(optimizer=optim.Adam(learning_rate=1e-3))

trainer = Trainer(max_epochs=100)
trainer.fit(model, train_loader, val_loader)
```

### Source Separation

Separate audio into stems using HTDemucs:

```python
import mlx_audio

stems = mlx_audio.separate("song.mp3")
stems.vocals.save("vocals.wav")
stems.drums.save("drums.wav")
stems.bass.save("bass.wav")
stems.other.save("other.wav")
```

## Architecture

```
mlx_audio/
├── primitives/    # Audio DSP (STFT, Mel, MFCC, etc.)
├── data/          # DataLoader and datasets
├── train/         # Training framework
├── models/        # Model implementations (Demucs, etc.)
├── functional/    # High-level API
├── hub/           # Model registry and caching
└── types/         # Result types
```

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.11+
- MLX 0.30.0+

## License

MIT
