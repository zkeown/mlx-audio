# Custom Models

This guide covers loading custom or fine-tuned models with mlx-audio.

## Loading from Local Path

### Converted Models

If you have a model converted to MLX format:

```python
import mlx_audio as ma

# Load from local directory
result = ma.transcribe("audio.wav", model="./my_whisper_model")
result = ma.separate("song.mp3", model="./my_htdemucs_model")
```

### Model Directory Structure

MLX models expect this structure:

```
my_model/
├── config.json        # Model configuration
├── weights.safetensors  # Model weights
└── tokenizer/         # (Optional) Tokenizer files
```

## Converting PyTorch Models

### Using Conversion Scripts

Each model type has a conversion script:

```bash
# Convert Whisper
cd python/mlx_audio/models/whisper
python convert.py --model-path ./pytorch_model --output ./mlx_model

# Convert HTDemucs
cd python/mlx_audio/models/demucs
python convert.py --model-path ./pytorch_model --output ./mlx_model

# Convert CLAP
cd python/mlx_audio/models/clap
python convert.py --model-path ./pytorch_model --output ./mlx_model
```

### Conversion Example

```python
from mlx_audio.models.whisper.convert import convert_whisper

# Convert a Whisper model
convert_whisper(
    pytorch_path="./whisper-finetuned",
    output_path="./whisper-finetuned-mlx",
)
```

## Fine-Tuned Models

### Loading Fine-Tuned Weights

```python
from mlx_audio.models import Whisper
import mlx.core as mx

# Load base model
model = Whisper.from_pretrained("whisper-small")

# Load fine-tuned weights
weights = mx.load("./finetuned_weights.safetensors")
model.load_weights(weights)

# Use the model
from mlx_audio.models.whisper import apply_model
result = apply_model(model, audio)
```

### LoRA Weights

For LoRA fine-tuned models:

```python
from mlx_audio.models import Whisper
from mlx_audio.train import LoRAUtils

# Load base model
model = Whisper.from_pretrained("whisper-small")

# Apply LoRA
model = LoRAUtils.apply(model, rank=8)

# Load LoRA weights
lora_weights = mx.load("./lora_weights.safetensors")
model.load_weights(lora_weights)
```

## HuggingFace Hub Models

### Loading from Hub

Models on HuggingFace Hub with MLX format:

```python
import mlx_audio as ma

# Load from Hub (if MLX-compatible)
result = ma.transcribe("audio.wav", model="username/whisper-finetuned-mlx")
```

### Pushing to Hub

Share your converted models:

```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="./my_model_mlx",
    repo_id="username/my-model-mlx",
    repo_type="model",
)
```

## Model Registry

### Registering Custom Models

```python
from mlx_audio.hub.registry import ModelRegistry

# Register a custom model
ModelRegistry.register(
    name="my-whisper",
    model_class="Whisper",
    path="./my_whisper_model",
)

# Now use it by name
result = ma.transcribe("audio.wav", model="my-whisper")
```

### Listing Available Models

```python
from mlx_audio.hub.registry import ModelRegistry

# List all registered models
for name, info in ModelRegistry.list_models().items():
    print(f"{name}: {info['path']}")
```

## Custom Model Classes

### Extending Base Models

```python
from mlx_audio.models import Whisper
import mlx.nn as nn

class CustomWhisper(Whisper):
    """Custom Whisper with additional processing."""

    def __init__(self, config):
        super().__init__(config)
        # Add custom layers
        self.custom_layer = nn.Linear(config.n_audio_state, config.n_audio_state)

    def encode(self, mel):
        # Custom encoding logic
        features = super().encode(mel)
        features = self.custom_layer(features)
        return features
```

### Using Custom Classes

```python
from mlx_audio.hub.cache import get_cache

cache = get_cache()
model = cache.get_model("whisper-small", CustomWhisper)
```

## Configuration

### Custom Config Files

Create a `config.json` for your model:

```json
{
    "n_mels": 128,
    "n_audio_ctx": 1500,
    "n_audio_state": 1024,
    "n_audio_head": 16,
    "n_audio_layer": 24,
    "n_vocab": 51865,
    "n_text_ctx": 448,
    "n_text_state": 1024,
    "n_text_head": 16,
    "n_text_layer": 24
}
```

### Loading with Custom Config

```python
from mlx_audio.models import Whisper, WhisperConfig
import json

# Load custom config
with open("./my_model/config.json") as f:
    config_dict = json.load(f)

config = WhisperConfig(**config_dict)
model = Whisper(config)
model.load_weights("./my_model/weights.safetensors")
```

## Troubleshooting

### Weight Mismatch Errors

If weights don't match:

```python
import mlx.core as mx

# Load weights and inspect
weights = mx.load("weights.safetensors")
for key in sorted(weights.keys())[:10]:
    print(f"{key}: {weights[key].shape}")

# Compare with model
model = Whisper.from_pretrained("whisper-small")
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")
```

### Missing Keys

For partial weight loading:

```python
# Load with strict=False to ignore missing keys
model.load_weights(weights, strict=False)
```

### Shape Mismatches

Ensure your config matches the weights:

```python
# Check config
print(model.config)

# Adjust if needed
model.config.n_audio_layer = 24  # Match your weights
```
