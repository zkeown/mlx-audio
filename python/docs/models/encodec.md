# EnCodec - Neural Audio Codec

EnCodec is a neural audio codec from Meta AI that compresses audio to very low bitrates while maintaining high quality. It's used internally by MusicGen and can be used standalone for audio compression.

## Quick Start

```python
from mlx_audio.models import EnCodec

# Load the codec
codec = EnCodec.from_pretrained("encodec-24khz")

# Encode audio to tokens
tokens = codec.encode(audio_array)

# Decode back to audio
reconstructed = codec.decode(tokens)
```

## Available Models

| Model | Sample Rate | Bandwidth | Use Case |
|-------|-------------|-----------|----------|
| `encodec-24khz` | 24 kHz | 1.5-24 kbps | General audio |
| `encodec-48khz` | 48 kHz | 3-24 kbps | High-quality audio |

## Basic Usage

### Loading the Model

```python
from mlx_audio.models import EnCodec

# 24kHz model (default)
codec = EnCodec.from_pretrained("encodec-24khz")

# 48kHz model for higher quality
codec = EnCodec.from_pretrained("encodec-48khz")
```

### Encode and Decode

```python
import mlx.core as mx
from mlx_audio.models import EnCodec

# Load model
codec = EnCodec.from_pretrained("encodec-24khz")

# Your audio (mono, at model's sample rate)
audio = mx.random.normal((1, 24000 * 5))  # 5 seconds at 24kHz

# Encode to discrete tokens
tokens = codec.encode(audio)
print(f"Token shape: {tokens.shape}")  # [batch, codebooks, time]

# Decode back to audio
reconstructed = codec.decode(tokens)
print(f"Audio shape: {reconstructed.shape}")
```

### Adjusting Bandwidth

Control the compression level:

```python
from mlx_audio.models import EnCodec

codec = EnCodec.from_pretrained("encodec-24khz")

# Lower bandwidth = more compression, lower quality
tokens_low = codec.encode(audio, bandwidth=1.5)  # 1.5 kbps

# Higher bandwidth = less compression, higher quality
tokens_high = codec.encode(audio, bandwidth=24.0)  # 24 kbps
```

## Technical Details

### Architecture

EnCodec uses:
- **Encoder**: Convolutional network that compresses audio to latent space
- **Quantizer**: Residual Vector Quantization (RVQ) for discrete tokens
- **Decoder**: Convolutional network that reconstructs audio

### Token Structure

```python
# Tokens shape: [batch, num_codebooks, time_steps]
# num_codebooks depends on bandwidth:
# - 1.5 kbps: 2 codebooks
# - 3 kbps: 4 codebooks
# - 6 kbps: 8 codebooks
# - 12 kbps: 16 codebooks
# - 24 kbps: 32 codebooks
```

### Sample Rates

| Model | Input Sample Rate | Frame Rate |
|-------|-------------------|------------|
| encodec-24khz | 24,000 Hz | 75 Hz |
| encodec-48khz | 48,000 Hz | 50 Hz |

## Use Cases

### Audio Compression

```python
from mlx_audio.models import EnCodec
import mlx.core as mx

codec = EnCodec.from_pretrained("encodec-24khz")

# Compress audio
tokens = codec.encode(audio, bandwidth=6.0)

# Store tokens (much smaller than raw audio)
# tokens can be saved as integers

# Reconstruct later
audio = codec.decode(tokens)
```

### Feature Extraction

Use EnCodec embeddings as audio features:

```python
from mlx_audio.models import EnCodec

codec = EnCodec.from_pretrained("encodec-24khz")

# Get continuous embeddings (before quantization)
embeddings = codec.encode_continuous(audio)
# embeddings shape: [batch, channels, time]
```

### Audio Generation (with MusicGen)

EnCodec is the audio tokenizer for MusicGen:

```python
import mlx_audio as ma

# MusicGen uses EnCodec internally
result = ma.generate("jazz piano")

# The generation process:
# 1. Text → MusicGen language model → tokens
# 2. Tokens → EnCodec decoder → audio
```

## Bandwidth vs Quality

| Bandwidth | Codebooks | Compression Ratio | Quality |
|-----------|-----------|-------------------|---------|
| 1.5 kbps | 2 | ~1000:1 | Acceptable |
| 3 kbps | 4 | ~500:1 | Good |
| 6 kbps | 8 | ~250:1 | Very Good |
| 12 kbps | 16 | ~125:1 | Excellent |
| 24 kbps | 32 | ~60:1 | Near-lossless |

*For comparison, MP3 at 128 kbps has ~11:1 compression*

## Working with Tokens

### Token Statistics

```python
from mlx_audio.models import EnCodec

codec = EnCodec.from_pretrained("encodec-24khz")
tokens = codec.encode(audio, bandwidth=6.0)

# Each codebook has 1024 possible values (0-1023)
print(f"Codebooks: {tokens.shape[1]}")
print(f"Time steps: {tokens.shape[2]}")
print(f"Vocab size: 1024")
```

### Streaming Encoding

For long audio, encode in chunks:

```python
from mlx_audio.models import EnCodec
import mlx.core as mx

codec = EnCodec.from_pretrained("encodec-24khz")

# Process in chunks
chunk_size = 24000 * 10  # 10 seconds
all_tokens = []

for i in range(0, audio.shape[-1], chunk_size):
    chunk = audio[:, i:i+chunk_size]
    tokens = codec.encode(chunk)
    all_tokens.append(tokens)

# Concatenate
all_tokens = mx.concatenate(all_tokens, axis=-1)
```

## Performance

### Speed

| Operation | 10 seconds | 1 minute |
|-----------|------------|----------|
| Encode | ~100ms | ~600ms |
| Decode | ~150ms | ~900ms |

*Benchmarks on M2 Max*

### Memory

| Model | Memory |
|-------|--------|
| encodec-24khz | ~150 MB |
| encodec-48khz | ~200 MB |

## Limitations

- **Lossy compression**: Some audio quality is lost during encoding
- **Mono output**: Stereo is converted to mono
- **Fixed sample rates**: Must use model's native sample rate
- **Latency**: Not suitable for real-time streaming (use for batch processing)

## Common Issues

### Audio sounds distorted

Try higher bandwidth:

```python
tokens = codec.encode(audio, bandwidth=24.0)  # Maximum quality
```

### Wrong sample rate

Resample before encoding:

```python
from mlx_audio.primitives import resample

# Resample to 24kHz for encodec-24khz
audio_24k = resample(audio, original_sr, 24000)
tokens = codec.encode(audio_24k)
```

### Memory issues with long audio

Process in chunks (see streaming example above).

## API Reference

For detailed API documentation, see the models API reference.
