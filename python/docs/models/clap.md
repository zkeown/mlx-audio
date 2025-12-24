# CLAP - Audio-Text Embeddings

CLAP (Contrastive Language-Audio Pretraining) encodes audio and text into a shared embedding space. This enables powerful capabilities like zero-shot classification, audio search, and audio-text similarity matching.

## Quick Start

```python
import mlx_audio as ma

# Zero-shot audio classification
result = ma.classify("sound.wav", labels=["dog barking", "car horn", "music"])
print(f"Detected: {result.predicted_class}")

# Audio embedding for similarity search
result = ma.embed(audio="audio.wav")
print(result.audio_embedding.shape)  # [1, 512]
```

## Available Models

| Model | Description |
|-------|-------------|
| `clap-htsat-fused` | Default model, best general performance |

## Core Functions

CLAP powers three main functions in mlx-audio:

| Function | Purpose |
|----------|---------|
| `embed()` | Generate audio and/or text embeddings |
| `classify()` | Zero-shot audio classification |
| `tag()` | Multi-label audio tagging |

## Audio Embeddings

Generate embeddings for audio files for similarity search, clustering, or as features for downstream tasks.

### Basic Embedding

```python
import mlx_audio as ma

# Get audio embedding
result = ma.embed(audio="audio.wav")
embedding = result.audio_embedding  # [1, 512]
```

### Text Embeddings

```python
import mlx_audio as ma

# Get text embedding
result = ma.embed(text="a dog barking loudly")
embedding = result.text_embedding  # [1, 512]

# Multiple text embeddings
result = ma.embed(text=["dog barking", "cat meowing", "bird singing"])
embeddings = result.text_embedding  # [3, 512]
```

### Audio-Text Similarity

```python
import mlx_audio as ma

# Compute similarity between audio and text descriptions
result = ma.embed(
    audio="sound.wav",
    text=["dog barking", "car horn", "music playing"],
    return_similarity=True
)

# Similarity scores
print(result.similarity)  # [1, 3] array

# Get best matching description
best_match = result.best_match()
print(f"Audio sounds like: {best_match}")
```

## Zero-Shot Classification

Classify audio without training, using natural language descriptions as class labels.

### Basic Classification

```python
import mlx_audio as ma

result = ma.classify(
    "sound.wav",
    labels=["dog barking", "cat meowing", "bird singing"]
)

print(f"Predicted: {result.predicted_class}")
print(f"Confidence: {result.confidence:.1%}")
```

### Top-K Predictions

```python
import mlx_audio as ma

result = ma.classify(
    "environmental_sound.wav",
    labels=["traffic", "rain", "wind", "birds", "construction"],
    top_k=3
)

for label, prob in zip(result.top_k_classes, result.top_k_probs):
    print(f"{label}: {prob:.1%}")
```

### Using Arrays

```python
import mlx_audio as ma
import numpy as np

# 48kHz audio (automatically resampled)
audio = np.random.randn(48000 * 5)  # 5 seconds

result = ma.classify(
    audio,
    sample_rate=48000,
    labels=["speech", "music", "noise"]
)
```

## Multi-Label Tagging

Tag audio with multiple labels that may all apply simultaneously.

### Basic Tagging

```python
import mlx_audio as ma

result = ma.tag(
    "music.wav",
    tags=["jazz", "piano", "upbeat", "slow", "vocals"]
)

print(f"Active tags: {result.active_tags}")
# ['jazz', 'piano', 'upbeat']
```

### Adjusting Threshold

```python
import mlx_audio as ma

# Lower threshold = more tags (higher recall)
result = ma.tag("music.wav", tags=tags, threshold=0.3)

# Higher threshold = fewer tags (higher precision)
result = ma.tag("music.wav", tags=tags, threshold=0.7)
```

### Tag Probabilities

```python
import mlx_audio as ma

result = ma.tag(
    "audio.wav",
    tags=["speech", "music", "noise", "silence"]
)

# Access all probabilities
for tag, prob in zip(result.tags, result.probabilities):
    status = "Active" if prob > result.threshold else "Inactive"
    print(f"{tag}: {prob:.1%} ({status})")
```

## Advanced Usage

### Similarity Search

Build a simple audio search system:

```python
import mlx_audio as ma
import mlx.core as mx

# Build index of audio embeddings
audio_files = ["audio1.wav", "audio2.wav", "audio3.wav"]
embeddings = []

for f in audio_files:
    result = ma.embed(audio=f)
    embeddings.append(result.audio_embedding)

index = mx.concatenate(embeddings, axis=0)  # [N, 512]

# Search by text query
query = ma.embed(text="upbeat electronic music")
similarities = query.text_embedding @ index.T  # [1, N]

# Get most similar audio
best_idx = int(mx.argmax(similarities))
print(f"Best match: {audio_files[best_idx]}")
```

### Audio Clustering

```python
import mlx_audio as ma
import numpy as np
from sklearn.cluster import KMeans

# Get embeddings for multiple audio files
embeddings = []
for audio_file in audio_files:
    result = ma.embed(audio=audio_file)
    embeddings.append(result.audio_embedding.numpy())

# Cluster with K-means
X = np.vstack(embeddings)
kmeans = KMeans(n_clusters=5)
clusters = kmeans.fit_predict(X)
```

### Custom Classifiers

Train a classifier on CLAP embeddings for your specific use case:

```python
import mlx_audio as ma
from sklearn.linear_model import LogisticRegression

# Extract features
X_train = []
y_train = []

for audio_file, label in training_data:
    result = ma.embed(audio=audio_file)
    X_train.append(result.audio_embedding.numpy().flatten())
    y_train.append(label)

# Train classifier
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Predict on new audio
result = ma.embed(audio="new_audio.wav")
prediction = clf.predict(result.audio_embedding.numpy())
```

## Working with Results

### CLAPEmbeddingResult

```python
result = ma.embed(audio="audio.wav", text=["label1", "label2"])

# Access embeddings
audio_emb = result.audio_embedding   # [1, 512] mx.array
text_emb = result.text_embedding     # [2, 512] mx.array

# Get similarity (if return_similarity=True)
sim = result.similarity              # [1, 2] mx.array

# Get best matching text
best = result.best_match()           # "label1" or "label2"

# Convert to numpy
numpy_emb = audio_emb.numpy()
```

### ClassificationResult

```python
result = ma.classify("audio.wav", labels=labels)

# Top prediction
print(result.predicted_class)        # str
print(result.confidence)             # float

# All probabilities
print(result.probabilities)          # mx.array [num_labels]
print(result.class_names)            # list[str]

# Top-k predictions
print(result.top_k_classes)          # list[str]
print(result.top_k_probs)            # list[float]
```

### TaggingResult

```python
result = ma.tag("audio.wav", tags=tags)

# Active tags (above threshold)
print(result.active_tags)            # list[str]

# All tags and probabilities
print(result.tags)                   # list[str]
print(result.probabilities)          # list[float]
print(result.threshold)              # float
```

## Tips for Better Results

### Label Design

Use descriptive, natural language labels:

```python
# Good labels (descriptive)
labels = [
    "a dog barking loudly",
    "a cat meowing softly",
    "birds chirping in a forest"
]

# Less effective (too short)
labels = ["dog", "cat", "bird"]
```

### Audio Quality

- CLAP works best with 48kHz audio (automatic resampling handles other rates)
- Longer audio clips (3-10 seconds) generally work better than very short clips
- Consider using `enhance()` for noisy audio

### Classification Strategies

```python
# For fine-grained classification, use more specific labels
labels = [
    "acoustic guitar playing melody",
    "electric guitar with distortion",
    "bass guitar playing deep notes"
]

# For coarse classification, use broader labels
labels = ["guitar", "piano", "drums", "vocals"]
```

## Technical Details

- **Embedding dimension**: 512
- **Audio sample rate**: 48,000 Hz (automatic resampling)
- **Audio length**: Variable (typically 1-10 seconds works best)
- **Text tokenizer**: RoBERTa-based (max 77 tokens)

## API Reference

::: mlx_audio.embed
    options:
      show_root_heading: false
      show_source: false

::: mlx_audio.classify
    options:
      show_root_heading: false
      show_source: false

::: mlx_audio.tag
    options:
      show_root_heading: false
      show_source: false
