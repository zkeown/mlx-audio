# Result Types

This reference documents all result types returned by mlx-audio functions.

## Overview

Each high-level function returns a specialized result type:

| Function | Result Type |
|----------|-------------|
| `separate()` | `SeparationResult` |
| `transcribe()` | `TranscriptionResult` |
| `generate()` | `GenerationResult` |
| `speak()` | `SpeechResult` |
| `embed()` | `CLAPEmbeddingResult` |
| `classify()` | `ClassificationResult` |
| `tag()` | `TaggingResult` |
| `detect_speech()` | `VADResult` |
| `enhance()` | `EnhancementResult` |
| `diarize()` | `DiarizationResult` |

## AudioData

Base type for audio data, used by all result types.

```python
class AudioData:
    array: mx.array        # Audio samples [channels, samples]
    sample_rate: int       # Sample rate in Hz

    def save(self, path: str | Path) -> None:
        """Save to audio file."""

    def numpy(self) -> np.ndarray:
        """Convert to numpy array."""

    @property
    def shape(self) -> tuple:
        """Shape of audio array."""

    @property
    def duration(self) -> float:
        """Duration in seconds."""
```

## SeparationResult

Returned by `separate()`.

```python
class SeparationResult:
    stems: dict[str, AudioData]  # Separated stems
    sample_rate: int             # Sample rate
    model_name: str              # Model used

    # Stem access as attributes
    @property
    def vocals(self) -> AudioData: ...
    @property
    def drums(self) -> AudioData: ...
    @property
    def bass(self) -> AudioData: ...
    @property
    def other(self) -> AudioData: ...
    @property
    def guitar(self) -> AudioData: ...  # 6-stem model only
    @property
    def piano(self) -> AudioData: ...   # 6-stem model only

    @property
    def stem_names(self) -> list[str]:
        """List of available stem names."""

    def items(self) -> Iterator[tuple[str, AudioData]]:
        """Iterate over (name, audio) pairs."""

    def save(self, output_dir: str | Path) -> None:
        """Save all stems to directory."""
```

### Usage

```python
result = ma.separate("song.mp3")

# Access by attribute
vocals = result.vocals

# Access by key
drums = result.stems["drums"]

# Iterate
for name, audio in result.items():
    audio.save(f"{name}.wav")

# Save all
result.save("./output")
```

## TranscriptionResult

Returned by `transcribe()`.

```python
class TranscriptionResult:
    text: str                    # Full transcription text
    segments: list[Segment]      # Timestamped segments
    language: str                # Detected language code
    duration: float              # Audio duration in seconds
    model_name: str              # Model used

    def save(self, path: str | Path, format: str = "txt") -> None:
        """Save to file (txt, srt, vtt, json)."""

    def to_dict(self) -> dict:
        """Convert to dictionary."""

class Segment:
    start: float                 # Start time in seconds
    end: float                   # End time in seconds
    text: str                    # Segment text
    words: list[Word] | None     # Word-level timing (if enabled)

class Word:
    word: str                    # Word text
    start: float                 # Start time
    end: float                   # End time
    probability: float           # Confidence
```

### Usage

```python
result = ma.transcribe("audio.wav", word_timestamps=True)

# Full text
print(result.text)

# Segments
for seg in result.segments:
    print(f"[{seg.start:.2f} - {seg.end:.2f}] {seg.text}")

    # Words (if enabled)
    if seg.words:
        for word in seg.words:
            print(f"  {word.word}: {word.start:.2f}s")

# Export
result.save("transcript.txt")
result.save("subtitles.srt", format="srt")
```

## GenerationResult

Returned by `generate()`.

```python
class GenerationResult:
    audio: AudioData             # Generated audio
    prompt: str                  # Original prompt
    duration: float              # Duration in seconds
    model_name: str              # Model used

    def save(self, path: str | Path) -> None:
        """Save audio to file."""
```

### Usage

```python
result = ma.generate("jazz piano")

print(f"Duration: {result.duration}s")
result.save("output.wav")
```

## SpeechResult

Returned by `speak()`.

```python
class SpeechResult:
    audio: AudioData             # Generated speech
    text: str                    # Input text
    duration: float              # Duration in seconds
    model_name: str              # Model used

    def save(self, path: str | Path) -> None:
        """Save audio to file."""
```

### Usage

```python
result = ma.speak("Hello world!")

result.save("speech.wav")
print(f"Duration: {result.duration}s")
```

## CLAPEmbeddingResult

Returned by `embed()`.

```python
class CLAPEmbeddingResult:
    audio_embeds: mx.array | None    # Audio embeddings [B, 512]
    text_embeds: mx.array | None     # Text embeddings [B, 512]
    similarity: mx.array | None      # Similarity matrix [audio, text]
    text_labels: list[str] | None    # Text labels provided
    model_name: str                  # Model used

    @property
    def audio_embedding(self) -> mx.array:
        """Get audio embedding (first if batch)."""

    @property
    def text_embedding(self) -> mx.array:
        """Get text embedding (first if batch)."""

    def best_match(self) -> str:
        """Get best matching text label."""
```

### Usage

```python
# Audio embedding
result = ma.embed(audio="audio.wav")
embedding = result.audio_embedding  # [1, 512]

# Text embedding
result = ma.embed(text=["dog", "cat", "bird"])
embeddings = result.text_embedding  # [3, 512]

# With similarity
result = ma.embed(
    audio="sound.wav",
    text=["dog", "cat", "bird"],
    return_similarity=True
)
print(result.similarity)  # [1, 3]
print(result.best_match())  # "dog"
```

## ClassificationResult

Returned by `classify()`.

```python
class ClassificationResult:
    predicted_class: str             # Top prediction
    probabilities: mx.array          # All probabilities
    class_names: list[str]           # Class labels
    top_k_classes: list[str]         # Top-k predictions
    top_k_probs: list[float]         # Top-k probabilities
    model_name: str                  # Model used

    @property
    def confidence(self) -> float:
        """Confidence of top prediction."""
```

### Usage

```python
result = ma.classify("sound.wav", labels=["dog", "cat", "bird"], top_k=3)

print(f"Predicted: {result.predicted_class}")
print(f"Confidence: {result.confidence:.1%}")

for cls, prob in zip(result.top_k_classes, result.top_k_probs):
    print(f"  {cls}: {prob:.1%}")
```

## TaggingResult

Returned by `tag()`.

```python
class TaggingResult:
    tags: list[str]                  # All tags
    probabilities: list[float]       # Tag probabilities
    threshold: float                 # Active threshold
    model_name: str                  # Model used

    @property
    def active_tags(self) -> list[str]:
        """Tags above threshold."""
```

### Usage

```python
result = ma.tag("music.wav", tags=["jazz", "piano", "vocals", "drums"])

print(f"Active: {result.active_tags}")

for tag, prob in zip(result.tags, result.probabilities):
    status = "Active" if prob > result.threshold else ""
    print(f"  {tag}: {prob:.1%} {status}")
```

## VADResult

Returned by `detect_speech()`.

```python
class VADResult:
    segments: list[VADSegment]       # Speech segments
    probabilities: mx.array | None   # Per-frame probabilities
    model_name: str                  # Model used

    @property
    def total_speech_duration(self) -> float:
        """Total speech duration in seconds."""

    def get_speech_audio(self, audio: mx.array) -> mx.array:
        """Extract speech portions from audio."""

class VADSegment:
    start: float                     # Start time
    end: float                       # End time

    @property
    def duration(self) -> float:
        """Segment duration."""
```

### Usage

```python
result = ma.detect_speech("audio.wav", return_probabilities=True)

for seg in result.segments:
    print(f"Speech: {seg.start:.2f}s - {seg.end:.2f}s")

print(f"Total speech: {result.total_speech_duration:.1f}s")
```

## EnhancementResult

Returned by `enhance()`.

```python
class EnhancementResult:
    audio: AudioData                 # Enhanced audio
    model_name: str                  # Model used
    metadata: dict                   # Additional info

    @property
    def duration(self) -> float:
        """Duration in seconds."""

    @property
    def before_after(self) -> tuple[AudioData, AudioData] | None:
        """Original and enhanced (if keep_original=True)."""

    def save(self, path: str | Path) -> None:
        """Save enhanced audio."""
```

### Usage

```python
result = ma.enhance("noisy.wav", keep_original=True)

result.save("clean.wav")

# Compare
original, enhanced = result.before_after
```

## DiarizationResult

Returned by `diarize()`.

```python
class DiarizationResult:
    segments: list[DiarizationSegment]  # Speaker segments
    num_speakers: int                    # Number of speakers
    model_name: str                      # Model used

class DiarizationSegment:
    start: float                         # Start time
    end: float                           # End time
    speaker: int | str                   # Speaker ID
```

### Usage

```python
result = ma.diarize("meeting.wav")

print(f"Detected {result.num_speakers} speakers")

for seg in result.segments:
    print(f"Speaker {seg.speaker}: {seg.start:.1f}s - {seg.end:.1f}s")
```

## Common Methods

All result types support:

```python
# Convert to dictionary
data = result.to_dict()

# Access model name
print(result.model_name)
```

## Type Hints

For IDE support, import result types:

```python
from mlx_audio.types import (
    SeparationResult,
    TranscriptionResult,
    GenerationResult,
    SpeechResult,
    CLAPEmbeddingResult,
    ClassificationResult,
    TaggingResult,
    VADResult,
    EnhancementResult,
    DiarizationResult,
)
```
