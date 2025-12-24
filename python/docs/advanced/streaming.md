# Streaming Audio Processing

mlx-audio provides a comprehensive streaming framework for real-time audio processing. This enables low-latency applications like live audio separation, real-time transcription, and audio monitoring.

## Overview

The streaming system is built around three core concepts:

| Component | Purpose |
|-----------|---------|
| **Source** | Provides audio input (file, microphone, callback) |
| **Processor** | Transforms audio (separation, effects, etc.) |
| **Sink** | Handles output (file, speaker, callback) |

These components are connected via a **Pipeline** that manages data flow and state.

## Quick Start

```python
from mlx_audio.streaming import (
    StreamingPipeline,
    FileSource,
    FileSink,
    HTDemucsStreamProcessor,
)
from mlx_audio.models import HTDemucs

# Load model
model = HTDemucs.from_pretrained("htdemucs_ft")

# Create pipeline
pipeline = StreamingPipeline(
    source=FileSource("song.mp3"),
    processor=HTDemucsStreamProcessor(model),
    sink=FileSink("vocals.wav", stem_index=3),  # Extract vocals
)

# Run pipeline
pipeline.start()
pipeline.wait()
```

## Sources

Sources provide audio data to the pipeline.

### FileSource

Read audio from a file:

```python
from mlx_audio.streaming import FileSource

# Basic file source
source = FileSource("audio.wav")

# With custom chunk size
source = FileSource("audio.mp3", chunk_size=4096)

# Force sample rate (auto-resamples if different)
source = FileSource("audio.wav", sample_rate=44100)
```

### MicrophoneSource

Capture audio from the system microphone:

```python
from mlx_audio.streaming import MicrophoneSource

# Default microphone
source = MicrophoneSource()

# Custom settings
source = MicrophoneSource(
    sample_rate=44100,
    channels=1,          # Mono
    chunk_size=1024,
    device_id=None,      # System default
)
```

### CallbackSource

Provide audio programmatically:

```python
from mlx_audio.streaming import CallbackSource
import numpy as np

def generate_audio():
    """Generator that yields audio chunks."""
    for _ in range(100):
        # Generate 1024 samples of stereo audio
        yield np.random.randn(2, 1024).astype(np.float32)

source = CallbackSource(
    callback=generate_audio,
    sample_rate=44100,
    channels=2,
)
```

## Processors

Processors transform audio chunks as they flow through the pipeline.

### HTDemucsStreamProcessor

Real-time source separation using HTDemucs:

```python
from mlx_audio.streaming import HTDemucsStreamProcessor
from mlx_audio.models import HTDemucs

model = HTDemucs.from_pretrained("htdemucs_ft")
processor = HTDemucsStreamProcessor(
    model,
    segment_length=6.0,    # Processing window (seconds)
    overlap=0.25,          # Overlap between segments (ratio)
)
```

### Built-in Processors

```python
from mlx_audio.streaming import GainProcessor, IdentityProcessor

# Apply gain (volume adjustment)
processor = GainProcessor(gain=0.5)  # Reduce volume by half

# Pass-through (useful for testing)
processor = IdentityProcessor()
```

### Custom Processors

Create your own processor by subclassing `StreamProcessor`:

```python
from mlx_audio.streaming import StreamProcessor, StreamChunk
import mlx.core as mx

class LowPassFilter(StreamProcessor):
    """Simple low-pass filter processor."""

    def __init__(self, cutoff: float = 0.1):
        super().__init__()
        self.cutoff = cutoff
        self._prev_sample = None

    def process(self, chunk: StreamChunk) -> StreamChunk:
        audio = chunk.audio

        # Simple first-order low-pass filter
        if self._prev_sample is None:
            self._prev_sample = mx.zeros((audio.shape[0], 1))

        filtered = mx.concatenate([self._prev_sample, audio], axis=1)
        output = self.cutoff * audio + (1 - self.cutoff) * filtered[:, :-1]

        self._prev_sample = audio[:, -1:]

        return StreamChunk(
            audio=output,
            sample_rate=chunk.sample_rate,
            channels=chunk.channels,
        )

    def reset(self):
        self._prev_sample = None
```

## Sinks

Sinks handle the output of processed audio.

### FileSink

Write audio to a file:

```python
from mlx_audio.streaming import FileSink

# Basic file sink
sink = FileSink("output.wav")

# For separation, specify which stem to save
sink = FileSink("vocals.wav", stem_index=3)  # Vocals from HTDemucs

# With specific format
sink = FileSink("output.flac", sample_rate=44100)
```

### MultiFileSink

Save multiple stems to separate files:

```python
from mlx_audio.streaming import MultiFileSink

# Save all stems from HTDemucs
sink = MultiFileSink(
    output_dir="./stems",
    stem_names=["drums", "bass", "other", "vocals"],
)
# Creates: ./stems/drums.wav, ./stems/bass.wav, etc.
```

### SpeakerSink

Play audio through speakers:

```python
from mlx_audio.streaming import SpeakerSink

# Play through default speakers
sink = SpeakerSink()

# Custom settings
sink = SpeakerSink(
    sample_rate=44100,
    channels=2,
    device_id=None,  # System default
)
```

### CallbackSink

Handle output programmatically:

```python
from mlx_audio.streaming import CallbackSink

def handle_audio(audio_chunk, sample_rate, channels):
    """Process each output chunk."""
    print(f"Received chunk: {audio_chunk.shape}")

sink = CallbackSink(callback=handle_audio)
```

## Pipeline

The `StreamingPipeline` connects sources, processors, and sinks.

### Basic Pipeline

```python
from mlx_audio.streaming import StreamingPipeline

pipeline = StreamingPipeline(
    source=source,
    processor=processor,
    sink=sink,
)

# Start processing (non-blocking)
pipeline.start()

# Wait for completion
pipeline.wait()
```

### Pipeline Control

```python
# Start processing
pipeline.start()

# Check if running
if pipeline.is_running:
    print("Processing...")

# Stop early
pipeline.stop()

# Wait with timeout
pipeline.wait(timeout=10.0)  # Wait up to 10 seconds
```

### Pipeline Statistics

```python
# Get processing statistics
stats = pipeline.stats

print(f"Chunks processed: {stats.chunks_processed}")
print(f"Total samples: {stats.samples_processed}")
print(f"Processing time: {stats.processing_time:.2f}s")
print(f"Real-time factor: {stats.realtime_factor:.2f}x")
```

## Real-Time Source Separation

Complete example for real-time music source separation:

```python
from mlx_audio.streaming import (
    StreamingPipeline,
    MicrophoneSource,
    SpeakerSink,
    HTDemucsStreamProcessor,
)
from mlx_audio.models import HTDemucs

# Load model
model = HTDemucs.from_pretrained("htdemucs_ft")

# Create real-time separation pipeline
pipeline = StreamingPipeline(
    source=MicrophoneSource(sample_rate=44100),
    processor=HTDemucsStreamProcessor(
        model,
        segment_length=4.0,  # Shorter for lower latency
        overlap=0.25,
    ),
    sink=SpeakerSink(stem_index=3),  # Play vocals only
)

print("Starting real-time vocal isolation...")
pipeline.start()

try:
    pipeline.wait()
except KeyboardInterrupt:
    pipeline.stop()
    print("Stopped")
```

## Processing Metrics

The streaming module includes metrics for evaluating separation quality:

```python
from mlx_audio.streaming import si_sdr, sdr, snr, correlation

# Compute Signal-to-Distortion Ratio (scale-invariant)
quality = si_sdr(estimated_audio, reference_audio)
print(f"SI-SDR: {quality:.2f} dB")

# Standard SDR
quality = sdr(estimated_audio, reference_audio)

# Signal-to-Noise Ratio
quality = snr(estimated_audio, reference_audio)

# Correlation coefficient
corr = correlation(estimated_audio, reference_audio)
```

### SeparationMetrics

Compute multiple metrics at once:

```python
from mlx_audio.streaming import SeparationMetrics

metrics = SeparationMetrics()
result = metrics.compute(estimated, reference)

print(f"SI-SDR: {result['si_sdr']:.2f} dB")
print(f"SDR: {result['sdr']:.2f} dB")
print(f"SNR: {result['snr']:.2f} dB")
```

## Buffer Management

For advanced use cases, you can work directly with the audio ring buffer:

```python
from mlx_audio.streaming import AudioRingBuffer

# Create buffer for 10 seconds of stereo audio at 44.1kHz
buffer = AudioRingBuffer(
    capacity=441000,  # 10 seconds
    channels=2,
)

# Write audio to buffer
buffer.write(audio_chunk)

# Read from buffer
chunk = buffer.read(4096)  # Read 4096 samples

# Check buffer state
print(f"Available samples: {buffer.available}")
print(f"Free space: {buffer.free}")
```

## Streaming Context

Manage streaming state across components:

```python
from mlx_audio.streaming import StreamingContext

context = StreamingContext(
    sample_rate=44100,
    channels=2,
    chunk_size=1024,
)

# Access state
print(f"Sample rate: {context.sample_rate}")
print(f"Chunk duration: {context.chunk_duration_ms:.2f} ms")
```

## Performance Tips

### Latency Optimization

- Use smaller `segment_length` for lower latency (but may affect quality)
- Use smaller `chunk_size` in sources/sinks
- Consider using `whisper-tiny` or `htdemucs` (non-ft) for faster processing

### Memory Management

- Ring buffers have fixed capacity; set appropriately for your use case
- For long-running pipelines, monitor memory usage
- Use `pipeline.reset()` between files to clear state

### Real-Time Performance

Check if processing keeps up with real-time:

```python
stats = pipeline.stats
if stats.realtime_factor < 1.0:
    print("Warning: Processing slower than real-time")
    print(f"Currently at {stats.realtime_factor:.2f}x real-time")
```

## API Reference

### Core Classes

::: mlx_audio.streaming.StreamingPipeline
    options:
      show_root_heading: true
      show_source: false

::: mlx_audio.streaming.StreamProcessor
    options:
      show_root_heading: true
      show_source: false
