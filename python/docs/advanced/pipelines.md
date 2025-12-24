# Audio Processing Pipelines

This guide shows how to combine multiple mlx-audio operations into powerful audio processing pipelines.

## Overview

Many real-world audio tasks require combining multiple operations. For example:
- Enhance audio → Transcribe
- Detect speech → Extract segments → Transcribe
- Separate → Process individual stems → Mix

## Common Pipelines

### Enhance Then Transcribe

Improve transcription accuracy by cleaning audio first:

```python
import mlx_audio as ma

# Step 1: Enhance noisy audio
enhanced = ma.enhance("noisy_recording.wav")
enhanced.save("temp_clean.wav")

# Step 2: Transcribe the cleaned audio
result = ma.transcribe("temp_clean.wav")
print(result.text)
```

### VAD-Guided Transcription

Only transcribe speech portions of long recordings:

```python
import mlx_audio as ma
import numpy as np

# Step 1: Detect speech segments
vad_result = ma.detect_speech("long_meeting.wav")

# Step 2: Load audio
audio = np.load("long_meeting.wav")  # Simplified

# Step 3: Transcribe each speech segment
for segment in vad_result.segments:
    start_sample = int(segment.start * 16000)
    end_sample = int(segment.end * 16000)
    segment_audio = audio[start_sample:end_sample]

    result = ma.transcribe(segment_audio, sample_rate=16000)
    print(f"[{segment.start:.1f}s - {segment.end:.1f}s]: {result.text}")
```

### Separate and Process Stems

Apply different processing to each stem:

```python
import mlx_audio as ma

# Step 1: Separate
stems = ma.separate("song.mp3")

# Step 2: Process vocals (enhance)
enhanced_vocals = ma.enhance(stems.vocals.numpy(), sample_rate=44100)

# Step 3: Save processed version
enhanced_vocals.save("vocals_enhanced.wav")
stems.drums.save("drums.wav")
stems.bass.save("bass.wav")
stems.other.save("other.wav")
```

### Karaoke Pipeline

Create a karaoke version by removing vocals:

```python
import mlx_audio as ma
import numpy as np

# Separate
stems = ma.separate("song.mp3")

# Mix instrumental (everything except vocals)
instrumental = (
    stems.drums.numpy() +
    stems.bass.numpy() +
    stems.other.numpy()
)

# Save
from scipy.io import wavfile
wavfile.write("karaoke.wav", 44100, instrumental.T)
```

### Transcription with Speaker Labels

Combine diarization with transcription:

```python
import mlx_audio as ma

# Step 1: Diarize
diarization = ma.diarize("meeting.wav")

# Step 2: Transcribe
transcription = ma.transcribe("meeting.wav")

# Step 3: Combine results
# (This is a simplified example - real alignment is more complex)
for segment in diarization.segments:
    print(f"Speaker {segment.speaker}: [{segment.start:.1f}s - {segment.end:.1f}s]")
```

## Building Custom Pipelines

### Pipeline Pattern

Create reusable pipeline functions:

```python
import mlx_audio as ma

def transcribe_with_enhancement(audio_path, output_path=None):
    """Enhanced transcription pipeline."""
    # Enhance
    enhanced = ma.enhance(audio_path)

    # Transcribe
    result = ma.transcribe(enhanced.audio.numpy(), sample_rate=enhanced.audio.sample_rate)

    # Save if requested
    if output_path:
        result.save(output_path, format="txt")

    return result

# Use the pipeline
result = transcribe_with_enhancement("noisy_recording.wav")
```

### Batch Processing

Process multiple files:

```python
import mlx_audio as ma
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

def process_file(audio_path):
    """Process a single file."""
    try:
        result = ma.transcribe(audio_path)
        output_path = Path(audio_path).with_suffix(".txt")
        result.save(output_path, format="txt")
        return True, audio_path
    except Exception as e:
        return False, f"{audio_path}: {e}"

# Process all WAV files in a directory
audio_files = list(Path("recordings").glob("*.wav"))

for audio_file in audio_files:
    success, msg = process_file(audio_file)
    if success:
        print(f"Processed: {msg}")
    else:
        print(f"Error: {msg}")
```

## Memory Management

When chaining operations, manage memory carefully:

```python
import mlx_audio as ma
import gc
import mlx.core as mx

def process_large_file(audio_path):
    # Step 1
    result1 = ma.enhance(audio_path)
    temp_audio = result1.audio.numpy()
    del result1
    gc.collect()
    mx.metal.clear_cache()

    # Step 2
    result2 = ma.transcribe(temp_audio, sample_rate=48000)
    del temp_audio
    gc.collect()
    mx.metal.clear_cache()

    return result2
```

## Tips

1. **Use intermediate files** for complex pipelines to save memory
2. **Clear caches** between heavy operations
3. **Match sample rates** - check each operation's expected input
4. **Handle errors** at each step for robust pipelines
5. **Log progress** for long-running batch jobs
