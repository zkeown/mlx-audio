# MLX Audio Lab

A SwiftUI demo app showcasing all MLX Audio capabilities on Apple Silicon.

## Features

The app includes 6 tabs demonstrating different audio ML capabilities:

| Tab | Model | Description |
|-----|-------|-------------|
| **Separate** | HTDemucs | Source separation into drums, bass, other, vocals |
| **Transcribe** | Whisper | Speech-to-text with language detection |
| **Generate** | MusicGen | Text-to-music generation |
| **Embed** | CLAP | Audio embeddings and zero-shot classification |
| **Live** | HTDemucs | Real-time stem isolation from microphone |
| **Banquet** | Banquet | Query-based separation with reference audio |

## Requirements

- macOS 14+ or iOS 17+
- Apple Silicon (M1/M2/M3)
- Xcode 15.2+
- Swift 6.0+

## Building

### From Xcode

1. Open the package in Xcode:
   ```bash
   cd swift/Examples/MLXAudioLab
   open Package.swift
   ```

2. Select the `MLXAudioLab` scheme and your target device

3. Build and run (Cmd+R)

### From Command Line

```bash
cd swift/Examples/MLXAudioLab
swift build
swift run MLXAudioLab
```

## Usage

### Model Downloads

Models are downloaded automatically from HuggingFace Hub when first accessed. Download progress is displayed in the UI. Downloaded models are cached in:
- macOS: `~/Documents/MLXAudioLab/Models/`
- iOS: App Documents directory

### Memory Management

The app automatically manages model memory based on device profile:
- **iPhone**: 2GB budget, evicts models when switching tabs
- **iPad**: 4GB budget, can keep 2 models loaded
- **Mac**: 8GB budget, can keep 4 models loaded
- **Mac Pro**: 16GB budget, can keep 8 models loaded

### Tab Details

#### Separate (HTDemucs)
1. Select an audio file (WAV, MP3, M4A, AIFF)
2. Choose model variant (standard, fine-tuned, or 6-stem)
3. Tap "Separate" and wait for processing
4. View and play individual stems
5. Export stems to files

#### Transcribe (Whisper)
1. Select an audio file or record from microphone
2. Choose model variant (tiny to large-v3)
3. Tap "Transcribe"
4. View timestamped text with language detection
5. Copy text or export as SRT subtitles

#### Generate (MusicGen)
1. Enter a text prompt describing desired music
2. Adjust duration (5-30 seconds)
3. Choose model size (small, medium, large)
4. Tap "Generate" and wait
5. Play and save generated audio

#### Embed & Classify (CLAP)
1. Select an audio file
2. **Text Search**: Type a description and compute similarity
3. **Classification**: Add labels and run zero-shot classification
4. Use presets for common label sets

#### Live (Real-time)
1. Select which stem to isolate
2. Tap "Start" to begin processing
3. Audio is captured, processed, and played back in real-time
4. Monitor latency and buffer statistics
5. Tap "Stop" when finished

#### Banquet (Query-based)
1. Select a reference (query) audio clip
2. Select a mixture audio file
3. Tap "Separate" to extract sounds similar to the query
4. Play and save extracted audio

## Architecture

```
MLXAudioLab/
├── MLXAudioLabApp.swift    # App entry point
├── ContentView.swift        # Tab container
├── Services/
│   ├── ModelManager.swift   # Model loading and caching
│   ├── HuggingFaceDownloader.swift  # Weight downloads
│   └── AudioLoader.swift    # Audio file I/O
├── Views/                   # SwiftUI views for each tab
├── ViewModels/              # Business logic per tab
└── Components/              # Reusable UI components
```

## Supported Models

| Model | Memory | Sample Rate |
|-------|--------|-------------|
| HTDemucs | 2000 MB | 44.1 kHz |
| HTDemucs Fine-tuned | 2000 MB | 44.1 kHz |
| HTDemucs 6-Stem | 2500 MB | 44.1 kHz |
| Whisper Tiny | 150 MB | 16 kHz |
| Whisper Small | 500 MB | 16 kHz |
| Whisper Medium | 1500 MB | 16 kHz |
| Whisper Large V3 Turbo | 1600 MB | 16 kHz |
| Whisper Large V3 | 3000 MB | 16 kHz |
| MusicGen Small | 1200 MB | 32 kHz |
| MusicGen Medium | 3500 MB | 32 kHz |
| MusicGen Large | 7000 MB | 32 kHz |
| CLAP Tiny | 150 MB | 48 kHz |
| CLAP Fused | 800 MB | 48 kHz |
| Banquet | ~1500 MB | 44.1 kHz |

## Permissions

The app requires the following permissions:
- **Microphone**: For Transcribe (recording) and Live tabs
- **Files**: For loading and saving audio files

## Troubleshooting

### Build Errors

If you encounter build errors, ensure:
1. You're using Xcode 15.2 or later
2. Swift 6.0 is selected
3. The parent MLXAudio package builds successfully:
   ```bash
   cd swift
   swift build
   ```

### Memory Issues

If you experience memory warnings:
- Switch to smaller model variants
- Close other memory-intensive apps
- On iOS, the app automatically evicts unused models

### Model Download Failures

If downloads fail:
- Check your internet connection
- Ensure sufficient disk space
- Delete and re-download: remove the model folder from Documents

## License

This demo app is part of the mlx-audio project. See the main repository for license information.
