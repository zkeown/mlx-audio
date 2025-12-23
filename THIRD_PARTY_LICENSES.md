# Third-Party Licenses

This document lists the licenses for third-party dependencies and pre-trained model weights used by mlx-audio.

## Core Dependencies

### MLX

- **Project**: [MLX](https://github.com/ml-explore/mlx)
- **License**: MIT
- **Copyright**: Apple Inc.

### mlx-swift

- **Project**: [mlx-swift](https://github.com/ml-explore/mlx-swift)
- **License**: MIT
- **Copyright**: Apple Inc.

## Model Weights

### Whisper

- **Project**: [Whisper](https://github.com/openai/whisper)
- **Organization**: OpenAI
- **License**: MIT
- **Models**: whisper-tiny, whisper-base, whisper-small, whisper-medium, whisper-large, whisper-large-v2, whisper-large-v3, whisper-large-v3-turbo
- **HuggingFace**: https://huggingface.co/openai/whisper-large-v3

### HTDemucs

- **Project**: [Demucs](https://github.com/facebookresearch/demucs)
- **Organization**: Meta AI (FAIR)
- **License**: MIT
- **Models**: htdemucs, htdemucs_ft, htdemucs_6s
- **HuggingFace**: https://huggingface.co/facebook/demucs

### MusicGen

- **Project**: [AudioCraft](https://github.com/facebookresearch/audiocraft)
- **Organization**: Meta AI (FAIR)
- **License**: CC-BY-NC 4.0 (Non-Commercial)
- **Models**: musicgen-small, musicgen-medium, musicgen-large, musicgen-melody
- **HuggingFace**: https://huggingface.co/facebook/musicgen-medium
- **Note**: These model weights are for non-commercial use only. Commercial use requires a separate license from Meta.

### EnCodec

- **Project**: [EnCodec](https://github.com/facebookresearch/encodec)
- **Organization**: Meta AI (FAIR)
- **License**: CC-BY-NC 4.0 (Non-Commercial)
- **Models**: encodec_24khz, encodec_48khz
- **HuggingFace**: https://huggingface.co/facebook/encodec_24khz
- **Note**: These model weights are for non-commercial use only. Commercial use requires a separate license from Meta.

### CLAP

- **Project**: [CLAP](https://github.com/LAION-AI/CLAP)
- **Organization**: LAION
- **License**: Apache 2.0
- **Models**: clap-htsat-fused, clap-htsat-unfused
- **HuggingFace**: https://huggingface.co/laion/clap-htsat-fused

### Parler-TTS

- **Project**: [Parler-TTS](https://github.com/huggingface/parler-tts)
- **Organization**: Hugging Face
- **License**: Apache 2.0
- **Models**: parler-tts-mini, parler-tts-large
- **HuggingFace**: https://huggingface.co/parler-tts/parler-tts-mini-v1

### Silero VAD

- **Project**: [Silero VAD](https://github.com/snakers4/silero-vad)
- **Organization**: Silero Team
- **License**: MIT
- **Models**: silero-vad
- **HuggingFace**: https://huggingface.co/silero/silero-vad

### DeepFilterNet

- **Project**: [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet)
- **Organization**: Hendrik Schröter
- **License**: MIT / Apache 2.0
- **Models**: deepfilternet2, deepfilternet3
- **HuggingFace**: https://huggingface.co/Rikorose/DeepFilterNet2

### ECAPA-TDNN

- **Project**: [SpeechBrain](https://github.com/speechbrain/speechbrain)
- **Organization**: SpeechBrain
- **License**: Apache 2.0
- **Models**: ecapa-tdnn
- **HuggingFace**: https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb

## Python Dependencies

The following Python packages are used (see pyproject.toml for complete list):

| Package | License |
|---------|---------|
| numpy | BSD-3-Clause |
| huggingface-hub | Apache 2.0 |
| librosa | ISC |
| soundfile | BSD-3-Clause |
| scipy | BSD-3-Clause |
| wandb | MIT |
| tensorboardX | MIT |
| mlflow | Apache 2.0 |

## Swift Dependencies

| Package | License |
|---------|---------|
| swift-atomics | Apache 2.0 |

## License Texts

### MIT License

```
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
```

### CC-BY-NC 4.0 Summary

The CC-BY-NC 4.0 license allows:
- Sharing and adapting the material
- Attribution must be given

But prohibits:
- Commercial use without separate license

Full license: https://creativecommons.org/licenses/by-nc/4.0/

### Apache 2.0 Summary

The Apache 2.0 license allows:
- Commercial use
- Modification and distribution
- Patent use

Requirements:
- License and copyright notice
- State changes

Full license: https://www.apache.org/licenses/LICENSE-2.0
