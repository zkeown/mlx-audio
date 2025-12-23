# Parity Testing Guide

This guide explains how to use and extend the parity testing infrastructure for verifying numerical equivalence between Python and Swift implementations of mlx-audio models.

## Overview

Parity testing ensures that Swift model implementations produce outputs numerically equivalent to their Python counterparts. This is critical for maintaining cross-language consistency and catching bugs during development.

### How It Works

1. **Python generates reference fixtures**: Scripts create `.safetensors` files containing model weights, inputs, and expected outputs
2. **Swift loads and compares**: Tests load fixtures, run the Swift model, and compare against expected outputs
3. **Tolerance-based comparison**: Small numerical differences (floating-point precision) are acceptable within defined tolerances

## Quick Start

### Generate All Fixtures

```bash
# From the python/ directory
cd python

# Generate fixtures for all models
python tests/parity/generate_references.py --model all

# Generate for specific models
python tests/parity/generate_references.py --model htdemucs clap

# Generate layer fixtures only (faster)
python tests/parity/generate_references.py --model all --layers-only
```

### Run Swift Parity Tests

```bash
# From the swift/ directory
cd swift

# Run all parity tests
swift test --filter ParityTests

# Run specific model's parity tests
swift test --filter HTDemucsParityTests
swift test --filter CLAPParityTests
```

## Tolerance Levels

Different tolerance levels are used depending on the scope of comparison:

| Level | Tolerance | Use Case |
|-------|-----------|----------|
| Layer | `1e-5` | Individual layer outputs (conv, attention, etc.) |
| Chain | `1e-4` | Multi-layer pipelines (encoder, decoder blocks) |
| Model | `1e-3` | Full model outputs (accumulated numerical error) |
| Cosine | `0.999` | Embedding similarity (L2-normalized vectors) |

### Why Different Tolerances?

- **Floating-point accumulation**: Error compounds through deep networks
- **Operation order**: Different optimization/fusing strategies may reorder operations
- **Precision differences**: Float32 vs Float16, fused multiply-add variations
- **Numerical stability**: Softmax, normalization can amplify small differences

## Fixture Directory Structure

```
swift/Tests/Fixtures/
├── HTDemucs/
│   ├── config.json
│   ├── dconv.safetensors
│   ├── dconv_weights.safetensors
│   ├── freq_encoder.safetensors
│   ├── freq_encoder_weights.safetensors
│   ├── small_model.safetensors
│   ├── small_model_weights.safetensors
│   └── ...
├── EnCodec/
│   ├── config.json
│   ├── encoder.safetensors
│   ├── decoder.safetensors
│   ├── quantizer.safetensors
│   └── ...
├── CLAP/
│   ├── config.json
│   ├── patch_embed.safetensors
│   ├── htsat_encoder.safetensors
│   ├── roberta_encoder.safetensors
│   └── ...
├── Whisper/
│   └── ...
└── MusicGen/
    ├── config.json
    ├── delay_pattern.safetensors
    ├── codebook_embed.safetensors
    └── ...
```

### Fixture File Contents

Each `.safetensors` file typically contains:
- `input`: Test input tensor(s)
- `output`: Expected output tensor(s)
- Additional intermediate values as needed

Weight files (`*_weights.safetensors`) contain the model parameters used to generate the fixtures.

## Adding Parity Tests for New Models

### 1. Create Python Fixture Generator

Create `python/tests/generate_{model}_fixtures.py`:

```python
#!/usr/bin/env python3
"""Generate test fixtures for {Model} Swift parity testing."""

import argparse
from pathlib import Path
import mlx.core as mx

def set_seed(seed: int = 42):
    mx.random.seed(seed)

def generate_layer_fixtures(output_dir: Path, config):
    """Generate fixtures for individual layers."""
    # Create layer
    layer = MyLayer(config)

    # Generate input
    input = mx.random.normal([batch_size, ...])

    # Forward pass
    output = layer(input)

    # Save fixtures
    mx.save_safetensors(
        str(output_dir / "my_layer.safetensors"),
        {"input": input, "output": output}
    )

    # Save weights
    layer.save_weights(str(output_dir / "my_layer_weights.safetensors"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    generate_layer_fixtures(args.output_dir, config)

if __name__ == "__main__":
    main()
```

### 2. Create Swift Parity Tests

Create `swift/Tests/MLXAudioModelsTests/{Model}ParityTests.swift`:

```swift
import XCTest
@testable import MLXAudioModels
import MLX

final class MyModelParityTests: XCTestCase {

    static let layerTolerance: Float = 1e-5
    static let modelTolerance: Float = 1e-3

    static var fixturesPath: URL {
        // Try environment variable first
        if let envPath = ProcessInfo.processInfo.environment["MYMODEL_FIXTURES_PATH"] {
            return URL(fileURLWithPath: envPath)
        }
        // Fallback to relative paths...
    }

    func loadFixture(_ name: String) throws -> [String: MLXArray] {
        let url = Self.fixturesPath.appendingPathComponent("\(name).safetensors")
        return try MLX.loadArrays(url: url)
    }

    func assertArraysEqual(
        _ actual: MLXArray,
        _ expected: MLXArray,
        tolerance: Float,
        message: String = ""
    ) {
        let diff = abs(actual - expected).max().item(Float.self)
        XCTAssertLessThan(diff, tolerance, message)
    }

    func testLayerParity() throws {
        let fixtures = try loadFixture("my_layer")
        let weights = try loadFixture("my_layer_weights")

        let input = fixtures["input"]!
        let expected = fixtures["output"]!

        let layer = MyLayer(config)
        try layer.update(parameters: ModuleParameters.unflattened(weights))

        let output = layer(input)

        assertArraysEqual(output, expected, tolerance: Self.layerTolerance)
    }
}
```

### 3. Register in Unified Generator

Add your model to `python/tests/parity/generate_references.py`:

```python
VALID_MODELS = {"htdemucs", "encodec", "whisper", "clap", "musicgen", "mymodel"}

GENERATOR_SCRIPTS = {
    # ...existing entries...
    "mymodel": "generate_mymodel_fixtures.py",
}
```

## Debugging Parity Failures

### 1. Check Shape Mismatch

First verify shapes match:

```swift
XCTAssertEqual(output.shape, expected.shape, "Shape mismatch")
```

### 2. Identify Divergence Point

For multi-layer models, test each layer individually to find where divergence occurs:

```swift
// Test layer by layer
let layer1Out = model.layer1(input)
assertArraysEqual(layer1Out, expectedLayer1, tolerance: Self.layerTolerance)

let layer2Out = model.layer2(layer1Out)
assertArraysEqual(layer2Out, expectedLayer2, tolerance: Self.layerTolerance)
```

### 3. Check Weight Loading

Verify weights loaded correctly:

```swift
let weights = try loadFixture("model_weights")
print("Weight keys: \(weights.keys.sorted())")
print("Expected: layers.0.weight shape \(weights["layers.0.weight"]?.shape ?? [])")
```

### 4. Compare Statistics

When values differ, check mean and max difference:

```swift
let diff = abs(actual - expected)
print("Max diff: \(diff.max().item(Float.self))")
print("Mean diff: \(diff.mean().item(Float.self))")
print("Actual range: \(actual.min().item(Float.self)) to \(actual.max().item(Float.self))")
print("Expected range: \(expected.min().item(Float.self)) to \(expected.max().item(Float.self))")
```

### 5. Common Issues

| Symptom | Likely Cause |
|---------|--------------|
| All zeros | Weight not loaded, wrong key name |
| Large constant offset | Bias not loaded |
| Scaled by constant | Different normalization factor |
| Transposed | Different data format (NCHW vs NHWC) |
| Permuted | Attention head reshaping difference |

## Known Differences

Some numerical differences are expected and acceptable:

### STFT/iSTFT
- Padding mode differences (`edge` vs `reflect`)
- Window normalization variations

### Attention
- Softmax numerical stability implementations
- Attention mask application order

### Normalization
- LayerNorm epsilon differences
- BatchNorm running stats handling

### Convolutions
- Group convolution implementation details
- Padding calculation edge cases

## CI Integration

Parity tests are automatically run on PRs via GitHub Actions. See `.github/workflows/parity-tests.yml` for configuration.

### Running Locally Before Push

```bash
# Generate fixtures
cd python
python tests/parity/generate_references.py --model all

# Run tests
cd ../swift
swift test --filter ParityTests
```

## Quality Benchmarks

In addition to numerical parity tests, quality benchmarks verify end-to-end model performance.

### Running Quality Benchmarks

```bash
cd python

# Install required dependencies
pip install mir_eval jiwer

# Run all benchmarks
pytest tests/benchmarks/ -v

# Run specific model benchmarks
pytest tests/benchmarks/test_htdemucs_quality.py -v
pytest tests/benchmarks/test_whisper_quality.py -v
pytest tests/benchmarks/test_clap_quality.py -v
pytest tests/benchmarks/test_encodec_quality.py -v
```

### Quality Metrics

| Model | Metric | Target | Description |
|-------|--------|--------|-------------|
| HTDemucs | SDR | within 0.1dB | Signal-to-Distortion Ratio for source separation |
| Whisper | WER | within 1% | Word Error Rate for transcription |
| CLAP | Cosine sim | > 0.999 | Audio-text embedding similarity |
| EnCodec | SDR | within 0.05dB | Reconstruction quality |
| MusicGen | Code match | 100% | Codec code sequence matching |

### SDR (Signal-to-Distortion Ratio)

Used for HTDemucs and EnCodec. Calculated using `mir_eval.separation.bss_eval_sources`.

```python
import mir_eval

sdr, sir, sar, perm = mir_eval.separation.bss_eval_sources(
    reference_sources,
    estimated_sources
)
```

### WER (Word Error Rate)

Used for Whisper. Calculated using `jiwer`.

```python
import jiwer

wer = jiwer.wer(reference_text, hypothesis_text)
```

### Cosine Similarity

Used for CLAP embeddings. Should be > 0.999 for identical inputs.

```python
import numpy as np

def cosine_similarity(a, b):
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b)
    return np.dot(a_norm, b_norm)
```

## Edge Case Testing

Edge case fixtures test model robustness on boundary conditions.

### Generate Edge Case Fixtures

```bash
cd python
python tests/generate_edge_case_fixtures.py --output-dir ../swift/Tests/Fixtures/EdgeCases
```

### Edge Cases Tested

- **Short audio**: < 0.1 second
- **Silent audio**: All zeros
- **Clipped audio**: Values at ±1.0
- **DC offset**: Non-zero mean
- **Impulse**: Single sample spike
- **Max amplitude**: All values at 1.0
- **Min amplitude**: Very quiet (1e-6 scale)

## Best Practices

1. **Always set random seed** for reproducible fixtures
2. **Test layers individually** before full model
3. **Use smaller configs** for faster test iteration
4. **Save intermediate activations** when debugging
5. **Document any tolerance exceptions** with justification
6. **Regenerate fixtures** after Python model changes
7. **Run quality benchmarks** before releasing
8. **Test edge cases** for robustness
