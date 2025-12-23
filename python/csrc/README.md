# C++ Extension

> *"When Python isn't fast enough, we go hardcore."*

This directory contains the optional C++ extension for mlx-audio-primitives, providing optimized implementations for performance-critical operations.

## Overview

The extension provides native implementations for:

| Function | Purpose | Why C++? |
|----------|---------|----------|
| `overlap_add` | ISTFT reconstruction | Fused kernel, atomic scatter-add |
| `frame_signal` | Signal framing | Optimized gather operations |
| `pad_signal` | Signal padding | All modes in single kernel |
| `generate_window` | Window functions | GPU-native generation |
| `mel_filterbank` | Mel filterbank matrix | Precision-matched to librosa |
| `hz_to_mel` / `mel_to_hz` | Frequency conversion | Slaney/HTK formulas |

## Building

The extension builds automatically when installing with pip:

```bash
pip install -e .
```

### Manual Build

```bash
# Create build directory
mkdir build && cd build

# Configure with CMake
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build . --parallel

# The resulting _ext.cpython-*.so will be in mlx_audio_primitives/
```

### Requirements

- CMake >= 3.25
- C++17 compiler (Clang/Apple Clang recommended)
- MLX >= 0.30.0 (provides nanobind integration)
- Metal SDK (for GPU kernels)

## Architecture

```
csrc/
├── CMakeLists.txt        # Build configuration
├── bindings.cpp          # Python bindings (nanobind)
├── bindings_wrappers.h   # Wrapper functions for optional args
│
├── overlap_add.h         # Overlap-add primitive header
├── overlap_add.cpp       # Overlap-add implementation
├── frame_signal.h        # Signal framing header
├── frame_signal.cpp      # Signal framing implementation
├── pad_signal.h          # Padding primitive header
├── pad_signal.cpp        # Padding implementation
├── windows.h             # Window functions header
├── windows.cpp           # Window functions implementation
├── mel_filterbank.h      # Mel filterbank header
├── mel_filterbank.cpp    # Mel filterbank implementation
│
└── metal/                # Metal shader sources
    ├── overlap_add.metal
    ├── frame_signal.metal
    ├── pad_signal.metal
    ├── windows.metal
    └── mel_filterbank.metal
```

## Python Fallback

The extension is **optional**. When not available, Python/Metal fallbacks are used:

```python
# mlx_audio_primitives/_extension.py
try:
    from . import _ext
    HAS_CPP_EXT = True
except ImportError:
    _ext = None
    HAS_CPP_EXT = False
```

Each function checks before using the extension:

```python
def _some_operation(...):
    if HAS_CPP_EXT and _ext is not None:
        return _ext.some_operation(...)

    # Python/Metal fallback
    ...
```

## Adding New Primitives

### 1. Create Header File

```cpp
// csrc/your_primitive.h
#pragma once

#include <mlx/mlx.h>

namespace mlx_audio {

/**
 * Brief description.
 *
 * @param input Input array of shape (...)
 * @param param Some parameter
 * @param s Stream or device for computation
 * @return Output array of shape (...)
 */
mlx::core::array your_primitive(
    const mlx::core::array& input,
    int param,
    mlx::core::StreamOrDevice s = {}
);

}  // namespace mlx_audio
```

### 2. Create Implementation

```cpp
// csrc/your_primitive.cpp
#include "your_primitive.h"

namespace mlx_audio {

array your_primitive(
    const array& input,
    int param,
    StreamOrDevice s
) {
    // Implementation using MLX ops
    // For Metal acceleration, register a custom kernel
    ...
}

}  // namespace mlx_audio
```

### 3. Create Metal Kernel (Optional)

```metal
// csrc/metal/your_primitive.metal
[[kernel]] void your_primitive_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& param [[buffer(2)]],
    uint idx [[thread_position_in_grid]]
) {
    // GPU implementation
    ...
}
```

### 4. Add to CMakeLists.txt

```cmake
# In csrc/CMakeLists.txt
set(SOURCES
    ...
    your_primitive.cpp
)

# If using Metal
set(METAL_SOURCES
    ...
    metal/your_primitive.metal
)
```

### 5. Add Python Bindings

```cpp
// In csrc/bindings.cpp
m.def(
    "your_primitive",
    &your_primitive_wrapper,
    "input"_a,
    "param"_a,
    "stream"_a = nb::none(),
    R"(
    Brief description.

    Parameters
    ----------
    input : array
        Input array.
    param : int
        Some parameter.
    stream : Stream, optional
        Computation stream.

    Returns
    -------
    array
        Output array.
    )"
);
```

### 6. Add Wrapper (for Optional Args)

```cpp
// In csrc/bindings_wrappers.h
inline array your_primitive_wrapper(
    const array& input,
    int param,
    std::optional<nb::object> stream
) {
    return mlx_audio::your_primitive(
        input,
        param,
        to_stream_or_device(stream)
    );
}
```

### 7. Add Python Fallback

```python
# In mlx_audio_primitives/your_module.py
from ._extension import HAS_CPP_EXT, _ext

def your_primitive(input, param):
    if HAS_CPP_EXT and _ext is not None:
        return _ext.your_primitive(input, param)

    # Python implementation
    ...
```

### 8. Add Tests

```python
# In tests/test_cpp_extension.py
class TestYourPrimitive:
    def test_basic_functionality(self):
        ...

    def test_gpu_cpu_consistency(self):
        ...
```

## Metal Kernel Guidelines

### Thread Dispatch

```metal
// Use 3D grid for batch + spatial dimensions
uint batch_idx = thread_position_in_grid.z;
uint frame_idx = thread_position_in_grid.y;
uint sample_idx = thread_position_in_grid.x;
```

### Atomic Operations

For scatter operations with potential race conditions:

```metal
atomic_fetch_add_explicit(
    (device atomic_float*)&output[idx],
    value,
    memory_order_relaxed
);
```

### Bounds Checking

Always validate thread indices:

```metal
if (idx >= array_size) return;
```

## Testing

```bash
# Run C++ extension tests
pytest tests/test_cpp_extension.py -v

# Test GPU/CPU consistency
pytest tests/test_cpp_extension.py::TestGPUvsCPU -v
```

## Debugging

### Check Extension Loaded

```python
from mlx_audio_primitives._extension import HAS_CPP_EXT, _ext
print(f"Extension loaded: {HAS_CPP_EXT}")
print(f"Available functions: {dir(_ext)}")
```

### Build with Debug Symbols

```bash
cmake .. -DCMAKE_BUILD_TYPE=Debug
cmake --build .
```

---

*Low-level performance, high-level convenience - that's how we roll.*
