# Pitch Detection Optimization Plan

## Executive Summary

The pitch detection algorithms in `primitives/pitch.py` contain significant optimization opportunities due to nested batch×frame loops with per-frame FFT operations. This plan details a **batched FFT approach** that can provide **10-50x speedups** while maintaining numerical parity with librosa.

**Scope**: `autocorrelation()`, `pitch_detect_acf()`, `periodicity()`, `yin()`, `pyin()`, and `_cumulative_mean_normalized_difference()`

**Risk Level**: Medium-High (algorithmic changes, but well-defined parity tests exist)

---

## Current Performance Bottlenecks

### 1. Nested Batch×Frame Loops with Per-Frame FFT

**Location**: `pitch_detect_acf()` lines 204-256

```python
for b in range(batch_size):          # O(B)
    for t in range(n_frames):        # O(T) - typically 40-200 frames/sec
        frame = y_np[b, start:end]
        Y = np.fft.rfft(...)         # FFT per frame!
        power = Y * np.conj(Y)
        r = np.fft.irfft(...)        # IFFT per frame!
        # ... peak detection
```

**Impact**: For 1 second of audio at 22050 Hz with default settings:
- `n_frames ≈ 43` frames
- Each frame: 2 FFT operations + peak search
- Total: 86 FFT calls per second of audio

**Similar patterns in**:
- `periodicity()` lines 342-362 (same nested loop structure)
- `yin()` lines 570-609 (nested batch→frame loop)
- `_cumulative_mean_normalized_difference()` lines 404-442 (per-frame FFT + sequential loops)
- `pyin()` lines 752-803 (per-frame threshold search)

### 2. Sequential Cumulative Sum in CMNDF

**Location**: `_cumulative_mean_normalized_difference()` lines 425-441

```python
for tau in range(1, max_period + 1):      # O(max_period) ≈ 400
    diff_values[tau] = 2 * (energy - acf[tau])

for tau in range(1, max_period + 1):      # Another O(max_period)
    cumsum += diff_values[tau]            # Sequential dependency
    if tau >= min_period:
        cmndf[t, tau_idx] = diff_values[tau] / mean
```

**Impact**: Two sequential loops per frame, cannot be parallelized with current structure.

### 3. Per-Frame Peak Detection with Python Loops

**Location**: `pitch_detect_acf()` lines 233-255

```python
peaks = []
for i in range(1, len(search_range) - 1):    # O(lag_range) ≈ 400
    if search_range[i] > search_range[i-1] and search_range[i] > search_range[i+1]:
        if search_range[i] > threshold:
            peaks.append((i, search_range[i]))
```

**Impact**: Python loop over autocorrelation lags for each frame.

---

## Optimization Strategy

### Phase 1: Batched FFT for Autocorrelation (HIGH IMPACT)

**Goal**: Process all frames with a single batched FFT call instead of per-frame FFT.

**Implementation for `pitch_detect_acf()` and `periodicity()`**:

```python
# BEFORE: O(batch × frames) FFT calls
for b in range(batch_size):
    for t in range(n_frames):
        Y = np.fft.rfft(frame, n=n_fft)
        ...

# AFTER: O(1) batched FFT call
# Step 1: Frame all signals at once using strided views
all_frames = frame_signal_batched(y_padded, frame_length, hop_length)
# Shape: (batch, n_frames, frame_length)

# Step 2: Reshape for batched FFT
all_frames_flat = all_frames.reshape(-1, frame_length)  # (batch*n_frames, frame_length)

# Step 3: Single batched FFT
Y = np.fft.rfft(all_frames_flat, n=n_fft, axis=-1)
power = Y * np.conj(Y)
acf = np.fft.irfft(power, n=n_fft, axis=-1)[:, :frame_length]

# Step 4: Reshape back
acf = acf.reshape(batch_size, n_frames, -1)  # (batch, n_frames, max_lag)
```

**Expected Speedup**: 10-20x for typical workloads (fewer FFT kernel launches, better memory locality)

**Files to modify**:
- `pitch_detect_acf()` lines 200-256
- `periodicity()` lines 339-363

---

### Phase 2: Vectorized CMNDF Computation (HIGH IMPACT)

**Goal**: Process all frames simultaneously in `_cumulative_mean_normalized_difference()`.

**Current signature**:
```python
def _cumulative_mean_normalized_difference(
    y_frames: np.ndarray,        # (n_frames, frame_length)
    min_period: int,
    max_period: int,
) -> np.ndarray:
```

**Optimized implementation**:

```python
def _cumulative_mean_normalized_difference_vectorized(
    y_frames: np.ndarray,
    min_period: int,
    max_period: int,
) -> np.ndarray:
    n_frames, frame_length = y_frames.shape
    n_lags = max_period - min_period

    # Batched FFT for all frames at once
    n_fft = 2 ** int(np.ceil(np.log2(2 * frame_length - 1)))
    Y = np.fft.rfft(y_frames, n=n_fft, axis=-1)           # (n_frames, n_fft//2+1)
    power = Y * np.conj(Y)
    acf = np.fft.irfft(power, n=n_fft, axis=-1)[:, :frame_length].real  # (n_frames, frame_length)

    # Vectorized difference function
    # d(tau) = 2 * (acf[:, 0] - acf[:, tau])
    energy = acf[:, 0:1]  # (n_frames, 1)
    diff = 2 * (energy - acf[:, 1:max_period+1])  # (n_frames, max_period)

    # Cumulative sum for CMNDF normalization
    # cumsum[tau] = sum(diff[1:tau+1])
    cumsum = np.cumsum(diff, axis=-1)  # (n_frames, max_period)

    # CMNDF: d'(tau) = d(tau) / (cumsum[tau] / tau)
    # Avoid division by zero
    tau_values = np.arange(1, max_period + 1)[None, :]  # (1, max_period)
    mean = cumsum / tau_values
    mean = np.maximum(mean, 1e-10)

    # Select valid tau range [min_period, max_period)
    cmndf = diff[:, min_period-1:max_period] / mean[:, min_period-1:max_period]

    return cmndf
```

**Expected Speedup**: 15-30x (single batched FFT + vectorized cumsum)

**Files to modify**:
- `_cumulative_mean_normalized_difference()` lines 373-443

---

### Phase 3: Vectorized Peak Detection (MEDIUM IMPACT)

**Goal**: Replace Python loop with vectorized operations.

**Current implementation** (per-frame):
```python
for i in range(1, len(search_range) - 1):
    if search_range[i] > search_range[i-1] and search_range[i] > search_range[i+1]:
        peaks.append(...)
```

**Vectorized implementation** (all frames at once):

```python
def find_first_peak_above_threshold(acf: np.ndarray, threshold: float) -> np.ndarray:
    """
    Find first peak above threshold for each frame.

    Args:
        acf: Autocorrelation values, shape (n_frames, n_lags)
        threshold: Minimum peak value

    Returns:
        peak_idx: Index of first valid peak for each frame, -1 if none found
    """
    n_frames, n_lags = acf.shape

    # Detect local maxima: acf[i] > acf[i-1] AND acf[i] > acf[i+1]
    is_peak = np.zeros((n_frames, n_lags), dtype=bool)
    is_peak[:, 1:-1] = (
        (acf[:, 1:-1] > acf[:, :-2]) &
        (acf[:, 1:-1] > acf[:, 2:]) &
        (acf[:, 1:-1] > threshold)
    )

    # Find first peak per frame using argmax trick
    # argmax returns first True when array is boolean
    peak_idx = np.argmax(is_peak, axis=-1)  # (n_frames,)

    # Mark frames with no valid peaks
    has_peak = np.any(is_peak, axis=-1)
    peak_idx = np.where(has_peak, peak_idx, -1)

    return peak_idx
```

**Expected Speedup**: 3-5x for peak detection phase

**Files to modify**:
- `pitch_detect_acf()` lines 228-255 (peak detection loop)
- `yin()` lines 582-608 (CMNDF minimum search)

---

### Phase 4: Optimize YIN's Frame Extraction (LOW-MEDIUM IMPACT)

**Current** (nested loop):
```python
for b in range(batch_size):
    frames = np.zeros((n_frames, frame_length), dtype=np.float32)
    for t in range(n_frames):
        start = t * hop_length
        frames[t] = y_np[b, start:end]
    cmndf = _cumulative_mean_normalized_difference(frames, ...)
```

**Optimized** (strided view + batch processing):
```python
# Use existing framing infrastructure
from mlx_audio.primitives._frame_impl import frame_signal_batched

# Frame all batches at once
y_mx = mx.array(y_np)
all_frames = frame_signal_batched(y_mx, frame_length, hop_length)  # (batch, n_frames, frame_length)
all_frames_np = np.array(all_frames)

# Reshape for CMNDF: (batch * n_frames, frame_length)
flat_frames = all_frames_np.reshape(-1, frame_length)

# Single call to vectorized CMNDF
cmndf_flat = _cumulative_mean_normalized_difference_vectorized(flat_frames, min_period, max_period)

# Reshape back: (batch, n_frames, n_lags)
cmndf = cmndf_flat.reshape(batch_size, n_frames, -1)
```

**Expected Speedup**: 2-3x (better memory access pattern + reuse existing optimized framing)

---

### Phase 5: PYIN Threshold Loop Optimization (MEDIUM IMPACT)

**Current** (nested loops):
```python
for t in range(n_frames):           # O(n_frames)
    for thresh in thresholds:       # O(100 thresholds)
        for tau_idx in range(...):  # O(n_lags)
            ...
```

**Optimized** (vectorized threshold search):
```python
# Broadcast CMNDF against all thresholds
# cmndf: (n_frames, n_lags)
# thresholds: (n_thresholds,)
below_thresh = cmndf[:, None, :] < thresholds[None, :, None]  # (n_frames, n_thresholds, n_lags)

# Find local minima (for YIN, we want minima, not maxima)
is_minimum = ...  # Similar vectorized approach

# First minimum below each threshold
first_valid = np.argmax(below_thresh & is_minimum, axis=-1)  # (n_frames, n_thresholds)
```

**Expected Speedup**: 10-20x (eliminates O(frames × thresholds × lags) Python loop)

---

## Implementation Order

| Priority | Phase | Function | Est. Speedup | Risk | Effort |
|----------|-------|----------|--------------|------|--------|
| 1 | 1a | `pitch_detect_acf()` batched FFT | 10-20x | Medium | 2-3 hrs |
| 2 | 2 | `_cumulative_mean_normalized_difference()` | 15-30x | Medium | 2-3 hrs |
| 3 | 1b | `periodicity()` batched FFT | 10-20x | Low | 1 hr |
| 4 | 3 | Vectorized peak detection | 3-5x | Low | 1-2 hrs |
| 5 | 4 | `yin()` frame extraction | 2-3x | Low | 1 hr |
| 6 | 5 | `pyin()` threshold optimization | 10-20x | Medium | 3-4 hrs |

**Total estimated effort**: 10-16 hours

---

## Testing Strategy

### 1. Numerical Parity Tests (CRITICAL)

All optimizations MUST pass existing parity tests:
```bash
pytest tests/primitives/test_pitch_parity.py -v
```

Key tolerances from existing tests:
- YIN pure tone detection: ±10 Hz from known frequency
- YIN vs librosa median: ±5 Hz
- PYIN pure tone: ±15 Hz
- Output length: within 2 frames of librosa

### 2. Unit Tests
```bash
pytest tests/primitives/test_pitch.py -v
```

### 3. Performance Benchmarks

Create `benchmarks/bench_pitch.py`:
```python
def bench_pitch_detect_acf():
    """Benchmark pitch_detect_acf with various input sizes."""
    for duration in [0.5, 1.0, 2.0, 5.0]:
        audio = generate_sine(440, 22050, duration)
        # Time the operation

def bench_yin():
    """Benchmark YIN algorithm."""
    ...

def bench_pyin():
    """Benchmark PYIN algorithm."""
    ...
```

### 4. Regression Prevention

For each optimization:
1. Run parity tests BEFORE changes
2. Make incremental changes
3. Run parity tests AFTER each change
4. Document any tolerance adjustments (should be rare)

---

## Risks and Mitigations

### Risk 1: Numerical Differences from Batching

**Issue**: Batched FFT may have different numerical properties than per-frame FFT.

**Mitigation**:
- Use same `n_fft` calculation as original
- Verify intermediate values (autocorrelation) match within `atol=1e-6`
- Test on edge cases (very short, very long, silence, pure tones)

### Risk 2: Memory Pressure from Large Batches

**Issue**: Processing all frames at once may require more memory.

**Mitigation**:
- For very long audio (>10s), process in chunks
- Add optional `max_batch_frames` parameter
- Profile memory usage before/after

### Risk 3: Breaking C++ Extension Path

**Issue**: `autocorrelation()` has C++ extension path that may need updates.

**Mitigation**:
- Keep C++ path unchanged initially
- Only optimize Python fallback first
- Document C++ extension update as future work

---

## Files to Modify

1. **`python/mlx_audio/primitives/pitch.py`**
   - `pitch_detect_acf()` - Phase 1a, 3
   - `periodicity()` - Phase 1b
   - `_cumulative_mean_normalized_difference()` - Phase 2
   - `yin()` - Phase 4
   - `pyin()` - Phase 5

2. **`python/benchmarks/bench_pitch.py`** (new file)
   - Benchmark suite for pitch detection

---

## Success Criteria

1. All existing parity tests pass unchanged
2. Performance improvement of at least 5x on typical workloads
3. No increase in memory usage for standard audio lengths (≤5s)
4. No new dependencies added

---

## Future Work (Out of Scope)

1. MLX-native FFT when available (currently uses NumPy)
2. C++ extension updates for batched processing
3. GPU-accelerated peak detection
4. Real-time streaming pitch detection
