// KVCache.swift
// Ring buffer KV cache for efficient Whisper decoding.
//
// Instead of concatenating K/V tensors each step (O(n) copying),
// this uses pre-allocated buffers with position tracking for O(1) append.

import Foundation
import MLX

/// Pre-allocated KV cache with O(1) append operations.
///
/// For autoregressive decoding, the naive approach concatenates
/// cached K/V with new K/V each step, resulting in O(n²) total
/// operations for n tokens. This cache pre-allocates fixed-size
/// buffers and tracks the current position for O(n) total operations.
///
/// The cache stores K/V for all layers in a single structure,
/// indexed by layer number.
///
/// Example:
/// ```swift
/// let cache = WhisperKVCache(maxLength: 512, nLayers: 6, hiddenDim: 512)
///
/// // In each decode step:
/// let (allK, allV) = cache.update(layer: layerIdx, k: newK, v: newV)
/// // Use allK, allV for attention...
///
/// // After processing all layers:
/// cache.step()
/// ```
///
/// ## Thread Safety
///
/// `WhisperKVCache` is marked `@unchecked Sendable` with the following constraints:
///
/// **Safe usage patterns:**
/// - Single-threaded decoding (typical use case)
/// - Sequential layer updates within a single decode step
/// - Reset between independent sequences
///
/// **Why @unchecked Sendable:**
/// 1. **MLXArray operations are GPU-serialized**: All array operations (`at[].add()`,
///    slicing) are dispatched to MLX's GPU command queue and execute in-order.
/// 2. **Intended single-writer pattern**: The cache is designed for sequential
///    autoregressive decoding where one thread updates layers in order.
/// 3. **Immutable configuration**: `maxLength`, `nLayers`, `hiddenDim`, `batchSize`
///    are set at init and never modified.
///
/// **Not recommended:**
/// - Concurrent `update()` calls to the same layer from multiple threads
/// - Calling `step()` while `update()` is in progress
///
/// For concurrent decoding of multiple sequences, create separate cache instances.
public class WhisperKVCache: @unchecked Sendable {

    /// Maximum sequence length (power of 2 for efficient indexing).
    public let maxLength: Int

    /// Number of transformer layers.
    public let nLayers: Int

    /// Hidden dimension per layer.
    public let hiddenDim: Int

    /// Batch size.
    public let batchSize: Int

    /// Current sequence length (number of tokens written).
    private var _length: Int = 0

    /// Pre-allocated key cache tensors, one per layer.
    /// Shape per layer: [batch, maxLength, hiddenDim]
    private var _keys: [MLXArray]

    /// Pre-allocated value cache tensors, one per layer.
    /// Shape per layer: [batch, maxLength, hiddenDim]
    private var _values: [MLXArray]

    /// Current sequence length.
    public var length: Int { _length }

    /// Offset for positional embeddings (same as length).
    public var offset: Int { _length }

    /// Initialize KV cache with pre-allocated buffers.
    ///
    /// - Parameters:
    ///   - maxLength: Maximum sequence length (will be rounded to power of 2)
    ///   - nLayers: Number of transformer layers
    ///   - hiddenDim: Hidden dimension (n_state)
    ///   - batchSize: Batch size (default: 1)
    ///   - dtype: Data type for cache arrays (default: float16)
    public init(
        maxLength: Int,
        nLayers: Int,
        hiddenDim: Int,
        batchSize: Int = 1,
        dtype: DType = .float16
    ) {
        // Round up to power of 2 for efficient indexing
        let roundedMaxLength = Self.nextPowerOf2(maxLength)
        self.maxLength = roundedMaxLength
        self.nLayers = nLayers
        self.hiddenDim = hiddenDim
        self.batchSize = batchSize

        // Pre-allocate cache tensors for all layers
        // Shape: [batch, maxLength, hiddenDim]
        var keys: [MLXArray] = []
        var values: [MLXArray] = []
        for _ in 0..<nLayers {
            keys.append(MLXArray.zeros([batchSize, roundedMaxLength, hiddenDim], dtype: dtype))
            values.append(MLXArray.zeros([batchSize, roundedMaxLength, hiddenDim], dtype: dtype))
        }
        self._keys = keys
        self._values = values

        // Force allocation
        eval(_keys + _values)
    }

    /// Round up to next power of 2.
    private static func nextPowerOf2(_ n: Int) -> Int {
        guard n > 0 else { return 1 }
        var value = n - 1
        value |= value >> 1
        value |= value >> 2
        value |= value >> 4
        value |= value >> 8
        value |= value >> 16
        return value + 1
    }

    /// Append new K/V to cache and return full sequence.
    ///
    /// This operation is O(1) for the append using at[].add() pattern.
    /// Returns views of all cached values up to current position.
    ///
    /// - Parameters:
    ///   - layer: Which layer's cache to update
    ///   - k: New key tensor [B, T_new, D] where T_new is typically 1
    ///   - v: New value tensor [B, T_new, D]
    /// - Returns: Tuple of (all_keys, all_values) [B, length + T_new, D]
    public func update(
        layer: Int,
        k: MLXArray,
        v: MLXArray
    ) -> (MLXArray, MLXArray) {
        let tNew = k.dim(1)

        // Calculate write positions
        let startPos = _length
        let endPos = startPos + tNew

        precondition(
            endPos <= maxLength,
            "KVCache overflow: sequence length \(endPos) exceeds maxLength \(maxLength). " +
            "Reset the cache or create one with larger maxLength."
        )

        // Write new K/V at current position using at[].add() pattern.
        // This avoids full tensor copies (O(1) instead of O(n)).
        // Pattern: cache = cache.at[indices].add(new - old)
        _keys[layer] = _keys[layer].at[0..., startPos..<endPos, 0...].add(
            k - _keys[layer][0..., startPos..<endPos, 0...]
        )
        _values[layer] = _values[layer].at[0..., startPos..<endPos, 0...].add(
            v - _values[layer][0..., startPos..<endPos, 0...]
        )

        // Return slice of valid cached values
        return (_keys[layer][0..., 0..<endPos, 0...], _values[layer][0..., 0..<endPos, 0...])
    }

    /// Advance position counter after all layers have been updated.
    ///
    /// Call this once per decode step, after all layers have called update().
    ///
    /// - Parameter nTokens: Number of tokens processed this step (typically 1)
    public func step(nTokens: Int = 1) {
        _length += nTokens
    }

    /// Reset cache for new sequence.
    ///
    /// Zeros out cache tensors and resets position to 0.
    public func reset() {
        _length = 0
        // Optionally zero out cache tensors (not strictly necessary but cleaner)
        for i in 0..<nLayers {
            _keys[i] = MLXArray.zeros(_keys[i].shape, dtype: _keys[i].dtype)
            _values[i] = MLXArray.zeros(_values[i].shape, dtype: _values[i].dtype)
        }
        eval(_keys + _values)
    }

    /// Get cached K/V for a layer (for compatibility with list-based cache).
    ///
    /// - Parameter layer: Layer index
    /// - Returns: Tuple of (keys, values) up to current length, or nil if empty
    public subscript(layer: Int) -> (MLXArray, MLXArray)? {
        guard _length > 0 else { return nil }
        return (
            _keys[layer][0..., 0..<_length, 0...],
            _values[layer][0..., 0..<_length, 0...]
        )
    }
}
