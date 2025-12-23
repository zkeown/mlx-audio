"""Ring buffer KV cache for efficient Whisper decoding.

Instead of concatenating K/V tensors each step (O(n) copying),
this uses pre-allocated buffers with position tracking for O(1) append.
"""

from __future__ import annotations

import mlx.core as mx


class KVCache:
    """Pre-allocated KV cache with O(1) append operations.

    For autoregressive decoding, the naive approach concatenates
    cached K/V with new K/V each step, resulting in O(n²) total
    operations for n tokens. This cache pre-allocates fixed-size
    buffers and tracks the current position for O(n) total operations.

    The cache stores K/V for all layers in a single structure,
    indexed by layer number.

    Attributes:
        max_length: Maximum sequence length (power of 2 for efficient modulo)
        n_layers: Number of transformer layers
        hidden_dim: Hidden dimension per layer
        length: Current sequence length
    """

    def __init__(
        self,
        max_length: int,
        n_layers: int,
        hidden_dim: int,
        batch_size: int = 1,
        dtype: mx.Dtype = mx.float16,
    ):
        """Initialize KV cache.

        Args:
            max_length: Maximum sequence length
            n_layers: Number of transformer layers
            hidden_dim: Hidden dimension (n_state)
            batch_size: Batch size
            dtype: Data type for cache arrays
        """
        # Round up to power of 2 for efficient indexing
        self.max_length = self._next_power_of_2(max_length)
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.dtype = dtype

        # Current write position (number of tokens written)
        self._length = 0

        # Pre-allocate cache tensors for all layers
        # Shape: [batch, max_length, hidden_dim]
        self._keys: list[mx.array] = []
        self._values: list[mx.array] = []
        for _ in range(n_layers):
            self._keys.append(
                mx.zeros((batch_size, self.max_length, hidden_dim), dtype=dtype)
            )
            self._values.append(
                mx.zeros((batch_size, self.max_length, hidden_dim), dtype=dtype)
            )

    @staticmethod
    def _next_power_of_2(n: int) -> int:
        """Round up to next power of 2."""
        if n <= 0:
            return 1
        n -= 1
        n |= n >> 1
        n |= n >> 2
        n |= n >> 4
        n |= n >> 8
        n |= n >> 16
        return n + 1

    @property
    def length(self) -> int:
        """Current sequence length."""
        return self._length

    @property
    def offset(self) -> int:
        """Offset for positional embeddings (same as length)."""
        return self._length

    def update(
        self,
        layer_idx: int,
        k: mx.array,
        v: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """Append new K/V to cache and return full sequence.

        This operation is O(1) for the append, but returns a view
        of all cached values which is O(n) but avoids copy overhead.

        Args:
            layer_idx: Which layer's cache to update
            k: New key tensor [B, T_new, D] where T_new is typically 1
            v: New value tensor [B, T_new, D]

        Returns:
            Tuple of (all_keys, all_values) [B, length + T_new, D]
        """
        T_new = k.shape[1]

        # Calculate write positions
        start_pos = self._length
        end_pos = start_pos + T_new

        if end_pos > self.max_length:
            raise ValueError(
                f"Sequence length {end_pos} exceeds max_length {self.max_length}"
            )

        # Write new K/V at current position
        # MLX doesn't support true in-place updates, but we can use
        # array slicing assignment which is efficient
        k_cache = self._keys[layer_idx]
        v_cache = self._values[layer_idx]

        # Create updated arrays using slice assignment
        # This creates new arrays but avoids copying the entire cache
        indices = mx.arange(start_pos, end_pos)
        k_cache = k_cache.at[:, indices, :].add(k - k_cache[:, start_pos:end_pos, :])
        v_cache = v_cache.at[:, indices, :].add(v - v_cache[:, start_pos:end_pos, :])

        # Store updated cache
        self._keys[layer_idx] = k_cache
        self._values[layer_idx] = v_cache

        # Return slice of valid cached values
        return k_cache[:, :end_pos, :], v_cache[:, :end_pos, :]

    def step(self, n_tokens: int = 1) -> None:
        """Advance position counter after all layers have been updated.

        Call this once per decode step, after all layers have called update().

        Args:
            n_tokens: Number of tokens processed this step (typically 1)
        """
        self._length += n_tokens

    def reset(self) -> None:
        """Reset cache for new sequence."""
        self._length = 0
        # Optionally zero out cache tensors (not strictly necessary)
        for i in range(self.n_layers):
            self._keys[i] = mx.zeros_like(self._keys[i])
            self._values[i] = mx.zeros_like(self._values[i])

    def get_offset(self) -> int:
        """Get offset for positional embeddings."""
        return self._length

    def __len__(self) -> int:
        """Return number of layers (for compatibility with list-based cache)."""
        return self.n_layers

    def __getitem__(self, layer_idx: int) -> tuple[mx.array, mx.array] | None:
        """Get cached K/V for a layer (for compatibility with list-based cache).

        Returns None if cache is empty, otherwise returns valid portion.
        """
        if self._length == 0:
            return None
        return (
            self._keys[layer_idx][:, : self._length, :],
            self._values[layer_idx][:, : self._length, :],
        )
