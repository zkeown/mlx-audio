"""Streaming context for state management across chunks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx
import numpy as np


@dataclass
class StreamingContext:
    """Manages state for continuous audio processing across chunks.

    This context maintains all state needed for seamless chunk-by-chunk
    processing, including filter states, overlap buffers, and position tracking.

    Attributes:
        sample_rate: Audio sample rate in Hz
        position: Current sample position in the stream
        chunk_counter: Number of chunks processed
        filter_states: Dictionary of filter states (e.g., pre-emphasis zi/zf)
        model_states: Dictionary of model-specific hidden states
        overlap_buffer: Accumulated samples for overlap-add blending
        weight_buffer: Weight accumulator for overlap-add normalization
        metadata: User-defined metadata dictionary

    Example:
        >>> ctx = StreamingContext(sample_rate=44100)
        >>> ctx.update_filter_state("preemphasis", zf_array)
        >>> zi = ctx.get_filter_state("preemphasis")
    """

    sample_rate: int
    position: int = 0
    chunk_counter: int = 0
    filter_states: dict[str, mx.array] = field(default_factory=dict)
    model_states: dict[str, Any] = field(default_factory=dict)
    overlap_buffer: mx.array | None = None
    weight_buffer: mx.array | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def time_position(self) -> float:
        """Current position in seconds."""
        return self.position / self.sample_rate

    def advance(self, num_samples: int) -> None:
        """Advance the stream position by the given number of samples."""
        self.position += num_samples
        self.chunk_counter += 1

    def update_filter_state(self, name: str, state: mx.array) -> None:
        """Update a filter's final state for use as initial state on next chunk.

        Args:
            name: Identifier for the filter (e.g., "preemphasis", "deemphasis")
            state: The filter's final state (zf) to store
        """
        self.filter_states[name] = state

    def get_filter_state(self, name: str) -> mx.array | None:
        """Get a filter's initial state for the current chunk.

        Args:
            name: Identifier for the filter

        Returns:
            The stored state, or None if no state exists for this filter
        """
        return self.filter_states.get(name)

    def update_model_state(self, name: str, state: Any) -> None:
        """Update model-specific state.

        Args:
            name: Identifier for the state
            state: The state to store (can be any pickleable object)
        """
        self.model_states[name] = state

    def get_model_state(self, name: str, default: Any = None) -> Any:
        """Get model-specific state.

        Args:
            name: Identifier for the state
            default: Value to return if state doesn't exist

        Returns:
            The stored state, or default if not found
        """
        return self.model_states.get(name, default)

    def set_overlap_buffer(
        self,
        output_buffer: mx.array,
        weight_buffer: mx.array,
    ) -> None:
        """Set the overlap-add buffers for blending.

        Args:
            output_buffer: Accumulated output samples
            weight_buffer: Accumulated weights for normalization
        """
        self.overlap_buffer = output_buffer
        self.weight_buffer = weight_buffer

    def checkpoint(self) -> dict[str, Any]:
        """Serialize the context state for persistence.

        Returns:
            Dictionary containing all state, with arrays converted to numpy
            for serialization compatibility.
        """
        state = {
            "sample_rate": self.sample_rate,
            "position": self.position,
            "chunk_counter": self.chunk_counter,
            "metadata": self.metadata,
            "filter_states": {
                k: np.array(v) for k, v in self.filter_states.items()
            },
            "model_states": {},
        }

        # Handle model states - convert mx.array to numpy
        for k, v in self.model_states.items():
            if isinstance(v, mx.array):
                state["model_states"][k] = {"type": "mx.array", "data": np.array(v)}
            else:
                state["model_states"][k] = {"type": "other", "data": v}

        # Handle overlap buffers
        if self.overlap_buffer is not None:
            state["overlap_buffer"] = np.array(self.overlap_buffer)
        if self.weight_buffer is not None:
            state["weight_buffer"] = np.array(self.weight_buffer)

        return state

    @classmethod
    def from_checkpoint(cls, data: dict[str, Any]) -> StreamingContext:
        """Restore context from a checkpoint.

        Args:
            data: Dictionary from a previous checkpoint() call

        Returns:
            Restored StreamingContext
        """
        ctx = cls(
            sample_rate=data["sample_rate"],
            position=data["position"],
            chunk_counter=data["chunk_counter"],
            metadata=data.get("metadata", {}),
        )

        # Restore filter states
        for k, v in data.get("filter_states", {}).items():
            ctx.filter_states[k] = mx.array(v)

        # Restore model states
        for k, v in data.get("model_states", {}).items():
            if v["type"] == "mx.array":
                ctx.model_states[k] = mx.array(v["data"])
            else:
                ctx.model_states[k] = v["data"]

        # Restore overlap buffers
        if "overlap_buffer" in data:
            ctx.overlap_buffer = mx.array(data["overlap_buffer"])
        if "weight_buffer" in data:
            ctx.weight_buffer = mx.array(data["weight_buffer"])

        return ctx

    def reset(self) -> None:
        """Reset the context to initial state, preserving sample_rate."""
        self.position = 0
        self.chunk_counter = 0
        self.filter_states.clear()
        self.model_states.clear()
        self.overlap_buffer = None
        self.weight_buffer = None
        self.metadata.clear()
