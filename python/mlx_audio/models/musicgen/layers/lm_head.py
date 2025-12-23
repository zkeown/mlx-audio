"""Language model head with delay pattern for MusicGen."""

from __future__ import annotations

from typing import TYPE_CHECKING

import mlx.core as mx
import mlx.nn as nn

if TYPE_CHECKING:
    from mlx_audio.models.musicgen.config import MusicGenConfig


class DelayPatternScheduler:
    """Manages the delay pattern for multi-codebook generation.

    In MusicGen, different codebooks are offset in time during generation.
    Codebook k has a delay of k timesteps. This allows the model to
    generate multiple codebooks in parallel while respecting dependencies.

    Example with 4 codebooks and 6 timesteps:
        Without delay:
            t=0  t=1  t=2  t=3  t=4  t=5
        k=0:  0    1    2    3    4    5
        k=1:  0    1    2    3    4    5
        k=2:  0    1    2    3    4    5
        k=3:  0    1    2    3    4    5

        With delay pattern:
            t=0  t=1  t=2  t=3  t=4  t=5  t=6  t=7  t=8
        k=0:  0    1    2    3    4    5    -    -    -
        k=1:  -    0    1    2    3    4    5    -    -
        k=2:  -    -    0    1    2    3    4    5    -
        k=3:  -    -    -    0    1    2    3    4    5

    Where '-' represents padding tokens.
    """

    def __init__(
        self,
        num_codebooks: int,
        pad_token_id: int = 2048,
    ):
        """Initialize delay pattern scheduler.

        Args:
            num_codebooks: Number of codebooks
            pad_token_id: Token ID to use for padding
        """
        self.num_codebooks = num_codebooks
        self.pad_token_id = pad_token_id

    def apply_delay_pattern(
        self,
        codes: mx.array,
    ) -> mx.array:
        """Apply delay pattern to codes.

        Args:
            codes: Input codes [B, K, T] where K is num_codebooks

        Returns:
            Delayed codes [B, K, T + K - 1]
        """
        batch_size, num_codebooks, seq_length = codes.shape
        delayed_length = seq_length + num_codebooks - 1

        # Build delayed codes by padding each codebook appropriately
        delayed_list = []
        for k in range(num_codebooks):
            # Pad left with k padding tokens, right with (num_codebooks - 1 - k) tokens
            left_pad = mx.full((batch_size, 1, k), self.pad_token_id, dtype=codes.dtype)
            right_pad = mx.full((batch_size, 1, num_codebooks - 1 - k), self.pad_token_id, dtype=codes.dtype)
            codebook_codes = codes[:, k:k+1, :]  # [B, 1, T]
            delayed_codebook = mx.concatenate([left_pad, codebook_codes, right_pad], axis=2)
            delayed_list.append(delayed_codebook)

        return mx.concatenate(delayed_list, axis=1)

    def revert_delay_pattern(
        self,
        delayed_codes: mx.array,
    ) -> mx.array:
        """Remove delay pattern from codes.

        Args:
            delayed_codes: Delayed codes [B, K, T_delayed]

        Returns:
            Original codes [B, K, T] where T = T_delayed - K + 1
        """
        batch_size, num_codebooks, delayed_length = delayed_codes.shape
        seq_length = delayed_length - num_codebooks + 1

        if seq_length <= 0:
            # Not enough tokens to revert
            return delayed_codes[:, :, :1]

        # Extract codes by slicing each codebook with its delay offset
        codes_list = []
        for k in range(num_codebooks):
            # Codebook k's data starts at position k
            codebook_codes = delayed_codes[:, k:k+1, k:k + seq_length]
            codes_list.append(codebook_codes)

        return mx.concatenate(codes_list, axis=1)

    def build_delay_pattern_mask(
        self,
        seq_length: int,
        device: mx.Device | None = None,
    ) -> mx.array:
        """Build attention mask for delay pattern.

        Creates a mask that allows each position to attend only to
        valid (non-padding) positions according to the delay pattern.

        Args:
            seq_length: Sequence length (after applying delay)
            device: Device to place the mask on

        Returns:
            Attention mask [K, T, T] where valid positions are 0
            and masked positions are -inf
        """
        # For each codebook k, position t can attend to positions 0..t
        # but only if those positions are valid (>= k for that codebook)

        # Build mask for each codebook separately using numpy-style indexing
        masks = []
        for k in range(self.num_codebooks):
            # Create a 2D mask for this codebook [T, T]
            # Position t can attend to positions k to t (inclusive)
            row_indices = mx.arange(seq_length)[:, None]  # [T, 1]
            col_indices = mx.arange(seq_length)[None, :]  # [1, T]

            # Valid if: col >= k (after delay start) AND col <= row (causal)
            valid = (col_indices >= k) & (col_indices <= row_indices)

            # Convert to attention mask: 0 for valid, -inf for masked
            codebook_mask = mx.where(valid, 0.0, float("-inf"))
            masks.append(codebook_mask)

        # Stack to [K, T, T]
        return mx.stack(masks, axis=0)

    def get_valid_positions(
        self,
        step: int,
    ) -> list[int]:
        """Get which codebooks have valid tokens at a given step.

        Args:
            step: Current generation step (0-indexed)

        Returns:
            List of codebook indices that have valid tokens at this step
        """
        # Codebook k has valid tokens at positions >= k
        return [k for k in range(self.num_codebooks) if step >= k]


class MusicGenLMHead(nn.Module):
    """Language model head for MusicGen with multiple codebooks.

    Projects hidden states to logits for each codebook independently.
    """

    def __init__(self, config: "MusicGenConfig"):
        """Initialize LM head.

        Args:
            config: MusicGen configuration
        """
        super().__init__()
        self.config = config
        self.num_codebooks = config.num_codebooks

        # Separate linear projection for each codebook
        # Output dimension is codebook_size + 1 for special tokens
        vocab_size = config.codebook_size + 1
        self.linears = [
            nn.Linear(config.hidden_size, vocab_size, bias=False)
            for _ in range(config.num_codebooks)
        ]

    def __call__(
        self,
        hidden_states: mx.array,
        codebook_idx: int | None = None,
    ) -> mx.array:
        """Project hidden states to logits.

        Args:
            hidden_states: Hidden states [B, T, D]
            codebook_idx: Optional specific codebook index.
                         If None, returns logits for all codebooks.

        Returns:
            Logits [B, T, V] if codebook_idx specified, else [B, K, T, V]
        """
        if codebook_idx is not None:
            # Single codebook
            return self.linears[codebook_idx](hidden_states)

        # All codebooks
        logits = []
        for linear in self.linears:
            logits.append(linear(hidden_states))

        # Stack to [B, K, T, V]
        return mx.stack(logits, axis=1)

    def get_codebook_logits(
        self,
        hidden_states: mx.array,
        codebook_idx: int,
    ) -> mx.array:
        """Get logits for a specific codebook.

        Args:
            hidden_states: Hidden states [B, T, D]
            codebook_idx: Codebook index

        Returns:
            Logits [B, T, V]
        """
        return self.linears[codebook_idx](hidden_states)
