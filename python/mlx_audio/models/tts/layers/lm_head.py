"""Language model head with delay pattern for Parler-TTS."""

from __future__ import annotations

from typing import TYPE_CHECKING

import mlx.core as mx
import mlx.nn as nn

if TYPE_CHECKING:
    from mlx_audio.models.tts.config import ParlerTTSConfig


class DelayPatternScheduler:
    """Manages the delay pattern for multi-codebook generation.

    In Parler-TTS (like MusicGen), different codebooks are offset in time
    during generation. Codebook k has a delay of k timesteps. This allows
    the model to generate multiple codebooks while respecting dependencies.

    Example with 4 codebooks and 6 timesteps:
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
        pad_token_id: int = 1024,
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
            right_pad = mx.full(
                (batch_size, 1, num_codebooks - 1 - k), self.pad_token_id, dtype=codes.dtype
            )
            codebook_codes = codes[:, k : k + 1, :]  # [B, 1, T]
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
            codebook_codes = delayed_codes[:, k : k + 1, k : k + seq_length]
            codes_list.append(codebook_codes)

        return mx.concatenate(codes_list, axis=1)

    def get_next_token_positions(
        self,
        step: int,
    ) -> list[tuple[int, int]]:
        """Get which codebook positions are valid at a given step.

        Args:
            step: Current generation step (0-indexed in delayed space)

        Returns:
            List of (codebook_idx, original_position) tuples for valid tokens
        """
        positions = []
        for k in range(self.num_codebooks):
            # Codebook k has valid tokens at positions >= k
            if step >= k:
                original_pos = step - k
                positions.append((k, original_pos))
        return positions


class ParlerTTSLMHead(nn.Module):
    """Language model head for Parler-TTS with multiple codebooks.

    Projects hidden states to logits for each codebook independently.
    """

    def __init__(self, config: "ParlerTTSConfig"):
        """Initialize LM head.

        Args:
            config: Parler-TTS configuration
        """
        super().__init__()
        self.config = config
        self.num_codebooks = config.num_codebooks

        # Separate linear projection for each codebook
        # Output dimension is codebook_size + 2 for special tokens (pad, bos)
        vocab_size = config.codebook_size + 2
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
