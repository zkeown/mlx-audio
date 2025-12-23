"""Checkpoint manager for mlx-train."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

from mlx_audio.train.checkpointing.state import TrainerState

if TYPE_CHECKING:
    import mlx.nn as nn
    import mlx.optimizers as optim


class CheckpointManager:
    """Manages saving and loading of training checkpoints.

    Checkpoint Format
    -----------------
    A checkpoint directory contains:

        checkpoint_dir/
            model.safetensors       # Model weights
            optimizer.safetensors   # Optimizer state
            trainer_state.json      # TrainerState as JSON
            callbacks.json          # Callback states as JSON

    This separation allows:
    - Loading just model weights for inference
    - Resuming training with full state
    - Inspecting training progress without loading model

    Why safetensors:
    - MLX native format with efficient memory mapping
    - Safe (no arbitrary code execution)
    - Fast loading with lazy evaluation

    Example:
        >>> manager = CheckpointManager("checkpoints")
        >>> manager.save("epoch_5", model, optimizer, state, callbacks)
        >>> state, callbacks = manager.load("epoch_5", model, optimizer)
    """

    MODEL_FILE = "model.safetensors"
    OPTIMIZER_FILE = "optimizer.safetensors"
    STATE_FILE = "trainer_state.json"
    CALLBACKS_FILE = "callbacks.json"
    BEST_LINK = "best"
    LAST_LINK = "last"

    def __init__(self, base_dir: str | Path) -> None:
        """Initialize the checkpoint manager.

        Args:
            base_dir: Base directory for all checkpoints
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        checkpoint_name: str,
        model: nn.Module,
        optimizer: optim.Optimizer,
        trainer_state: TrainerState,
        callback_states: dict[str, dict[str, Any]],
        is_best: bool = False,
    ) -> Path:
        """Save a complete checkpoint.

        Args:
            checkpoint_name: Name for this checkpoint (e.g., "epoch_10")
            model: The MLX nn.Module
            optimizer: The MLX optimizer
            trainer_state: Current TrainerState
            callback_states: States from all callbacks
            is_best: If True, update "best" symlink

        Returns:
            Path to saved checkpoint directory
        """
        checkpoint_dir = self.base_dir / checkpoint_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save model weights
        model_path = checkpoint_dir / self.MODEL_FILE
        model.save_weights(str(model_path))

        # 2. Save optimizer state
        optimizer_path = checkpoint_dir / self.OPTIMIZER_FILE
        self._save_optimizer_state(optimizer, optimizer_path)

        # 3. Save trainer state
        state_path = checkpoint_dir / self.STATE_FILE
        with open(state_path, "w") as f:
            json.dump(trainer_state.to_dict(), f, indent=2)

        # 4. Save callback states
        callbacks_path = checkpoint_dir / self.CALLBACKS_FILE
        with open(callbacks_path, "w") as f:
            json.dump(callback_states, f, indent=2)

        # 5. Update symlinks
        self._update_symlink(self.LAST_LINK, checkpoint_name)
        if is_best:
            self._update_symlink(self.BEST_LINK, checkpoint_name)

        return checkpoint_dir

    def load(
        self,
        checkpoint_name: str,
        model: nn.Module,
        optimizer: optim.Optimizer,
    ) -> tuple[TrainerState, dict[str, dict[str, Any]]]:
        """Load a checkpoint and restore all state.

        Args:
            checkpoint_name: Name of checkpoint to load (or "best"/"last")
            model: The MLX nn.Module to load weights into
            optimizer: The MLX optimizer to restore state into

        Returns:
            Tuple of (TrainerState, callback_states dict)
        """
        # Resolve symlinks
        checkpoint_dir = self.base_dir / checkpoint_name
        if checkpoint_dir.is_symlink():
            checkpoint_dir = checkpoint_dir.resolve()

        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_dir}")

        # 1. Load model weights
        model_path = checkpoint_dir / self.MODEL_FILE
        model.load_weights(str(model_path))

        # 2. Load optimizer state
        optimizer_path = checkpoint_dir / self.OPTIMIZER_FILE
        self._load_optimizer_state(optimizer, optimizer_path)

        # 3. Load trainer state
        state_path = checkpoint_dir / self.STATE_FILE
        with open(state_path) as f:
            trainer_state = TrainerState.from_dict(json.load(f))

        # 4. Load callback states
        callbacks_path = checkpoint_dir / self.CALLBACKS_FILE
        callback_states: dict[str, dict[str, Any]] = {}
        if callbacks_path.exists():
            with open(callbacks_path) as f:
                callback_states = json.load(f)

        return trainer_state, callback_states

    def load_model_only(self, checkpoint_name: str, model: nn.Module) -> None:
        """Load only model weights (for inference or fine-tuning).

        Args:
            checkpoint_name: Name of checkpoint to load
            model: The MLX nn.Module to load weights into
        """
        checkpoint_dir = self.base_dir / checkpoint_name
        if checkpoint_dir.is_symlink():
            checkpoint_dir = checkpoint_dir.resolve()

        model_path = checkpoint_dir / self.MODEL_FILE
        model.load_weights(str(model_path))

    def list_checkpoints(self) -> list[str]:
        """List all available checkpoints."""
        return [d.name for d in self.base_dir.iterdir() if d.is_dir() and not d.is_symlink()]

    def get_best_checkpoint(self) -> str | None:
        """Get the name of the best checkpoint, if exists."""
        best_link = self.base_dir / self.BEST_LINK
        if best_link.exists() or best_link.is_symlink():
            return best_link.resolve().name
        return None

    def get_last_checkpoint(self) -> str | None:
        """Get the name of the last checkpoint, if exists."""
        last_link = self.base_dir / self.LAST_LINK
        if last_link.exists() or last_link.is_symlink():
            return last_link.resolve().name
        return None

    def _save_optimizer_state(
        self,
        optimizer: optim.Optimizer,
        path: Path,
    ) -> None:
        """Save optimizer state to safetensors."""
        if not hasattr(optimizer, "state") or not optimizer.state:
            # No state to save (optimizer hasn't been used yet)
            mx.save_safetensors(str(path), {})
            return

        # Flatten the optimizer state tree
        flat_state = tree_flatten(optimizer.state)

        # Convert to dict format for safetensors
        # Keys need to be strings, values need to be arrays
        state_dict = {}
        for i, (key, value) in enumerate(flat_state):
            if value is not None and isinstance(value, mx.array):
                # Use index-based keys to preserve order
                state_dict[f"{i}:{key}"] = value

        mx.save_safetensors(str(path), state_dict)

    def _load_optimizer_state(
        self,
        optimizer: optim.Optimizer,
        path: Path,
    ) -> None:
        """Load optimizer state from safetensors."""
        if not path.exists():
            return

        state_dict = mx.load(str(path))
        if not state_dict:
            return

        # Reconstruct the flattened list
        items = []
        for key, value in sorted(state_dict.items(), key=lambda x: int(x[0].split(":")[0])):
            # Extract original key from "index:key" format
            original_key = key.split(":", 1)[1] if ":" in key else key
            items.append((original_key, value))

        # Unflatten and set state
        optimizer.state = tree_unflatten(items)

    def _update_symlink(self, link_name: str, target_name: str) -> None:
        """Create or update a symlink."""
        link_path = self.base_dir / link_name
        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()
        link_path.symlink_to(target_name)
