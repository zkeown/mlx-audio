"""Voice Activity Detection model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import mlx.core as mx
import mlx.nn as nn

from mlx_audio.models.vad.config import VADConfig
from mlx_audio.models.vad.layers import StackedLSTM, VADDecoder, VADEncoder

if TYPE_CHECKING:
    pass


class SileroVAD(nn.Module):
    """Silero-compatible Voice Activity Detection model.

    LSTM-based model for detecting speech in audio. Processes audio in
    fixed-size windows and outputs speech probability for each window.

    The model maintains internal state (LSTM hidden states and context buffer)
    for streaming inference.

    Args:
        config: VAD configuration

    Example:
        >>> config = VADConfig.silero_vad_16k()
        >>> model = SileroVAD(config)
        >>> audio = mx.random.normal((512,))  # 32ms at 16kHz
        >>> prob, state = model(audio)
        >>> print(f"Speech probability: {float(prob):.3f}")

        >>> # Streaming usage
        >>> state = None
        >>> for chunk in audio_chunks:
        ...     prob, state = model(chunk, state=state)
        ...     if prob > 0.5:
        ...         print("Speech detected")
    """

    def __init__(self, config: VADConfig | None = None) -> None:
        super().__init__()

        if config is None:
            config = VADConfig.silero_vad_16k()
        self.config = config

        # Audio encoder
        self.encoder = VADEncoder(
            input_size=config.window_size_samples,
            hidden_size=config.hidden_size,
        )

        # Stacked LSTM for temporal modeling
        self.lstm = StackedLSTM(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
        )

        # Decoder for speech probability
        self.decoder = VADDecoder(hidden_size=config.hidden_size)

    def __call__(
        self,
        audio: mx.array,
        state: dict | None = None,
    ) -> tuple[mx.array, dict]:
        """Process audio and return speech probability.

        Args:
            audio: Audio samples [batch, samples] or [samples].
                   Expected length: config.window_size_samples
            state: Optional state dict containing:
                   - 'h': LSTM hidden state [num_layers, batch, hidden]
                   - 'c': LSTM cell state [num_layers, batch, hidden]
                   - 'context': Context buffer [batch, context_samples]

        Returns:
            Tuple of:
                - Speech probability [batch, 1] or scalar in [0, 1]
                - Updated state dict
        """
        # Handle batch dimension
        if audio.ndim == 1:
            audio = audio[None, :]
            squeeze_batch = True
        else:
            squeeze_batch = False

        batch_size = audio.shape[0]

        # Initialize state if not provided
        if state is None:
            h = mx.zeros((self.config.num_layers, batch_size, self.config.hidden_size))
            c = mx.zeros((self.config.num_layers, batch_size, self.config.hidden_size))
            context = mx.zeros((batch_size, self.config.context_size_samples))
        else:
            h = state.get("h")
            c = state.get("c")
            context = state.get("context")

            # Validate state shapes
            if h is None or c is None or context is None:
                h = mx.zeros((self.config.num_layers, batch_size, self.config.hidden_size))
                c = mx.zeros((self.config.num_layers, batch_size, self.config.hidden_size))
                context = mx.zeros((batch_size, self.config.context_size_samples))

        # Prepend context from previous window
        audio_with_context = mx.concatenate([context, audio], axis=-1)

        # Encode audio to single feature vector per window
        features = self.encoder(audio_with_context)  # [batch, hidden]

        # Add sequence dimension for LSTM: [batch, hidden] -> [batch, 1, hidden]
        # This matches Silero architecture: 1 timestep per window
        features = features[:, None, :]

        # Process through LSTM (single timestep)
        lstm_out, (new_h, new_c) = self.lstm(features, hidden=(h, c))

        # Extract the single timestep output: [batch, 1, hidden] -> [batch, hidden]
        final_features = lstm_out[:, 0, :]

        # Get speech probability
        prob = self.decoder(final_features)

        # Update context buffer (last context_size samples from current window)
        new_context = audio[:, -self.config.context_size_samples:]

        # Build new state
        new_state = {
            "h": new_h,
            "c": new_c,
            "context": new_context,
        }

        if squeeze_batch:
            prob = prob[0]

        return prob, new_state

    def reset_state(self, batch_size: int = 1) -> dict:
        """Create fresh initial state.

        Args:
            batch_size: Batch size for state tensors

        Returns:
            Initial state dict
        """
        return {
            "h": mx.zeros((self.config.num_layers, batch_size, self.config.hidden_size)),
            "c": mx.zeros((self.config.num_layers, batch_size, self.config.hidden_size)),
            "context": mx.zeros((batch_size, self.config.context_size_samples)),
        }

    def process_audio(
        self,
        audio: mx.array,
        threshold: float | None = None,
    ) -> tuple[mx.array, list[tuple[float, float]]]:
        """Process complete audio and return per-frame probabilities and segments.

        Convenience method for non-streaming inference on complete audio.

        Args:
            audio: Complete audio [samples] or [batch, samples]
            threshold: Speech probability threshold (default: config.threshold)

        Returns:
            Tuple of:
                - Per-window probabilities [num_windows] or [batch, num_windows]
                - List of (start_sec, end_sec) speech segments
        """
        if threshold is None:
            threshold = self.config.threshold

        # Handle batch dimension
        if audio.ndim == 1:
            audio = audio[None, :]
            squeeze_batch = True
        else:
            squeeze_batch = False

        audio.shape[0]
        num_samples = audio.shape[1]
        window_size = self.config.window_size_samples
        sample_rate = self.config.sample_rate

        # Calculate number of windows
        num_windows = (num_samples + window_size - 1) // window_size

        # Pad audio to fit exact windows
        padded_length = num_windows * window_size
        if num_samples < padded_length:
            padding = padded_length - num_samples
            audio = mx.pad(audio, [(0, 0), (0, padding)])

        # Process each window
        probs = []
        state = None

        for i in range(num_windows):
            start = i * window_size
            end = start + window_size
            window = audio[:, start:end]

            prob, state = self(window, state=state)
            probs.append(prob)

        # Stack probabilities
        probs = mx.stack(probs, axis=-1)
        if probs.ndim == 3:
            probs = probs.squeeze(1)  # Remove extra dim from decoder output

        # Convert to segments (for first batch item)
        segments = self._probs_to_segments(
            probs[0] if probs.ndim == 2 else probs,
            threshold=threshold,
            sample_rate=sample_rate,
            window_size=window_size,
        )

        if squeeze_batch:
            probs = probs[0]

        return probs, segments

    def _probs_to_segments(
        self,
        probs: mx.array,
        threshold: float,
        sample_rate: int,
        window_size: int,
    ) -> list[tuple[float, float]]:
        """Convert per-window probabilities to speech segments.

        Args:
            probs: Speech probabilities [num_windows]
            threshold: Speech threshold
            sample_rate: Audio sample rate
            window_size: Window size in samples

        Returns:
            List of (start_sec, end_sec) tuples
        """
        probs_np = probs.tolist() if hasattr(probs, "tolist") else list(probs)
        window_duration = window_size / sample_rate

        segments = []
        in_speech = False
        speech_start = 0.0

        for i, p in enumerate(probs_np):
            time = i * window_duration

            if p > threshold and not in_speech:
                # Speech onset
                in_speech = True
                speech_start = time
            elif p <= threshold and in_speech:
                # Speech offset
                in_speech = False
                segments.append((speech_start, time + window_duration))

        # Handle case where speech extends to end
        if in_speech:
            final_time = len(probs_np) * window_duration
            segments.append((speech_start, final_time))

        return segments

    @classmethod
    def from_pretrained(
        cls,
        path: str | Path,
        **kwargs,
    ) -> SileroVAD:
        """Load pretrained VAD model.

        Args:
            path: Path to model directory containing:
                  - config.json
                  - model.safetensors (or weights.npz)
            **kwargs: Additional arguments to override config

        Returns:
            Loaded SileroVAD model
        """
        path = Path(path)

        # Load config
        config_path = path / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config_dict = json.load(f)
            config = VADConfig.from_dict(config_dict)
        else:
            # Default to 16kHz config
            config = VADConfig.silero_vad_16k()

        # Override config with kwargs
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        # Create model
        model = cls(config)

        # Load weights
        weights_path = path / "model.safetensors"
        if weights_path.exists():
            model.load_weights(str(weights_path))
        else:
            # Try .npz format
            npz_path = path / "weights.npz"
            if npz_path.exists():
                model.load_weights(str(npz_path))

        return model

    def save_pretrained(self, path: str | Path) -> None:
        """Save model to directory.

        Args:
            path: Output directory
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save config
        config_path = path / "config.json"
        with open(config_path, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        # Save weights
        weights_path = path / "model.safetensors"
        self.save_weights(str(weights_path))


__all__ = ["SileroVAD"]
