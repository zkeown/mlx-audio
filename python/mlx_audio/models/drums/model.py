"""Main drum transcription model.

Combines CNN encoder, transformer, and prediction heads into
a complete drum transcription model.

Ported from PyTorch implementation.
"""

import mlx.core as mx
import mlx.nn as nn

from mlx_audio.models.drums.config import DrumTranscriberConfig
from mlx_audio.models.drums.encoder import LightweightEncoder, SpectrogramEncoder
from mlx_audio.models.drums.heads import DualHead
from mlx_audio.models.drums.transformer import LocalAttentionTransformer, TemporalTransformer


class DrumTranscriber(nn.Module):
    """Drum transcription model.

    Architecture:
        Input: Mel-spectrogram (batch, 1, time, n_mels)
        -> CNN Encoder: Extract local spectral features
        -> Transformer: Model temporal dependencies
        -> Dual Head: Predict onsets and velocities
        Output: onset_logits (batch, time, num_classes)
                velocity (batch, time, num_classes)
    """

    def __init__(self, config: DrumTranscriberConfig | None = None):
        """Initialize model.

        Args:
            config: Model configuration. Uses defaults if None.
        """
        super().__init__()

        self.config = config or DrumTranscriberConfig()

        # Build encoder
        if self.config.encoder_type == "lightweight":
            self.encoder = LightweightEncoder(
                n_mels=self.config.n_mels,
                embed_dim=self.config.embed_dim,
                drop_rate=self.config.encoder_dropout,
            )
        else:
            self.encoder = SpectrogramEncoder(
                n_mels=self.config.n_mels,
                embed_dim=self.config.embed_dim,
                base_channels=self.config.base_channels,
                drop_rate=self.config.encoder_dropout,
            )

        # Build transformer
        if self.config.use_local_attention:
            self.transformer = LocalAttentionTransformer(
                embed_dim=self.config.embed_dim,
                num_layers=self.config.num_layers,
                num_heads=self.config.num_heads,
                window_size=self.config.window_size,
                mlp_ratio=self.config.mlp_ratio,
                dropout=self.config.dropout,
                max_len=self.config.max_seq_len,
            )
        else:
            self.transformer = TemporalTransformer(
                embed_dim=self.config.embed_dim,
                num_layers=self.config.num_layers,
                num_heads=self.config.num_heads,
                mlp_ratio=self.config.mlp_ratio,
                dropout=self.config.dropout,
                max_len=self.config.max_seq_len,
            )

        # Build prediction head
        self.head = DualHead(
            embed_dim=self.config.embed_dim,
            num_classes=self.config.num_classes,
            hidden_dim=self.config.head_hidden_dim,
            dropout=self.config.dropout,
            share_layers=self.config.share_head_layers,
        )

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        """Forward pass.

        Args:
            x: Mel-spectrogram (batch, 1, time, n_mels)

        Returns:
            Tuple of:
                - onset_logits (batch, time, num_classes)
                - velocity (batch, time, num_classes) in [0, 1]
        """
        # Encode: (B, 1, T, F) -> (B, T, embed_dim)
        features = self.encoder(x)

        # Temporal modeling: (B, T, embed_dim) -> (B, T, embed_dim)
        features = self.transformer(features)

        # Predict: (B, T, embed_dim) -> (onset_logits, velocity)
        onset_logits, velocity = self.head(features)

        return onset_logits, velocity

    def predict(
        self,
        x: mx.array,
        threshold: float = 0.5,
    ) -> tuple[mx.array, mx.array, mx.array]:
        """Make predictions with thresholding.

        Args:
            x: Mel-spectrogram (batch, 1, time, n_mels)
            threshold: Onset detection threshold

        Returns:
            Tuple of:
                - onset_probs (batch, time, num_classes) probabilities
                - onset_binary (batch, time, num_classes) binary predictions
                - velocity_midi (batch, time, num_classes) in [0, 127]
        """
        onset_logits, velocity = self(x)

        # Convert logits to probabilities
        onset_probs = mx.sigmoid(onset_logits)

        # Threshold for binary predictions
        onset_binary = (onset_probs > threshold).astype(mx.float32)

        # Scale velocity to MIDI range
        velocity_midi = mx.round(velocity * 127).clip(0, 127)

        return onset_probs, onset_binary, velocity_midi

    @property
    def num_parameters(self) -> int:
        """Total number of parameters."""
        from mlx.utils import tree_flatten

        return sum(p.size for _, p in tree_flatten(self.parameters()))


def create_model(
    preset: str = "standard",
    **kwargs,
) -> DrumTranscriber:
    """Create a model with preset configuration.

    Args:
        preset: Configuration preset
            - "standard": Full model for best accuracy
            - "lightweight": Smaller model for faster inference
            - "fast": Optimized for real-time inference
        **kwargs: Override any config parameters

    Returns:
        DrumTranscriber model
    """
    presets = {
        "standard": DrumTranscriberConfig(
            encoder_type="standard",
            embed_dim=512,
            num_layers=4,
            num_heads=8,
        ),
        "lightweight": DrumTranscriberConfig(
            encoder_type="lightweight",
            embed_dim=256,
            num_layers=3,
            num_heads=4,
        ),
        "fast": DrumTranscriberConfig(
            encoder_type="lightweight",
            embed_dim=256,
            num_layers=2,
            num_heads=4,
            use_local_attention=True,
            window_size=32,
        ),
    }

    if preset not in presets:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(presets.keys())}")

    config = presets[preset]

    # Override with kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown config parameter: {key}")

    return DrumTranscriber(config)


def test_model():
    """Test model forward pass."""
    import time

    # Create model
    model = create_model("standard")
    print(f"Model parameters: {model.num_parameters:,}")

    # Test input: 5 seconds at 100fps, 128 mel bins
    x = mx.random.normal((2, 1, 500, 128))

    # Forward pass
    start = time.time()
    onset_logits, velocity = model(x)
    mx.eval(onset_logits, velocity)  # Force evaluation
    elapsed = time.time() - start

    print(f"Input shape: {x.shape}")
    print(f"Onset shape: {onset_logits.shape}")
    print(f"Velocity shape: {velocity.shape}")
    print(f"Forward time: {elapsed:.3f}s")
    print(f"Realtime factor: {5.0 / elapsed:.1f}x")


if __name__ == "__main__":
    test_model()
