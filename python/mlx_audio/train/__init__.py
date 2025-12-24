"""mlx-train: A Lightning-like training framework for Apple's MLX.

Stop copying training loops from mlx-examples.

Example:
    >>> from mlx_audio.train import TrainModule, Trainer, OptimizerConfig
    >>> from mlx_audio.train.callbacks import ModelCheckpoint, EarlyStopping
    >>> from mlx_audio.train.schedulers import WarmupCosineScheduler
    >>> from mlx_audio.train.loggers import WandbLogger
    >>>
    >>> class MyModel(TrainModule):
    ...     def compute_loss(self, batch):
    ...         x, y = batch
    ...         logits = self(x)
    ...         loss = mx.mean(nn.losses.cross_entropy(logits, y))
    ...         return loss, {"accuracy": mx.mean(mx.argmax(logits, -1) == y)}
    ...
    ...     def configure_optimizers(self):
    ...         return OptimizerConfig(optimizer=optim.AdamW(learning_rate=1e-4))
    >>>
    >>> trainer = Trainer(
    ...     max_epochs=10,
    ...     callbacks=[EarlyStopping(monitor="val_loss", patience=3)],
    ... )
    >>> trainer.fit(model, train_loader, val_loader)
"""

__version__ = "0.1.0"

from mlx_audio.train.callbacks.base import Callback, CallbackContext, CallbackPriority
from mlx_audio.train.module import OptimizerConfig, TrainModule
from mlx_audio.train.precompute import (
    EmbeddingPreprocessor,
    MelSpectrogramPreprocessor,
    PrecomputeStats,
    Preprocessor,
    precompute_dataset,
    run_precompute_cli,
)
from mlx_audio.train.trainer import Trainer

__all__ = [
    "Callback",
    "CallbackContext",
    "CallbackPriority",
    "EmbeddingPreprocessor",
    "MelSpectrogramPreprocessor",
    "OptimizerConfig",
    "PrecomputeStats",
    "Preprocessor",
    "precompute_dataset",
    "run_precompute_cli",
    "Trainer",
    "TrainModule",
]
