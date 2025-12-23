"""Loggers for mlx-train."""

from mlx_audio.train.loggers.base import Logger
from mlx_audio.train.loggers.wandb import WandbLogger

__all__ = ["Logger", "WandbLogger"]
