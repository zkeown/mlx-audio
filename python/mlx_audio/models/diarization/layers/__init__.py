"""ECAPA-TDNN layers."""

from .pooling import AttentiveStatisticsPooling
from .se_res2net import SERes2NetBlock

__all__ = [
    "SERes2NetBlock",
    "AttentiveStatisticsPooling",
]
