"""
Shared validation utilities for parameter checking.

These utilities provide consistent error messages across the library.
"""

from __future__ import annotations


def validate_positive(value: int, name: str) -> None:
    """
    Validate that a value is positive.

    Parameters
    ----------
    value : int
        Value to validate.
    name : str
        Parameter name for error message.

    Raises
    ------
    ValueError
        If value is not positive.
    """
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def validate_non_negative(value: float, name: str) -> None:
    """
    Validate that a value is non-negative.

    Parameters
    ----------
    value : float
        Value to validate.
    name : str
        Parameter name for error message.

    Raises
    ------
    ValueError
        If value is negative.
    """
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")


def validate_range(
    value: float,
    name: str,
    min_val: float | None = None,
    max_val: float | None = None,
    min_inclusive: bool = True,
    max_inclusive: bool = True,
) -> None:
    """
    Validate that a value is within a specified range.

    Parameters
    ----------
    value : float
        Value to validate.
    name : str
        Parameter name for error message.
    min_val : float, optional
        Minimum allowed value.
    max_val : float, optional
        Maximum allowed value.
    min_inclusive : bool, default=True
        Whether min_val is inclusive.
    max_inclusive : bool, default=True
        Whether max_val is inclusive.

    Raises
    ------
    ValueError
        If value is outside the specified range.
    """
    if min_val is not None:
        if min_inclusive and value < min_val:
            raise ValueError(f"{name} must be >= {min_val}, got {value}")
        elif not min_inclusive and value <= min_val:
            raise ValueError(f"{name} must be > {min_val}, got {value}")

    if max_val is not None:
        if max_inclusive and value > max_val:
            raise ValueError(f"{name} must be <= {max_val}, got {value}")
        elif not max_inclusive and value >= max_val:
            raise ValueError(f"{name} must be < {max_val}, got {value}")
