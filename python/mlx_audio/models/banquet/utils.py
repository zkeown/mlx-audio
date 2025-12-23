"""Utility functions for Banquet model."""

from __future__ import annotations

from typing import List, Tuple

import mlx.core as mx
import numpy as np


def band_widths_from_specs(band_specs: List[Tuple[int, int]]) -> List[int]:
    """Get bandwidth for each band specification."""
    return [end - start for start, end in band_specs]


def check_nonzero_bandwidth(band_specs: List[Tuple[int, int]]) -> None:
    """Check that all bands have positive bandwidth."""
    for start, end in band_specs:
        if end - start <= 0:
            raise ValueError("Bands cannot be zero-width")


def check_no_overlap(band_specs: List[Tuple[int, int]]) -> None:
    """Check that bands do not overlap."""
    end_prev = -1
    for start, end in band_specs:
        if start <= end_prev:
            raise ValueError("Bands cannot overlap")
        end_prev = end


def check_no_gap(band_specs: List[Tuple[int, int]]) -> None:
    """Check that there are no gaps between bands."""
    start, _ = band_specs[0]
    assert start == 0, "First band must start at 0"

    end_prev = -1
    for start, end in band_specs:
        if start - end_prev > 1:
            raise ValueError("Bands cannot leave gap")
        end_prev = end


def hz_to_midi(hz: float) -> float:
    """Convert frequency in Hz to MIDI note number."""
    return 12.0 * np.log2(hz / 440.0) + 69.0


def midi_to_hz(midi: float) -> float:
    """Convert MIDI note number to frequency in Hz."""
    return 440.0 * np.power(2.0, (midi - 69.0) / 12.0)


class MusicalBandsplitSpecification:
    """Musical band specification for 64-band frequency decomposition.

    Uses a musical filterbank based on MIDI note spacing for perceptually
    uniform frequency bands.
    """

    def __init__(
        self,
        nfft: int = 2048,
        fs: int = 44100,
        n_bands: int = 64,
        f_min: float = 0.0,
        f_max: float | None = None,
    ) -> None:
        self.nfft = nfft
        self.fs = fs
        self.n_bands = n_bands
        self.n_freqs = nfft // 2 + 1

        f_max = f_max or fs / 2
        f_min = f_min or fs / nfft
        df = fs / nfft

        # Calculate octave spacing
        n_octaves = np.log2(f_max / f_min)
        n_octaves_per_band = n_octaves / n_bands
        bandwidth_mult = np.power(2.0, n_octaves_per_band)

        # Convert to MIDI for linear spacing
        low_midi = max(0, hz_to_midi(f_min))
        high_midi = hz_to_midi(f_max)
        midi_points = np.linspace(low_midi, high_midi, n_bands)
        hz_pts = midi_to_hz(midi_points)

        # Calculate band boundaries
        low_pts = hz_pts / bandwidth_mult
        high_pts = hz_pts * bandwidth_mult

        low_bins = np.floor(low_pts / df).astype(int)
        high_bins = np.ceil(high_pts / df).astype(int)

        # Create filterbank
        fb = np.zeros((n_bands, self.n_freqs))
        for i in range(n_bands):
            fb[i, low_bins[i] : high_bins[i] + 1] = 1.0

        # Extend first and last bands to cover full spectrum
        fb[0, : low_bins[0]] = 1.0
        fb[-1, high_bins[-1] + 1 :] = 1.0

        self.filterbank = fb

        # Normalize for frequency weights
        weight_per_bin = np.sum(fb, axis=0, keepdims=True)
        weight_per_bin = np.maximum(weight_per_bin, 1e-8)  # Avoid division by zero
        normalized_fb = fb / weight_per_bin

        # Extract band specs and frequency weights
        freq_weights = []
        band_specs = []
        for i in range(n_bands):
            active_bins = np.nonzero(fb[i, :])[0]
            if len(active_bins) == 0:
                continue
            start_idx = int(active_bins[0])
            end_idx = int(active_bins[-1] + 1)
            band_specs.append((start_idx, end_idx))
            freq_weights.append(mx.array(normalized_fb[i, start_idx:end_idx]))

        self._band_specs = band_specs
        self._freq_weights = freq_weights

    def get_band_specs(self) -> List[Tuple[int, int]]:
        """Get list of (start_bin, end_bin) for each band."""
        return self._band_specs

    def get_freq_weights(self) -> List[mx.array]:
        """Get frequency weights for each band."""
        return self._freq_weights

    def get_band_widths(self) -> List[int]:
        """Get bandwidth for each band."""
        return band_widths_from_specs(self._band_specs)
