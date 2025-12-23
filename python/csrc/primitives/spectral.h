#pragma once

#include <mlx/array.h>
#include <mlx/stream.h>
#include <mlx/utils.h>

namespace mlx_audio {

/**
 * Compute spectral centroid.
 *
 * The spectral centroid is the center of mass of the spectrum:
 *     centroid = sum(f * S) / sum(S)
 *
 * Parameters
 * ----------
 * S : mlx::core::array
 *     Magnitude spectrogram of shape (freq_bins, n_frames) or
 *     (batch, freq_bins, n_frames).
 * frequencies : mlx::core::array
 *     Frequency values for each bin, shape (freq_bins,).
 * s : mlx::core::StreamOrDevice
 *     Stream for computation.
 *
 * Returns
 * -------
 * mlx::core::array
 *     Spectral centroid for each frame.
 *     Shape: (1, n_frames) or (batch, 1, n_frames).
 */
mlx::core::array spectral_centroid(
    const mlx::core::array& S,
    const mlx::core::array& frequencies,
    mlx::core::StreamOrDevice s = {});

/**
 * Compute spectral bandwidth.
 *
 * bandwidth = (sum(|f - centroid|^p * S) / sum(S))^(1/p)
 *
 * Parameters
 * ----------
 * S : mlx::core::array
 *     Magnitude spectrogram.
 * frequencies : mlx::core::array
 *     Frequency values for each bin.
 * centroid : mlx::core::array
 *     Pre-computed spectral centroid (optional, will compute if empty).
 * p : float
 *     Power for bandwidth calculation. Default: 2.0.
 * s : mlx::core::StreamOrDevice
 *     Stream for computation.
 *
 * Returns
 * -------
 * mlx::core::array
 *     Spectral bandwidth for each frame.
 */
mlx::core::array spectral_bandwidth(
    const mlx::core::array& S,
    const mlx::core::array& frequencies,
    const mlx::core::array& centroid,
    float p = 2.0f,
    mlx::core::StreamOrDevice s = {});

/**
 * Compute spectral rolloff.
 *
 * The rolloff frequency is defined as the frequency below which
 * roll_percent of the spectral energy is contained.
 *
 * Parameters
 * ----------
 * S : mlx::core::array
 *     Magnitude spectrogram.
 * frequencies : mlx::core::array
 *     Frequency values for each bin.
 * roll_percent : float
 *     Percentage of energy threshold. Default: 0.85.
 * s : mlx::core::StreamOrDevice
 *     Stream for computation.
 *
 * Returns
 * -------
 * mlx::core::array
 *     Spectral rolloff for each frame.
 */
mlx::core::array spectral_rolloff(
    const mlx::core::array& S,
    const mlx::core::array& frequencies,
    float roll_percent = 0.85f,
    mlx::core::StreamOrDevice s = {});

/**
 * Compute spectral flatness.
 *
 * Spectral flatness measures how tone-like vs noise-like a signal is:
 *     flatness = exp(mean(log(S))) / mean(S)
 *
 * Parameters
 * ----------
 * S : mlx::core::array
 *     Magnitude spectrogram.
 * amin : float
 *     Minimum amplitude for numerical stability. Default: 1e-10.
 * s : mlx::core::StreamOrDevice
 *     Stream for computation.
 *
 * Returns
 * -------
 * mlx::core::array
 *     Spectral flatness for each frame.
 */
mlx::core::array spectral_flatness(
    const mlx::core::array& S,
    float amin = 1e-10f,
    mlx::core::StreamOrDevice s = {});

}  // namespace mlx_audio
