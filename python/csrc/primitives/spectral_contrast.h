#pragma once

#include <mlx/array.h>
#include <mlx/stream.h>
#include <mlx/utils.h>

namespace mlx_audio {

/**
 * Compute spectral contrast.
 *
 * Spectral contrast measures the difference between peaks and valleys
 * in each frequency band. It's useful for music classification.
 *
 * Parameters
 * ----------
 * S : mlx::core::array
 *     Magnitude spectrogram of shape (freq_bins, n_frames) or
 *     (batch, freq_bins, n_frames).
 * frequencies : mlx::core::array
 *     Frequency values for each bin, shape (freq_bins,).
 * fmin : float
 *     Minimum frequency for band computation. Default: 200.0.
 * n_bands : int
 *     Number of octave bands. Default: 6.
 * quantile : float
 *     Quantile for peak/valley estimation. Default: 0.02.
 * linear : bool
 *     If True, return linear contrast. If False, return log contrast.
 *     Default: false.
 * s : mlx::core::StreamOrDevice
 *     Stream for computation.
 *
 * Returns
 * -------
 * mlx::core::array
 *     Spectral contrast for each band and frame.
 *     Shape: (n_bands + 1, n_frames) or (batch, n_bands + 1, n_frames).
 */
mlx::core::array spectral_contrast(
    const mlx::core::array& S,
    const mlx::core::array& frequencies,
    float fmin = 200.0f,
    int n_bands = 6,
    float quantile = 0.02f,
    bool linear = false,
    mlx::core::StreamOrDevice s = {});

}  // namespace mlx_audio
