#pragma once

#include <mlx/array.h>
#include <mlx/stream.h>
#include <mlx/utils.h>

namespace mlx_audio {

/**
 * Compute autocorrelation of a signal using FFT (Wiener-Khinchin theorem).
 *
 * r[k] = IFFT(|FFT(y)|^2)
 *
 * Parameters
 * ----------
 * signal : mlx::core::array
 *     Input signal of shape (samples,) or (batch, samples).
 * max_lag : int
 *     Maximum lag to compute. If <= 0, uses signal length.
 * normalize : bool
 *     If true, normalize by r[0] so that r[0] = 1.
 * center : bool
 *     If true, subtract mean before computing autocorrelation.
 * s : mlx::core::StreamOrDevice
 *     Stream for computation.
 *
 * Returns
 * -------
 * mlx::core::array
 *     Autocorrelation values for lags 0 to max_lag-1.
 *     Shape: (max_lag,) for 1D input, (batch, max_lag) for batched.
 */
mlx::core::array autocorrelation(
    const mlx::core::array& signal,
    int max_lag = -1,
    bool normalize = true,
    bool center = true,
    mlx::core::StreamOrDevice s = {});

}  // namespace mlx_audio
