#pragma once

#include <mlx/array.h>
#include <mlx/stream.h>
#include <mlx/utils.h>

namespace mlx_audio {

/**
 * Resample a signal using FFT-based bandlimited interpolation.
 *
 * This implements the same algorithm as scipy.signal.resample and
 * librosa.resample with res_type='fft'.
 *
 * Parameters
 * ----------
 * signal : mlx::core::array
 *     Input signal of shape (samples,) or (batch, samples).
 * num_samples : int
 *     Number of samples in the resampled signal.
 * s : mlx::core::StreamOrDevice
 *     Stream for computation.
 *
 * Returns
 * -------
 * mlx::core::array
 *     Resampled signal with shape (num_samples,) or (batch, num_samples).
 */
mlx::core::array resample_fft(
    const mlx::core::array& signal,
    int num_samples,
    mlx::core::StreamOrDevice s = {});

/**
 * Resample audio from one sample rate to another.
 *
 * Parameters
 * ----------
 * signal : mlx::core::array
 *     Input signal of shape (samples,) or (batch, samples).
 * orig_sr : int
 *     Original sample rate.
 * target_sr : int
 *     Target sample rate.
 * fix : bool
 *     If true, adjust output length to exactly target_sr/orig_sr * len(signal).
 * scale : bool
 *     If true, scale output amplitude by the resampling ratio.
 * s : mlx::core::StreamOrDevice
 *     Stream for computation.
 *
 * Returns
 * -------
 * mlx::core::array
 *     Resampled signal.
 */
mlx::core::array resample(
    const mlx::core::array& signal,
    int orig_sr,
    int target_sr,
    bool fix = true,
    bool scale = false,
    mlx::core::StreamOrDevice s = {});

}  // namespace mlx_audio
