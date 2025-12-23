#pragma once

#include <mlx/mlx.h>

namespace mlx_audio {

/**
 * Overlap-add reconstruction with window normalization.
 *
 * Reconstructs a time-domain signal from overlapping windowed frames
 * using the overlap-add method with squared window normalization.
 *
 * Uses a gather-based approach where each output sample computes its
 * value by iterating over contributing frames, avoiding atomic operations.
 *
 * @param frames Input frames of shape (batch, n_frames, n_fft)
 * @param window Window function of shape (n_fft,)
 * @param hop_length Number of samples between consecutive frames
 * @param output_length Desired length of output signal
 * @param s Stream or device for computation
 * @return Reconstructed signal of shape (batch, output_length)
 */
mlx::core::array overlap_add(
    const mlx::core::array& frames,
    const mlx::core::array& window,
    int hop_length,
    int output_length,
    mlx::core::StreamOrDevice s = {});

}  // namespace mlx_audio
