#pragma once

#include <mlx/mlx.h>

namespace mlx_audio {

/**
 * Frame a signal into overlapping windows.
 *
 * Splits a signal into overlapping frames for STFT processing.
 *
 * @param signal Input signal of shape (batch, samples) or (samples,)
 * @param frame_length Length of each frame (typically n_fft)
 * @param hop_length Number of samples between consecutive frames
 * @param s Stream or device for computation
 * @return Framed signal of shape (batch, n_frames, frame_length)
 */
mlx::core::array frame_signal(
    const mlx::core::array& signal,
    int frame_length,
    int hop_length,
    mlx::core::StreamOrDevice s = {});

}  // namespace mlx_audio
