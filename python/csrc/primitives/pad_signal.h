#pragma once

#include <mlx/mlx.h>
#include <string>

namespace mlx_audio {

/**
 * Pad a signal on both sides.
 *
 * Supports multiple padding modes for STFT center padding.
 *
 * @param signal Input signal of shape (batch, samples)
 * @param pad_length Number of samples to pad on each side
 * @param mode Padding mode: "constant", "reflect", or "edge"
 * @param s Stream or device for computation
 * @return Padded signal of shape (batch, samples + 2 * pad_length)
 */
mlx::core::array pad_signal(
    const mlx::core::array& signal,
    int pad_length,
    const std::string& mode = "constant",
    mlx::core::StreamOrDevice s = {});

}  // namespace mlx_audio
