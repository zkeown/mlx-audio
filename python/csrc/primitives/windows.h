#pragma once

#include <mlx/mlx.h>
#include <string>

namespace mlx_audio {

/**
 * Generate a window function.
 *
 * Generates various window functions commonly used in spectral analysis.
 *
 * @param window_type Window type: "hann", "hamming", "blackman", "bartlett", "rectangular"
 * @param length Window length
 * @param periodic If true, create periodic (DFT-even) window for FFT
 * @param s Stream or device for computation
 * @return Window array of shape (length,) with dtype float32
 */
mlx::core::array generate_window(
    const std::string& window_type,
    int length,
    bool periodic = true,
    mlx::core::StreamOrDevice s = {});

}  // namespace mlx_audio
