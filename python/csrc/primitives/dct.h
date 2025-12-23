#pragma once

#include <string>
#include <mlx/array.h>
#include <mlx/stream.h>
#include <mlx/utils.h>

namespace mlx_audio {

/**
 * Compute the Type-II Discrete Cosine Transform.
 *
 * The DCT-II is defined as:
 *     C[k] = sum_{n=0}^{N-1} x[n] * cos(pi * k * (2n + 1) / (2N))
 *
 * Parameters
 * ----------
 * x : mlx::core::array
 *     Input array.
 * n : int
 *     Number of output coefficients. If <= 0, uses input size along axis.
 * axis : int
 *     Axis along which to compute DCT. Default: -1 (last axis).
 * norm : std::string
 *     Normalization mode: "ortho" for orthonormal, "" for standard.
 * s : mlx::core::StreamOrDevice
 *     Stream for computation.
 *
 * Returns
 * -------
 * mlx::core::array
 *     DCT coefficients.
 */
mlx::core::array dct(
    const mlx::core::array& x,
    int n = -1,
    int axis = -1,
    const std::string& norm = "ortho",
    mlx::core::StreamOrDevice s = {});

/**
 * Get or create a cached DCT-II basis matrix.
 *
 * Parameters
 * ----------
 * n_out : int
 *     Number of output coefficients.
 * n_in : int
 *     Number of input samples.
 * norm : std::string
 *     Normalization mode.
 * s : mlx::core::StreamOrDevice
 *     Stream for computation.
 *
 * Returns
 * -------
 * mlx::core::array
 *     DCT basis matrix of shape (n_out, n_in).
 */
mlx::core::array get_dct_matrix(
    int n_out,
    int n_in,
    const std::string& norm = "ortho",
    mlx::core::StreamOrDevice s = {});

}  // namespace mlx_audio
