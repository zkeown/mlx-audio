#pragma once

#include <mlx/array.h>
#include <mlx/stream.h>
#include <mlx/utils.h>

namespace mlx_audio {

/**
 * Parallel PYIN threshold candidate detection.
 *
 * Evaluates all thresholds in parallel on GPU to find pitch candidates
 * for each frame. This is the computationally intensive part of PYIN.
 *
 * @param cmndf Cumulative mean normalized difference function.
 *              Shape: (n_frames, n_lags)
 * @param thresholds Array of threshold values to test.
 *                   Shape: (n_thresholds,)
 * @param min_period Minimum period (lag offset) for frequency calculation.
 * @param sr Sample rate for converting periods to frequencies.
 * @param s Stream for computation.
 *
 * @return Tuple of (candidates, weights, n_candidates) where:
 *         - candidates: Shape (n_frames, n_thresholds) - frequency candidates
 *         - weights: Shape (n_frames, n_thresholds) - candidate weights (1 - cmndf)
 *         - n_candidates: Shape (n_frames,) - number of valid candidates per frame
 */
std::tuple<mlx::core::array, mlx::core::array, mlx::core::array>
pyin_candidates(
    const mlx::core::array& cmndf,
    const mlx::core::array& thresholds,
    int min_period,
    int sr,
    mlx::core::StreamOrDevice s = {});

/**
 * Compute weighted median from candidates.
 *
 * For each frame, computes the weighted median of the pitch candidates.
 *
 * @param candidates Frequency candidates. Shape: (n_frames, n_thresholds)
 * @param weights Candidate weights. Shape: (n_frames, n_thresholds)
 * @param n_candidates Number of valid candidates per frame. Shape: (n_frames,)
 * @param s Stream for computation.
 *
 * @return Tuple of (f0, voiced_prob) where:
 *         - f0: Fundamental frequency per frame. Shape: (n_frames,)
 *         - voiced_prob: Voicing probability. Shape: (n_frames,)
 */
std::pair<mlx::core::array, mlx::core::array>
pyin_weighted_median(
    const mlx::core::array& candidates,
    const mlx::core::array& weights,
    const mlx::core::array& n_candidates,
    mlx::core::StreamOrDevice s = {});

}  // namespace mlx_audio
