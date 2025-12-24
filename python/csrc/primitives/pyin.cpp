/**
 * PYIN parallel threshold candidate detection implementation.
 *
 * Parallelizes the computationally intensive threshold evaluation
 * across frames and thresholds using Metal GPU.
 */

#include "primitives/pyin.h"
#include "primitives/metal_utils.h"

#include <algorithm>
#include <cmath>
#include <vector>
#include <mlx/ops.h>
#include <mlx/transforms.h>

namespace mlx_audio {

#ifdef MLX_BUILD_METAL
std::tuple<mlx::core::array, mlx::core::array, mlx::core::array>
pyin_candidates_metal(
    const mlx::core::array& cmndf,
    const mlx::core::array& thresholds,
    int min_period,
    int sr,
    mlx::core::StreamOrDevice s) {

    using namespace mlx::core;

    int n_frames = cmndf.shape(0);
    int n_lags = cmndf.shape(1);
    int n_thresholds = thresholds.shape(0);

    // Ensure inputs are contiguous float32
    auto cmndf_f = astype(cmndf, float32, s);
    auto thresh_f = astype(thresholds, float32, s);

    // Flatten and reshape to ensure contiguous
    cmndf_f = flatten(cmndf_f, s);
    cmndf_f = reshape(cmndf_f, {n_frames, n_lags}, s);
    thresh_f = flatten(thresh_f, s);
    thresh_f = reshape(thresh_f, {n_thresholds}, s);

    // Get Metal device and load library
    auto& d = get_metal_device(s);
    auto stream = to_stream(s);
    auto lib = d.get_library(METAL_LIB_NAME);

    // Allocate output arrays
    auto candidates = zeros({n_frames, n_thresholds}, float32, s);
    auto weights = zeros({n_frames, n_thresholds}, float32, s);
    auto n_candidates = zeros({n_frames}, int32, s);

    // Evaluate inputs
    eval({cmndf_f, thresh_f, candidates, weights, n_candidates});

    // Launch candidate detection kernel
    auto kernel = d.get_kernel("pyin_candidates_float", lib);
    auto& enc = d.get_command_encoder(stream.index);
    enc.set_compute_pipeline_state(kernel);

    enc.set_input_array(cmndf_f, 0);
    enc.set_input_array(thresh_f, 1);
    enc.set_output_array(candidates, 2);
    enc.set_output_array(weights, 3);
    enc.set_output_array(n_candidates, 4);  // Atomic output
    enc.set_bytes(n_frames, 5);
    enc.set_bytes(n_lags, 6);
    enc.set_bytes(n_thresholds, 7);
    enc.set_bytes(min_period, 8);
    enc.set_bytes(sr, 9);

    // Dispatch: 2D grid (n_frames, n_thresholds)
    auto [tg0, tg1] = get_threadgroup_size_2d(n_frames, n_thresholds);
    MTL::Size grid_dims = MTL::Size(n_frames, n_thresholds, 1);
    MTL::Size group_dims = MTL::Size(tg0, tg1, 1);
    enc.dispatch_threads(grid_dims, group_dims);

    return {candidates, weights, n_candidates};
}

std::pair<mlx::core::array, mlx::core::array>
pyin_weighted_median_metal(
    const mlx::core::array& candidates,
    const mlx::core::array& weights,
    const mlx::core::array& n_candidates,
    mlx::core::StreamOrDevice s) {

    using namespace mlx::core;

    int n_frames = candidates.shape(0);
    int n_thresholds = candidates.shape(1);

    // Ensure inputs are contiguous
    auto cands = astype(candidates, float32, s);
    auto wgts = astype(weights, float32, s);
    auto n_cands = astype(n_candidates, int32, s);

    cands = flatten(cands, s);
    cands = reshape(cands, {n_frames, n_thresholds}, s);
    wgts = flatten(wgts, s);
    wgts = reshape(wgts, {n_frames, n_thresholds}, s);
    n_cands = flatten(n_cands, s);
    n_cands = reshape(n_cands, {n_frames}, s);

    // Get Metal device and load library
    auto& d = get_metal_device(s);
    auto stream = to_stream(s);
    auto lib = d.get_library(METAL_LIB_NAME);

    // Allocate output arrays
    auto f0 = zeros({n_frames}, float32, s);
    auto voiced_prob = zeros({n_frames}, float32, s);

    // Evaluate inputs
    eval({cands, wgts, n_cands, f0, voiced_prob});

    // Launch weighted median kernel
    auto kernel = d.get_kernel("pyin_weighted_median_float", lib);
    auto& enc = d.get_command_encoder(stream.index);
    enc.set_compute_pipeline_state(kernel);

    enc.set_input_array(cands, 0);
    enc.set_input_array(wgts, 1);
    enc.set_input_array(n_cands, 2);
    enc.set_output_array(f0, 3);
    enc.set_output_array(voiced_prob, 4);
    enc.set_bytes(n_frames, 5);
    enc.set_bytes(n_thresholds, 6);

    // Dispatch: 1D grid (n_frames,)
    int tg = get_threadgroup_size_1d(n_frames);
    MTL::Size grid_dims = MTL::Size(n_frames, 1, 1);
    MTL::Size group_dims = MTL::Size(tg, 1, 1);
    enc.dispatch_threads(grid_dims, group_dims);

    return {f0, voiced_prob};
}
#endif  // MLX_BUILD_METAL

std::tuple<mlx::core::array, mlx::core::array, mlx::core::array>
pyin_candidates_cpu(
    const mlx::core::array& cmndf,
    const mlx::core::array& thresholds,
    int min_period,
    int sr,
    mlx::core::StreamOrDevice s) {

    using namespace mlx::core;

    int n_frames = cmndf.shape(0);
    int n_lags = cmndf.shape(1);
    int n_thresholds = thresholds.shape(0);

    // Evaluate to get data
    auto cmndf_f = astype(cmndf, float32, s);
    auto thresh_f = astype(thresholds, float32, s);
    eval({cmndf_f, thresh_f});

    // Allocate output
    std::vector<float> candidates_data(n_frames * n_thresholds, 0.0f);
    std::vector<float> weights_data(n_frames * n_thresholds, 0.0f);
    std::vector<int> n_candidates_data(n_frames, 0);

    auto cmndf_ptr = cmndf_f.data<float>();
    auto thresh_ptr = thresh_f.data<float>();

    // Process each frame and threshold
    for (int frame = 0; frame < n_frames; frame++) {
        for (int t = 0; t < n_thresholds; t++) {
            float thresh = thresh_ptr[t];
            int cmndf_offset = frame * n_lags;
            int out_idx = frame * n_thresholds + t;

            // Scan for first local minimum below threshold
            for (int tau = 0; tau < n_lags - 1; tau++) {
                float val = cmndf_ptr[cmndf_offset + tau];
                float next_val = cmndf_ptr[cmndf_offset + tau + 1];

                if (val < thresh && val < next_val) {
                    // Found valid candidate
                    int period = min_period + tau;
                    float freq = static_cast<float>(sr) / static_cast<float>(period);
                    float weight = 1.0f - val;

                    candidates_data[out_idx] = freq;
                    weights_data[out_idx] = weight;
                    n_candidates_data[frame]++;
                    break;
                }
            }
        }
    }

    auto candidates = array(candidates_data.data(), {n_frames, n_thresholds}, float32);
    auto weights = array(weights_data.data(), {n_frames, n_thresholds}, float32);
    auto n_candidates = array(n_candidates_data.data(), {n_frames}, int32);

    return {candidates, weights, n_candidates};
}

std::pair<mlx::core::array, mlx::core::array>
pyin_weighted_median_cpu(
    const mlx::core::array& candidates,
    const mlx::core::array& weights,
    const mlx::core::array& n_candidates,
    mlx::core::StreamOrDevice s) {

    using namespace mlx::core;

    int n_frames = candidates.shape(0);
    int n_thresholds = candidates.shape(1);

    // Evaluate inputs
    auto cands = astype(candidates, float32, s);
    auto wgts = astype(weights, float32, s);
    auto n_cands = astype(n_candidates, int32, s);
    eval({cands, wgts, n_cands});

    auto cands_ptr = cands.data<float>();
    auto wgts_ptr = wgts.data<float>();
    auto n_cands_ptr = n_cands.data<int>();

    // Allocate output
    std::vector<float> f0_data(n_frames, 0.0f);
    std::vector<float> prob_data(n_frames, 0.0f);

    for (int frame = 0; frame < n_frames; frame++) {
        int nc = n_cands_ptr[frame];
        if (nc == 0) {
            continue;
        }

        // Voiced probability = fraction of thresholds that found a candidate
        prob_data[frame] = static_cast<float>(nc) / static_cast<float>(n_thresholds);

        int offset = frame * n_thresholds;

        // Collect valid candidates
        std::vector<std::pair<float, float>> valid_pairs;
        for (int i = 0; i < n_thresholds; i++) {
            float c = cands_ptr[offset + i];
            if (c > 0.0f) {
                valid_pairs.emplace_back(c, wgts_ptr[offset + i]);
            }
        }

        if (valid_pairs.empty()) {
            continue;
        }

        // Sort by frequency
        std::sort(valid_pairs.begin(), valid_pairs.end());

        // Normalize weights
        float weight_sum = 0.0f;
        for (const auto& p : valid_pairs) {
            weight_sum += p.second;
        }
        if (weight_sum < 1e-10f) {
            weight_sum = 1.0f;
        }

        // Compute weighted median
        float cum_weight = 0.0f;
        for (const auto& p : valid_pairs) {
            cum_weight += p.second / weight_sum;
            if (cum_weight >= 0.5f) {
                f0_data[frame] = p.first;
                break;
            }
        }
    }

    auto f0 = array(f0_data.data(), {n_frames}, float32);
    auto voiced_prob = array(prob_data.data(), {n_frames}, float32);

    return {f0, voiced_prob};
}

std::tuple<mlx::core::array, mlx::core::array, mlx::core::array>
pyin_candidates(
    const mlx::core::array& cmndf,
    const mlx::core::array& thresholds,
    int min_period,
    int sr,
    mlx::core::StreamOrDevice s) {

    using namespace mlx::core;

    // Validate inputs
    if (cmndf.ndim() != 2) {
        throw std::invalid_argument("cmndf must be 2D (n_frames, n_lags)");
    }
    if (thresholds.ndim() != 1) {
        throw std::invalid_argument("thresholds must be 1D (n_thresholds,)");
    }
    if (min_period < 1) {
        throw std::invalid_argument("min_period must be >= 1");
    }
    if (sr < 1) {
        throw std::invalid_argument("sr must be >= 1");
    }

#ifdef MLX_BUILD_METAL
    if (should_use_metal(s)) {
        return pyin_candidates_metal(cmndf, thresholds, min_period, sr, s);
    }
#endif

    return pyin_candidates_cpu(cmndf, thresholds, min_period, sr, s);
}

std::pair<mlx::core::array, mlx::core::array>
pyin_weighted_median(
    const mlx::core::array& candidates,
    const mlx::core::array& weights,
    const mlx::core::array& n_candidates,
    mlx::core::StreamOrDevice s) {

    using namespace mlx::core;

    // Validate inputs
    if (candidates.ndim() != 2) {
        throw std::invalid_argument("candidates must be 2D (n_frames, n_thresholds)");
    }
    if (weights.ndim() != 2) {
        throw std::invalid_argument("weights must be 2D (n_frames, n_thresholds)");
    }
    if (n_candidates.ndim() != 1) {
        throw std::invalid_argument("n_candidates must be 1D (n_frames,)");
    }
    if (candidates.shape(0) != weights.shape(0) ||
        candidates.shape(0) != n_candidates.shape(0)) {
        throw std::invalid_argument("Input shapes must match in first dimension");
    }
    if (candidates.shape(1) != weights.shape(1)) {
        throw std::invalid_argument("candidates and weights must have same shape");
    }

#ifdef MLX_BUILD_METAL
    if (should_use_metal(s)) {
        return pyin_weighted_median_metal(candidates, weights, n_candidates, s);
    }
#endif

    return pyin_weighted_median_cpu(candidates, weights, n_candidates, s);
}

}  // namespace mlx_audio
