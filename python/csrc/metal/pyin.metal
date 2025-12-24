// PYIN parallel threshold candidate detection Metal kernels

#include <metal_stdlib>
using namespace metal;

/**
 * Parallel PYIN threshold candidate detection kernel.
 *
 * Evaluates all thresholds in parallel to find pitch candidates for each frame.
 * This is the most computationally intensive part of PYIN.
 *
 * Grid: (n_frames, n_thresholds, 1)
 *
 * For each (frame, threshold) pair:
 * 1. Scan CMNDF to find first local minimum below threshold
 * 2. Store frequency candidate and weight (1 - cmndf_value)
 */
[[kernel]] void pyin_candidates_float(
    device const float* cmndf [[buffer(0)]],       // (n_frames, n_lags)
    device const float* thresholds [[buffer(1)]],  // (n_thresholds,)
    device float* candidates [[buffer(2)]],        // (n_frames, n_thresholds)
    device float* weights [[buffer(3)]],           // (n_frames, n_thresholds)
    device atomic_int* n_candidates [[buffer(4)]], // (n_frames,) - atomic counter
    constant int& n_frames [[buffer(5)]],
    constant int& n_lags [[buffer(6)]],
    constant int& n_thresholds [[buffer(7)]],
    constant int& min_period [[buffer(8)]],
    constant int& sr [[buffer(9)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int frame_idx = gid.x;
    int thresh_idx = gid.y;

    if (frame_idx >= n_frames || thresh_idx >= n_thresholds) {
        return;
    }

    float thresh = thresholds[thresh_idx];
    int cmndf_offset = frame_idx * n_lags;
    int out_idx = frame_idx * n_thresholds + thresh_idx;

    // Initialize output
    candidates[out_idx] = 0.0f;
    weights[out_idx] = 0.0f;

    // Scan for first local minimum below threshold
    // Local minimum: cmndf[tau] < cmndf[tau+1] (rising after)
    for (int tau = 0; tau < n_lags - 1; tau++) {
        float val = cmndf[cmndf_offset + tau];
        float next_val = cmndf[cmndf_offset + tau + 1];

        if (val < thresh && val < next_val) {
            // Found valid candidate
            int period = min_period + tau;
            float freq = float(sr) / float(period);
            float weight = 1.0f - val;

            candidates[out_idx] = freq;
            weights[out_idx] = weight;

            // Increment frame's candidate count atomically
            atomic_fetch_add_explicit(&n_candidates[frame_idx], 1, memory_order_relaxed);
            break;
        }
    }
}

/**
 * Compute weighted median from candidates for each frame.
 *
 * Grid: (n_frames, 1, 1)
 *
 * For each frame:
 * 1. Collect all non-zero candidates
 * 2. Sort by frequency
 * 3. Compute weighted median
 * 4. Output f0 and voiced probability
 */
[[kernel]] void pyin_weighted_median_float(
    device const float* candidates [[buffer(0)]],  // (n_frames, n_thresholds)
    device const float* weights [[buffer(1)]],     // (n_frames, n_thresholds)
    device const int* n_candidates_in [[buffer(2)]], // (n_frames,)
    device float* f0 [[buffer(3)]],                // (n_frames,)
    device float* voiced_prob [[buffer(4)]],       // (n_frames,)
    constant int& n_frames [[buffer(5)]],
    constant int& n_thresholds [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    int frame_idx = gid;

    if (frame_idx >= n_frames) {
        return;
    }

    int nc = n_candidates_in[frame_idx];

    if (nc == 0) {
        f0[frame_idx] = 0.0f;
        voiced_prob[frame_idx] = 0.0f;
        return;
    }

    // Voiced probability = fraction of thresholds that found a candidate
    voiced_prob[frame_idx] = float(nc) / float(n_thresholds);

    int offset = frame_idx * n_thresholds;

    // Collect valid candidates into thread-local arrays
    // Use fixed-size arrays (Metal doesn't support VLAs)
    // Limit to reasonable max to avoid register pressure
    constexpr int MAX_CANDS = 128;
    float local_cands[MAX_CANDS];
    float local_weights[MAX_CANDS];
    int valid_count = 0;

    for (int i = 0; i < n_thresholds && valid_count < MAX_CANDS; i++) {
        float c = candidates[offset + i];
        if (c > 0.0f) {
            local_cands[valid_count] = c;
            local_weights[valid_count] = weights[offset + i];
            valid_count++;
        }
    }

    if (valid_count == 0) {
        f0[frame_idx] = 0.0f;
        return;
    }

    // Simple insertion sort (small arrays, O(n^2) is fine)
    for (int i = 1; i < valid_count; i++) {
        float key_c = local_cands[i];
        float key_w = local_weights[i];
        int j = i - 1;
        while (j >= 0 && local_cands[j] > key_c) {
            local_cands[j + 1] = local_cands[j];
            local_weights[j + 1] = local_weights[j];
            j--;
        }
        local_cands[j + 1] = key_c;
        local_weights[j + 1] = key_w;
    }

    // Normalize weights
    float weight_sum = 0.0f;
    for (int i = 0; i < valid_count; i++) {
        weight_sum += local_weights[i];
    }
    if (weight_sum < 1e-10f) {
        weight_sum = 1.0f;
    }
    for (int i = 0; i < valid_count; i++) {
        local_weights[i] /= weight_sum;
    }

    // Compute weighted median via cumulative weights
    float cum_weight = 0.0f;
    int median_idx = 0;
    for (int i = 0; i < valid_count; i++) {
        cum_weight += local_weights[i];
        if (cum_weight >= 0.5f) {
            median_idx = i;
            break;
        }
    }

    f0[frame_idx] = local_cands[median_idx];
}
