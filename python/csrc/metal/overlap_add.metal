#include <metal_stdlib>
using namespace metal;

/**
 * Fused overlap-add kernel with inline normalization.
 *
 * Each thread handles one output sample, gathering contributions from
 * all frames that overlap at that position and normalizing in-place.
 * This eliminates a separate normalization pass, saving kernel launch
 * overhead and memory bandwidth.
 *
 * Performance optimization: Fusing gather + normalize into one kernel
 * reduces memory round-trips from 2 to 1 per output element.
 */
template <typename T>
[[kernel]] void overlap_add_fused_kernel(
    device const T* frames [[buffer(0)]],        // (batch, n_frames, n_fft)
    device const T* window [[buffer(1)]],        // (n_fft,)
    device T* output [[buffer(2)]],              // (batch, output_length)
    constant const int& batch_size [[buffer(3)]],
    constant const int& n_frames [[buffer(4)]],
    constant const int& n_fft [[buffer(5)]],
    constant const int& hop_length [[buffer(6)]],
    constant const int& output_length [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]) {

    int out_idx = gid.x;
    int batch_idx = gid.y;

    if (out_idx >= output_length || batch_idx >= batch_size) {
        return;
    }

    // Compute which frames contribute to this output position
    // Frame f contributes to positions [f * hop_length, f * hop_length + n_fft)
    int first_frame = max(0, (out_idx - n_fft + 1 + hop_length - 1) / hop_length);
    int last_frame = min(n_frames - 1, out_idx / hop_length);

    T sum = T(0);
    T win_sq_sum = T(0);

    for (int f = first_frame; f <= last_frame; f++) {
        int sample_in_frame = out_idx - f * hop_length;
        if (sample_in_frame >= 0 && sample_in_frame < n_fft) {
            T win_val = window[sample_in_frame];
            T frame_val = frames[batch_idx * n_frames * n_fft + f * n_fft + sample_in_frame];
            sum += win_val * frame_val;
            win_sq_sum += win_val * win_val;
        }
    }

    // Normalize and store output in one step (fused operation)
    // Epsilon 1e-8 prevents division by zero for edge samples
    output[batch_idx * output_length + out_idx] = sum / max(win_sq_sum, T(1e-8));
}

/**
 * Legacy overlap-add kernel (unfused) - kept for compatibility.
 * Prefer overlap_add_fused_kernel for better performance.
 */
template <typename T>
[[kernel]] void overlap_add_kernel(
    device const T* frames [[buffer(0)]],        // (batch, n_frames, n_fft)
    device const T* window [[buffer(1)]],        // (n_fft,)
    device T* output [[buffer(2)]],              // (batch, output_length)
    device T* window_sum [[buffer(3)]],          // (output_length,)
    constant const int& batch_size [[buffer(4)]],
    constant const int& n_frames [[buffer(5)]],
    constant const int& n_fft [[buffer(6)]],
    constant const int& hop_length [[buffer(7)]],
    constant const int& output_length [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]]) {

    int out_idx = gid.x;
    int batch_idx = gid.y;

    if (out_idx >= output_length || batch_idx >= batch_size) {
        return;
    }

    // Compute which frames contribute to this output position
    int first_frame = max(0, (out_idx - n_fft + 1 + hop_length - 1) / hop_length);
    int last_frame = min(n_frames - 1, out_idx / hop_length);

    T sum = T(0);
    T win_sq_sum = T(0);

    for (int f = first_frame; f <= last_frame; f++) {
        int sample_in_frame = out_idx - f * hop_length;
        if (sample_in_frame >= 0 && sample_in_frame < n_fft) {
            T win_val = window[sample_in_frame];
            T frame_val = frames[batch_idx * n_frames * n_fft + f * n_fft + sample_in_frame];
            sum += win_val * frame_val;
            if (batch_idx == 0) {
                win_sq_sum += win_val * win_val;
            }
        }
    }

    output[batch_idx * output_length + out_idx] = sum;
    if (batch_idx == 0) {
        window_sum[out_idx] = max(win_sq_sum, T(1e-8));
    }
}

/**
 * Normalization kernel - divides output by window_sum.
 * Used with unfused overlap_add_kernel.
 */
template <typename T>
[[kernel]] void normalize_kernel(
    device T* output [[buffer(0)]],              // (batch, output_length)
    device const T* window_sum [[buffer(1)]],    // (output_length,)
    constant const int& batch_size [[buffer(2)]],
    constant const int& output_length [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]) {

    int out_idx = gid.x;
    int batch_idx = gid.y;

    if (out_idx >= output_length || batch_idx >= batch_size) {
        return;
    }

    output[batch_idx * output_length + out_idx] /= window_sum[out_idx];
}

// Explicit instantiations
template [[host_name("overlap_add_fused_float")]] [[kernel]] void overlap_add_fused_kernel<float>(
    device const float*, device const float*, device float*,
    constant const int&, constant const int&, constant const int&,
    constant const int&, constant const int&, uint2);

template [[host_name("overlap_add_float")]] [[kernel]] void overlap_add_kernel<float>(
    device const float*, device const float*, device float*, device float*,
    constant const int&, constant const int&, constant const int&,
    constant const int&, constant const int&, uint2);

template [[host_name("normalize_float")]] [[kernel]] void normalize_kernel<float>(
    device float*, device const float*,
    constant const int&, constant const int&, uint2);
