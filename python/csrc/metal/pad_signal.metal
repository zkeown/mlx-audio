#include <metal_stdlib>
using namespace metal;

/**
 * Reflect padding kernel.
 *
 * Pads signal on both sides using reflection.
 * Pattern: [d c b a | a b c d e f | f e d c]
 */
template <typename T>
[[kernel]] void pad_reflect_kernel(
    device const T* signal [[buffer(0)]],        // (batch, signal_length)
    device T* output [[buffer(1)]],              // (batch, signal_length + 2 * pad_length)
    constant const int& batch_size [[buffer(2)]],
    constant const int& signal_length [[buffer(3)]],
    constant const int& pad_length [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]) {

    int out_idx = gid.x;
    int batch_idx = gid.y;

    int output_length = signal_length + 2 * pad_length;

    if (out_idx >= output_length || batch_idx >= batch_size) {
        return;
    }

    int src_idx;

    if (out_idx < pad_length) {
        // Left padding region: reflect from signal[1:pad_length+1]
        src_idx = pad_length - out_idx;
    }
    else if (out_idx >= pad_length + signal_length) {
        // Right padding region: reflect from signal[-pad_length-1:-1]
        int offset = out_idx - (pad_length + signal_length);
        src_idx = signal_length - 2 - offset;
    }
    else {
        // Middle region: direct copy
        src_idx = out_idx - pad_length;
    }

    output[batch_idx * output_length + out_idx] = signal[batch_idx * signal_length + src_idx];
}

/**
 * Edge padding kernel.
 *
 * Pads signal on both sides by replicating edge values.
 */
template <typename T>
[[kernel]] void pad_edge_kernel(
    device const T* signal [[buffer(0)]],        // (batch, signal_length)
    device T* output [[buffer(1)]],              // (batch, signal_length + 2 * pad_length)
    constant const int& batch_size [[buffer(2)]],
    constant const int& signal_length [[buffer(3)]],
    constant const int& pad_length [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]) {

    int out_idx = gid.x;
    int batch_idx = gid.y;

    int output_length = signal_length + 2 * pad_length;

    if (out_idx >= output_length || batch_idx >= batch_size) {
        return;
    }

    int src_idx;

    if (out_idx < pad_length) {
        // Left padding: replicate first element
        src_idx = 0;
    }
    else if (out_idx >= pad_length + signal_length) {
        // Right padding: replicate last element
        src_idx = signal_length - 1;
    }
    else {
        // Middle region: direct copy
        src_idx = out_idx - pad_length;
    }

    output[batch_idx * output_length + out_idx] = signal[batch_idx * signal_length + src_idx];
}

/**
 * Constant (zero) padding kernel.
 */
template <typename T>
[[kernel]] void pad_constant_kernel(
    device const T* signal [[buffer(0)]],        // (batch, signal_length)
    device T* output [[buffer(1)]],              // (batch, signal_length + 2 * pad_length)
    constant const int& batch_size [[buffer(2)]],
    constant const int& signal_length [[buffer(3)]],
    constant const int& pad_length [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]) {

    int out_idx = gid.x;
    int batch_idx = gid.y;

    int output_length = signal_length + 2 * pad_length;

    if (out_idx >= output_length || batch_idx >= batch_size) {
        return;
    }

    T value;

    if (out_idx < pad_length || out_idx >= pad_length + signal_length) {
        // Padding region: zero
        value = T(0);
    }
    else {
        // Middle region: copy from signal
        value = signal[batch_idx * signal_length + (out_idx - pad_length)];
    }

    output[batch_idx * output_length + out_idx] = value;
}

// Explicit instantiations
template [[host_name("pad_reflect_float")]] [[kernel]] void pad_reflect_kernel<float>(
    device const float*, device float*,
    constant const int&, constant const int&, constant const int&, uint2);

template [[host_name("pad_edge_float")]] [[kernel]] void pad_edge_kernel<float>(
    device const float*, device float*,
    constant const int&, constant const int&, constant const int&, uint2);

template [[host_name("pad_constant_float")]] [[kernel]] void pad_constant_kernel<float>(
    device const float*, device float*,
    constant const int&, constant const int&, constant const int&, uint2);
