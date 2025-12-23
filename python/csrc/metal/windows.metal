#include <metal_stdlib>
using namespace metal;

constant float PI = 3.14159265358979323846f;

/**
 * Hann window kernel.
 * w[k] = 0.5 - 0.5 * cos(2 * pi * k / (n - 1))
 */
[[kernel]] void hann_window_kernel(
    device float* window [[buffer(0)]],
    constant const int& length [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {

    if (gid >= uint(length)) {
        return;
    }

    if (length <= 1) {
        window[gid] = 1.0f;
        return;
    }

    float k = float(gid);
    float denom = float(length - 1);
    window[gid] = 0.5f - 0.5f * cos(2.0f * PI * k / denom);
}

/**
 * Hamming window kernel.
 * w[k] = 0.54 - 0.46 * cos(2 * pi * k / (n - 1))
 */
[[kernel]] void hamming_window_kernel(
    device float* window [[buffer(0)]],
    constant const int& length [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {

    if (gid >= uint(length)) {
        return;
    }

    if (length <= 1) {
        window[gid] = 1.0f;
        return;
    }

    float k = float(gid);
    float denom = float(length - 1);
    window[gid] = 0.54f - 0.46f * cos(2.0f * PI * k / denom);
}

/**
 * Blackman window kernel.
 * w[k] = 0.42 - 0.5 * cos(2*pi*k/(n-1)) + 0.08 * cos(4*pi*k/(n-1))
 */
[[kernel]] void blackman_window_kernel(
    device float* window [[buffer(0)]],
    constant const int& length [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {

    if (gid >= uint(length)) {
        return;
    }

    if (length <= 1) {
        window[gid] = 1.0f;
        return;
    }

    float k = float(gid);
    float denom = float(length - 1);
    float val = 0.42f
              - 0.5f * cos(2.0f * PI * k / denom)
              + 0.08f * cos(4.0f * PI * k / denom);
    // Clamp to non-negative
    window[gid] = max(val, 0.0f);
}

/**
 * Bartlett (triangular) window kernel.
 * w[k] = 1 - |2 * k / (n - 1) - 1|
 */
[[kernel]] void bartlett_window_kernel(
    device float* window [[buffer(0)]],
    constant const int& length [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {

    if (gid >= uint(length)) {
        return;
    }

    if (length <= 1) {
        window[gid] = 1.0f;
        return;
    }

    float k = float(gid);
    float denom = float(length - 1);
    window[gid] = 1.0f - abs(2.0f * k / denom - 1.0f);
}

/**
 * Rectangular (boxcar) window kernel.
 * All ones.
 */
[[kernel]] void rectangular_window_kernel(
    device float* window [[buffer(0)]],
    constant const int& length [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {

    if (gid >= uint(length)) {
        return;
    }

    window[gid] = 1.0f;
}
