#include "primitives/windows.h"
#include "primitives/metal_utils.h"

#include <cmath>
#include <mlx/ops.h>
#include <mlx/transforms.h>

namespace mlx_audio {

using namespace mlx::core;

namespace {

constexpr double PI = 3.14159265358979323846;

#ifdef MLX_BUILD_METAL
array generate_window_metal(
    const std::string& window_type,
    int n,
    StreamOrDevice s) {

    // Get Metal device and load library
    auto& d = get_metal_device(s);
    auto stream = to_stream(s);
    auto lib = d.get_library(METAL_LIB_NAME);

    // Select kernel based on window type
    std::string kernel_name;
    if (window_type == "hann" || window_type == "hanning") {
        kernel_name = "hann_window_kernel";
    } else if (window_type == "hamming") {
        kernel_name = "hamming_window_kernel";
    } else if (window_type == "blackman") {
        kernel_name = "blackman_window_kernel";
    } else if (window_type == "bartlett" || window_type == "triangular") {
        kernel_name = "bartlett_window_kernel";
    } else if (window_type == "rectangular" || window_type == "boxcar" || window_type == "ones") {
        kernel_name = "rectangular_window_kernel";
    } else {
        // Unknown type - throw exception instead of silently returning invalid array
        throw std::invalid_argument(
            "Unknown window type: '" + window_type + "'. "
            "Supported: hann, hamming, blackman, bartlett, rectangular");
    }

    auto kernel = d.get_kernel(kernel_name, lib);

    // Allocate output
    auto window = zeros({n}, float32, s);

    // Evaluate to allocate
    eval({window});

    // Get command encoder
    auto& enc = d.get_command_encoder(stream.index);
    enc.set_compute_pipeline_state(kernel);

    // Set buffers
    enc.set_output_array(window, 0);       // window
    enc.set_bytes(n, 1);                   // length

    // Dispatch threads: 1D grid
    int tg = get_threadgroup_size_1d(n);
    MTL::Size grid_dims = MTL::Size(n, 1, 1);
    MTL::Size group_dims = MTL::Size(tg, 1, 1);
    enc.dispatch_threads(grid_dims, group_dims);

    return window;
}
#endif  // MLX_BUILD_METAL

// Helper to enforce perfect symmetry by averaging with reversed array
array enforce_symmetry(const array& window, int n, StreamOrDevice s) {
    auto reversed = slice(window, {n-1}, {-n-1}, {-1}, s);
    auto symmetric = divide(add(window, reversed, s), array(2.0f), s);
    eval({symmetric});
    return symmetric;
}

array hann_window(int n, StreamOrDevice s) {
    if (n <= 1) {
        return ones({n}, float32, s);
    }

    // w[k] = 0.5 - 0.5 * cos(2 * pi * k / (n - 1))
    // Compute in float64 for precision, use explicit ops with stream parameter.
    auto k = arange(n, float64, s);
    auto denom = array(static_cast<double>(n - 1), float64);
    auto two_pi = array(2.0 * PI, float64);
    auto arg = divide(multiply(two_pi, k, s), denom, s);
    auto cosval = cos(arg, s);
    auto half = array(0.5, float64);
    auto window = subtract(half, multiply(half, cosval, s), s);
    auto result = astype(window, float32, s);
    eval({result});

    return enforce_symmetry(result, n, s);
}

array hamming_window(int n, StreamOrDevice s) {
    if (n <= 1) {
        return ones({n}, float32, s);
    }

    // w[k] = 0.54 - 0.46 * cos(2 * pi * k / (n - 1))
    // Compute in float64 for precision, use explicit ops with stream parameter.
    auto k = arange(n, float64, s);
    auto denom = array(static_cast<double>(n - 1), float64);
    auto two_pi = array(2.0 * PI, float64);
    auto arg = divide(multiply(two_pi, k, s), denom, s);
    auto cosval = cos(arg, s);
    auto a0 = array(0.54, float64);
    auto a1 = array(0.46, float64);
    auto window = subtract(a0, multiply(a1, cosval, s), s);
    auto result = astype(window, float32, s);
    eval({result});

    return enforce_symmetry(result, n, s);
}

array blackman_window(int n, StreamOrDevice s) {
    if (n <= 1) {
        return ones({n}, float32, s);
    }

    // w[k] = 0.42 - 0.5 * cos(2*pi*k/(n-1)) + 0.08 * cos(4*pi*k/(n-1))
    // Compute in float64 for precision, use explicit ops with stream parameter.
    auto k = arange(n, float64, s);
    auto denom = array(static_cast<double>(n - 1), float64);
    auto two_pi = array(2.0 * PI, float64);
    auto four_pi = array(4.0 * PI, float64);
    auto arg1 = divide(multiply(two_pi, k, s), denom, s);
    auto arg2 = divide(multiply(four_pi, k, s), denom, s);
    auto cos1 = cos(arg1, s);
    auto cos2 = cos(arg2, s);

    auto a0 = array(0.42, float64);
    auto a1 = array(0.5, float64);
    auto a2 = array(0.08, float64);
    auto window = add(subtract(a0, multiply(a1, cos1, s), s), multiply(a2, cos2, s), s);

    // Clamp to non-negative
    auto zero = array(0.0, float64);
    window = maximum(window, zero, s);

    auto result = astype(window, float32, s);
    eval({result});

    return enforce_symmetry(result, n, s);
}

array bartlett_window(int n, StreamOrDevice s) {
    if (n <= 1) {
        return ones({n}, float32, s);
    }

    // w[k] = 1 - |2 * k / (n - 1) - 1|
    // Compute in float64 for precision, use explicit ops with stream parameter.
    auto k = arange(n, float64, s);
    auto denom = array(static_cast<double>(n - 1), float64);
    auto two = array(2.0, float64);
    auto one = array(1.0, float64);
    auto scaled = divide(multiply(two, k, s), denom, s);
    auto shifted = subtract(scaled, one, s);
    auto window = subtract(one, abs(shifted, s), s);

    auto result = astype(window, float32, s);
    eval({result});

    return enforce_symmetry(result, n, s);
}

array rectangular_window(int n, StreamOrDevice s) {
    return ones({n}, float32, s);
}

}  // namespace

array generate_window(
    const std::string& window_type,
    int length,
    bool periodic,
    StreamOrDevice s) {

    if (length <= 0) {
        throw std::invalid_argument("Window length must be positive");
    }

    // For periodic (fftbins=True), compute n+1 points and drop the last
    int n = periodic ? length + 1 : length;

    // Window functions are computed on CPU with float64 precision for perfect symmetry.
    // This matches scipy/librosa behavior. Windows are typically small (< 8192) and only
    // computed once per session (cached), so CPU execution is acceptable.
    // Metal float32 kernels exist but produce asymmetric windows due to precision limits.
    auto cpu_stream = default_stream(Device::cpu);

    // CPU implementation with float64 precision
    array window = ones({n}, float32, cpu_stream);  // Initialize with valid array

    if (window_type == "hann" || window_type == "hanning") {
        window = hann_window(n, cpu_stream);
    }
    else if (window_type == "hamming") {
        window = hamming_window(n, cpu_stream);
    }
    else if (window_type == "blackman") {
        window = blackman_window(n, cpu_stream);
    }
    else if (window_type == "bartlett" || window_type == "triangular") {
        window = bartlett_window(n, cpu_stream);
    }
    else if (window_type == "rectangular" || window_type == "boxcar" || window_type == "ones") {
        window = rectangular_window(n, cpu_stream);
    }
    else {
        throw std::invalid_argument(
            "Unknown window type: '" + window_type + "'. "
            "Supported: hann, hamming, blackman, bartlett, rectangular");
    }

    // For periodic windows, drop the last sample
    if (periodic && n > length) {
        window = slice(window, {0}, {length}, cpu_stream);
    }

    return window;
}

}  // namespace mlx_audio
