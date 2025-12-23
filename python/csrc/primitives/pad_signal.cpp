#include "primitives/pad_signal.h"
#include "primitives/metal_utils.h"

#include <string>
#include <utility>
#include <vector>
#include <mlx/ops.h>
#include <mlx/transforms.h>

namespace mlx_audio {

#ifdef MLX_BUILD_METAL
mlx::core::array pad_signal_metal(
    const mlx::core::array& signal,
    int pad_length,
    const std::string& mode,
    mlx::core::StreamOrDevice s) {

    using namespace mlx::core;

    int batch_size = signal.shape(0);
    int signal_length = signal.shape(1);
    int output_length = signal_length + 2 * pad_length;

    // Ensure contiguous float32 input
    auto signal_f = astype(signal, float32, s);
    signal_f = flatten(signal_f, s);
    signal_f = reshape(signal_f, {batch_size, signal_length}, s);

    // Get Metal device and load library
    auto& d = get_metal_device(s);
    auto stream = to_stream(s);
    auto lib = d.get_library(METAL_LIB_NAME);

    // Select kernel based on mode
    std::string kernel_name;
    if (mode == "constant") {
        kernel_name = "pad_constant_float";
    } else if (mode == "edge") {
        kernel_name = "pad_edge_float";
    } else if (mode == "reflect") {
        kernel_name = "pad_reflect_float";
    } else {
        throw std::invalid_argument(
            "Unknown pad mode: '" + mode + "'. Supported: constant, reflect, edge");
    }

    auto kernel = d.get_kernel(kernel_name, lib);

    // Allocate output
    auto output = zeros({batch_size, output_length}, float32, s);

    // Evaluate inputs
    eval({signal_f, output});

    // Get command encoder
    auto& enc = d.get_command_encoder(stream.index);
    enc.set_compute_pipeline_state(kernel);

    // Set buffers - must match kernel signature
    enc.set_input_array(signal_f, 0);      // signal
    enc.set_output_array(output, 1);       // output
    enc.set_bytes(batch_size, 2);          // batch_size
    enc.set_bytes(signal_length, 3);       // signal_length
    enc.set_bytes(pad_length, 4);          // pad_length

    // Dispatch threads: 2D grid (output_length, batch_size)
    auto [tg0, tg1] = get_threadgroup_size_2d(output_length, batch_size);
    MTL::Size grid_dims = MTL::Size(output_length, batch_size, 1);
    MTL::Size group_dims = MTL::Size(tg0, tg1, 1);
    enc.dispatch_threads(grid_dims, group_dims);

    return output;
}
#endif  // MLX_BUILD_METAL

mlx::core::array pad_signal_cpu(
    const mlx::core::array& signal,
    int pad_length,
    const std::string& mode,
    mlx::core::StreamOrDevice s) {

    using namespace mlx::core;

    int batch_size = signal.shape(0);
    int signal_length = signal.shape(1);

    if (mode == "constant") {
        // Zero padding - use MLX pad operation
        std::vector<std::pair<int, int>> pad_width = {{0, 0}, {pad_length, pad_length}};
        return pad(signal, pad_width, array(0.0f), "constant", s);
    }
    else if (mode == "edge") {
        // Edge padding - replicate edge values
        std::vector<std::pair<int, int>> pad_width = {{0, 0}, {pad_length, pad_length}};
        return pad(signal, pad_width, array(0.0f), "edge", s);
    }
    else if (mode == "reflect") {
        // Reflect padding: [d c b a | a b c d e f | f e d c]
        // MLX may not support reflect mode directly, implement manually

        if (pad_length > signal_length - 1) {
            throw std::invalid_argument(
                "pad_length for reflect mode must be <= signal_length - 1");
        }

        // Left reflection: signal[:, pad_length:0:-1]
        // This is signal[:, 1:pad_length+1] reversed
        std::vector<int> left_indices;
        for (int i = pad_length; i > 0; --i) {
            left_indices.push_back(i);
        }
        auto left_idx = array(left_indices.data(), {static_cast<int>(left_indices.size())}, int32);
        auto left_pad = take(signal, left_idx, 1, s);

        // Right reflection: signal[:, -2:-pad_length-2:-1]
        std::vector<int> right_indices;
        for (int i = signal_length - 2; i > signal_length - 2 - pad_length; --i) {
            right_indices.push_back(i);
        }
        auto right_idx = array(right_indices.data(), {static_cast<int>(right_indices.size())}, int32);
        auto right_pad = take(signal, right_idx, 1, s);

        // Concatenate: left_pad | signal | right_pad
        return concatenate({left_pad, signal, right_pad}, 1, s);
    }
    else {
        throw std::invalid_argument(
            "Unknown pad mode: '" + mode + "'. Supported: constant, reflect, edge");
    }
}

mlx::core::array pad_signal(
    const mlx::core::array& signal,
    int pad_length,
    const std::string& mode,
    mlx::core::StreamOrDevice s) {

    using namespace mlx::core;

    // Validate inputs
    if (signal.ndim() != 2) {
        throw std::invalid_argument(
            "signal must be 2-dimensional (batch, samples)");
    }
    if (pad_length < 0) {
        throw std::invalid_argument("pad_length must be non-negative");
    }
    if (pad_length == 0) {
        return signal;
    }

#ifdef MLX_BUILD_METAL
    // Try Metal GPU path
    if (should_use_metal(s)) {
        return pad_signal_metal(signal, pad_length, mode, s);
    }
#endif

    // CPU fallback
    return pad_signal_cpu(signal, pad_length, mode, s);
}

}  // namespace mlx_audio
