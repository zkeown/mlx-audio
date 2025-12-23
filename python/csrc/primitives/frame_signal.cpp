#include "primitives/frame_signal.h"
#include "primitives/metal_utils.h"

#include <mlx/ops.h>
#include <mlx/transforms.h>

namespace mlx_audio {

#ifdef MLX_BUILD_METAL
mlx::core::array frame_signal_metal(
    const mlx::core::array& signal,
    int frame_length,
    int hop_length,
    mlx::core::StreamOrDevice s) {

    using namespace mlx::core;

    // Handle 1D input by adding batch dimension
    bool input_is_1d = signal.ndim() == 1;
    array y = input_is_1d ? reshape(signal, {1, -1}, s) : signal;

    int batch_size = y.shape(0);
    int signal_length = y.shape(1);
    int n_frames = 1 + (signal_length - frame_length) / hop_length;

    // Ensure contiguous float32 input
    auto y_f = astype(y, float32, s);
    y_f = flatten(y_f, s);
    y_f = reshape(y_f, {batch_size, signal_length}, s);

    // Get Metal device and load library
    auto& d = get_metal_device(s);
    auto stream = to_stream(s);

    auto lib = d.get_library(METAL_LIB_NAME);
    auto kernel = d.get_kernel("frame_signal_float", lib);

    // Allocate output
    auto frames = zeros({batch_size, n_frames, frame_length}, float32, s);

    // Evaluate inputs
    eval({y_f, frames});

    // Get command encoder
    auto& enc = d.get_command_encoder(stream.index);
    enc.set_compute_pipeline_state(kernel);

    // Set buffers - must match kernel signature
    enc.set_input_array(y_f, 0);           // signal
    enc.set_output_array(frames, 1);       // frames
    enc.set_bytes(batch_size, 2);          // batch_size
    enc.set_bytes(signal_length, 3);       // signal_length
    enc.set_bytes(n_frames, 4);            // n_frames
    enc.set_bytes(frame_length, 5);        // frame_length
    enc.set_bytes(hop_length, 6);          // hop_length

    // Dispatch threads: 3D grid (frame_length, n_frames, batch_size)
    auto [tg0, tg1, tg2] = get_threadgroup_size_3d(frame_length, n_frames, batch_size);
    MTL::Size grid_dims = MTL::Size(frame_length, n_frames, batch_size);
    MTL::Size group_dims = MTL::Size(tg0, tg1, tg2);
    enc.dispatch_threads(grid_dims, group_dims);

    // Remove batch dimension if input was 1D
    if (input_is_1d) {
        frames = squeeze(frames, 0, s);
    }

    return frames;
}
#endif  // MLX_BUILD_METAL

mlx::core::array frame_signal_cpu(
    const mlx::core::array& signal,
    int frame_length,
    int hop_length,
    mlx::core::StreamOrDevice s) {

    using namespace mlx::core;

    // Handle 1D input by adding batch dimension
    bool input_is_1d = signal.ndim() == 1;
    array y = input_is_1d ? reshape(signal, {1, -1}, s) : signal;

    int batch_size = y.shape(0);
    int signal_length = y.shape(1);

    // Compute number of frames
    int n_frames = 1 + (signal_length - frame_length) / hop_length;

    // Create frame start indices: [0, hop_length, 2*hop_length, ...]
    auto frame_starts = arange(n_frames, s) * hop_length;  // (n_frames,)

    // Create sample offsets within each frame: [0, 1, 2, ..., frame_length-1]
    auto sample_offsets = arange(frame_length, s);  // (frame_length,)

    // Broadcast to get all indices: (n_frames, frame_length)
    auto indices = reshape(frame_starts, {n_frames, 1}, s) +
                   reshape(sample_offsets, {1, frame_length}, s);

    // Flatten indices for take operation
    auto flat_indices = flatten(indices, s);  // (n_frames * frame_length,)

    // Gather frames using take along the last axis for each batch element
    // y shape: (batch, signal_length)
    // Result shape: (batch, n_frames * frame_length)
    auto gathered = take(y, flat_indices, 1, s);

    // Reshape to (batch, n_frames, frame_length)
    auto frames = reshape(gathered, {batch_size, n_frames, frame_length}, s);

    // Remove batch dimension if input was 1D
    if (input_is_1d) {
        frames = squeeze(frames, 0, s);
    }

    return frames;
}

mlx::core::array frame_signal(
    const mlx::core::array& signal,
    int frame_length,
    int hop_length,
    mlx::core::StreamOrDevice s) {

    using namespace mlx::core;

    // Validate inputs
    if (signal.ndim() < 1 || signal.ndim() > 2) {
        throw std::invalid_argument(
            "signal must be 1-dimensional (samples,) or 2-dimensional (batch, samples)");
    }
    if (frame_length <= 0) {
        throw std::invalid_argument("frame_length must be positive");
    }
    if (hop_length <= 0) {
        throw std::invalid_argument("hop_length must be positive");
    }

    // Check signal length
    int signal_length = signal.ndim() == 1 ? signal.shape(0) : signal.shape(1);
    if (signal_length < frame_length) {
        throw std::invalid_argument(
            "signal length must be >= frame_length");
    }

#ifdef MLX_BUILD_METAL
    // Try Metal GPU path
    if (should_use_metal(s)) {
        return frame_signal_metal(signal, frame_length, hop_length, s);
    }
#endif

    // CPU fallback
    return frame_signal_cpu(signal, frame_length, hop_length, s);
}

}  // namespace mlx_audio
