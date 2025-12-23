#include <metal_stdlib>
using namespace metal;

/**
 * Frame signal kernel - extracts overlapping frames from a signal.
 *
 * Each thread handles one sample in one frame.
 */
template <typename T>
[[kernel]] void frame_signal_kernel(
    device const T* signal [[buffer(0)]],        // (batch, signal_length)
    device T* frames [[buffer(1)]],              // (batch, n_frames, frame_length)
    constant const int& batch_size [[buffer(2)]],
    constant const int& signal_length [[buffer(3)]],
    constant const int& n_frames [[buffer(4)]],
    constant const int& frame_length [[buffer(5)]],
    constant const int& hop_length [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]]) {

    int sample_idx = gid.x;      // Sample index within frame
    int frame_idx = gid.y;       // Frame index
    int batch_idx = gid.z;       // Batch index

    if (sample_idx >= frame_length || frame_idx >= n_frames || batch_idx >= batch_size) {
        return;
    }

    // Compute source index in signal
    int src_idx = frame_idx * hop_length + sample_idx;

    // Compute destination index in frames
    int dst_idx = batch_idx * n_frames * frame_length + frame_idx * frame_length + sample_idx;

    // Copy sample
    frames[dst_idx] = signal[batch_idx * signal_length + src_idx];
}

// Explicit instantiation for float32 (the only supported type for audio signals).
// float16 is not instantiated because audio processing requires float32 precision.
template [[host_name("frame_signal_float")]] [[kernel]] void frame_signal_kernel<float>(
    device const float*, device float*,
    constant const int&, constant const int&, constant const int&,
    constant const int&, constant const int&, uint3);
