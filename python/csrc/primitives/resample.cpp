#include "primitives/resample.h"

#include <mlx/ops.h>
#include <mlx/fft.h>
#include <cmath>

namespace mlx_audio {

mlx::core::array resample_fft(
    const mlx::core::array& signal,
    int num_samples,
    mlx::core::StreamOrDevice s) {

    using namespace mlx::core;

    // Validate inputs
    if (signal.ndim() < 1 || signal.ndim() > 2) {
        throw std::invalid_argument(
            "signal must be 1-dimensional (samples,) or 2-dimensional (batch, samples)");
    }
    if (num_samples <= 0) {
        throw std::invalid_argument("num_samples must be positive");
    }

    // Handle 1D input by adding batch dimension
    bool input_is_1d = signal.ndim() == 1;
    array y = input_is_1d ? reshape(signal, {1, -1}, s) : signal;

    int batch_size = y.shape(0);
    int orig_length = y.shape(1);

    // If same length, return input
    if (num_samples == orig_length) {
        return signal;
    }

    // Ensure float32
    y = astype(y, float32, s);

    // FFT-based resampling (Fourier method)
    // 1. Compute FFT
    // 2. Zero-pad or truncate in frequency domain
    // 3. Compute inverse FFT

    // Compute full FFT
    auto Y = fft::fft(y, orig_length, -1, s);

    // Compute scaling factor
    float scale_factor = static_cast<float>(num_samples) / static_cast<float>(orig_length);

    array Y_resampled = Y;  // Initialize to avoid default constructor

    if (num_samples > orig_length) {
        // Upsampling: zero-pad the frequency spectrum
        // Insert zeros in the middle (between positive and negative frequencies)
        int n_pos = (orig_length + 1) / 2;  // Number of positive freq bins including DC
        int n_pad = num_samples - orig_length;

        // Split into positive and negative frequencies
        auto Y_pos = slice(Y, {0, 0}, {batch_size, n_pos}, s);
        auto Y_neg = slice(Y, {0, n_pos}, {batch_size, orig_length}, s);

        // Create zero padding
        auto zeros_pad = zeros({batch_size, n_pad}, complex64, s);

        // Concatenate: [positive freqs, zeros, negative freqs]
        Y_resampled = concatenate({Y_pos, zeros_pad, Y_neg}, -1, s);

    } else {
        // Downsampling: truncate the frequency spectrum
        // Keep only the low frequencies
        int n_keep_pos = (num_samples + 1) / 2;  // Positive frequencies to keep
        int n_keep_neg = num_samples - n_keep_pos;  // Negative frequencies to keep

        // Get positive frequencies (first n_keep_pos)
        auto Y_pos = slice(Y, {0, 0}, {batch_size, n_keep_pos}, s);

        // Get negative frequencies (last n_keep_neg)
        auto Y_neg = slice(Y, {0, orig_length - n_keep_neg}, {batch_size, orig_length}, s);

        // Concatenate
        Y_resampled = concatenate({Y_pos, Y_neg}, -1, s);
    }

    // Inverse FFT
    auto y_resampled = fft::ifft(Y_resampled, num_samples, -1, s);

    // Take real part and scale
    y_resampled = real(y_resampled, s);
    y_resampled = multiply(y_resampled, array(scale_factor, float32), s);

    // Remove batch dimension if input was 1D
    if (input_is_1d) {
        y_resampled = squeeze(y_resampled, 0, s);
    }

    return y_resampled;
}

mlx::core::array resample(
    const mlx::core::array& signal,
    int orig_sr,
    int target_sr,
    bool fix,
    bool scale,
    mlx::core::StreamOrDevice s) {

    using namespace mlx::core;

    // Validate inputs
    if (orig_sr <= 0 || target_sr <= 0) {
        throw std::invalid_argument("Sample rates must be positive");
    }

    // Same sample rate - return input
    if (orig_sr == target_sr) {
        return signal;
    }

    // Handle 1D input
    bool input_is_1d = signal.ndim() == 1;
    array y = input_is_1d ? reshape(signal, {1, -1}, s) : signal;

    int orig_length = y.shape(1);

    // Compute target length
    double ratio = static_cast<double>(target_sr) / static_cast<double>(orig_sr);
    int target_length;
    if (fix) {
        target_length = static_cast<int>(std::round(orig_length * ratio));
    } else {
        target_length = static_cast<int>(std::ceil(orig_length * ratio));
    }

    // Perform FFT-based resampling
    auto result = resample_fft(y, target_length, s);

    // Scale if requested
    if (scale) {
        result = multiply(result, array(static_cast<float>(ratio), float32), s);
    }

    // Remove batch dimension if input was 1D
    if (input_is_1d) {
        result = squeeze(result, 0, s);
    }

    return result;
}

}  // namespace mlx_audio
