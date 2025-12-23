#include "primitives/autocorrelation.h"

#include <mlx/ops.h>
#include <mlx/fft.h>
#include <cmath>
#include <vector>

namespace mlx_audio {

mlx::core::array autocorrelation(
    const mlx::core::array& signal,
    int max_lag,
    bool normalize,
    bool center,
    mlx::core::StreamOrDevice s) {

    using namespace mlx::core;

    // Validate inputs
    if (signal.ndim() < 1 || signal.ndim() > 2) {
        throw std::invalid_argument(
            "signal must be 1-dimensional (samples,) or 2-dimensional (batch, samples)");
    }

    // Handle 1D input by adding batch dimension
    bool input_is_1d = signal.ndim() == 1;
    array y = input_is_1d ? reshape(signal, {1, -1}, s) : signal;

    int batch_size = y.shape(0);
    int n = y.shape(1);

    // Set max_lag to signal length if not specified
    if (max_lag <= 0) {
        max_lag = n;
    }
    max_lag = std::min(max_lag, n);

    // Ensure float32
    y = astype(y, float32, s);

    // Center the signal (subtract mean)
    if (center) {
        auto mean_val = mean(y, std::vector<int>{-1}, true, s);
        y = subtract(y, mean_val, s);
    }

    // Use FFT for efficient autocorrelation (Wiener-Khinchin theorem)
    // Zero-pad to avoid circular correlation
    int n_fft = 1;
    while (n_fft < 2 * n - 1) {
        n_fft *= 2;
    }

    // FFT along last axis
    auto Y = fft::rfft(y, n_fft, -1, s);

    // Power spectrum: |FFT(y)|^2 = Y * conj(Y)
    auto power = multiply(Y, conjugate(Y, s), s);

    // Inverse FFT to get autocorrelation
    auto r = fft::irfft(power, n_fft, -1, s);

    // Take only positive lags up to max_lag
    // r shape: (batch, n_fft)
    // We want r[:, :max_lag]
    r = slice(r, {0, 0}, {batch_size, max_lag}, s);

    // Normalize if requested
    if (normalize) {
        // Normalize by r[0] (variance)
        // r[:, 0:1] for broadcasting
        auto r0 = slice(r, {0, 0}, {batch_size, 1}, s);
        // Avoid division by zero
        r0 = maximum(r0, array(1e-10f, float32), s);
        r = divide(r, r0, s);
    }

    // Remove batch dimension if input was 1D
    if (input_is_1d) {
        r = squeeze(r, 0, s);
    }

    return r;
}

}  // namespace mlx_audio
