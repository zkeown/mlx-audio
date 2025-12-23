#include "primitives/spectral.h"

#include <mlx/ops.h>
#include <cmath>

namespace mlx_audio {

mlx::core::array spectral_centroid(
    const mlx::core::array& S,
    const mlx::core::array& frequencies,
    mlx::core::StreamOrDevice s) {

    using namespace mlx::core;

    // Validate inputs
    if (S.ndim() < 2 || S.ndim() > 3) {
        throw std::invalid_argument(
            "S must be 2D (freq_bins, n_frames) or 3D (batch, freq_bins, n_frames)");
    }
    if (frequencies.ndim() != 1) {
        throw std::invalid_argument("frequencies must be 1D (freq_bins,)");
    }

    // Ensure float32
    auto mag = astype(S, float32, s);
    auto freqs = astype(frequencies, float32, s);

    // Handle 2D vs 3D input
    bool is_2d = mag.ndim() == 2;
    if (is_2d) {
        // Add batch dimension: (freq_bins, n_frames) -> (1, freq_bins, n_frames)
        mag = reshape(mag, {1, mag.shape(0), mag.shape(1)}, s);
    }

    // mag shape: (batch, freq_bins, n_frames)
    // freqs shape: (freq_bins,)

    // Reshape freqs for broadcasting: (freq_bins,) -> (1, freq_bins, 1)
    auto freqs_broadcast = reshape(freqs, {1, static_cast<int>(freqs.shape(0)), 1}, s);

    // centroid = sum(f * S, axis=freq) / sum(S, axis=freq)
    // Sum over frequency axis (axis=1)
    auto weighted_sum = sum(multiply(freqs_broadcast, mag, s), {1}, true, s);
    auto total_sum = sum(mag, {1}, true, s);

    // Avoid division by zero
    total_sum = maximum(total_sum, array(1e-10f, float32), s);

    auto centroid = divide(weighted_sum, total_sum, s);

    // Remove batch dimension if input was 2D
    if (is_2d) {
        centroid = squeeze(centroid, 0, s);
    }

    return centroid;
}

mlx::core::array spectral_bandwidth(
    const mlx::core::array& S,
    const mlx::core::array& frequencies,
    const mlx::core::array& centroid,
    float p,
    mlx::core::StreamOrDevice s) {

    using namespace mlx::core;

    // Validate inputs
    if (S.ndim() < 2 || S.ndim() > 3) {
        throw std::invalid_argument(
            "S must be 2D (freq_bins, n_frames) or 3D (batch, freq_bins, n_frames)");
    }
    if (frequencies.ndim() != 1) {
        throw std::invalid_argument("frequencies must be 1D (freq_bins,)");
    }

    // Ensure float32
    auto mag = astype(S, float32, s);
    auto freqs = astype(frequencies, float32, s);

    // Handle 2D vs 3D input
    bool is_2d = mag.ndim() == 2;
    if (is_2d) {
        mag = reshape(mag, {1, mag.shape(0), mag.shape(1)}, s);
    }

    // Get or compute centroid
    array cent = (centroid.size() == 0)
        ? spectral_centroid(S, frequencies, s)
        : astype(centroid, float32, s);

    if (centroid.size() == 0 && is_2d) {
        cent = reshape(cent, {1, 1, cent.shape(-1)}, s);
    } else if (centroid.size() > 0 && cent.ndim() == 2) {
        // Ensure centroid has right shape for broadcasting
        cent = reshape(cent, {1, cent.shape(0), cent.shape(1)}, s);
    }

    // mag shape: (batch, freq_bins, n_frames)
    // freqs shape: (freq_bins,)
    // cent shape: (batch, 1, n_frames)

    // Reshape freqs for broadcasting: (freq_bins,) -> (1, freq_bins, 1)
    auto freqs_broadcast = reshape(freqs, {1, static_cast<int>(freqs.shape(0)), 1}, s);

    // |f - centroid|^p
    auto deviation = abs(subtract(freqs_broadcast, cent, s), s);
    auto deviation_p = power(deviation, array(p, float32), s);

    // bandwidth = (sum(|f - centroid|^p * S) / sum(S))^(1/p)
    auto weighted_sum = sum(multiply(deviation_p, mag, s), {1}, true, s);
    auto total_sum = sum(mag, {1}, true, s);

    total_sum = maximum(total_sum, array(1e-10f, float32), s);

    auto bandwidth = power(divide(weighted_sum, total_sum, s), array(1.0f / p, float32), s);

    if (is_2d) {
        bandwidth = squeeze(bandwidth, 0, s);
    }

    return bandwidth;
}

mlx::core::array spectral_rolloff(
    const mlx::core::array& S,
    const mlx::core::array& frequencies,
    float roll_percent,
    mlx::core::StreamOrDevice s) {

    using namespace mlx::core;

    // Validate inputs
    if (S.ndim() < 2 || S.ndim() > 3) {
        throw std::invalid_argument(
            "S must be 2D (freq_bins, n_frames) or 3D (batch, freq_bins, n_frames)");
    }
    if (frequencies.ndim() != 1) {
        throw std::invalid_argument("frequencies must be 1D (freq_bins,)");
    }
    if (roll_percent < 0.0f || roll_percent > 1.0f) {
        throw std::invalid_argument("roll_percent must be between 0 and 1");
    }

    // Ensure float32
    auto mag = astype(S, float32, s);
    auto freqs = astype(frequencies, float32, s);

    // Handle 2D vs 3D input
    bool is_2d = mag.ndim() == 2;
    if (is_2d) {
        mag = reshape(mag, {1, mag.shape(0), mag.shape(1)}, s);
    }

    // mag shape: (batch, freq_bins, n_frames)
    int batch_size = mag.shape(0);
    int n_freq = mag.shape(1);
    int n_frames = mag.shape(2);

    // Compute cumulative sum along frequency axis
    auto cum_sum = cumsum(mag, 1, false, false, s);  // (batch, freq_bins, n_frames)

    // Total energy per frame
    auto total_energy = slice(cum_sum, {0, n_freq - 1, 0}, {batch_size, n_freq, n_frames}, s);
    // Shape: (batch, 1, n_frames)

    // Threshold
    auto threshold = multiply(total_energy, array(roll_percent, float32), s);

    // Find first bin where cumsum >= threshold
    // This is tricky without a proper searchsorted - we'll use a comparison approach
    // cum_sum >= threshold -> find first True along freq axis
    auto exceeds = greater_equal(cum_sum, threshold, s);  // (batch, freq_bins, n_frames)

    // Convert to float and use argmax to find first True
    auto exceeds_float = astype(exceeds, float32, s);

    // To find first True, we can multiply by -arange and find argmin
    // Or use cumsum trick: first True is where cumsum == 1 for first time
    // Simpler: use argmax on the boolean (finds first True if any)
    auto indices = argmax(exceeds_float, 1, true, s);  // (batch, 1, n_frames)

    // Convert indices to frequencies
    // indices contains the frequency bin index
    auto indices_float = astype(indices, float32, s);

    // For safety, clamp indices to valid range
    indices_float = clip(indices_float, array(0.0f, float32), array(static_cast<float>(n_freq - 1), float32), s);

    // Linear interpolation would be better, but for now just use the bin frequency
    // rolloff = freqs[indices]
    // We need to gather along the frequency axis
    indices = astype(indices_float, int32, s);

    // Reshape for take_along_axis: indices needs to match mag's ndim
    auto rolloff = take_along_axis(
        reshape(freqs, {1, n_freq, 1}, s),  // Broadcast freqs
        indices,
        1,
        s
    );

    if (is_2d) {
        rolloff = squeeze(rolloff, 0, s);
    }

    return rolloff;
}

mlx::core::array spectral_flatness(
    const mlx::core::array& S,
    float amin,
    mlx::core::StreamOrDevice s) {

    using namespace mlx::core;

    // Validate inputs
    if (S.ndim() < 2 || S.ndim() > 3) {
        throw std::invalid_argument(
            "S must be 2D (freq_bins, n_frames) or 3D (batch, freq_bins, n_frames)");
    }

    // Ensure float32
    auto mag = astype(S, float32, s);

    // Handle 2D vs 3D input
    bool is_2d = mag.ndim() == 2;
    if (is_2d) {
        mag = reshape(mag, {1, mag.shape(0), mag.shape(1)}, s);
    }

    // mag shape: (batch, freq_bins, n_frames)
    int n_freq = mag.shape(1);

    // Clamp to minimum value for numerical stability
    mag = maximum(mag, array(amin, float32), s);

    // flatness = exp(mean(log(S))) / mean(S)
    //          = geometric_mean(S) / arithmetic_mean(S)

    // Log mean (geometric mean in log space)
    auto log_mag = log(mag, s);
    auto log_mean = mean(log_mag, {1}, true, s);  // Mean over freq axis
    auto geometric_mean = exp(log_mean, s);

    // Arithmetic mean
    auto arith_mean = mean(mag, {1}, true, s);

    // Flatness
    auto flatness = divide(geometric_mean, arith_mean, s);

    if (is_2d) {
        flatness = squeeze(flatness, 0, s);
    }

    return flatness;
}

}  // namespace mlx_audio
