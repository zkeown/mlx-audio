#include "primitives/spectral_contrast.h"
#include "primitives/metal_utils.h"

#include <algorithm>
#include <cmath>
#include <vector>
#include <mlx/ops.h>
#include <mlx/transforms.h>

namespace mlx_audio {

#ifdef MLX_BUILD_METAL
mlx::core::array spectral_contrast_metal(
    const mlx::core::array& S,
    const mlx::core::array& frequencies,
    float fmin,
    int n_bands,
    float quantile,
    bool linear,
    mlx::core::StreamOrDevice s) {

    using namespace mlx::core;

    // Handle 2D vs 3D input
    bool is_2d = S.ndim() == 2;
    array mag = astype(S, float32, s);
    if (is_2d) {
        mag = reshape(mag, {1, mag.shape(0), mag.shape(1)}, s);
    }

    int batch_size = mag.shape(0);
    int n_bins = mag.shape(1);
    int n_frames = mag.shape(2);

    // Ensure contiguous inputs
    mag = flatten(mag, s);
    mag = reshape(mag, {batch_size, n_bins, n_frames}, s);

    auto freq = astype(frequencies, float32, s);
    freq = flatten(freq, s);
    freq = reshape(freq, {n_bins}, s);

    // Get Metal device and load library
    auto& d = get_metal_device(s);
    auto stream = to_stream(s);
    auto lib = d.get_library(METAL_LIB_NAME);

    // Allocate output arrays
    auto peak = zeros({batch_size, n_bands + 1, n_frames}, float32, s);
    auto valley = zeros({batch_size, n_bands + 1, n_frames}, float32, s);

    // Evaluate inputs
    eval({mag, freq, peak, valley});

    // Launch spectral contrast kernel
    auto kernel = d.get_kernel("spectral_contrast_float", lib);
    auto& enc = d.get_command_encoder(stream.index);
    enc.set_compute_pipeline_state(kernel);

    enc.set_input_array(mag, 0);
    enc.set_input_array(freq, 1);
    enc.set_output_array(peak, 2);
    enc.set_output_array(valley, 3);
    enc.set_bytes(batch_size, 4);
    enc.set_bytes(n_bins, 5);
    enc.set_bytes(n_frames, 6);
    enc.set_bytes(n_bands, 7);
    enc.set_bytes(fmin, 8);
    enc.set_bytes(quantile, 9);

    // Dispatch: 3D grid (n_frames, n_bands+1, batch_size)
    auto [tg0, tg1, tg2] = get_threadgroup_size_3d(n_frames, n_bands + 1, batch_size);
    MTL::Size grid_dims = MTL::Size(n_frames, n_bands + 1, batch_size);
    MTL::Size group_dims = MTL::Size(tg0, tg1, tg2);
    enc.dispatch_threads(grid_dims, group_dims);

    if (is_2d) {
        if (linear) {
            // Linear contrast: peak - valley
            return squeeze(peak - valley, 0, s);
        } else {
            // Log contrast via second kernel
            int total_size = batch_size * (n_bands + 1) * n_frames;
            auto contrast = zeros({batch_size, n_bands + 1, n_frames}, float32, s);
            eval({peak, valley, contrast});

            auto log_kernel = d.get_kernel("spectral_contrast_log_float", lib);
            auto& enc2 = d.get_command_encoder(stream.index);
            enc2.set_compute_pipeline_state(log_kernel);

            enc2.set_input_array(peak, 0);
            enc2.set_input_array(valley, 1);
            enc2.set_output_array(contrast, 2);
            enc2.set_bytes(total_size, 3);
            float amin = 1e-10f;
            enc2.set_bytes(amin, 4);

            int tg = get_threadgroup_size_1d(total_size);
            MTL::Size grid1d = MTL::Size(total_size, 1, 1);
            MTL::Size group1d = MTL::Size(tg, 1, 1);
            enc2.dispatch_threads(grid1d, group1d);

            return squeeze(contrast, 0, s);
        }
    } else {
        if (linear) {
            return peak - valley;
        } else {
            // Log contrast via second kernel
            int total_size = batch_size * (n_bands + 1) * n_frames;
            auto contrast = zeros({batch_size, n_bands + 1, n_frames}, float32, s);
            eval({peak, valley, contrast});

            auto log_kernel = d.get_kernel("spectral_contrast_log_float", lib);
            auto& enc2 = d.get_command_encoder(stream.index);
            enc2.set_compute_pipeline_state(log_kernel);

            enc2.set_input_array(peak, 0);
            enc2.set_input_array(valley, 1);
            enc2.set_output_array(contrast, 2);
            enc2.set_bytes(total_size, 3);
            float amin = 1e-10f;
            enc2.set_bytes(amin, 4);

            int tg = get_threadgroup_size_1d(total_size);
            MTL::Size grid1d = MTL::Size(total_size, 1, 1);
            MTL::Size group1d = MTL::Size(tg, 1, 1);
            enc2.dispatch_threads(grid1d, group1d);

            return contrast;
        }
    }
}
#endif  // MLX_BUILD_METAL

mlx::core::array spectral_contrast_cpu(
    const mlx::core::array& S,
    const mlx::core::array& frequencies,
    float fmin,
    int n_bands,
    float quantile,
    bool linear,
    mlx::core::StreamOrDevice s) {

    using namespace mlx::core;

    // Handle 2D vs 3D input
    bool is_2d = S.ndim() == 2;
    array mag = astype(S, float32, s);
    if (is_2d) {
        mag = reshape(mag, {1, mag.shape(0), mag.shape(1)}, s);
    }

    int batch_size = mag.shape(0);
    int n_bins = mag.shape(1);
    int n_frames = mag.shape(2);

    auto freq = astype(frequencies, float32, s);

    // This CPU implementation mirrors the NumPy version for correctness
    // Evaluate to get actual values for indexing
    eval({mag, freq});

    // Get frequency values for band computation
    std::vector<float> freq_vals(n_bins);
    {
        auto freq_data = freq.data<float>();
        for (int i = 0; i < n_bins; i++) {
            freq_vals[i] = freq_data[i];
        }
    }

    // Compute octave band edges
    std::vector<float> octa(n_bands + 2);
    octa[0] = 0.0f;
    for (int i = 0; i <= n_bands; i++) {
        octa[i + 1] = fmin * std::pow(2.0f, static_cast<float>(i));
    }

    // Allocate outputs
    std::vector<float> peak_data(batch_size * (n_bands + 1) * n_frames, 0.0f);
    std::vector<float> valley_data(batch_size * (n_bands + 1) * n_frames, 0.0f);

    // Get magnitude data
    auto mag_data = mag.data<float>();

    // Process each band
    for (int k = 0; k <= n_bands; k++) {
        float f_low = octa[k];
        float f_high = octa[k + 1];

        // Find bins in current band
        int first_idx = -1;
        int last_idx = -1;
        for (int i = 0; i < n_bins; i++) {
            if (freq_vals[i] >= f_low && freq_vals[i] <= f_high) {
                if (first_idx < 0) first_idx = i;
                last_idx = i;
            }
        }

        if (first_idx < 0) continue;

        // Include neighbor bin at lower edge (except for first band)
        if (k > 0 && first_idx > 0) {
            first_idx--;
        }

        // Extend last band to Nyquist
        if (k == n_bands && last_idx + 1 < n_bins) {
            last_idx = n_bins - 1;
        }

        // Calculate n_quantile before removing last bin
        int band_size = last_idx - first_idx + 1;
        int n_quant = std::max(1, static_cast<int>(std::round(quantile * band_size)));

        // Remove last bin for all bands except the last
        if (k < n_bands && band_size > 1) {
            last_idx--;
            band_size = last_idx - first_idx + 1;
        }

        if (band_size == 0) continue;

        n_quant = std::min(n_quant, band_size);

        // Process each batch and frame
        for (int b = 0; b < batch_size; b++) {
            for (int f = 0; f < n_frames; f++) {
                // Extract sub-band values
                std::vector<float> sub_band(band_size);
                for (int i = 0; i < band_size; i++) {
                    int bin_idx = first_idx + i;
                    sub_band[i] = mag_data[b * n_bins * n_frames + bin_idx * n_frames + f];
                }

                // Compute valley (mean of bottom quantile)
                std::vector<float> sorted = sub_band;
                std::partial_sort(sorted.begin(), sorted.begin() + n_quant, sorted.end());
                float valley_sum = 0.0f;
                for (int i = 0; i < n_quant; i++) {
                    valley_sum += sorted[i];
                }
                valley_sum /= n_quant;

                // Compute peak (mean of top quantile)
                sorted = sub_band;
                std::partial_sort(sorted.begin(), sorted.begin() + n_quant, sorted.end(),
                                  std::greater<float>());
                float peak_sum = 0.0f;
                for (int i = 0; i < n_quant; i++) {
                    peak_sum += sorted[i];
                }
                peak_sum /= n_quant;

                // Store results
                int out_idx = b * (n_bands + 1) * n_frames + k * n_frames + f;
                peak_data[out_idx] = peak_sum;
                valley_data[out_idx] = valley_sum;
            }
        }
    }

    // Convert to arrays - note: Shape is SmallVector<int>
    auto peak = array(peak_data.data(), {batch_size, n_bands + 1, n_frames}, float32);
    auto valley = array(valley_data.data(), {batch_size, n_bands + 1, n_frames}, float32);

    if (is_2d) {
        if (linear) {
            return squeeze(peak - valley, 0, s);
        } else {
            // Log contrast: 10 * log10(peak) - 10 * log10(valley)
            float amin = 1e-10f;
            auto peak_clamped = maximum(peak, array(amin, float32), s);
            auto valley_clamped = maximum(valley, array(amin, float32), s);

            // 10 * log10(x) = 10 / ln(10) * ln(x)
            float log10_factor = 10.0f / std::log(10.0f);
            auto peak_db = log10_factor * log(peak_clamped, s);
            auto valley_db = log10_factor * log(valley_clamped, s);
            return squeeze(peak_db - valley_db, 0, s);
        }
    } else {
        if (linear) {
            return peak - valley;
        } else {
            // Log contrast: 10 * log10(peak) - 10 * log10(valley)
            float amin = 1e-10f;
            auto peak_clamped = maximum(peak, array(amin, float32), s);
            auto valley_clamped = maximum(valley, array(amin, float32), s);

            // 10 * log10(x) = 10 / ln(10) * ln(x)
            float log10_factor = 10.0f / std::log(10.0f);
            auto peak_db = log10_factor * log(peak_clamped, s);
            auto valley_db = log10_factor * log(valley_clamped, s);
            return peak_db - valley_db;
        }
    }
}

mlx::core::array spectral_contrast(
    const mlx::core::array& S,
    const mlx::core::array& frequencies,
    float fmin,
    int n_bands,
    float quantile,
    bool linear,
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
    if (n_bands < 1) {
        throw std::invalid_argument("n_bands must be >= 1");
    }
    if (quantile < 0.0f || quantile > 1.0f) {
        throw std::invalid_argument("quantile must be between 0 and 1");
    }
    if (fmin <= 0.0f) {
        throw std::invalid_argument("fmin must be positive");
    }

#ifdef MLX_BUILD_METAL
    if (should_use_metal(s)) {
        return spectral_contrast_metal(S, frequencies, fmin, n_bands, quantile, linear, s);
    }
#endif

    return spectral_contrast_cpu(S, frequencies, fmin, n_bands, quantile, linear, s);
}

}  // namespace mlx_audio
