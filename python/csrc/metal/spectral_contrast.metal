#include <metal_stdlib>
using namespace metal;

/**
 * Spectral contrast kernel - computes peak and valley values for octave bands.
 *
 * This kernel parallelizes the band computation across threads.
 * Each thread processes one (batch, band, frame) combination.
 *
 * The algorithm:
 * 1. For each band, determine the frequency bin range
 * 2. Extract the sub-band from the spectrogram
 * 3. Use bitonic sort to find top-k and bottom-k quantiles
 * 4. Compute mean of quantile values for peak and valley
 */

// Constants
constant int MAX_BAND_SIZE = 512;  // Max bins per band

/**
 * Compute spectral contrast for a single band and frame.
 *
 * This kernel uses a simplified approach:
 * - Each thread handles one (batch, band, frame)
 * - Uses partial sorting via parallel min/max finding
 */
template <typename T>
[[kernel]] void spectral_contrast_kernel(
    device const T* S [[buffer(0)]],              // (batch, n_bins, n_frames)
    device const T* freq [[buffer(1)]],           // (n_bins,)
    device T* peak [[buffer(2)]],                 // (batch, n_bands+1, n_frames)
    device T* valley [[buffer(3)]],               // (batch, n_bands+1, n_frames)
    constant const int& batch_size [[buffer(4)]],
    constant const int& n_bins [[buffer(5)]],
    constant const int& n_frames [[buffer(6)]],
    constant const int& n_bands [[buffer(7)]],
    constant const T& fmin [[buffer(8)]],
    constant const T& quantile [[buffer(9)]],
    uint3 gid [[thread_position_in_grid]]) {

    int frame_idx = gid.x;
    int band_idx = gid.y;
    int batch_idx = gid.z;

    if (frame_idx >= n_frames || band_idx > n_bands || batch_idx >= batch_size) {
        return;
    }

    // Compute octave band edges
    // Band k covers [fmin * 2^(k-1), fmin * 2^k] for k > 0
    // Band 0 covers [0, fmin]
    T f_low, f_high;
    if (band_idx == 0) {
        f_low = T(0);
        f_high = fmin;
    } else {
        f_low = fmin * pow(T(2), T(band_idx - 1));
        f_high = fmin * pow(T(2), T(band_idx));
    }

    // Find bins in current band
    int first_bin = -1;
    int last_bin = -1;

    for (int i = 0; i < n_bins; i++) {
        T f = freq[i];
        if (f >= f_low && f <= f_high) {
            if (first_bin < 0) first_bin = i;
            last_bin = i;
        }
    }

    // Handle empty bands
    if (first_bin < 0 || last_bin < 0) {
        int out_idx = batch_idx * (n_bands + 1) * n_frames + band_idx * n_frames + frame_idx;
        peak[out_idx] = T(0);
        valley[out_idx] = T(0);
        return;
    }

    // Include neighbor bin at lower edge (except for first band)
    if (band_idx > 0 && first_bin > 0) {
        first_bin = first_bin - 1;
    }

    // Extend last band to Nyquist
    if (band_idx == n_bands && last_bin + 1 < n_bins) {
        last_bin = n_bins - 1;
    }

    // Calculate n_quantile before removing last bin
    int band_size = last_bin - first_bin + 1;
    int n_quantile = max(1, int(round(quantile * T(band_size))));

    // Remove last bin for all bands except the last (after calculating n_quantile)
    if (band_idx < n_bands && band_size > 1) {
        last_bin = last_bin - 1;
        band_size = last_bin - first_bin + 1;
    }

    if (band_size == 0) {
        int out_idx = batch_idx * (n_bands + 1) * n_frames + band_idx * n_frames + frame_idx;
        peak[out_idx] = T(0);
        valley[out_idx] = T(0);
        return;
    }

    // Extract sub-band values into thread-local storage
    // For efficiency, we use a small fixed buffer
    T sub_band[MAX_BAND_SIZE];
    int actual_size = min(band_size, MAX_BAND_SIZE);

    for (int i = 0; i < actual_size; i++) {
        int bin_idx = first_bin + i;
        int s_idx = batch_idx * n_bins * n_frames + bin_idx * n_frames + frame_idx;
        sub_band[i] = S[s_idx];
    }

    // Simple partial sort: find bottom n_quantile values for valley
    // and top n_quantile values for peak
    // This is O(n * k) which is acceptable for small bands

    T valley_sum = T(0);
    T peak_sum = T(0);

    n_quantile = min(n_quantile, actual_size);

    if (n_quantile >= actual_size) {
        // Use all elements
        for (int i = 0; i < actual_size; i++) {
            valley_sum += sub_band[i];
            peak_sum += sub_band[i];
        }
        valley_sum /= T(actual_size);
        peak_sum /= T(actual_size);
    } else {
        // Find bottom k for valley (using selection)
        // Copy array for modification
        T sorted[MAX_BAND_SIZE];
        for (int i = 0; i < actual_size; i++) {
            sorted[i] = sub_band[i];
        }

        // Partial bubble sort to find k smallest
        for (int i = 0; i < n_quantile; i++) {
            for (int j = i + 1; j < actual_size; j++) {
                if (sorted[j] < sorted[i]) {
                    T tmp = sorted[i];
                    sorted[i] = sorted[j];
                    sorted[j] = tmp;
                }
            }
            valley_sum += sorted[i];
        }
        valley_sum /= T(n_quantile);

        // Find top k for peak (largest elements at the end)
        for (int i = 0; i < n_quantile; i++) {
            int idx = actual_size - 1 - i;
            for (int j = 0; j < idx; j++) {
                if (sorted[j] > sorted[idx]) {
                    T tmp = sorted[j];
                    sorted[j] = sorted[idx];
                    sorted[idx] = tmp;
                }
            }
            peak_sum += sorted[idx];
        }
        peak_sum /= T(n_quantile);
    }

    // Write output
    int out_idx = batch_idx * (n_bands + 1) * n_frames + band_idx * n_frames + frame_idx;
    peak[out_idx] = peak_sum;
    valley[out_idx] = valley_sum;
}

/**
 * Compute log contrast from peak and valley.
 * contrast = 10 * log10(peak) - 10 * log10(valley)
 */
template <typename T>
[[kernel]] void spectral_contrast_log_kernel(
    device const T* peak [[buffer(0)]],           // (batch, n_bands+1, n_frames)
    device const T* valley [[buffer(1)]],         // (batch, n_bands+1, n_frames)
    device T* contrast [[buffer(2)]],             // (batch, n_bands+1, n_frames)
    constant const int& total_size [[buffer(3)]],
    constant const T& amin [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {

    if (gid >= uint(total_size)) {
        return;
    }

    T p = max(peak[gid], amin);
    T v = max(valley[gid], amin);

    // 10 * log10(x) = 10 * log(x) / log(10)
    const T log10_factor = T(10.0) / log(T(10.0));
    T peak_db = log10_factor * log(p);
    T valley_db = log10_factor * log(v);

    contrast[gid] = peak_db - valley_db;
}

// Explicit instantiations
template [[host_name("spectral_contrast_float")]] [[kernel]] void spectral_contrast_kernel<float>(
    device const float*, device const float*, device float*, device float*,
    constant const int&, constant const int&, constant const int&, constant const int&,
    constant const float&, constant const float&, uint3);

template [[host_name("spectral_contrast_log_float")]] [[kernel]] void spectral_contrast_log_kernel<float>(
    device const float*, device const float*, device float*,
    constant const int&, constant const float&, uint);
