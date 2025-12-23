#include <metal_stdlib>
using namespace metal;

// Constants for Slaney mel scale
// Note: LOGSTEP = log(6.4) / 27.0 = 0.0687697...
// Pre-computed to avoid global constructor issue in Metal
constant float F_MIN = 0.0f;
constant float F_SP = 200.0f / 3.0f;
constant float MIN_LOG_HZ = 1000.0f;
constant float MIN_LOG_MEL = (MIN_LOG_HZ - F_MIN) / F_SP;  // = 15.0
constant float LOGSTEP = 0.06875177742094912f;  // = log(6.4) / 27.0

/**
 * Hz to mel conversion (Slaney formula).
 */
float hz_to_mel_slaney(float freq) {
    if (freq < MIN_LOG_HZ) {
        return (freq - F_MIN) / F_SP;
    } else {
        return MIN_LOG_MEL + log(freq / MIN_LOG_HZ) / LOGSTEP;
    }
}

/**
 * Hz to mel conversion (HTK formula).
 */
float hz_to_mel_htk(float freq) {
    return 2595.0f * log10(1.0f + freq / 700.0f);
}

/**
 * Mel to Hz conversion (Slaney formula).
 */
float mel_to_hz_slaney(float mel) {
    if (mel < MIN_LOG_MEL) {
        return F_MIN + F_SP * mel;
    } else {
        return MIN_LOG_HZ * exp(LOGSTEP * (mel - MIN_LOG_MEL));
    }
}

/**
 * Mel to Hz conversion (HTK formula).
 */
float mel_to_hz_htk(float mel) {
    return 700.0f * (pow(10.0f, mel / 2595.0f) - 1.0f);
}

/**
 * Mel filterbank kernel.
 *
 * Each thread computes one element of the filterbank matrix.
 */
[[kernel]] void mel_filterbank_kernel(
    device float* filterbank [[buffer(0)]],      // (n_mels, n_freqs)
    constant const int& sr [[buffer(1)]],
    constant const int& n_fft [[buffer(2)]],
    constant const int& n_mels [[buffer(3)]],
    constant const float& fmin [[buffer(4)]],
    constant const float& fmax [[buffer(5)]],
    constant const int& htk [[buffer(6)]],
    constant const int& do_slaney_norm [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]) {

    int freq_idx = gid.x;        // FFT bin index
    int mel_idx = gid.y;         // Mel band index

    int n_freqs = 1 + n_fft / 2;

    if (freq_idx >= n_freqs || mel_idx >= n_mels) {
        return;
    }

    // Compute mel boundaries
    float mel_min = htk ? hz_to_mel_htk(fmin) : hz_to_mel_slaney(fmin);
    float mel_max = htk ? hz_to_mel_htk(fmax) : hz_to_mel_slaney(fmax);

    // Mel spacing
    float mel_step = (mel_max - mel_min) / float(n_mels + 1);

    // Mel points for this band
    float mel_lower = mel_min + float(mel_idx) * mel_step;
    float mel_center = mel_min + float(mel_idx + 1) * mel_step;
    float mel_upper = mel_min + float(mel_idx + 2) * mel_step;

    // Convert to Hz
    float f_lower = htk ? mel_to_hz_htk(mel_lower) : mel_to_hz_slaney(mel_lower);
    float f_center = htk ? mel_to_hz_htk(mel_center) : mel_to_hz_slaney(mel_center);
    float f_upper = htk ? mel_to_hz_htk(mel_upper) : mel_to_hz_slaney(mel_upper);

    // FFT bin frequency
    float freq = float(freq_idx) * float(sr) / float(n_fft);

    // Compute triangular filter value
    float lower_slope = (freq - f_lower) / (f_center - f_lower + 1e-10f);
    float upper_slope = (f_upper - freq) / (f_upper - f_center + 1e-10f);

    float value = max(0.0f, min(lower_slope, upper_slope));

    // Slaney normalization
    if (do_slaney_norm) {
        float bandwidth = f_upper - f_lower;
        value *= 2.0f / bandwidth;
    }

    filterbank[mel_idx * n_freqs + freq_idx] = value;
}
