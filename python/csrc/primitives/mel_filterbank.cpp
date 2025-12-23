#include "primitives/mel_filterbank.h"
#include "primitives/metal_utils.h"

#include <cmath>
#include <string>
#include <mlx/ops.h>
#include <mlx/transforms.h>

namespace mlx_audio {

namespace {

// Slaney mel scale constants - use double precision
constexpr double F_MIN = 0.0;
constexpr double F_SP = 200.0 / 3.0;  // Spacing in Hz for linear region
constexpr double MIN_LOG_HZ = 1000.0;  // Boundary between linear and log
constexpr double MIN_LOG_MEL = (MIN_LOG_HZ - F_MIN) / F_SP;  // Mel at boundary = 15.0
constexpr double LOGSTEP = 0.06875177742094912;  // log(6.4) / 27.0

// Float64 versions of hz_to_mel and mel_to_hz for internal use
mlx::core::array hz_to_mel_f64(
    const mlx::core::array& frequencies,
    bool htk,
    mlx::core::StreamOrDevice s) {

    using namespace mlx::core;

    if (htk) {
        // HTK formula: mel = 2595 * log10(1 + f / 700)
        auto f_scaled = divide(frequencies, array(700.0, float64), s);
        auto f_plus_one = add(array(1.0, float64), f_scaled, s);
        return multiply(array(2595.0, float64), log10(f_plus_one, s), s);
    }
    else {
        // Slaney formula (librosa default)
        // Linear below 1000 Hz, logarithmic above
        auto linear_part = divide(subtract(frequencies, array(F_MIN, float64), s), array(F_SP, float64), s);
        auto log_arg = divide(frequencies, array(MIN_LOG_HZ, float64), s);
        auto log_part = add(array(MIN_LOG_MEL, float64), divide(log(log_arg, s), array(LOGSTEP, float64), s), s);

        return where(less(frequencies, array(MIN_LOG_HZ, float64), s), linear_part, log_part, s);
    }
}

mlx::core::array mel_to_hz_f64(
    const mlx::core::array& mels,
    bool htk,
    mlx::core::StreamOrDevice s) {

    using namespace mlx::core;

    if (htk) {
        // HTK formula: f = 700 * (10^(mel / 2595) - 1)
        auto mel_scaled = divide(mels, array(2595.0, float64), s);
        auto pow_result = power(array(10.0, float64), mel_scaled, s);
        return multiply(array(700.0, float64), subtract(pow_result, array(1.0, float64), s), s);
    }
    else {
        // Slaney formula (inverse)
        auto linear_part = add(array(F_MIN, float64), multiply(array(F_SP, float64), mels, s), s);
        auto exp_arg = multiply(array(LOGSTEP, float64), subtract(mels, array(MIN_LOG_MEL, float64), s), s);
        auto log_part = multiply(array(MIN_LOG_HZ, float64), exp(exp_arg, s), s);

        return where(less(mels, array(MIN_LOG_MEL, float64), s), linear_part, log_part, s);
    }
}

}  // namespace

mlx::core::array hz_to_mel(
    const mlx::core::array& frequencies,
    bool htk,
    mlx::core::StreamOrDevice s) {

    using namespace mlx::core;

    // Force CPU for precision
    auto cpu_stream = default_stream(Device::cpu);
    auto freq_f64 = astype(frequencies, float64, cpu_stream);
    eval({freq_f64});

    auto result = hz_to_mel_f64(freq_f64, htk, cpu_stream);
    auto result_f32 = astype(result, float32, cpu_stream);
    eval({result_f32});

    return result_f32;
}

mlx::core::array mel_to_hz(
    const mlx::core::array& mels,
    bool htk,
    mlx::core::StreamOrDevice s) {

    using namespace mlx::core;

    // Force CPU for precision
    auto cpu_stream = default_stream(Device::cpu);
    auto mels_f64 = astype(mels, float64, cpu_stream);
    eval({mels_f64});

    auto result = mel_to_hz_f64(mels_f64, htk, cpu_stream);
    auto result_f32 = astype(result, float32, cpu_stream);
    eval({result_f32});

    return result_f32;
}

mlx::core::array mel_filterbank(
    int sr,
    int n_fft,
    int n_mels,
    float fmin,
    std::optional<float> fmax_opt,
    bool htk,
    const std::string& norm,
    mlx::core::StreamOrDevice s) {

    using namespace mlx::core;

    // Input validation
    if (sr <= 0) {
        throw std::invalid_argument("Sample rate (sr) must be positive");
    }
    if (n_fft <= 0) {
        throw std::invalid_argument("n_fft must be positive");
    }
    if (n_mels <= 0) {
        throw std::invalid_argument("n_mels must be positive");
    }
    if (fmin < 0) {
        throw std::invalid_argument("fmin must be non-negative");
    }

    double fmax = fmax_opt.value_or(static_cast<double>(sr) / 2.0);

    if (fmin >= fmax) {
        throw std::invalid_argument("fmin must be less than fmax");
    }
    if (fmax > static_cast<double>(sr) / 2.0) {
        throw std::invalid_argument(
            "fmax cannot exceed Nyquist frequency (sr / 2)");
    }

    // Mel filterbanks are computed once and cached, so use CPU with float64
    // precision to match librosa exactly. Skip Metal path for accuracy.
    auto cpu_stream = default_stream(Device::cpu);

    // Number of frequency bins
    int n_freqs = 1 + n_fft / 2;

    // Center freqs of each FFT bin (matches librosa.fft_frequencies)
    auto fft_freqs = linspace(0.0, static_cast<double>(sr) / 2.0, n_freqs, float64, cpu_stream);
    eval({fft_freqs});

    // Mel frequencies: n_mels + 2 points (matches librosa.mel_frequencies)
    auto fmin_arr = array(static_cast<double>(fmin), float64);
    auto fmax_arr = array(fmax, float64);
    auto mel_min = hz_to_mel_f64(fmin_arr, htk, cpu_stream);
    auto mel_max = hz_to_mel_f64(fmax_arr, htk, cpu_stream);
    eval({mel_min, mel_max});

    auto mel_points = linspace(mel_min.item<double>(), mel_max.item<double>(), n_mels + 2, float64, cpu_stream);
    eval({mel_points});
    auto mel_f = mel_to_hz_f64(mel_points, htk, cpu_stream);
    eval({mel_f});

    // fdiff = np.diff(mel_f) - differences between consecutive mel frequencies
    // Shape: (n_mels + 1,)
    auto mel_f_hi = slice(mel_f, {1}, {n_mels + 2}, cpu_stream);
    auto mel_f_lo = slice(mel_f, {0}, {n_mels + 1}, cpu_stream);
    auto fdiff = subtract(mel_f_hi, mel_f_lo, cpu_stream);
    eval({fdiff});

    // ramps = np.subtract.outer(mel_f, fftfreqs)
    // Shape: (n_mels + 2, n_freqs) - each row is mel_f[i] - fftfreqs
    auto mel_f_col = reshape(mel_f, {n_mels + 2, 1}, cpu_stream);
    auto fft_freqs_row = reshape(fft_freqs, {1, n_freqs}, cpu_stream);
    auto ramps = subtract(mel_f_col, fft_freqs_row, cpu_stream);
    eval({ramps});

    // Build each mel band row and stack them
    // For each mel band i:
    //   lower = -ramps[i] / fdiff[i]
    //   upper = ramps[i + 2] / fdiff[i + 1]
    //   weights[i] = max(0, min(lower, upper))
    std::vector<array> rows;
    rows.reserve(n_mels);

    for (int i = 0; i < n_mels; i++) {
        // Get ramps[i] and ramps[i+2]
        auto ramps_i = squeeze(slice(ramps, {i, 0}, {i + 1, n_freqs}, cpu_stream), cpu_stream);
        auto ramps_i2 = squeeze(slice(ramps, {i + 2, 0}, {i + 3, n_freqs}, cpu_stream), cpu_stream);

        // Get fdiff[i] and fdiff[i+1]
        auto fdiff_i = slice(fdiff, {i}, {i + 1}, cpu_stream);
        auto fdiff_i1 = slice(fdiff, {i + 1}, {i + 2}, cpu_stream);

        // lower = -ramps[i] / fdiff[i]
        auto lower = divide(negative(ramps_i, cpu_stream), fdiff_i, cpu_stream);

        // upper = ramps[i + 2] / fdiff[i + 1]
        auto upper = divide(ramps_i2, fdiff_i1, cpu_stream);

        // weights[i] = max(0, min(lower, upper))
        auto zero = array(0.0, float64);
        auto row = maximum(zero, minimum(lower, upper, cpu_stream), cpu_stream);

        rows.push_back(row);
    }

    // Stack all rows into final weights matrix
    auto weights = stack(rows, 0, cpu_stream);
    eval({weights});

    // Normalize
    if (norm == "slaney") {
        // enorm = 2.0 / (mel_f[2:n_mels+2] - mel_f[0:n_mels])
        auto mel_hi = slice(mel_f, {2}, {n_mels + 2}, cpu_stream);
        auto mel_lo = slice(mel_f, {0}, {n_mels}, cpu_stream);
        auto bandwidth = subtract(mel_hi, mel_lo, cpu_stream);
        auto enorm = divide(array(2.0, float64), bandwidth, cpu_stream);
        eval({enorm});

        // weights *= enorm[:, np.newaxis]
        auto enorm_col = reshape(enorm, {n_mels, 1}, cpu_stream);
        weights = multiply(weights, enorm_col, cpu_stream);
        eval({weights});
    }
    else if (!norm.empty()) {
        throw std::invalid_argument(
            "Unknown norm: '" + norm + "'. Supported: 'slaney', empty string for none");
    }

    // Convert to float32 for output
    auto result = astype(weights, float32, cpu_stream);
    eval({result});

    return result;
}

}  // namespace mlx_audio
