#pragma once

#include <optional>
#include <mlx/mlx.h>
#include "primitives/windows.h"
#include "primitives/frame_signal.h"
#include "primitives/overlap_add.h"
#include "primitives/pad_signal.h"
#include "primitives/mel_filterbank.h"
#include "primitives/autocorrelation.h"
#include "primitives/resample.h"
#include "primitives/dct.h"
#include "primitives/spectral.h"

namespace mlx_audio {

// Wrapper functions that take optional stream for proper nanobind binding

inline mlx::core::array generate_window_wrapper(
    const std::string& window_type,
    int length,
    bool periodic,
    std::optional<mlx::core::Stream> stream) {
    if (stream.has_value()) {
        return generate_window(window_type, length, periodic, stream.value());
    }
    return generate_window(window_type, length, periodic);
}

inline mlx::core::array frame_signal_wrapper(
    const mlx::core::array& signal,
    int frame_length,
    int hop_length,
    std::optional<mlx::core::Stream> stream) {
    if (stream.has_value()) {
        return frame_signal(signal, frame_length, hop_length, stream.value());
    }
    return frame_signal(signal, frame_length, hop_length);
}

inline mlx::core::array overlap_add_wrapper(
    const mlx::core::array& frames,
    const mlx::core::array& window,
    int hop_length,
    int output_length,
    std::optional<mlx::core::Stream> stream) {
    if (stream.has_value()) {
        return overlap_add(frames, window, hop_length, output_length, stream.value());
    }
    return overlap_add(frames, window, hop_length, output_length);
}

inline mlx::core::array pad_signal_wrapper(
    const mlx::core::array& signal,
    int pad_length,
    const std::string& mode,
    std::optional<mlx::core::Stream> stream) {
    if (stream.has_value()) {
        return pad_signal(signal, pad_length, mode, stream.value());
    }
    return pad_signal(signal, pad_length, mode);
}

inline mlx::core::array hz_to_mel_wrapper(
    const mlx::core::array& frequencies,
    bool htk,
    std::optional<mlx::core::Stream> stream) {
    if (stream.has_value()) {
        return hz_to_mel(frequencies, htk, stream.value());
    }
    return hz_to_mel(frequencies, htk);
}

inline mlx::core::array mel_to_hz_wrapper(
    const mlx::core::array& mels,
    bool htk,
    std::optional<mlx::core::Stream> stream) {
    if (stream.has_value()) {
        return mel_to_hz(mels, htk, stream.value());
    }
    return mel_to_hz(mels, htk);
}

inline mlx::core::array mel_filterbank_wrapper(
    int sr,
    int n_fft,
    int n_mels,
    float fmin,
    std::optional<float> fmax,
    bool htk,
    const std::string& norm,
    std::optional<mlx::core::Stream> stream) {
    if (stream.has_value()) {
        return mel_filterbank(sr, n_fft, n_mels, fmin, fmax, htk, norm, stream.value());
    }
    return mel_filterbank(sr, n_fft, n_mels, fmin, fmax, htk, norm);
}

// Autocorrelation wrapper
inline mlx::core::array autocorrelation_wrapper(
    const mlx::core::array& signal,
    int max_lag,
    bool normalize,
    bool center,
    std::optional<mlx::core::Stream> stream) {
    if (stream.has_value()) {
        return autocorrelation(signal, max_lag, normalize, center, stream.value());
    }
    return autocorrelation(signal, max_lag, normalize, center);
}

// Resample wrappers
inline mlx::core::array resample_fft_wrapper(
    const mlx::core::array& signal,
    int num_samples,
    std::optional<mlx::core::Stream> stream) {
    if (stream.has_value()) {
        return resample_fft(signal, num_samples, stream.value());
    }
    return resample_fft(signal, num_samples);
}

inline mlx::core::array resample_wrapper(
    const mlx::core::array& signal,
    int orig_sr,
    int target_sr,
    bool fix,
    bool scale,
    std::optional<mlx::core::Stream> stream) {
    if (stream.has_value()) {
        return resample(signal, orig_sr, target_sr, fix, scale, stream.value());
    }
    return resample(signal, orig_sr, target_sr, fix, scale);
}

// DCT wrappers
inline mlx::core::array get_dct_matrix_wrapper(
    int n_out,
    int n_in,
    const std::string& norm,
    std::optional<mlx::core::Stream> stream) {
    if (stream.has_value()) {
        return get_dct_matrix(n_out, n_in, norm, stream.value());
    }
    return get_dct_matrix(n_out, n_in, norm);
}

inline mlx::core::array dct_wrapper(
    const mlx::core::array& x,
    int n,
    int axis,
    const std::string& norm,
    std::optional<mlx::core::Stream> stream) {
    if (stream.has_value()) {
        return dct(x, n, axis, norm, stream.value());
    }
    return dct(x, n, axis, norm);
}

// Spectral feature wrappers
inline mlx::core::array spectral_centroid_wrapper(
    const mlx::core::array& S,
    const mlx::core::array& frequencies,
    std::optional<mlx::core::Stream> stream) {
    if (stream.has_value()) {
        return spectral_centroid(S, frequencies, stream.value());
    }
    return spectral_centroid(S, frequencies);
}

inline mlx::core::array spectral_bandwidth_wrapper(
    const mlx::core::array& S,
    const mlx::core::array& frequencies,
    const mlx::core::array& centroid,
    float p,
    std::optional<mlx::core::Stream> stream) {
    if (stream.has_value()) {
        return spectral_bandwidth(S, frequencies, centroid, p, stream.value());
    }
    return spectral_bandwidth(S, frequencies, centroid, p);
}

inline mlx::core::array spectral_rolloff_wrapper(
    const mlx::core::array& S,
    const mlx::core::array& frequencies,
    float roll_percent,
    std::optional<mlx::core::Stream> stream) {
    if (stream.has_value()) {
        return spectral_rolloff(S, frequencies, roll_percent, stream.value());
    }
    return spectral_rolloff(S, frequencies, roll_percent);
}

inline mlx::core::array spectral_flatness_wrapper(
    const mlx::core::array& S,
    float amin,
    std::optional<mlx::core::Stream> stream) {
    if (stream.has_value()) {
        return spectral_flatness(S, amin, stream.value());
    }
    return spectral_flatness(S, amin);
}

}  // namespace mlx_audio
