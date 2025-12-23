#pragma once

#include <mlx/mlx.h>
#include <optional>
#include <string>

namespace mlx_audio {

/**
 * Convert frequency in Hz to mel scale.
 *
 * @param frequencies Frequencies in Hz
 * @param htk If true, use HTK formula; otherwise use Slaney (librosa default)
 * @param s Stream or device for computation
 * @return Frequencies in mel scale
 */
mlx::core::array hz_to_mel(
    const mlx::core::array& frequencies,
    bool htk = false,
    mlx::core::StreamOrDevice s = {});

/**
 * Convert mel scale to frequency in Hz.
 *
 * @param mels Frequencies in mel scale
 * @param htk If true, use HTK formula; otherwise use Slaney (librosa default)
 * @param s Stream or device for computation
 * @return Frequencies in Hz
 */
mlx::core::array mel_to_hz(
    const mlx::core::array& mels,
    bool htk = false,
    mlx::core::StreamOrDevice s = {});

/**
 * Create a mel-scale filterbank matrix.
 *
 * @param sr Sample rate of the audio
 * @param n_fft FFT size
 * @param n_mels Number of mel bands
 * @param fmin Minimum frequency in Hz
 * @param fmax Maximum frequency in Hz (defaults to sr/2 if not specified)
 * @param htk If true, use HTK formula for mel scale
 * @param norm Normalization mode: "slaney" or empty for none
 * @param s Stream or device for computation
 * @return Filterbank matrix of shape (n_mels, n_fft // 2 + 1)
 */
mlx::core::array mel_filterbank(
    int sr,
    int n_fft,
    int n_mels = 128,
    float fmin = 0.0f,
    std::optional<float> fmax = std::nullopt,
    bool htk = false,
    const std::string& norm = "slaney",
    mlx::core::StreamOrDevice s = {});

}  // namespace mlx_audio
