#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>

#include "mlx/mlx.h"
#include "bindings_wrappers.h"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_ext, m) {
    m.doc() = "MLX Audio Primitives C++ Extension";

    // Core STFT operations
    m.def(
        "overlap_add",
        &mlx_audio::overlap_add_wrapper,
        "frames"_a,
        "window"_a,
        "hop_length"_a,
        "output_length"_a,
        "stream"_a = nb::none(),
        R"(
        Perform overlap-add reconstruction with window normalization.

        Reconstructs a time-domain signal from overlapping windowed frames
        using the overlap-add method with squared window normalization.

        Parameters
        ----------
        frames : mlx.core.array
            Input frames of shape (batch, n_frames, n_fft).
        window : mlx.core.array
            Window function of shape (n_fft,).
        hop_length : int
            Number of samples between consecutive frames.
        output_length : int
            Desired length of output signal.
        stream : mlx.core.Stream, optional
            Stream for computation.

        Returns
        -------
        mlx.core.array
            Reconstructed signal of shape (batch, output_length).
        )");

    m.def(
        "frame_signal",
        &mlx_audio::frame_signal_wrapper,
        "signal"_a,
        "frame_length"_a,
        "hop_length"_a,
        "stream"_a = nb::none(),
        R"(
        Frame a signal into overlapping windows.

        Parameters
        ----------
        signal : mlx.core.array
            Input signal of shape (batch, samples) or (samples,).
        frame_length : int
            Length of each frame (typically n_fft).
        hop_length : int
            Number of samples between consecutive frames.
        stream : mlx.core.Stream, optional
            Stream for computation.

        Returns
        -------
        mlx.core.array
            Framed signal of shape (batch, n_frames, frame_length) or
            (n_frames, frame_length) if input was 1D.
        )");

    m.def(
        "pad_signal",
        &mlx_audio::pad_signal_wrapper,
        "signal"_a,
        "pad_length"_a,
        "mode"_a = "constant",
        "stream"_a = nb::none(),
        R"(
        Pad signal on both sides.

        Parameters
        ----------
        signal : mlx.core.array
            Input signal of shape (batch, samples).
        pad_length : int
            Number of samples to pad on each side.
        mode : str, optional
            Padding mode: 'constant', 'reflect', or 'edge'.
            Default: 'constant'.
        stream : mlx.core.Stream, optional
            Stream for computation.

        Returns
        -------
        mlx.core.array
            Padded signal of shape (batch, samples + 2 * pad_length).
        )");

    // Window functions
    m.def(
        "generate_window",
        &mlx_audio::generate_window_wrapper,
        "window_type"_a,
        "length"_a,
        "periodic"_a = true,
        "stream"_a = nb::none(),
        R"(
        Generate a window function.

        Parameters
        ----------
        window_type : str
            Window type: 'hann', 'hamming', 'blackman', 'bartlett', 'rectangular'.
        length : int
            Window length.
        periodic : bool, optional
            If True, create periodic (DFT-even) window for FFT. Default: True.
        stream : mlx.core.Stream, optional
            Stream for computation.

        Returns
        -------
        mlx.core.array
            Window array of shape (length,) with dtype float32.
        )");

    // Mel-scale operations
    m.def(
        "hz_to_mel",
        &mlx_audio::hz_to_mel_wrapper,
        "frequencies"_a,
        "htk"_a = false,
        "stream"_a = nb::none(),
        R"(
        Convert frequency in Hz to mel scale.

        Parameters
        ----------
        frequencies : mlx.core.array
            Frequencies in Hz.
        htk : bool, optional
            If True, use HTK formula. If False, use Slaney (librosa default).
            Default: False.
        stream : mlx.core.Stream, optional
            Stream for computation.

        Returns
        -------
        mlx.core.array
            Frequencies in mel scale.
        )");

    m.def(
        "mel_to_hz",
        &mlx_audio::mel_to_hz_wrapper,
        "mels"_a,
        "htk"_a = false,
        "stream"_a = nb::none(),
        R"(
        Convert mel scale to frequency in Hz.

        Parameters
        ----------
        mels : mlx.core.array
            Frequencies in mel scale.
        htk : bool, optional
            If True, use HTK formula. If False, use Slaney (librosa default).
            Default: False.
        stream : mlx.core.Stream, optional
            Stream for computation.

        Returns
        -------
        mlx.core.array
            Frequencies in Hz.
        )");

    m.def(
        "mel_filterbank",
        &mlx_audio::mel_filterbank_wrapper,
        "sr"_a,
        "n_fft"_a,
        "n_mels"_a = 128,
        "fmin"_a = 0.0f,
        "fmax"_a = nb::none(),
        "htk"_a = false,
        "norm"_a = "slaney",
        "stream"_a = nb::none(),
        R"(
        Create a mel-scale filterbank matrix.

        Parameters
        ----------
        sr : int
            Sample rate of the audio.
        n_fft : int
            FFT size.
        n_mels : int, optional
            Number of mel bands. Default: 128.
        fmin : float, optional
            Minimum frequency in Hz. Default: 0.0.
        fmax : float, optional
            Maximum frequency in Hz. Default: sr / 2.
        htk : bool, optional
            If True, use HTK formula for mel scale. Default: False.
        norm : str, optional
            Normalization mode: 'slaney' or empty string for none.
            Default: 'slaney'.
        stream : mlx.core.Stream, optional
            Stream for computation.

        Returns
        -------
        mlx.core.array
            Filterbank matrix of shape (n_mels, n_fft // 2 + 1).
        )");

    // Autocorrelation
    m.def(
        "autocorrelation",
        &mlx_audio::autocorrelation_wrapper,
        "signal"_a,
        "max_lag"_a = -1,
        "normalize"_a = true,
        "center"_a = true,
        "stream"_a = nb::none(),
        R"(
        Compute autocorrelation using FFT (Wiener-Khinchin theorem).

        Parameters
        ----------
        signal : mlx.core.array
            Input signal of shape (samples,) or (batch, samples).
        max_lag : int, optional
            Maximum lag to compute. Default: signal length.
        normalize : bool, optional
            If True, normalize by r[0] (variance). Default: True.
        center : bool, optional
            If True, subtract mean before computing. Default: True.
        stream : mlx.core.Stream, optional
            Stream for computation.

        Returns
        -------
        mlx.core.array
            Autocorrelation for lags 0 to max_lag-1.
        )");

    // Resampling
    m.def(
        "resample_fft",
        &mlx_audio::resample_fft_wrapper,
        "signal"_a,
        "num_samples"_a,
        "stream"_a = nb::none(),
        R"(
        Resample signal to target number of samples using FFT.

        Parameters
        ----------
        signal : mlx.core.array
            Input signal of shape (samples,) or (batch, samples).
        num_samples : int
            Target number of samples.
        stream : mlx.core.Stream, optional
            Stream for computation.

        Returns
        -------
        mlx.core.array
            Resampled signal.
        )");

    m.def(
        "resample",
        &mlx_audio::resample_wrapper,
        "signal"_a,
        "orig_sr"_a,
        "target_sr"_a,
        "fix"_a = true,
        "scale"_a = false,
        "stream"_a = nb::none(),
        R"(
        Resample signal from orig_sr to target_sr using FFT.

        Parameters
        ----------
        signal : mlx.core.array
            Input signal of shape (samples,) or (batch, samples).
        orig_sr : int
            Original sample rate.
        target_sr : int
            Target sample rate.
        fix : bool, optional
            If True, use round for length calculation. Default: True.
        scale : bool, optional
            If True, scale output by sample rate ratio. Default: False.
        stream : mlx.core.Stream, optional
            Stream for computation.

        Returns
        -------
        mlx.core.array
            Resampled signal.
        )");

    // DCT
    m.def(
        "get_dct_matrix",
        &mlx_audio::get_dct_matrix_wrapper,
        "n_out"_a,
        "n_in"_a,
        "norm"_a = "ortho",
        "stream"_a = nb::none(),
        R"(
        Get a cached DCT-II basis matrix.

        Parameters
        ----------
        n_out : int
            Number of output coefficients.
        n_in : int
            Number of input samples.
        norm : str, optional
            Normalization: 'ortho' or ''. Default: 'ortho'.
        stream : mlx.core.Stream, optional
            Stream for computation.

        Returns
        -------
        mlx.core.array
            DCT basis matrix of shape (n_out, n_in).
        )");

    m.def(
        "dct",
        &mlx_audio::dct_wrapper,
        "x"_a,
        "n"_a = -1,
        "axis"_a = -1,
        "norm"_a = "ortho",
        "stream"_a = nb::none(),
        R"(
        Compute Discrete Cosine Transform (DCT-II).

        Parameters
        ----------
        x : mlx.core.array
            Input array.
        n : int, optional
            Number of output coefficients. Default: input size.
        axis : int, optional
            Axis along which to compute DCT. Default: -1.
        norm : str, optional
            Normalization: 'ortho' or ''. Default: 'ortho'.
        stream : mlx.core.Stream, optional
            Stream for computation.

        Returns
        -------
        mlx.core.array
            DCT coefficients.
        )");

    // Spectral features
    m.def(
        "spectral_centroid",
        &mlx_audio::spectral_centroid_wrapper,
        "S"_a,
        "frequencies"_a,
        "stream"_a = nb::none(),
        R"(
        Compute spectral centroid.

        centroid = sum(f * S) / sum(S)

        Parameters
        ----------
        S : mlx.core.array
            Magnitude spectrogram of shape (freq_bins, n_frames) or
            (batch, freq_bins, n_frames).
        frequencies : mlx.core.array
            Frequency values for each bin, shape (freq_bins,).
        stream : mlx.core.Stream, optional
            Stream for computation.

        Returns
        -------
        mlx.core.array
            Spectral centroid for each frame.
        )");

    m.def(
        "spectral_bandwidth",
        &mlx_audio::spectral_bandwidth_wrapper,
        "S"_a,
        "frequencies"_a,
        "centroid"_a,
        "p"_a = 2.0f,
        "stream"_a = nb::none(),
        R"(
        Compute spectral bandwidth.

        bandwidth = (sum(|f - centroid|^p * S) / sum(S))^(1/p)

        Parameters
        ----------
        S : mlx.core.array
            Magnitude spectrogram.
        frequencies : mlx.core.array
            Frequency values for each bin.
        centroid : mlx.core.array
            Pre-computed centroid (pass empty array to compute automatically).
        p : float, optional
            Power for bandwidth calculation. Default: 2.0.
        stream : mlx.core.Stream, optional
            Stream for computation.

        Returns
        -------
        mlx.core.array
            Spectral bandwidth for each frame.
        )");

    m.def(
        "spectral_rolloff",
        &mlx_audio::spectral_rolloff_wrapper,
        "S"_a,
        "frequencies"_a,
        "roll_percent"_a = 0.85f,
        "stream"_a = nb::none(),
        R"(
        Compute spectral rolloff frequency.

        The rolloff frequency is the frequency below which roll_percent
        of the spectral energy is contained.

        Parameters
        ----------
        S : mlx.core.array
            Magnitude spectrogram.
        frequencies : mlx.core.array
            Frequency values for each bin.
        roll_percent : float, optional
            Energy threshold percentage. Default: 0.85.
        stream : mlx.core.Stream, optional
            Stream for computation.

        Returns
        -------
        mlx.core.array
            Spectral rolloff for each frame.
        )");

    m.def(
        "spectral_flatness",
        &mlx_audio::spectral_flatness_wrapper,
        "S"_a,
        "amin"_a = 1e-10f,
        "stream"_a = nb::none(),
        R"(
        Compute spectral flatness.

        flatness = exp(mean(log(S))) / mean(S)

        Parameters
        ----------
        S : mlx.core.array
            Magnitude spectrogram.
        amin : float, optional
            Minimum amplitude for numerical stability. Default: 1e-10.
        stream : mlx.core.Stream, optional
            Stream for computation.

        Returns
        -------
        mlx.core.array
            Spectral flatness for each frame.
        )");
}
