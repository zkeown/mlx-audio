"""
Spectral gating noise reduction.

Provides non-neural audio enhancement via frequency-domain noise suppression.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
from scipy.ndimage import minimum_filter1d, uniform_filter

from mlx_audio.constants import DIVISION_EPSILON

from ._validation import validate_positive, validate_range
from .stft import istft, magnitude, phase, stft


def spectral_gate(
    audio: mx.array,
    noise_profile: mx.array | None = None,
    *,
    sr: int = 22050,
    n_fft: int = 2048,
    hop_length: int | None = None,
    win_length: int | None = None,
    window: str = "hann",
    threshold_db: float = -20.0,
    prop_decrease: float = 1.0,
    n_std_thresh: float = 1.5,
    time_constant_s: float = 0.05,
    freq_smooth_hz: float = 100.0,
    time_smooth_ms: float = 10.0,
    stationary: bool = True,
    n_grad_freq: int = 2,
    n_grad_time: int = 4,
) -> mx.array:
    """
    Apply spectral gating noise reduction.

    Non-neural noise reduction using frequency-dependent thresholding.
    Suitable for stationary noise (HVAC, fans, hiss) and mild enhancement.

    Parameters
    ----------
    audio : mx.array
        Input audio. Shape: (samples,) or (batch, samples).
    noise_profile : mx.array, optional
        Noise-only audio segment for threshold estimation.
        If None, noise floor is estimated from quietest frames.
    sr : int, default=22050
        Sample rate.
    n_fft : int, default=2048
        FFT size.
    hop_length : int, optional
        Hop length. Default: n_fft // 4.
    win_length : int, optional
        Window length. Default: n_fft.
    window : str, default='hann'
        Window function.
    threshold_db : float, default=-20.0
        Threshold in dB below noise floor for gating.
    prop_decrease : float, default=1.0
        Proportion of noise to remove (0-1). 1.0 = full removal.
    n_std_thresh : float, default=1.5
        Number of standard deviations above mean for threshold.
    time_constant_s : float, default=0.05
        Time constant for adaptive noise estimation.
    freq_smooth_hz : float, default=100.0
        Frequency smoothing bandwidth (Hz).
    time_smooth_ms : float, default=10.0
        Time smoothing window (ms).
    stationary : bool, default=True
        If True, use stationary noise estimation.
        If False, use adaptive noise tracking.
    n_grad_freq : int, default=2
        Frequency gradient smoothing width.
    n_grad_time : int, default=4
        Time gradient smoothing width.

    Returns
    -------
    mx.array
        Enhanced audio with same shape as input.

    Notes
    -----
    This implements spectral gating similar to the noisereduce library.
    For best results:
    - Provide a noise_profile from a noise-only segment
    - Adjust threshold_db based on SNR (-20 for mild, -30 for aggressive)
    - Use prop_decrease < 1.0 for more natural results

    Examples
    --------
    >>> # Basic noise reduction
    >>> clean = spectral_gate(noisy_audio, sr=22050)

    >>> # With noise profile from first second
    >>> noise_sample = audio[:sr]  # First second as noise
    >>> clean = spectral_gate(audio, noise_profile=noise_sample, sr=sr)

    >>> # Gentle noise reduction (preserve more signal)
    >>> clean = spectral_gate(audio, threshold_db=-15, prop_decrease=0.8)
    """
    if hop_length is None:
        hop_length = n_fft // 4
    if win_length is None:
        win_length = n_fft

    validate_positive(n_fft, "n_fft")
    validate_positive(hop_length, "hop_length")
    validate_range(prop_decrease, "prop_decrease", 0.0, 1.0)

    # Handle batched input
    input_is_1d = audio.ndim == 1
    if input_is_1d:
        audio = audio[None, :]

    audio_np = np.array(audio)
    batch_size, n_samples = audio_np.shape

    # Compute STFT
    audio_stft = stft(
        audio,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
    )

    # Get magnitude and phase
    audio_mag = np.array(magnitude(audio_stft))
    audio_phase = np.array(phase(audio_stft))

    # Estimate noise threshold
    if noise_profile is not None:
        # Use provided noise profile
        noise_stft = stft(
            noise_profile if noise_profile.ndim > 1 else noise_profile[None, :],
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=True,
        )
        noise_mag = np.array(magnitude(noise_stft))

        # Mean and std across time
        noise_mean = np.mean(noise_mag, axis=-1, keepdims=True)
        noise_std = np.std(noise_mag, axis=-1, keepdims=True)
        noise_thresh = noise_mean + n_std_thresh * noise_std
    else:
        # Estimate noise from signal (use quietest 10% of frames)
        # audio_mag shape: (batch, freq_bins, frames)
        if stationary:
            # Use global statistics
            # Calculate energy per frame (mean over frequency bins)
            frame_energy = np.mean(audio_mag ** 2, axis=1)  # (batch, frames)
            n_frames = frame_energy.shape[-1]
            n_quiet = max(1, int(0.1 * n_frames))

            noise_thresh_list = []
            for b in range(batch_size):
                quiet_idx = np.argsort(frame_energy[b])[:n_quiet]
                # Index with [:, quiet_idx] to get (freq_bins, n_quiet)
                quiet_frames = audio_mag[b][:, quiet_idx]
                # Mean over quiet frames (axis=1) to get (freq_bins,)
                noise_mean = np.mean(quiet_frames, axis=1, keepdims=True)
                noise_std = np.std(quiet_frames, axis=1, keepdims=True)
                # Shape: (freq_bins, 1)
                noise_thresh_list.append(noise_mean + n_std_thresh * noise_std)
            # Stack for batch dim: (batch, freq_bins, 1)
            noise_thresh = np.stack(noise_thresh_list, axis=0)
        else:
            # Adaptive noise tracking (running minimum)
            # Time constant in frames
            tc_frames = max(1, int(time_constant_s * sr / hop_length))

            noise_thresh = minimum_filter1d(
                audio_mag, size=tc_frames, axis=-1, mode="nearest"
            )
            noise_thresh = noise_thresh * (1 + n_std_thresh)

    # Apply threshold offset in dB
    threshold_linear = 10 ** (threshold_db / 20.0)
    noise_thresh = noise_thresh * threshold_linear

    # Compute soft mask
    # mask = 1 where signal > threshold, smoothly decreasing below
    mask = (audio_mag - noise_thresh) / (audio_mag + DIVISION_EPSILON)
    mask = np.clip(mask, 0.0, 1.0)

    # Apply smoothing to mask
    if freq_smooth_hz > 0 or time_smooth_ms > 0:
        # Frequency smoothing (convert Hz to bins)
        freq_bins = int(freq_smooth_hz * n_fft / sr)
        freq_bins = max(1, freq_bins)

        # Time smoothing (convert ms to frames)
        time_frames = int(time_smooth_ms * sr / (1000 * hop_length))
        time_frames = max(1, time_frames)

        # Apply uniform filter across entire batch at once
        # size=(1, freq_bins, time_frames) means no filtering across batch dim
        mask = uniform_filter(mask, size=(1, freq_bins, time_frames), mode="nearest")

    # Gradient smoothing for smoother transitions
    if n_grad_freq > 1 or n_grad_time > 1:
        # Apply uniform filter across entire batch at once
        mask = uniform_filter(mask, size=(1, n_grad_freq, n_grad_time), mode="nearest")

    # Apply prop_decrease (don't remove all noise)
    mask = 1.0 - prop_decrease * (1.0 - mask)

    # Apply mask to magnitude
    enhanced_mag = audio_mag * mask

    # Reconstruct complex spectrogram
    enhanced_stft = enhanced_mag * np.exp(1j * audio_phase)
    enhanced_stft = mx.array(enhanced_stft.astype(np.complex64))

    # Inverse STFT
    enhanced = istft(
        enhanced_stft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        length=n_samples,
    )

    # Remove batch dimension if input was 1D
    if input_is_1d:
        enhanced = enhanced[0]

    return enhanced


def spectral_gate_adaptive(
    audio: mx.array,
    *,
    sr: int = 22050,
    n_fft: int = 2048,
    hop_length: int | None = None,
    threshold_db: float = -20.0,
    prop_decrease: float = 1.0,
    look_ahead_ms: float = 100.0,
    look_back_ms: float = 100.0,
) -> mx.array:
    """
    Adaptive spectral gating with local noise estimation.

    This variant estimates the noise floor locally in time, making it
    suitable for non-stationary noise that varies over time.

    Parameters
    ----------
    audio : mx.array
        Input audio. Shape: (samples,) or (batch, samples).
    sr : int, default=22050
        Sample rate.
    n_fft : int, default=2048
        FFT size.
    hop_length : int, optional
        Hop length. Default: n_fft // 4.
    threshold_db : float, default=-20.0
        Threshold in dB for noise gating.
    prop_decrease : float, default=1.0
        Proportion of noise to remove (0-1).
    look_ahead_ms : float, default=100.0
        Look-ahead window for noise estimation (ms).
    look_back_ms : float, default=100.0
        Look-back window for noise estimation (ms).

    Returns
    -------
    mx.array
        Enhanced audio.

    Examples
    --------
    >>> # For time-varying noise
    >>> clean = spectral_gate_adaptive(audio, sr=22050)
    """
    if hop_length is None:
        hop_length = n_fft // 4

    validate_positive(n_fft, "n_fft")
    validate_positive(hop_length, "hop_length")
    validate_range(prop_decrease, "prop_decrease", 0.0, 1.0)

    # Convert time windows to frames
    look_ahead_frames = max(1, int(look_ahead_ms * sr / (1000 * hop_length)))
    look_back_frames = max(1, int(look_back_ms * sr / (1000 * hop_length)))
    total_window = look_ahead_frames + look_back_frames + 1

    # Handle batched input
    input_is_1d = audio.ndim == 1
    if input_is_1d:
        audio = audio[None, :]

    n_samples = audio.shape[-1]

    # Compute STFT
    audio_stft = stft(
        audio,
        n_fft=n_fft,
        hop_length=hop_length,
        center=True,
    )

    audio_mag = np.array(magnitude(audio_stft))
    audio_phase = np.array(phase(audio_stft))

    batch_size, n_freq, n_frames = audio_mag.shape

    # Compute local minimum as noise estimate
    noise_floor = minimum_filter1d(
        audio_mag, size=total_window, axis=-1, mode="nearest"
    )

    # Apply threshold
    threshold_linear = 10 ** (threshold_db / 20.0)
    noise_thresh = noise_floor * (1.0 + threshold_linear)

    # Compute mask
    mask = (audio_mag - noise_thresh) / (audio_mag + DIVISION_EPSILON)
    mask = np.clip(mask, 0.0, 1.0)

    # Apply prop_decrease
    mask = 1.0 - prop_decrease * (1.0 - mask)

    # Smooth mask (vectorized across batch)
    mask = uniform_filter(mask, size=(1, 3, 5), mode="nearest")

    # Apply mask and reconstruct
    enhanced_mag = audio_mag * mask
    enhanced_stft = enhanced_mag * np.exp(1j * audio_phase)
    enhanced_stft = mx.array(enhanced_stft.astype(np.complex64))

    enhanced = istft(
        enhanced_stft,
        hop_length=hop_length,
        center=True,
        length=n_samples,
    )

    if input_is_1d:
        enhanced = enhanced[0]

    return enhanced
