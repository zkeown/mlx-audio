"""Audio quality metrics for validation.

Implements SDR (Signal-to-Distortion Ratio) and SI-SDR (Scale-Invariant SDR)
for evaluating source separation quality.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np


def si_sdr(estimate: mx.array | np.ndarray, reference: mx.array | np.ndarray) -> float:
    """Compute Scale-Invariant Signal-to-Distortion Ratio.

    SI-SDR is defined as:
        SI-SDR = 10 * log10(||s_target||^2 / ||e_noise||^2)

    where s_target is the projection of estimate onto reference,
    and e_noise is the residual.

    Args:
        estimate: Estimated signal [..., samples]
        reference: Reference/ground truth signal [..., samples]

    Returns:
        SI-SDR in dB (higher is better)

    Example:
        >>> estimate = model_output[0, 3]  # vocals stem
        >>> reference = ground_truth_vocals
        >>> score = si_sdr(estimate, reference)
        >>> print(f"SI-SDR: {score:.2f} dB")
    """
    if isinstance(estimate, mx.array):
        estimate = np.array(estimate)
    if isinstance(reference, mx.array):
        reference = np.array(reference)

    # Flatten to 1D
    estimate = estimate.flatten().astype(np.float64)
    reference = reference.flatten().astype(np.float64)

    # Remove mean (zero-mean signals)
    estimate = estimate - np.mean(estimate)
    reference = reference - np.mean(reference)

    # Compute scaling factor: alpha = <s, s_hat> / <s, s>
    dot = np.dot(reference, estimate)
    s_ref_energy = np.dot(reference, reference) + 1e-8

    # s_target = alpha * reference
    alpha = dot / s_ref_energy
    s_target = alpha * reference

    # e_noise = estimate - s_target
    e_noise = estimate - s_target

    # SI-SDR = 10 * log10(||s_target||^2 / ||e_noise||^2)
    si_sdr_value = 10 * np.log10(
        (np.dot(s_target, s_target) + 1e-8) / (np.dot(e_noise, e_noise) + 1e-8)
    )

    return float(si_sdr_value)


def sdr(estimate: mx.array | np.ndarray, reference: mx.array | np.ndarray) -> float:
    """Compute Signal-to-Distortion Ratio (simplified BSSEval-style).

    SDR = 10 * log10(||reference||^2 / ||reference - estimate||^2)

    This is a simplified SDR that doesn't account for permitted distortions.
    For full BSSEval metrics, use the museval library.

    Args:
        estimate: Estimated signal [..., samples]
        reference: Reference/ground truth signal [..., samples]

    Returns:
        SDR in dB (higher is better)
    """
    if isinstance(estimate, mx.array):
        estimate = np.array(estimate)
    if isinstance(reference, mx.array):
        reference = np.array(reference)

    estimate = estimate.flatten().astype(np.float64)
    reference = reference.flatten().astype(np.float64)

    noise = reference - estimate
    ref_energy = np.dot(reference, reference) + 1e-8
    noise_energy = np.dot(noise, noise) + 1e-8

    return float(10 * np.log10(ref_energy / noise_energy))


def snr(signal: mx.array | np.ndarray, noise: mx.array | np.ndarray) -> float:
    """Compute Signal-to-Noise Ratio.

    Args:
        signal: Clean signal
        noise: Noise component

    Returns:
        SNR in dB
    """
    if isinstance(signal, mx.array):
        signal = np.array(signal)
    if isinstance(noise, mx.array):
        noise = np.array(noise)

    signal = signal.flatten().astype(np.float64)
    noise = noise.flatten().astype(np.float64)

    signal_power = np.dot(signal, signal) + 1e-8
    noise_power = np.dot(noise, noise) + 1e-8

    return float(10 * np.log10(signal_power / noise_power))


def correlation(
    estimate: mx.array | np.ndarray, reference: mx.array | np.ndarray
) -> float:
    """Compute Pearson correlation coefficient.

    Args:
        estimate: Estimated signal
        reference: Reference signal

    Returns:
        Correlation coefficient (-1 to 1, higher absolute value is better)
    """
    if isinstance(estimate, mx.array):
        estimate = np.array(estimate)
    if isinstance(reference, mx.array):
        reference = np.array(reference)

    estimate = estimate.flatten()
    reference = reference.flatten()

    return float(np.corrcoef(estimate, reference)[0, 1])


def mae(estimate: mx.array | np.ndarray, reference: mx.array | np.ndarray) -> float:
    """Compute Mean Absolute Error.

    Args:
        estimate: Estimated signal
        reference: Reference signal

    Returns:
        MAE (lower is better)
    """
    if isinstance(estimate, mx.array):
        estimate = np.array(estimate)
    if isinstance(reference, mx.array):
        reference = np.array(reference)

    return float(np.mean(np.abs(estimate - reference)))


def mse(estimate: mx.array | np.ndarray, reference: mx.array | np.ndarray) -> float:
    """Compute Mean Squared Error.

    Args:
        estimate: Estimated signal
        reference: Reference signal

    Returns:
        MSE (lower is better)
    """
    if isinstance(estimate, mx.array):
        estimate = np.array(estimate)
    if isinstance(reference, mx.array):
        reference = np.array(reference)

    return float(np.mean((estimate - reference) ** 2))


class SeparationMetrics:
    """Compute comprehensive separation metrics for all stems.

    Example:
        >>> metrics = SeparationMetrics()
        >>> results = metrics.evaluate(
        ...     estimates=model_output[0],  # [4, 2, samples]
        ...     references=ground_truth,     # dict or [4, 2, samples]
        ...     stem_names=["drums", "bass", "other", "vocals"]
        ... )
        >>> print(results)
    """

    STEM_NAMES = ["drums", "bass", "other", "vocals"]

    def evaluate(
        self,
        estimates: mx.array | np.ndarray,
        references: mx.array | np.ndarray | dict[str, np.ndarray],
        stem_names: list[str] | None = None,
    ) -> dict[str, dict[str, float]]:
        """Evaluate separation quality for all stems.

        Args:
            estimates: Model output [num_stems, channels, samples]
            references: Ground truth stems (array or dict by stem name)
            stem_names: Names of stems (default: drums, bass, other, vocals)

        Returns:
            Dict mapping stem names to metric dicts
        """
        if stem_names is None:
            stem_names = self.STEM_NAMES

        if isinstance(estimates, mx.array):
            estimates = np.array(estimates)

        if isinstance(references, dict):
            ref_array = np.stack([references[name] for name in stem_names])
        elif isinstance(references, mx.array):
            ref_array = np.array(references)
        else:
            ref_array = references

        results = {}
        for i, stem in enumerate(stem_names):
            est = estimates[i]
            ref = ref_array[i]

            results[stem] = {
                "si_sdr": si_sdr(est, ref),
                "sdr": sdr(est, ref),
                "correlation": correlation(est, ref),
                "mae": mae(est, ref),
            }

        # Compute mean across stems
        results["mean"] = {
            metric: np.mean([results[stem][metric] for stem in stem_names])
            for metric in ["si_sdr", "sdr", "correlation", "mae"]
        }

        return results

    def compare(
        self,
        estimates_a: mx.array | np.ndarray,
        estimates_b: mx.array | np.ndarray,
        references: mx.array | np.ndarray | dict[str, np.ndarray],
        stem_names: list[str] | None = None,
    ) -> dict[str, dict[str, float]]:
        """Compare two sets of estimates against the same reference.

        Useful for comparing streaming vs batch processing.

        Args:
            estimates_a: First model output (e.g., streaming)
            estimates_b: Second model output (e.g., batch)
            references: Ground truth stems

        Returns:
            Dict with metrics for both and their differences
        """
        results_a = self.evaluate(estimates_a, references, stem_names)
        results_b = self.evaluate(estimates_b, references, stem_names)

        comparison = {
            "a": results_a,
            "b": results_b,
            "diff": {},
        }

        stem_names = stem_names or self.STEM_NAMES
        for stem in stem_names + ["mean"]:
            comparison["diff"][stem] = {
                metric: results_a[stem][metric] - results_b[stem][metric]
                for metric in ["si_sdr", "sdr", "correlation"]
            }

        return comparison
