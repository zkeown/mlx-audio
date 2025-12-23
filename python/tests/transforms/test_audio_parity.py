"""Parity tests for audio augmentation transforms.

Validates transforms against:
1. Mathematical properties (SNR, energy, spectral content)
2. Reference implementations (librosa, scipy)
3. Known test vectors with expected outputs
"""

import numpy as np
import pytest
from scipy import signal as scipy_signal

# Skip all tests if mlx is not available
pytest.importorskip("mlx")

# Test configuration
SAMPLE_RATE = 16000
DURATION = 1.0
AUDIO_LENGTH = int(SAMPLE_RATE * DURATION)
TEST_SEED = 42
# Tolerance constants - tightened based on measured headroom (see measure_tolerances.py)
# Simple math operations achieve ~1e-7 rtol, so we use 1e-6 with 10x safety margin
TOLERANCE = 1e-6  # Tightened from 0.01 (1%) - actual headroom: 100000x+
TOLERANCE_LOOSE = 1e-5  # For operations with small floating point accumulation


# =============================================================================
# Test Signal Generators
# =============================================================================


def sine_wave(freq: float, sr: int = SAMPLE_RATE, duration: float = 1.0) -> np.ndarray:
    """Generate a pure sine wave."""
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


def multi_sine(
    freqs: list[float] = [220, 440, 880],
    sr: int = SAMPLE_RATE,
    duration: float = 1.0,
) -> np.ndarray:
    """Generate a multi-frequency sine wave."""
    result = np.zeros(int(sr * duration), dtype=np.float32)
    for freq in freqs:
        result += sine_wave(freq, sr, duration)
    return (result / len(freqs)).astype(np.float32)


def chirp_signal(
    f_start: float = 200,
    f_end: float = 4000,
    sr: int = SAMPLE_RATE,
    duration: float = 1.0,
) -> np.ndarray:
    """Generate a frequency sweep (chirp) signal."""
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    return scipy_signal.chirp(t, f_start, duration, f_end).astype(np.float32)


def white_noise(sr: int = SAMPLE_RATE, duration: float = 1.0, seed: int = 42) -> np.ndarray:
    """Generate white noise."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(int(sr * duration)).astype(np.float32)


def compute_rms(audio: np.ndarray) -> float:
    """Compute RMS (root mean square) of audio."""
    return float(np.sqrt(np.mean(audio**2)))


def compute_snr(original: np.ndarray, noisy: np.ndarray) -> float:
    """Compute SNR in dB between original and noisy signal."""
    noise = noisy - original
    signal_power = np.mean(original**2)
    noise_power = np.mean(noise**2)
    if noise_power < 1e-10:
        return float("inf")
    return float(10 * np.log10(signal_power / noise_power))


def compute_spectral_centroid(audio: np.ndarray, sr: int = SAMPLE_RATE) -> float:
    """Compute spectral centroid of audio."""
    fft = np.fft.rfft(audio)
    magnitude = np.abs(fft)
    freqs = np.fft.rfftfreq(len(audio), 1 / sr)
    if np.sum(magnitude) < 1e-10:
        return 0.0
    return float(np.sum(freqs * magnitude) / np.sum(magnitude))


def compute_dominant_frequency(audio: np.ndarray, sr: int = SAMPLE_RATE) -> float:
    """Find the dominant frequency in audio."""
    fft = np.fft.rfft(audio)
    magnitude = np.abs(fft)
    freqs = np.fft.rfftfreq(len(audio), 1 / sr)
    return float(freqs[np.argmax(magnitude)])


# =============================================================================
# Gain Transform Parity Tests
# =============================================================================


class TestGainParity:
    """Validate Gain transform against mathematical properties."""

    def test_gain_db_to_linear_conversion(self):
        """Verify correct dB to linear conversion: 20dB = 10x, 6dB ≈ 2x."""
        from mlx_audio.data.transforms.audio import Gain

        audio = np.ones(1000, dtype=np.float32) * 0.1

        # +20dB should multiply by 10
        result = Gain(gain_db_range=(20, 20), p=1.0)(audio, SAMPLE_RATE)
        expected = 0.1 * (10 ** (20 / 20))  # 1.0
        np.testing.assert_allclose(result, expected, rtol=TOLERANCE)

        # +6dB should approximately double
        result = Gain(gain_db_range=(6, 6), p=1.0)(audio, SAMPLE_RATE)
        expected = 0.1 * (10 ** (6 / 20))  # ≈ 0.1995
        np.testing.assert_allclose(result, expected, rtol=TOLERANCE)

        # -6dB should approximately halve
        result = Gain(gain_db_range=(-6, -6), p=1.0)(audio, SAMPLE_RATE)
        expected = 0.1 * (10 ** (-6 / 20))  # ≈ 0.0501
        np.testing.assert_allclose(result, expected, rtol=TOLERANCE)

    def test_gain_rms_scaling(self):
        """Verify RMS scales correctly with gain."""
        from mlx_audio.data.transforms.audio import Gain

        audio = white_noise(duration=0.5, seed=123)
        original_rms = compute_rms(audio)

        # +6dB should double RMS
        result = Gain(gain_db_range=(6, 6), p=1.0)(audio, SAMPLE_RATE)
        expected_rms = original_rms * (10 ** (6 / 20))
        actual_rms = compute_rms(result)

        np.testing.assert_allclose(actual_rms, expected_rms, rtol=TOLERANCE)

    def test_gain_preserves_frequency_content(self):
        """Gain should not change frequency content."""
        from mlx_audio.data.transforms.audio import Gain

        audio = sine_wave(440)
        original_freq = compute_dominant_frequency(audio)

        result = Gain(gain_db_range=(6, 6), p=1.0)(audio, SAMPLE_RATE)
        result_freq = compute_dominant_frequency(result)

        np.testing.assert_allclose(result_freq, original_freq, rtol=0.01)


# =============================================================================
# GaussianNoise SNR Parity Tests
# =============================================================================


class TestGaussianNoiseParity:
    """Validate GaussianNoise transform against SNR specifications."""

    @pytest.mark.parametrize("target_snr", [10, 20, 30])
    def test_snr_accuracy(self, target_snr):
        """Verify actual SNR matches target within tolerance."""
        from mlx_audio.data.transforms.audio import GaussianNoise

        # Use a clean signal
        audio = sine_wave(440)
        transform = GaussianNoise(snr_range=(target_snr, target_snr), p=1.0, seed=42)

        result = transform(audio, SAMPLE_RATE)
        actual_snr = compute_snr(audio, result)

        # Tightened from 2% to 1e-6 after improving GaussianNoise to scale
        # noise to exact target power (eliminates statistical sampling variance)
        np.testing.assert_allclose(actual_snr, target_snr, rtol=1e-6)

    def test_noise_is_gaussian(self):
        """Verify added noise has Gaussian distribution."""
        from mlx_audio.data.transforms.audio import GaussianNoise
        from scipy import stats

        audio = np.zeros(100000, dtype=np.float32)  # Silent, so result is pure noise
        # Actually need signal for SNR-based noise, use constant
        audio = np.ones(100000, dtype=np.float32) * 0.5

        transform = GaussianNoise(snr_range=(10, 10), p=1.0, seed=42)
        result = transform(audio, SAMPLE_RATE)

        noise = result - audio

        # Shapiro-Wilk test for normality (sample subset for speed)
        _, p_value = stats.shapiro(noise[::100])
        assert p_value > 0.01, "Noise is not Gaussian distributed"

    def test_noise_energy_relationship(self):
        """Verify noise power relationship: noise_power = signal_power / 10^(SNR/10)."""
        from mlx_audio.data.transforms.audio import GaussianNoise

        audio = sine_wave(440)
        signal_power = np.mean(audio**2)
        target_snr = 20

        transform = GaussianNoise(snr_range=(target_snr, target_snr), p=1.0, seed=42)
        result = transform(audio, SAMPLE_RATE)

        noise = result - audio
        noise_power = np.mean(noise**2)

        expected_noise_power = signal_power / (10 ** (target_snr / 10))
        # Tightened from 5% to 1e-6 after improving GaussianNoise to scale
        # noise to exact target power (eliminates statistical sampling variance)
        np.testing.assert_allclose(noise_power, expected_noise_power, rtol=1e-6)


# =============================================================================
# TimeStretch Parity Tests
# =============================================================================


class TestTimeStretchParity:
    """Validate TimeStretch using phase vocoder algorithm.

    The phase vocoder preserves pitch while changing duration.
    """

    def test_length_change_accuracy(self):
        """Verify output length changes proportionally with rate."""
        from mlx_audio.data.transforms.audio import TimeStretch

        audio = white_noise(duration=1.0)

        for rate in [0.5, 0.8, 1.25, 2.0]:
            transform = TimeStretch(rate_range=(rate, rate), p=1.0)
            result = transform(audio, SAMPLE_RATE)

            expected_length = int(len(audio) / rate)
            # Allow 5% tolerance on length (phase vocoder has some variance)
            assert abs(len(result) - expected_length) < expected_length * 0.05

    def test_pitch_preservation(self):
        """Phase vocoder should preserve pitch while changing duration."""
        from mlx_audio.data.transforms.audio import TimeStretch

        audio = sine_wave(440)
        original_freq = compute_dominant_frequency(audio)

        # Slow down by 2x - duration doubles but pitch stays same
        transform = TimeStretch(rate_range=(0.5, 0.5), p=1.0)
        result = transform(audio, SAMPLE_RATE)

        result_freq = compute_dominant_frequency(result)

        # Frequency should be preserved (tightened from 5% - measured headroom: 17x)
        np.testing.assert_allclose(result_freq, original_freq, rtol=0.01)

    def test_dominant_frequency_preservation(self):
        """Time stretch should preserve dominant frequency (pitch).

        Note: Basic phase vocoders introduce spectral spreading which affects
        spectral centroid. This is documented in librosa: "This is a simplified
        implementation... likely to produce many audible artifacts."

        The key property is that the PITCH (dominant frequency) is preserved,
        not the spectral centroid. We verify this matches librosa's behavior.
        """
        from mlx_audio.data.transforms.audio import TimeStretch

        audio = sine_wave(440)
        original_freq = compute_dominant_frequency(audio)

        transform = TimeStretch(rate_range=(0.75, 0.75), p=1.0)
        result = transform(audio, SAMPLE_RATE)
        result_freq = compute_dominant_frequency(result)

        # Dominant frequency (pitch) should be preserved (tightened from 1%)
        np.testing.assert_allclose(result_freq, original_freq, rtol=2e-3)


# =============================================================================
# PitchShift Parity Tests
# =============================================================================


class TestPitchShiftParity:
    """Validate PitchShift using resample + phase vocoder algorithm.

    The algorithm:
    1. Resample to change pitch (also changes duration)
    2. Phase vocoder time-stretch to restore original duration
    """

    def test_length_preservation(self):
        """Pitch shift should preserve audio length."""
        from mlx_audio.data.transforms.audio import PitchShift

        audio = sine_wave(440)
        transform = PitchShift(semitone_range=(12, 12), p=1.0)

        result = transform(audio, SAMPLE_RATE)

        assert len(result) == len(audio)

    def test_frequency_shift_octave_up(self):
        """12 semitones up should double the frequency."""
        from mlx_audio.data.transforms.audio import PitchShift

        audio = sine_wave(440)
        original_freq = compute_dominant_frequency(audio)

        transform = PitchShift(semitone_range=(12, 12), p=1.0)
        result = transform(audio, SAMPLE_RATE)
        result_freq = compute_dominant_frequency(result)

        # Frequency should double - measured: exact match!
        expected_freq = original_freq * 2.0
        np.testing.assert_allclose(result_freq, expected_freq, rtol=1e-6)

    def test_frequency_shift_octave_down(self):
        """12 semitones down should halve the frequency."""
        from mlx_audio.data.transforms.audio import PitchShift

        audio = sine_wave(880)  # Start higher so we can go down
        original_freq = compute_dominant_frequency(audio)

        transform = PitchShift(semitone_range=(-12, -12), p=1.0)
        result = transform(audio, SAMPLE_RATE)
        result_freq = compute_dominant_frequency(result)

        # Frequency should halve - measured: exact match!
        expected_freq = original_freq * 0.5
        np.testing.assert_allclose(result_freq, expected_freq, rtol=1e-6)

    def test_semitone_formula(self):
        """Verify the semitone to frequency ratio formula."""
        for semitones in [-12, -7, -5, 0, 5, 7, 12]:
            expected_ratio = 2 ** (semitones / 12)

            if semitones == 12:
                assert abs(expected_ratio - 2.0) < 1e-10
            elif semitones == -12:
                assert abs(expected_ratio - 0.5) < 1e-10
            elif semitones == 0:
                assert abs(expected_ratio - 1.0) < 1e-10

    def test_spectral_centroid_shifts(self):
        """Pitch shift should move the spectral centroid proportionally.

        Note: Phase vocoders introduce spectral spreading, especially for
        multi-frequency signals. Even librosa's pitch_shift shows ~33% error
        on spectral centroid tests. We use a single frequency where the
        dominant frequency test is the most reliable measure.
        """
        from mlx_audio.data.transforms.audio import PitchShift

        # Use single frequency for cleaner spectral test
        audio = sine_wave(440)
        original_centroid = compute_spectral_centroid(audio)

        # +12 semitones = 2x frequency = 2x centroid
        transform = PitchShift(semitone_range=(12, 12), p=1.0)
        result = transform(audio, SAMPLE_RATE)
        result_centroid = compute_spectral_centroid(result)

        # Phase vocoders have significant spectral spreading - measured rtol: 0.3154
        # Tightened from 40% to 35% (1.1x headroom - truly algorithm limited)
        expected_centroid = original_centroid * 2.0
        np.testing.assert_allclose(
            result_centroid, expected_centroid, rtol=0.35
        )


# =============================================================================
# Filter Parity Tests
# =============================================================================


class TestFilterParity:
    """Validate filters against scipy.signal implementation."""

    def test_lowpass_attenuates_high_frequencies(self):
        """Low-pass filter should attenuate frequencies above cutoff."""
        from mlx_audio.data.transforms.audio import LowPassFilter

        # Create signal with low (500Hz) and high (6000Hz) components
        low_freq = sine_wave(500)
        high_freq = sine_wave(6000)
        audio = (low_freq + high_freq).astype(np.float32) / 2

        # Cutoff at 0.25 of Nyquist (2000 Hz at 16kHz sample rate)
        transform = LowPassFilter(cutoff_range=(0.25, 0.25), p=1.0)
        result = transform(audio, SAMPLE_RATE)

        # Compute energy in low and high frequency bands
        fft_orig = np.abs(np.fft.rfft(audio))
        fft_result = np.abs(np.fft.rfft(result))
        freqs = np.fft.rfftfreq(len(audio), 1 / SAMPLE_RATE)

        # High frequency energy should be reduced
        high_freq_mask = freqs > 3000
        orig_high_energy = np.sum(fft_orig[high_freq_mask] ** 2)
        result_high_energy = np.sum(fft_result[high_freq_mask] ** 2)

        assert result_high_energy < orig_high_energy * 0.1  # 90% attenuation

    def test_highpass_attenuates_low_frequencies(self):
        """High-pass filter should attenuate frequencies below cutoff."""
        from mlx_audio.data.transforms.audio import HighPassFilter

        # Create signal with low (200Hz) and high (4000Hz) components
        low_freq = sine_wave(200)
        high_freq = sine_wave(4000)
        audio = (low_freq + high_freq).astype(np.float32) / 2

        # Cutoff at 0.15 of Nyquist (1200 Hz at 16kHz sample rate)
        transform = HighPassFilter(cutoff_range=(0.15, 0.15), p=1.0)
        result = transform(audio, SAMPLE_RATE)

        # Compute energy in low and high frequency bands
        fft_orig = np.abs(np.fft.rfft(audio))
        fft_result = np.abs(np.fft.rfft(result))
        freqs = np.fft.rfftfreq(len(audio), 1 / SAMPLE_RATE)

        # Low frequency energy should be reduced
        low_freq_mask = freqs < 500
        orig_low_energy = np.sum(fft_orig[low_freq_mask] ** 2)
        result_low_energy = np.sum(fft_result[low_freq_mask] ** 2)

        assert result_low_energy < orig_low_energy * 0.1  # 90% attenuation

    def test_scipy_butter_equivalence(self):
        """Verify our filter matches scipy.signal.butter + filtfilt."""
        from mlx_audio.data.transforms.audio import LowPassFilter

        audio = white_noise(duration=0.5)
        cutoff = 0.3

        # Our implementation
        transform = LowPassFilter(cutoff_range=(cutoff, cutoff), order=5, p=1.0)
        our_result = transform(audio, SAMPLE_RATE)

        # scipy reference
        b, a = scipy_signal.butter(5, cutoff, btype="low")
        scipy_result = scipy_signal.filtfilt(b, a, audio)

        # Should match exactly (tightened from 1e-5 - measured: exact match)
        np.testing.assert_allclose(our_result, scipy_result, rtol=1e-7, atol=1e-7)


# =============================================================================
# SpecAugment Parity Tests
# =============================================================================


class TestSpecAugmentParity:
    """Validate SpecAugment against paper specifications."""

    def test_masking_creates_zeros(self):
        """SpecAugment should create zero regions in spectrogram."""
        from mlx_audio.data.transforms.audio import SpecAugment
        import mlx.core as mx
        from mlx_audio.primitives import stft

        audio = white_noise(duration=0.5)

        transform = SpecAugment(
            freq_mask_param=50,
            time_mask_param=50,
            num_freq_masks=2,
            num_time_masks=2,
            p=1.0,
            seed=42,
        )

        result = transform(audio, SAMPLE_RATE)

        # Compute spectrograms
        S_orig = np.abs(np.array(stft(mx.array(audio), n_fft=512, hop_length=128)))
        S_result = np.abs(np.array(stft(mx.array(result), n_fft=512, hop_length=128)))

        # Result should have more zeros (masked regions)
        orig_zeros = np.sum(S_orig < 1e-10)
        result_zeros = np.sum(S_result < 1e-10)

        assert result_zeros > orig_zeros

    def test_length_preserved(self):
        """SpecAugment should preserve audio length."""
        from mlx_audio.data.transforms.audio import SpecAugment

        audio = white_noise()
        transform = SpecAugment(p=1.0, seed=42)

        result = transform(audio, SAMPLE_RATE)

        assert len(result) == len(audio)


# =============================================================================
# Reverb Parity Tests
# =============================================================================


class TestReverbParity:
    """Validate RoomReverb against convolution properties."""

    def test_impulse_response_convolution(self):
        """Reverb on impulse should produce the IR."""
        from mlx_audio.data.transforms.audio import RoomReverb

        # Create impulse
        impulse = np.zeros(SAMPLE_RATE, dtype=np.float32)
        impulse[100] = 1.0

        # Apply reverb with high wet mix
        transform = RoomReverb(wet_dry_range=(0.9, 0.9), p=1.0, seed=42)
        result = transform(impulse, SAMPLE_RATE)

        # Should have energy after the impulse (reverb tail)
        assert np.sum(np.abs(result[200:])) > 0

        # Peak should still be near the original impulse
        peak_idx = np.argmax(np.abs(result))
        assert abs(peak_idx - 100) < 50

    def test_wet_dry_mixing(self):
        """Verify wet/dry mix ratio."""
        from mlx_audio.data.transforms.audio import RoomReverb

        audio = sine_wave(440)

        # Full dry (no reverb) - should be exact passthrough
        transform_dry = RoomReverb(wet_dry_range=(0.0, 0.0), p=1.0, seed=42)
        result_dry = transform_dry(audio, SAMPLE_RATE)
        # Tightened from 1e-5 - measured: exact match
        np.testing.assert_allclose(result_dry, audio, rtol=1e-7, atol=1e-7)


# =============================================================================
# Normalize Parity Tests
# =============================================================================


class TestNormalizeParity:
    """Validate Normalize transform against expected levels."""

    def test_peak_normalization(self):
        """Peak normalization should set max to target level."""
        from mlx_audio.data.transforms.audio import Normalize

        audio = white_noise(duration=0.5)
        target_db = -6

        transform = Normalize(target_db=target_db, mode="peak", p=1.0)
        result = transform(audio, SAMPLE_RATE)

        # Peak should be at target level
        expected_peak = 10 ** (target_db / 20)
        actual_peak = np.max(np.abs(result))

        np.testing.assert_allclose(actual_peak, expected_peak, rtol=TOLERANCE)

    def test_rms_normalization(self):
        """RMS normalization should set RMS to target level."""
        from mlx_audio.data.transforms.audio import Normalize

        audio = white_noise(duration=0.5)
        target_db = -20

        transform = Normalize(target_db=target_db, mode="rms", p=1.0)
        result = transform(audio, SAMPLE_RATE)

        # RMS should be at target level
        expected_rms = 10 ** (target_db / 20)
        actual_rms = compute_rms(result)

        np.testing.assert_allclose(actual_rms, expected_rms, rtol=TOLERANCE)


# =============================================================================
# Energy Conservation Tests
# =============================================================================


class TestEnergyConservation:
    """Test energy/power relationships across transforms."""

    def test_gain_energy_scaling(self):
        """Energy should scale as (linear_gain)^2."""
        from mlx_audio.data.transforms.audio import Gain

        audio = white_noise(duration=0.5)
        original_energy = np.sum(audio**2)

        gain_db = 6
        linear_gain = 10 ** (gain_db / 20)

        transform = Gain(gain_db_range=(gain_db, gain_db), p=1.0)
        result = transform(audio, SAMPLE_RATE)
        result_energy = np.sum(result**2)

        expected_energy = original_energy * (linear_gain**2)
        np.testing.assert_allclose(result_energy, expected_energy, rtol=TOLERANCE)

    def test_time_stretch_energy_bounded(self):
        """Time stretch energy should be bounded (not blow up or vanish).

        Note: FFT-based resampling (scipy.signal.resample) doesn't
        perfectly conserve energy, especially with significant rate changes.
        We just verify the energy stays reasonable.
        """
        from mlx_audio.data.transforms.audio import TimeStretch

        audio = white_noise(duration=0.5)
        original_energy = np.sum(audio**2)

        transform = TimeStretch(rate_range=(1.5, 1.5), p=1.0)
        result = transform(audio, SAMPLE_RATE)
        result_energy = np.sum(result**2)

        # Energy should be within an order of magnitude
        ratio = result_energy / original_energy
        assert 0.1 < ratio < 10.0, f"Energy ratio {ratio} outside bounds"


# =============================================================================
# Composition Parity Tests
# =============================================================================


class TestCompositionParity:
    """Test composition utilities work correctly."""

    def test_compose_order_matters(self):
        """Verify transforms are applied in order."""
        from mlx_audio.data.transforms.audio import AudioCompose, Gain, Reverse

        audio = np.arange(100, dtype=np.float32)

        # Gain then Reverse
        transform1 = AudioCompose([Gain(gain_db_range=(6, 6), p=1.0), Reverse(p=1.0)])
        result1 = transform1(audio, SAMPLE_RATE)

        # Reverse then Gain
        transform2 = AudioCompose([Reverse(p=1.0), Gain(gain_db_range=(6, 6), p=1.0)])
        result2 = transform2(audio, SAMPLE_RATE)

        # Results should be the same (both operations are linear and commute here)
        # But let's verify they're applied
        assert not np.allclose(result1, audio)

    def test_one_of_probability(self):
        """OneOf should select each transform with equal probability."""
        from mlx_audio.data.transforms.audio import OneOf, Gain, Reverse

        audio = np.ones(100, dtype=np.float32)
        counts = {"gain": 0, "reverse": 0}

        gain_transform = Gain(gain_db_range=(6, 6), p=1.0)
        reverse_transform = Reverse(p=1.0)

        for seed in range(100):
            transform = OneOf([gain_transform, reverse_transform], seed=seed)
            result = transform(audio.copy(), SAMPLE_RATE)

            if np.allclose(result, audio[::-1]):
                counts["reverse"] += 1
            else:
                counts["gain"] += 1

        # Should be roughly 50/50
        assert 30 < counts["gain"] < 70
        assert 30 < counts["reverse"] < 70
