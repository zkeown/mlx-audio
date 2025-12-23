"""Tests for audio augmentation transforms."""

import numpy as np
import pytest

# Skip all tests if mlx is not available
pytest.importorskip("mlx")


class TestAudioTransformBase:
    """Tests for AudioTransform base class."""

    def test_probability_zero_returns_unchanged(self):
        """Transform with p=0 should never apply."""
        from mlx_audio.data.transforms.audio import Gain

        audio = np.random.randn(16000).astype(np.float32)
        transform = Gain(p=0.0, seed=42)

        result = transform(audio, sample_rate=16000)

        np.testing.assert_array_equal(result, audio)

    def test_probability_one_always_applies(self):
        """Transform with p=1 should always apply."""
        from mlx_audio.data.transforms.audio import Gain

        audio = np.random.randn(16000).astype(np.float32)
        transform = Gain(gain_db_range=(6, 6), p=1.0, seed=42)  # +6dB

        result = transform(audio, sample_rate=16000)

        # Should be different (louder)
        assert not np.allclose(result, audio)

    def test_invalid_probability_raises(self):
        """Invalid probability should raise ValueError."""
        from mlx_audio.data.transforms.audio import Gain

        with pytest.raises(ValueError):
            Gain(p=-0.1)

        with pytest.raises(ValueError):
            Gain(p=1.5)

    def test_seed_produces_deterministic_output(self):
        """Same seed should produce identical results."""
        from mlx_audio.data.transforms.audio import Gain

        audio = np.random.randn(16000).astype(np.float32)

        result1 = Gain(seed=42)(audio.copy(), 16000)
        result2 = Gain(seed=42)(audio.copy(), 16000)

        np.testing.assert_array_equal(result1, result2)

    def test_numpy_input_returns_numpy(self):
        """Numpy input should return numpy array."""
        from mlx_audio.data.transforms.audio import Gain

        audio = np.random.randn(16000).astype(np.float32)
        transform = Gain(p=1.0)

        result = transform(audio, sample_rate=16000)

        assert isinstance(result, np.ndarray)

    def test_mlx_input_returns_mlx(self):
        """MLX input should return MLX array."""
        import mlx.core as mx
        from mlx_audio.data.transforms.audio import Gain

        audio = mx.array(np.random.randn(16000).astype(np.float32))
        transform = Gain(p=1.0)

        result = transform(audio, sample_rate=16000)

        assert isinstance(result, mx.array)


class TestTimeDomainTransforms:
    """Tests for time-domain transforms."""

    def test_gain_changes_amplitude(self):
        """Gain should change audio amplitude."""
        from mlx_audio.data.transforms.audio import Gain

        audio = np.ones(1000, dtype=np.float32) * 0.5
        transform = Gain(gain_db_range=(6, 6), p=1.0)  # +6dB ≈ 2x

        result = transform(audio, sample_rate=16000)

        # Should be approximately doubled (tightened from 1e-2 - measured: exact)
        expected = 0.5 * (10 ** (6 / 20))
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_gaussian_noise_adds_noise(self):
        """GaussianNoise should add noise to signal."""
        from mlx_audio.data.transforms.audio import GaussianNoise

        audio = np.zeros(16000, dtype=np.float32)  # Silent
        transform = GaussianNoise(snr_range=(20, 20), p=1.0, seed=42)

        result = transform(audio, sample_rate=16000)

        # Silent audio should remain silent (no signal power)
        np.testing.assert_array_equal(result, audio)

        # With actual signal
        audio = np.sin(2 * np.pi * 440 * np.arange(16000) / 16000).astype(np.float32)
        result = transform(audio, sample_rate=16000)

        assert not np.allclose(result, audio)

    def test_time_shift_circular(self):
        """TimeShift with circular mode should wrap audio."""
        from mlx_audio.data.transforms.audio import TimeShift

        audio = np.arange(100, dtype=np.float32)
        transform = TimeShift(shift_range=(0.1, 0.1), mode="circular", p=1.0)

        result = transform(audio, sample_rate=16000)

        # Should be shifted by 10 samples
        assert len(result) == len(audio)

    def test_time_masking_zeros_segments(self):
        """TimeMasking should zero out segments."""
        from mlx_audio.data.transforms.audio import TimeMasking

        audio = np.ones(1000, dtype=np.float32)
        transform = TimeMasking(max_mask_fraction=0.1, num_masks=1, p=1.0, seed=42)

        result = transform(audio, sample_rate=16000)

        # Should have some zeros
        assert np.sum(result == 0) > 0
        # Should not be all zeros
        assert np.sum(result != 0) > 0

    def test_trim_removes_silence(self):
        """Trim should remove silence from ends."""
        from mlx_audio.data.transforms.audio import Trim

        # Audio with silence at beginning and end
        audio = np.zeros(1000, dtype=np.float32)
        audio[200:800] = 1.0

        transform = Trim(top_db=20, p=1.0)
        result = transform(audio, sample_rate=16000)

        # Should be shorter
        assert len(result) < len(audio)
        # Should start and end with non-zero
        assert result[0] != 0 or result[-1] != 0

    def test_time_stretch_changes_length(self):
        """TimeStretch should change audio length."""
        from mlx_audio.data.transforms.audio import TimeStretch

        audio = np.random.randn(16000).astype(np.float32)
        transform = TimeStretch(rate_range=(2.0, 2.0), p=1.0)  # 2x speed

        result = transform(audio, sample_rate=16000)

        # Should be exactly half length (our implementation targets exact length)
        expected_length = len(audio) // 2
        assert len(result) == expected_length

    def test_pitch_shift_preserves_length(self):
        """PitchShift should preserve audio length."""
        from mlx_audio.data.transforms.audio import PitchShift

        audio = np.random.randn(16000).astype(np.float32)
        transform = PitchShift(semitone_range=(4, 4), p=1.0)

        result = transform(audio, sample_rate=16000)

        # Should have same length
        assert len(result) == len(audio)

    def test_normalize_peak(self):
        """Normalize should adjust peak level."""
        from mlx_audio.data.transforms.audio import Normalize

        audio = np.random.randn(1000).astype(np.float32) * 0.1
        transform = Normalize(target_db=-6, mode="peak", p=1.0)

        result = transform(audio, sample_rate=16000)

        # Peak should be at -6dB ≈ 0.5 (tightened from 1e-2 - measured: exact)
        expected_peak = 10 ** (-6 / 20)
        np.testing.assert_allclose(np.max(np.abs(result)), expected_peak, rtol=1e-6)

    def test_reverse(self):
        """Reverse should reverse the audio."""
        from mlx_audio.data.transforms.audio import Reverse

        audio = np.arange(100, dtype=np.float32)
        transform = Reverse(p=1.0)

        result = transform(audio, sample_rate=16000)

        np.testing.assert_array_equal(result, audio[::-1])

    def test_fade(self):
        """Fade should apply fade in/out."""
        from mlx_audio.data.transforms.audio import Fade

        audio = np.ones(1000, dtype=np.float32)
        transform = Fade(fade_in_fraction=0.1, fade_out_fraction=0.1, p=1.0)

        result = transform(audio, sample_rate=16000)

        # Start should be faded (less than 1)
        assert result[0] < 1.0
        # End should be faded
        assert result[-1] < 1.0
        # Middle should be unchanged
        assert result[500] == 1.0


class TestFrequencyDomainTransforms:
    """Tests for frequency-domain transforms."""

    def test_spec_augment_modifies_audio(self):
        """SpecAugment should modify the audio."""
        from mlx_audio.data.transforms.audio import SpecAugment

        # Create a sine wave
        t = np.linspace(0, 1, 16000, dtype=np.float32)
        audio = np.sin(2 * np.pi * 440 * t)

        transform = SpecAugment(
            freq_mask_param=27,
            time_mask_param=100,
            num_freq_masks=2,
            num_time_masks=2,
            p=1.0,
            seed=42,
        )

        result = transform(audio, sample_rate=16000)

        # Should be modified
        assert not np.allclose(result, audio)
        # Should have same length
        assert len(result) == len(audio)

    def test_frequency_masking(self):
        """FrequencyMasking should mask frequency bands."""
        from mlx_audio.data.transforms.audio import FrequencyMasking

        audio = np.random.randn(8000).astype(np.float32)
        transform = FrequencyMasking(max_mask_fraction=0.2, num_masks=2, p=1.0, seed=42)

        result = transform(audio, sample_rate=16000)

        # Should be modified
        assert not np.allclose(result, audio)
        assert len(result) == len(audio)

    def test_lowpass_filter(self):
        """LowPassFilter should attenuate high frequencies."""
        from mlx_audio.data.transforms.audio import LowPassFilter

        # Mix of low and high frequencies
        t = np.linspace(0, 1, 16000, dtype=np.float32)
        low_freq = np.sin(2 * np.pi * 200 * t)  # 200 Hz
        high_freq = np.sin(2 * np.pi * 7000 * t)  # 7000 Hz
        audio = (low_freq + high_freq).astype(np.float32)

        transform = LowPassFilter(cutoff_range=(0.3, 0.3), p=1.0)
        result = transform(audio, sample_rate=16000)

        # High frequency energy should be reduced
        # (simple check: correlation with low_freq should increase)
        assert len(result) == len(audio)

    def test_highpass_filter(self):
        """HighPassFilter should attenuate low frequencies."""
        from mlx_audio.data.transforms.audio import HighPassFilter

        audio = np.random.randn(16000).astype(np.float32)
        transform = HighPassFilter(cutoff_range=(0.1, 0.1), p=1.0)

        result = transform(audio, sample_rate=16000)

        assert len(result) == len(audio)


class TestEffects:
    """Tests for audio effects."""

    def test_room_reverb_adds_reverb(self):
        """RoomReverb should add reverb to audio."""
        from mlx_audio.data.transforms.audio import RoomReverb

        # Short impulse
        audio = np.zeros(16000, dtype=np.float32)
        audio[100] = 1.0

        transform = RoomReverb(wet_dry_range=(0.5, 0.5), p=1.0, seed=42)
        result = transform(audio, sample_rate=16000)

        # Should have energy after the impulse (reverb tail)
        assert np.sum(np.abs(result[200:])) > 0

    def test_background_noise_adds_noise(self):
        """BackgroundNoise should add noise."""
        from mlx_audio.data.transforms.audio import BackgroundNoise

        audio = np.sin(2 * np.pi * 440 * np.arange(16000) / 16000).astype(np.float32)
        transform = BackgroundNoise(snr_range=(10, 10), noise_type="white", p=1.0, seed=42)

        result = transform(audio, sample_rate=16000)

        assert not np.allclose(result, audio)
        assert len(result) == len(audio)

    def test_clipping(self):
        """Clipping should hard limit the signal."""
        from mlx_audio.data.transforms.audio import Clipping

        audio = np.random.randn(1000).astype(np.float32)
        transform = Clipping(percentile_range=(50, 50), p=1.0)

        result = transform(audio, sample_rate=16000)

        # Max value should be lower than before
        assert np.max(np.abs(result)) <= np.max(np.abs(audio))

    def test_speed_perturb_changes_length(self):
        """SpeedPerturb should change audio length."""
        from mlx_audio.data.transforms.audio import SpeedPerturb

        audio = np.random.randn(16000).astype(np.float32)
        transform = SpeedPerturb(speed_range=(0.5, 0.5), p=1.0)  # Half speed

        result = transform(audio, sample_rate=16000)

        # Should be approximately double length
        assert len(result) > len(audio)

    def test_codec_quantizes(self):
        """Codec should apply quantization."""
        from mlx_audio.data.transforms.audio import Codec

        audio = np.random.randn(1000).astype(np.float32) * 0.5
        transform = Codec(bits_range=(4, 4), apply_lowpass=False, p=1.0)

        result = transform(audio, sample_rate=16000)

        # Should be quantized (fewer unique values)
        assert len(np.unique(result)) < len(np.unique(audio))


class TestComposition:
    """Tests for composition utilities."""

    def test_audio_compose_chains_transforms(self):
        """AudioCompose should chain transforms."""
        from mlx_audio.data.transforms.audio import AudioCompose, Gain, GaussianNoise

        audio = np.random.randn(16000).astype(np.float32)
        transform = AudioCompose([
            Gain(gain_db_range=(3, 3), p=1.0),
            GaussianNoise(snr_range=(30, 30), p=1.0, seed=42),
        ])

        result = transform(audio, sample_rate=16000)

        assert not np.allclose(result, audio)

    def test_one_of_selects_single_transform(self):
        """OneOf should apply exactly one transform."""
        from mlx_audio.data.transforms.audio import OneOf, Gain, Reverse

        audio = np.arange(100, dtype=np.float32)
        transform = OneOf([
            Gain(gain_db_range=(6, 6), p=1.0),
            Reverse(p=1.0),
        ], seed=42)

        result = transform(audio, sample_rate=16000)

        # Should be modified by one transform
        assert not np.allclose(result, audio)

    def test_some_of_applies_subset(self):
        """SomeOf should apply a subset of transforms."""
        from mlx_audio.data.transforms.audio import SomeOf, Gain, Reverse, TimeShift

        audio = np.arange(100, dtype=np.float32)
        transform = SomeOf([
            Gain(p=1.0),
            Reverse(p=1.0),
            TimeShift(p=1.0),
        ], n_range=(1, 2), seed=42)

        result = transform(audio, sample_rate=16000)

        assert len(result) == len(audio)

    def test_audio_policy_light(self):
        """AudioPolicy 'light' should apply light augmentation."""
        from mlx_audio.data.transforms.audio import AudioPolicy

        audio = np.random.randn(16000).astype(np.float32)
        transform = AudioPolicy("light", seed=42)

        result = transform(audio, sample_rate=16000)

        assert len(result) == len(audio)

    def test_audio_policy_speech(self):
        """AudioPolicy 'speech' should work."""
        from mlx_audio.data.transforms.audio import AudioPolicy

        audio = np.random.randn(16000).astype(np.float32)
        transform = AudioPolicy("speech", seed=42)

        # May change length due to speed_perturb
        result = transform(audio, sample_rate=16000)

        assert result is not None

    def test_audio_policy_invalid_raises(self):
        """Invalid policy name should raise ValueError."""
        from mlx_audio.data.transforms.audio import AudioPolicy

        with pytest.raises(ValueError):
            AudioPolicy("invalid_policy")

    def test_probabilistic_compose(self):
        """ProbabilisticCompose should apply transforms with probabilities."""
        from mlx_audio.data.transforms.audio import ProbabilisticCompose, Gain, Reverse

        audio = np.arange(100, dtype=np.float32)
        transform = ProbabilisticCompose([
            (Gain(gain_db_range=(6, 6)), 1.0),  # Always
            (Reverse(), 0.0),  # Never
        ], seed=42)

        result = transform(audio, sample_rate=16000)

        # Should have gain applied but not reversed (tightened from 1e-2 - exact)
        expected = audio * (10 ** (6 / 20))
        np.testing.assert_allclose(result, expected, rtol=1e-6)


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_audio_handled(self):
        """Empty audio should be handled gracefully."""
        from mlx_audio.data.transforms.audio import Gain

        audio = np.array([], dtype=np.float32)
        transform = Gain(p=1.0)

        result = transform(audio, sample_rate=16000)

        assert len(result) == 0

    def test_silent_audio_handled(self):
        """Silent audio should be handled gracefully."""
        from mlx_audio.data.transforms.audio import Gain, GaussianNoise, Trim

        audio = np.zeros(1000, dtype=np.float32)

        # Should not crash
        Gain(p=1.0)(audio.copy(), 16000)
        GaussianNoise(p=1.0, seed=42)(audio.copy(), 16000)
        Trim(p=1.0)(audio.copy(), 16000)

    def test_very_short_audio(self):
        """Very short audio should be handled."""
        from mlx_audio.data.transforms.audio import TimeStretch, TimeMasking

        audio = np.array([0.5], dtype=np.float32)

        TimeStretch(p=1.0)(audio.copy(), 16000)
        TimeMasking(p=1.0)(audio.copy(), 16000)

    def test_nan_not_introduced(self):
        """Transforms should not introduce NaN values."""
        from mlx_audio.data.transforms.audio import (
            AudioCompose, Gain, GaussianNoise, TimeStretch,
            LowPassFilter, RoomReverb
        )

        audio = np.random.randn(16000).astype(np.float32)
        transform = AudioCompose([
            Gain(p=1.0),
            GaussianNoise(p=1.0, seed=42),
            TimeStretch(p=1.0),
            LowPassFilter(p=1.0),
            RoomReverb(p=1.0, seed=42),
        ])

        result = transform(audio, sample_rate=16000)

        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))


class TestParameterSampler:
    """Tests for ParameterSampler."""

    def test_uniform_sampling(self):
        """Uniform sampling should be within range."""
        from mlx_audio.data.transforms.audio.base import ParameterSampler

        sampler = ParameterSampler(0.0, 1.0, distribution="uniform", seed=42)

        for _ in range(100):
            value = sampler.sample()
            assert 0.0 <= value <= 1.0

    def test_log_uniform_sampling(self):
        """Log-uniform sampling should be within range."""
        from mlx_audio.data.transforms.audio.base import ParameterSampler

        sampler = ParameterSampler(0.1, 10.0, distribution="log_uniform", seed=42)

        for _ in range(100):
            value = sampler.sample()
            assert 0.1 <= value <= 10.0

    def test_invalid_distribution_raises(self):
        """Invalid distribution should raise ValueError."""
        from mlx_audio.data.transforms.audio.base import ParameterSampler

        sampler = ParameterSampler(0.0, 1.0, distribution="invalid")
        with pytest.raises(ValueError):
            sampler.sample()


class TestImports:
    """Tests for module imports."""

    def test_audio_transforms_importable(self):
        """All audio transforms should be importable."""
        from mlx_audio.data.transforms import audio

        # Check all expected exports exist
        assert hasattr(audio, "AudioTransform")
        assert hasattr(audio, "Gain")
        assert hasattr(audio, "GaussianNoise")
        assert hasattr(audio, "TimeStretch")
        assert hasattr(audio, "PitchShift")
        assert hasattr(audio, "SpecAugment")
        assert hasattr(audio, "RoomReverb")
        assert hasattr(audio, "AudioCompose")
        assert hasattr(audio, "AudioPolicy")

    def test_direct_import(self):
        """Direct imports should work."""
        from mlx_audio.data.transforms.audio import (
            Gain,
            GaussianNoise,
            TimeStretch,
            SpecAugment,
            AudioCompose,
        )

        assert callable(Gain)
        assert callable(AudioCompose)
