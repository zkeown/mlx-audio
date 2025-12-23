#!/usr/bin/env python3
"""
Generate DSP Reference Data for Swift Parity Tests

This script generates reference data files that Swift parity tests
can load to verify numerical correctness of DSP implementations.

Run this script to create reference data before running Swift tests:
    cd python && python tests/parity/generate_swift_dsp_parity_data.py

Reference data is saved to /tmp/dsp_parity/
"""

import json
import os
from pathlib import Path

import numpy as np

try:
    import librosa

    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    print("Warning: librosa not installed. Some reference data will be skipped.")

try:
    import scipy.signal

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not installed. Some reference data will be skipped.")


OUTPUT_DIR = Path("/tmp/dsp_parity")


def save_npy(name: str, array: np.ndarray):
    """Save numpy array to .npy file."""
    path = OUTPUT_DIR / f"{name}.npy"
    np.save(path, array.astype(np.float32))
    print(f"  Saved: {path}")


def save_json(name: str, data: dict):
    """Save dictionary to JSON file."""
    path = OUTPUT_DIR / f"{name}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved: {path}")


def generate_test_signals():
    """Generate standard test signals."""
    print("\n=== Generating Test Signals ===")

    # 1. Pure sine wave (440 Hz, 22050 Hz sample rate, 0.5 seconds)
    sr = 22050
    duration = 0.5
    frequency = 440.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    sine_wave = np.sin(2 * np.pi * frequency * t).astype(np.float32)
    save_npy("sine_440hz_22050sr", sine_wave)
    save_json(
        "sine_440hz_22050sr_info",
        {
            "sample_rate": sr,
            "frequency": frequency,
            "duration": duration,
            "num_samples": len(sine_wave),
        },
    )

    # 2. Chirp signal (20 Hz to 8000 Hz sweep)
    sr = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    chirp = scipy.signal.chirp(t, f0=20, f1=8000, t1=duration, method="linear").astype(np.float32)
    save_npy("chirp_20_8000hz_16000sr", chirp)

    # 3. Random noise with seed for reproducibility
    np.random.seed(42)
    noise = np.random.randn(22050).astype(np.float32)
    save_npy("noise_seed42_22050", noise)

    # 4. Short signal for edge case testing
    short_signal = np.array([0.1, 0.5, 0.9, 0.5, 0.1, -0.5, -0.9, -0.5], dtype=np.float32)
    save_npy("short_8_samples", short_signal)

    # 5. Whisper-compatible signal (16000 Hz, 1 second)
    sr = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    whisper_signal = (np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 880 * t)).astype(
        np.float32
    )
    save_npy("whisper_test_16000sr", whisper_signal)


def generate_stft_references():
    """Generate STFT reference data."""
    if not HAS_LIBROSA:
        print("\nSkipping STFT references (librosa not installed)")
        return

    print("\n=== Generating STFT References ===")

    # Load test signals
    sine_wave = np.load(OUTPUT_DIR / "sine_440hz_22050sr.npy")
    noise = np.load(OUTPUT_DIR / "noise_seed42_22050.npy")

    # Test case 1: Standard STFT parameters
    config1 = {"n_fft": 2048, "hop_length": 512, "center": True}
    stft1 = librosa.stft(sine_wave, **config1)
    save_npy("stft_sine_nfft2048_hop512_mag", np.abs(stft1))
    save_npy("stft_sine_nfft2048_hop512_phase", np.angle(stft1))
    save_json("stft_sine_nfft2048_hop512_info", {**config1, "shape": list(stft1.shape)})

    # Test case 2: Different FFT sizes
    for n_fft in [512, 1024, 2048, 4096]:
        hop_length = n_fft // 4
        stft_result = librosa.stft(noise, n_fft=n_fft, hop_length=hop_length, center=True)
        save_npy(f"stft_noise_nfft{n_fft}_hop{hop_length}_mag", np.abs(stft_result))
        save_json(
            f"stft_noise_nfft{n_fft}_hop{hop_length}_info",
            {"n_fft": n_fft, "hop_length": hop_length, "center": True, "shape": list(stft_result.shape)},
        )

    # Test case 3: Non-centered STFT
    stft_no_center = librosa.stft(sine_wave, n_fft=1024, hop_length=256, center=False)
    save_npy("stft_sine_nfft1024_hop256_no_center_mag", np.abs(stft_no_center))
    save_json(
        "stft_sine_nfft1024_hop256_no_center_info",
        {"n_fft": 1024, "hop_length": 256, "center": False, "shape": list(stft_no_center.shape)},
    )

    # Test case 4: Whisper-like parameters (16kHz, 400-sample FFT)
    whisper_signal = np.load(OUTPUT_DIR / "whisper_test_16000sr.npy")
    stft_whisper = librosa.stft(whisper_signal, n_fft=400, hop_length=160, center=True)
    save_npy("stft_whisper_nfft400_hop160_mag", np.abs(stft_whisper))
    save_json(
        "stft_whisper_nfft400_hop160_info",
        {"n_fft": 400, "hop_length": 160, "center": True, "sample_rate": 16000, "shape": list(stft_whisper.shape)},
    )


def generate_istft_references():
    """Generate ISTFT (roundtrip) reference data."""
    if not HAS_LIBROSA:
        print("\nSkipping ISTFT references (librosa not installed)")
        return

    print("\n=== Generating ISTFT References ===")

    # Test roundtrip reconstruction
    noise = np.load(OUTPUT_DIR / "noise_seed42_22050.npy")

    n_fft = 2048
    hop_length = 512

    # Forward STFT
    stft_result = librosa.stft(noise, n_fft=n_fft, hop_length=hop_length, center=True)

    # Inverse STFT
    reconstructed = librosa.istft(stft_result, hop_length=hop_length, length=len(noise))
    save_npy("istft_roundtrip_input", noise)
    save_npy("istft_roundtrip_output", reconstructed.astype(np.float32))
    save_json(
        "istft_roundtrip_info",
        {"n_fft": n_fft, "hop_length": hop_length, "original_length": len(noise)},
    )


def generate_mel_filterbank_references():
    """Generate mel filterbank reference data."""
    if not HAS_LIBROSA:
        print("\nSkipping mel filterbank references (librosa not installed)")
        return

    print("\n=== Generating Mel Filterbank References ===")

    test_cases = [
        {"sr": 22050, "n_fft": 2048, "n_mels": 128, "name": "standard"},
        {"sr": 16000, "n_fft": 400, "n_mels": 80, "name": "whisper"},
        {"sr": 44100, "n_fft": 4096, "n_mels": 128, "name": "highres"},
        {"sr": 8000, "n_fft": 256, "n_mels": 40, "name": "lowres"},
    ]

    for case in test_cases:
        name = case.pop("name")

        # With Slaney normalization
        fb_slaney = librosa.filters.mel(**case, norm="slaney")
        save_npy(f"mel_filterbank_{name}_slaney", fb_slaney)
        save_json(f"mel_filterbank_{name}_slaney_info", {**case, "norm": "slaney", "shape": list(fb_slaney.shape)})

        # Without normalization
        fb_none = librosa.filters.mel(**case, norm=None)
        save_npy(f"mel_filterbank_{name}_none", fb_none)
        save_json(f"mel_filterbank_{name}_none_info", {**case, "norm": None, "shape": list(fb_none.shape)})


def generate_mel_spectrogram_references():
    """Generate mel spectrogram reference data."""
    if not HAS_LIBROSA:
        print("\nSkipping mel spectrogram references (librosa not installed)")
        return

    print("\n=== Generating Mel Spectrogram References ===")

    # Standard mel spectrogram
    noise = np.load(OUTPUT_DIR / "noise_seed42_22050.npy")
    sr = 22050
    n_fft = 2048
    hop_length = 512
    n_mels = 128

    mel_spec = librosa.feature.melspectrogram(
        y=noise, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    save_npy("mel_spectrogram_standard", mel_spec)
    save_json(
        "mel_spectrogram_standard_info",
        {"sr": sr, "n_fft": n_fft, "hop_length": hop_length, "n_mels": n_mels, "shape": list(mel_spec.shape)},
    )

    # Log mel spectrogram
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    save_npy("mel_spectrogram_standard_log", log_mel)

    # Whisper-like mel spectrogram
    whisper_signal = np.load(OUTPUT_DIR / "whisper_test_16000sr.npy")
    whisper_mel = librosa.feature.melspectrogram(
        y=whisper_signal, sr=16000, n_fft=400, hop_length=160, n_mels=80
    )
    save_npy("mel_spectrogram_whisper", whisper_mel)
    save_json(
        "mel_spectrogram_whisper_info",
        {"sr": 16000, "n_fft": 400, "hop_length": 160, "n_mels": 80, "shape": list(whisper_mel.shape)},
    )


def generate_window_references():
    """Generate window function reference data."""
    print("\n=== Generating Window Function References ===")

    window_sizes = [256, 512, 1024, 2048]

    for size in window_sizes:
        # Hann window
        hann = scipy.signal.get_window("hann", size).astype(np.float32)
        save_npy(f"window_hann_{size}", hann)

        # Hamming window
        hamming = scipy.signal.get_window("hamming", size).astype(np.float32)
        save_npy(f"window_hamming_{size}", hamming)

        # Blackman window
        blackman = scipy.signal.get_window("blackman", size).astype(np.float32)
        save_npy(f"window_blackman_{size}", blackman)


def generate_mfcc_references():
    """Generate MFCC reference data."""
    if not HAS_LIBROSA:
        print("\nSkipping MFCC references (librosa not installed)")
        return

    print("\n=== Generating MFCC References ===")

    noise = np.load(OUTPUT_DIR / "noise_seed42_22050.npy")
    sr = 22050

    # Standard MFCCs
    mfccs = librosa.feature.mfcc(y=noise, sr=sr, n_mfcc=13)
    save_npy("mfcc_standard_13", mfccs)
    save_json("mfcc_standard_13_info", {"sr": sr, "n_mfcc": 13, "shape": list(mfccs.shape)})

    # More MFCCs
    mfccs_40 = librosa.feature.mfcc(y=noise, sr=sr, n_mfcc=40)
    save_npy("mfcc_standard_40", mfccs_40)
    save_json("mfcc_standard_40_info", {"sr": sr, "n_mfcc": 40, "shape": list(mfccs_40.shape)})


def main():
    """Generate all reference data."""
    print("=" * 60)
    print("Generating DSP Reference Data for Swift Parity Tests")
    print("=" * 60)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {OUTPUT_DIR}")

    # Generate all references
    generate_test_signals()
    generate_stft_references()
    generate_istft_references()
    generate_mel_filterbank_references()
    generate_mel_spectrogram_references()
    generate_window_references()
    generate_mfcc_references()

    print("\n" + "=" * 60)
    print("Reference data generation complete!")
    print(f"Files saved to: {OUTPUT_DIR}")
    print("\nTo run Swift parity tests:")
    print("  cd swift && swift test --filter ParityTests")
    print("=" * 60)


if __name__ == "__main__":
    main()
