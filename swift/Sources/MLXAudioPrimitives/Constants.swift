// Constants.swift
// Spectral analysis constants for mel filterbanks and spectrograms.
//
// These constants define parameters for mel-frequency analysis
// and spectrogram computation, matching the Python implementation.

import Foundation

// MARK: - Mel Filterbank Sizes

/// Whisper v1/v2 mel filterbank bins.
public let whisperNMels = 80

/// Whisper v3 mel filterbank bins.
public let whisperV3NMels = 128

/// CLAP mel filterbank bins.
public let clapNMels = 64

/// Default mel filterbank bins for general use.
public let defaultNMels = 128

// MARK: - Frequency Bounds

/// Whisper maximum frequency for mel filterbank (Nyquist at 16kHz).
public let whisperFMax: Double = 8000.0

/// CLAP spectrogram size.
public let clapSpecSize = 256

// MARK: - Slaney Mel Formula Constants

/// These match librosa's default 'slaney' mel scale.
public enum SlaneyConstants {
    /// Slaney mel scale minimum frequency.
    public static let fMin: Double = 0.0

    /// Slaney mel scale frequency spacing (66.67 Hz).
    public static let fSp: Double = 200.0 / 3.0

    /// Slaney mel scale transition to log spacing.
    public static let minLogHz: Double = 1000.0

    /// Slaney mel scale log step divisor.
    public static let logstepDivisor: Double = 27.0

    /// Slaney mel scale log step size.
    public static let logstep: Double = log(6.4) / logstepDivisor

    /// Minimum mel value at log transition point.
    public static let minLogMel: Double = (minLogHz - fMin) / fSp
}

// MARK: - HTK Mel Formula Constants

public enum HTKConstants {
    /// HTK mel scale conversion factor.
    public static let melFactor: Double = 2595.0

    /// HTK mel scale base frequency.
    public static let melBase: Double = 700.0
}

// MARK: - Window Function Coefficients

/// Generalized cosine window coefficients.
/// Reference: Harris, F.J. (1978). "On the use of windows for harmonic analysis"
public enum WindowCoefficients {
    /// Hann window: w[k] = 0.5 - 0.5 * cos(2*pi*k/(n-1))
    public static let hann: [Double] = [0.5, 0.5]

    /// Hamming window: w[k] = 0.54 - 0.46 * cos(2*pi*k/(n-1))
    public static let hamming: [Double] = [0.54, 0.46]

    /// Blackman window: w[k] = 0.42 - 0.5*cos(...) + 0.08*cos(...)
    public static let blackman: [Double] = [0.42, 0.5, 0.08]
}

// MARK: - STFT Defaults

public enum STFTDefaults {
    /// Default FFT size.
    public static let nFFT = 2048

    /// Default hop length ratio (nFFT / 4).
    public static let hopLengthRatio = 4

    /// Epsilon for window normalization to avoid division by zero.
    public static let windowSumEpsilon: Float = 1e-8
}

// MARK: - MFCC Defaults

public enum MFCCDefaults {
    /// Default number of MFCCs.
    public static let nMFCC = 20

    /// Default number of mel bands.
    public static let nMels = 128

    /// Default sample rate.
    public static let sampleRate = 22050
}

// MARK: - Decibel Conversion

public enum DecibelConstants {
    /// Default reference value for power to dB conversion.
    public static let ref: Float = 1.0

    /// Default minimum amplitude for log conversion.
    public static let amin: Float = 1e-10

    /// Default top dB for dynamic range compression.
    public static let topDb: Float = 80.0
}
