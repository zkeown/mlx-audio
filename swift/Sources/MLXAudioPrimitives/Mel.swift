// Mel.swift
// Mel-scale filterbank and mel spectrogram.
//
// Provides mel filterbank construction and mel spectrogram computation,
// matching librosa conventions.

import Accelerate
import Foundation
@preconcurrency import MLX

// MARK: - Configuration

/// Normalization mode for mel filterbank.
public enum MelNorm: String, Sendable {
    /// Normalize by bandwidth (area under each filter = 1).
    case slaney
}

/// Configuration for mel filterbank and spectrogram.
public struct MelConfig: Sendable {
    /// Sample rate of the audio. Default: 22050.
    public var sampleRate: Int

    /// Number of mel bands. Default: 128.
    public var nMels: Int

    /// Minimum frequency (Hz). Default: 0.0.
    public var fMin: Float

    /// Maximum frequency (Hz). Default: sampleRate / 2 (Nyquist).
    public var fMax: Float?

    /// Use HTK formula for mel scale. If false, use Slaney (librosa default). Default: false.
    public var htk: Bool

    /// Normalization mode. Default: .slaney.
    public var norm: MelNorm?

    /// Creates a new mel configuration with default values.
    public init(
        sampleRate: Int = MFCCDefaults.sampleRate,
        nMels: Int = defaultNMels,
        fMin: Float = 0.0,
        fMax: Float? = nil,
        htk: Bool = false,
        norm: MelNorm? = .slaney
    ) {
        self.sampleRate = sampleRate
        self.nMels = nMels
        self.fMin = fMin
        self.fMax = fMax
        self.htk = htk
        self.norm = norm
    }

    /// Resolved maximum frequency (uses Nyquist if nil).
    public var resolvedFMax: Float {
        fMax ?? Float(sampleRate) / 2.0
    }
}

// MARK: - Mel Scale Conversion

/// Convert Hz to mel scale.
///
/// - Parameters:
///   - frequencies: Frequencies in Hz.
///   - htk: If true, use HTK formula. If false, use Slaney (librosa default).
/// - Returns: Frequencies in mel scale.
///
/// Slaney formula (librosa default):
/// - Below 1000 Hz: mel = (hz - 0) / 66.67
/// - Above 1000 Hz: mel = mel_min_log + log(hz/1000) / logstep
///
/// HTK formula:
/// - mel = 2595 * log10(1 + hz / 700)
public func hzToMel(_ frequencies: [Double], htk: Bool = false) -> [Double] {
    if htk {
        // HTK formula: mel = 2595 * log10(1 + f / 700)
        return frequencies.map { hz in
            HTKConstants.melFactor * log10(1.0 + hz / HTKConstants.melBase)
        }
    } else {
        // Slaney formula
        return frequencies.map { hz in
            if hz < SlaneyConstants.minLogHz {
                return (hz - SlaneyConstants.fMin) / SlaneyConstants.fSp
            } else {
                return SlaneyConstants.minLogMel +
                    log(hz / SlaneyConstants.minLogHz) / SlaneyConstants.logstep
            }
        }
    }
}

/// Convert mel scale to Hz.
///
/// - Parameters:
///   - mels: Frequencies in mel scale.
///   - htk: If true, use HTK formula. If false, use Slaney (librosa default).
/// - Returns: Frequencies in Hz.
public func melToHz(_ mels: [Double], htk: Bool = false) -> [Double] {
    if htk {
        // HTK formula: f = 700 * (10^(mel / 2595) - 1)
        return mels.map { mel in
            HTKConstants.melBase * (pow(10.0, mel / HTKConstants.melFactor) - 1.0)
        }
    } else {
        // Slaney formula (inverse)
        return mels.map { mel in
            if mel < SlaneyConstants.minLogMel {
                return SlaneyConstants.fMin + SlaneyConstants.fSp * mel
            } else {
                return SlaneyConstants.minLogHz *
                    exp(SlaneyConstants.logstep * (mel - SlaneyConstants.minLogMel))
            }
        }
    }
}

// MARK: - Mel Filterbank Cache

/// Thread-safe cache for mel filterbanks using NSLock.
private final class MelFilterbankCache: @unchecked Sendable {
    private var cache: [String: MLXArray] = [:]
    private let maxSize: Int
    private let lock = NSLock()

    init(maxSize: Int = 64) {
        self.maxSize = maxSize
    }

    func get(key: String) -> MLXArray? {
        lock.lock()
        defer { lock.unlock() }
        return cache[key]
    }

    func set(key: String, value: MLXArray) {
        lock.lock()
        defer { lock.unlock() }
        if cache.count >= maxSize {
            cache.removeAll()
        }
        cache[key] = value
    }

    func clear() {
        lock.lock()
        defer { lock.unlock() }
        cache.removeAll()
    }
}

private let melFilterbankCache = MelFilterbankCache()

// MARK: - Mel Filterbank

/// Create a mel-scale filterbank matrix.
///
/// Results are cached for repeated calls with identical parameters.
///
/// - Parameters:
///   - nFFT: FFT size.
///   - config: Mel configuration.
/// - Returns: Mel filterbank matrix of shape (nMels, nFFT // 2 + 1).
/// - Throws: `MelError` if parameters are invalid.
///
/// Example:
/// ```swift
/// let melFB = try melFilterbank(nFFT: 2048, config: MelConfig(sampleRate: 22050, nMels: 128))
/// // melFB.shape == [128, 1025]
/// ```
public func melFilterbank(nFFT: Int, config: MelConfig = MelConfig()) throws -> MLXArray {
    // Validate parameters
    guard config.nMels > 0 else {
        throw MelError.invalidParameter("nMels must be positive, got \(config.nMels)")
    }
    guard config.fMin >= 0 else {
        throw MelError.invalidParameter("fMin must be non-negative, got \(config.fMin)")
    }

    let fMax = config.resolvedFMax
    let nyquist = Float(config.sampleRate) / 2.0

    guard config.fMin < fMax else {
        throw MelError.invalidParameter("fMin (\(config.fMin)) must be less than fMax (\(fMax))")
    }
    guard fMax <= nyquist else {
        throw MelError.invalidParameter("fMax (\(fMax)) cannot exceed Nyquist frequency (\(nyquist))")
    }

    // Check cache
    let cacheKey = "\(config.sampleRate)_\(nFFT)_\(config.nMels)_\(config.fMin)_\(fMax)_\(config.htk)_\(config.norm?.rawValue ?? "none")"

    if let cached = melFilterbankCache.get(key: cacheKey) {
        return cached
    }

    // Compute filterbank using Accelerate for precision
    let filterbank = computeMelFilterbank(
        sampleRate: config.sampleRate,
        nFFT: nFFT,
        nMels: config.nMels,
        fMin: Double(config.fMin),
        fMax: Double(fMax),
        htk: config.htk,
        norm: config.norm
    )

    // Cache and return
    melFilterbankCache.set(key: cacheKey, value: filterbank)

    return filterbank
}

/// Compute mel filterbank using double precision.
private func computeMelFilterbank(
    sampleRate: Int,
    nFFT: Int,
    nMels: Int,
    fMin: Double,
    fMax: Double,
    htk: Bool,
    norm: MelNorm?
) -> MLXArray {
    // Number of frequency bins
    let nFreqs = 1 + nFFT / 2

    // Frequencies of FFT bins
    var fftFreqs = [Double](repeating: 0, count: nFreqs)
    let freqStep = Double(sampleRate) / 2.0 / Double(nFreqs - 1)
    for i in 0..<nFreqs {
        fftFreqs[i] = Double(i) * freqStep
    }

    // Mel scale boundaries
    let melMin = hzToMel([fMin], htk: htk)[0]
    let melMax = hzToMel([fMax], htk: htk)[0]

    // Mel points: nMels + 2 points (including edges)
    var melPoints = [Double](repeating: 0, count: nMels + 2)
    let melStep = (melMax - melMin) / Double(nMels + 1)
    for i in 0..<(nMels + 2) {
        melPoints[i] = melMin + Double(i) * melStep
    }

    // Convert back to Hz
    let hzPoints = melToHz(melPoints, htk: htk)

    // Create filterbank matrix
    var filterbank = [Float](repeating: 0, count: nMels * nFreqs)

    // Build triangular filters
    for m in 0..<nMels {
        let fLower = hzPoints[m]
        let fCenter = hzPoints[m + 1]
        let fUpper = hzPoints[m + 2]

        for k in 0..<nFreqs {
            let freq = fftFreqs[k]

            // Lower slope: (freq - fLower) / (fCenter - fLower)
            // Upper slope: (fUpper - freq) / (fUpper - fCenter)
            var value: Double = 0

            if freq >= fLower && freq <= fCenter {
                let denom = fCenter - fLower
                if denom > 1e-10 {
                    value = (freq - fLower) / denom
                }
            } else if freq > fCenter && freq <= fUpper {
                let denom = fUpper - fCenter
                if denom > 1e-10 {
                    value = (fUpper - freq) / denom
                }
            }

            filterbank[m * nFreqs + k] = Float(max(0, value))
        }
    }

    // Apply normalization
    if norm == .slaney {
        // Normalize by bandwidth (area under each filter = 1)
        for m in 0..<nMels {
            let bandwidth = hzPoints[m + 2] - hzPoints[m]
            if bandwidth > 1e-10 {
                let normFactor = Float(2.0 / bandwidth)
                for k in 0..<nFreqs {
                    filterbank[m * nFreqs + k] *= normFactor
                }
            }
        }
    }

    // Convert to MLXArray
    return MLXArray(filterbank).reshaped([nMels, nFreqs])
}

// MARK: - Mel Spectrogram

/// Compute mel spectrogram from audio waveform.
///
/// - Parameters:
///   - y: Audio waveform. Shape: (samples,) or (batch, samples).
///   - nFFT: FFT size. Default: 2048.
///   - hopLength: Hop length. Default: nFFT / 4.
///   - stftConfig: STFT configuration.
///   - melConfig: Mel configuration.
///   - power: Exponent for magnitude spectrogram. 1.0 for amplitude, 2.0 for power. Default: 2.0.
/// - Returns: Mel spectrogram.
///   - Shape: (nMels, nFrames) for 1D input.
///   - Shape: (batch, nMels, nFrames) for 2D input.
/// - Throws: `STFTError` or `MelError` if parameters are invalid.
///
/// Example:
/// ```swift
/// let melSpec = try melspectrogram(signal, nFFT: 2048, melConfig: MelConfig(nMels: 128))
/// ```
public func melspectrogram(
    _ y: MLXArray,
    nFFT: Int = STFTDefaults.nFFT,
    hopLength: Int? = nil,
    stftConfig: STFTConfig? = nil,
    melConfig: MelConfig = MelConfig(),
    power: Float = 2.0
) throws -> MLXArray {
    // Create STFT config
    var config = stftConfig ?? STFTConfig()
    config.nFFT = nFFT
    if let hop = hopLength {
        config.hopLength = hop
    }

    // Compute STFT
    let S = try stft(y, config: config)

    // Get mel filterbank
    let melBasis = try melFilterbank(nFFT: nFFT, config: melConfig)

    // Compute magnitude spectrogram
    var magSpec = S.magnitude()

    // Apply power
    if power != 1.0 {
        magSpec = MLX.pow(magSpec, MLXArray(power))
    }

    // Apply mel filterbank: melBasis @ magSpec
    // melBasis: (nMels, freqBins)
    // magSpec: (freqBins, nFrames) or (batch, freqBins, nFrames)
    let isBatched = magSpec.ndim == 3

    if isBatched {
        // For batched input, we need to handle the batch dimension
        // melSpec[b] = melBasis @ magSpec[b]
        let batchSize = magSpec.shape[0]
        var melSpecs = [MLXArray]()

        for b in 0..<batchSize {
            let batchMag = magSpec[b]  // (freqBins, nFrames)
            let batchMel = MLX.matmul(melBasis, batchMag)
            melSpecs.append(batchMel.expandedDimensions(axis: 0))
        }

        return MLX.concatenated(melSpecs, axis: 0)
    } else {
        // Single input: direct matmul
        return MLX.matmul(melBasis, magSpec)
    }
}

// MARK: - Decibel Conversion

/// Convert power spectrogram to decibel scale.
///
/// - Parameters:
///   - S: Power spectrogram.
///   - ref: Reference value for dB computation. Default: 1.0.
///   - amin: Minimum amplitude to avoid log(0). Default: 1e-10.
///   - topDb: Maximum dynamic range in dB. Default: 80.0.
/// - Returns: Spectrogram in decibels.
///
/// Formula: S_db = 10 * log10(S / ref)
/// Clipped to [max(S_db) - topDb, max(S_db)]
public func powerToDb(
    _ S: MLXArray,
    ref: Float = DecibelConstants.ref,
    amin: Float = DecibelConstants.amin,
    topDb: Float = DecibelConstants.topDb
) -> MLXArray {
    // Clip to minimum amplitude
    let clipped = MLX.maximum(S, MLXArray(amin))

    // Convert to dB: 10 * log10(S / ref)
    let refArray = MLXArray(ref)
    var SDb = 10.0 * MLX.log10(clipped / refArray)

    // Apply top_db clipping
    if topDb > 0 {
        let maxDb = MLX.max(SDb)
        let minDb = maxDb - MLXArray(topDb)
        SDb = MLX.maximum(SDb, minDb)
    }

    return SDb
}

/// Convert amplitude spectrogram to decibel scale.
///
/// - Parameters:
///   - S: Amplitude spectrogram.
///   - ref: Reference value for dB computation. Default: 1.0.
///   - amin: Minimum amplitude to avoid log(0). Default: 1e-5.
///   - topDb: Maximum dynamic range in dB. Default: 80.0.
/// - Returns: Spectrogram in decibels.
///
/// Formula: S_db = 20 * log10(S / ref)
public func amplitudeToDb(
    _ S: MLXArray,
    ref: Float = DecibelConstants.ref,
    amin: Float = 1e-5,
    topDb: Float = DecibelConstants.topDb
) -> MLXArray {
    // Clip to minimum amplitude
    let clipped = MLX.maximum(S, MLXArray(amin))

    // Convert to dB: 20 * log10(S / ref)
    let refArray = MLXArray(ref)
    var SDb = 20.0 * MLX.log10(clipped / refArray)

    // Apply top_db clipping
    if topDb > 0 {
        let maxDb = MLX.max(SDb)
        let minDb = maxDb - MLXArray(topDb)
        SDb = MLX.maximum(SDb, minDb)
    }

    return SDb
}

// MARK: - Errors

/// Errors that can occur during mel operations.
public enum MelError: Error, LocalizedError {
    case invalidParameter(String)

    public var errorDescription: String? {
        switch self {
        case .invalidParameter(let message):
            return "Invalid mel parameter: \(message)"
        }
    }
}
