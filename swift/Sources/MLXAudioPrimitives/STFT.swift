// STFT.swift
// Short-Time Fourier Transform (STFT) and Inverse STFT.
//
// Provides librosa-compatible STFT and ISTFT implementations for MLX-Swift.

import Foundation
@preconcurrency import MLX

// MARK: - Configuration Types

/// Padding mode for signal padding.
public enum STFTPadMode: String, Sendable {
    /// Pad with zeros.
    case constant
    /// Repeat edge values.
    case edge
}

/// Configuration for STFT/ISTFT operations.
public struct STFTConfig: Sendable {
    /// FFT size. Default: 2048.
    public var nFFT: Int

    /// Number of samples between frames. Default: nFFT / 4.
    public var hopLength: Int?

    /// Window length. Default: nFFT.
    public var winLength: Int?

    /// Window function type. Default: .hann.
    public var window: WindowType

    /// If true, pad signal so frame t is centered at y[t * hopLength]. Default: true.
    public var center: Bool

    /// Padding mode if center is true. Default: .constant.
    public var padMode: STFTPadMode

    /// Creates a new STFT configuration with default values.
    public init(
        nFFT: Int = STFTDefaults.nFFT,
        hopLength: Int? = nil,
        winLength: Int? = nil,
        window: WindowType = .hann,
        center: Bool = true,
        padMode: STFTPadMode = .constant
    ) {
        self.nFFT = nFFT
        self.hopLength = hopLength
        self.winLength = winLength
        self.window = window
        self.center = center
        self.padMode = padMode
    }

    /// Resolved hop length (uses default if nil).
    public var resolvedHopLength: Int {
        hopLength ?? nFFT / STFTDefaults.hopLengthRatio
    }

    /// Resolved window length (uses nFFT if nil).
    public var resolvedWinLength: Int {
        winLength ?? nFFT
    }
}

// MARK: - STFT

/// Short-Time Fourier Transform.
///
/// - Parameters:
///   - y: Input signal. Shape: (samples,) or (batch, samples).
///   - config: STFT configuration.
/// - Returns: Complex STFT matrix.
///   - Shape: (nFFT/2 + 1, nFrames) for 1D input.
///   - Shape: (batch, nFFT/2 + 1, nFrames) for 2D input.
/// - Throws: `STFTError` if parameters are invalid.
///
/// Example:
/// ```swift
/// let signal = MLXArray.random.normal([22050])  // 1 second at 22050 Hz
/// let S = try stft(signal, config: STFTConfig(nFFT: 2048, hopLength: 512))
/// // S.shape == [1025, 44]
/// ```
public func stft(_ y: MLXArray, config: STFTConfig = STFTConfig()) throws -> ComplexArray {
    let nFFT = config.nFFT
    let hopLength = config.resolvedHopLength
    let winLength = config.resolvedWinLength

    // Validate parameters
    guard hopLength > 0 else {
        throw STFTError.invalidParameter("hopLength must be positive, got \(hopLength)")
    }
    guard winLength > 0 else {
        throw STFTError.invalidParameter("winLength must be positive, got \(winLength)")
    }
    guard winLength <= nFFT else {
        throw STFTError.invalidParameter("winLength (\(winLength)) must be <= nFFT (\(nFFT))")
    }

    // Handle batched input
    let inputIs1D = y.ndim == 1
    var signal = inputIs1D ? y.expandedDimensions(axis: 0) : y

    // Get window (with caching and potential padding)
    let window = getPaddedWindow(config.window, winLength: winLength, nFFT: nFFT)

    // Center padding if enabled
    if config.center {
        let padLength = nFFT / 2
        signal = padSignal(signal, padLength: padLength, mode: config.padMode)
    }

    // Frame the signal
    let frames = try frameSignal(signal, frameLength: nFFT, hopLength: hopLength)

    // Apply window and compute FFT
    let windowedFrames = frames * window

    // Compute real FFT - returns complex-valued array with dtype complex64
    // MLX-Swift uses top-level rfft function
    let fftResult = rfft(windowedFrames, axis: -1)

    // Extract real and imaginary parts using MLX-Swift API
    let realPart = fftResult.realPart()
    let imagPart = fftResult.imaginaryPart()

    // Create ComplexArray
    var stftMatrix = ComplexArray(real: realPart, imag: imagPart)

    // Transpose to (batch, freqBins, nFrames) to match librosa convention
    stftMatrix = stftMatrix.transposed([0, 2, 1])

    // Remove batch dimension if input was 1D
    if inputIs1D {
        stftMatrix = stftMatrix.squeezed(axis: 0)
    }

    return stftMatrix
}

/// Inverse Short-Time Fourier Transform.
///
/// - Parameters:
///   - stftMatrix: Complex STFT matrix.
///     - Shape: (nFFT/2 + 1, nFrames) for 1D output.
///     - Shape: (batch, nFFT/2 + 1, nFrames) for 2D output.
///   - config: STFT configuration (should match the one used in stft()).
///   - length: If provided, the output is trimmed or zero-padded to this length.
/// - Returns: Reconstructed time-domain signal.
///   - Shape: (samples,) for 2D input.
///   - Shape: (batch, samples) for 3D input.
/// - Throws: `STFTError` if parameters are invalid.
///
/// Example:
/// ```swift
/// let S = try stft(signal, config: config)
/// let reconstructed = try istft(S, config: config, length: signal.shape[0])
/// ```
public func istft(
    _ stftMatrix: ComplexArray,
    config: STFTConfig = STFTConfig(),
    length: Int? = nil
) throws -> MLXArray {
    let nFFT = config.nFFT
    let hopLength = config.resolvedHopLength
    let winLength = config.resolvedWinLength

    // Handle batched input
    let inputIs2D = stftMatrix.ndim == 2
    var matrix = inputIs2D ? stftMatrix.expandedDimensions(axis: 0) : stftMatrix

    let batchSize = matrix.shape[0]
    let freqBins = matrix.shape[1]
    let nFrames = matrix.shape[2]

    // Infer nFFT from frequency bins if needed
    let inferredNFFT = 2 * (freqBins - 1)
    guard inferredNFFT == nFFT else {
        throw STFTError.invalidParameter(
            "STFT matrix frequency bins (\(freqBins)) inconsistent with nFFT (\(nFFT))"
        )
    }

    // Get window
    let window = getPaddedWindow(config.window, winLength: winLength, nFFT: nFFT)

    // Transpose to (batch, nFrames, freqBins) for irfft
    matrix = matrix.transposed([0, 2, 1])

    // Reconstruct complex array for irfft
    // MLX irfft expects a complex64 array: real + i * imag
    let i = MLXArray(real: 0, imaginary: 1)
    let complexInput = matrix.real + i * matrix.imag

    // Inverse FFT: (batch, nFrames, nFFT)
    let frames = irfft(complexInput, n: nFFT, axis: -1)

    // Determine output length
    let paddedLength: Int
    if let targetLength = length {
        if config.center {
            paddedLength = targetLength + nFFT
        } else {
            paddedLength = targetLength
        }
    } else {
        paddedLength = nFFT + (nFrames - 1) * hopLength
    }

    // Perform overlap-add reconstruction
    var y = overlapAddSimple(frames, hopLength: hopLength, window: window, outputLength: paddedLength, batchSize: batchSize)

    // Trim center padding if needed
    if config.center {
        let padLength = nFFT / 2
        if let targetLength = length {
            y = y[0..., padLength..<(padLength + targetLength)]
        } else {
            let endIdx = y.shape[1] - padLength
            if endIdx > padLength {
                y = y[0..., padLength..<endIdx]
            } else {
                y = y[0..., 0..<0]
            }
        }
    } else if let targetLength = length {
        let currentLength = y.shape[1]
        if targetLength < currentLength {
            y = y[0..., 0..<targetLength]
        } else if targetLength > currentLength {
            let padAmount = targetLength - currentLength
            y = MLX.padded(y, widths: [[0, 0], [0, padAmount]])
        }
    }

    // Remove batch dimension if input was 2D
    if inputIs2D {
        y = y.squeezed(axis: 0)
    }

    return y
}

// MARK: - Convenience Functions

/// Compute magnitude spectrogram from complex STFT.
///
/// - Parameter stftMatrix: Complex STFT matrix (output of stft()).
/// - Returns: Magnitude spectrogram (same shape, real-valued).
public func magnitude(_ stftMatrix: ComplexArray) -> MLXArray {
    return stftMatrix.magnitude()
}

/// Compute phase spectrogram from complex STFT.
///
/// - Parameter stftMatrix: Complex STFT matrix (output of stft()).
/// - Returns: Phase spectrogram in radians (same shape, real-valued).
public func phase(_ stftMatrix: ComplexArray) -> MLXArray {
    return stftMatrix.phase()
}

// MARK: - Internal Functions

/// Get window, padding to nFFT if needed.
private func getPaddedWindow(_ type: WindowType, winLength: Int, nFFT: Int) -> MLXArray {
    var window = getWindow(type, length: winLength, periodic: true)

    // Pad if needed (center padding)
    if winLength < nFFT {
        let padLeft = (nFFT - winLength) / 2
        let padRight = nFFT - winLength - padLeft
        window = MLX.padded(window, widths: [[padLeft, padRight]])
    }

    return window
}

/// Pad signal on both sides.
private func padSignal(_ y: MLXArray, padLength: Int, mode: STFTPadMode) -> MLXArray {
    switch mode {
    case .constant:
        return MLX.padded(y, widths: [[0, 0], [padLength, padLength]])
    case .edge:
        return MLX.padded(y, widths: [[0, 0], [padLength, padLength]], mode: .edge)
    }
}

/// Frame signal into overlapping windows.
private func frameSignal(_ y: MLXArray, frameLength: Int, hopLength: Int) throws -> MLXArray {
    let signalLength = y.shape[1]

    guard signalLength >= frameLength else {
        throw STFTError.signalTooShort(
            "Signal length (\(signalLength)) must be >= frame length (\(frameLength))"
        )
    }

    let nFrames = 1 + (signalLength - frameLength) / hopLength

    return frameSignalGather(y, frameLength: frameLength, hopLength: hopLength, nFrames: nFrames)
}

/// Frame signal using gather operations.
private func frameSignalGather(
    _ y: MLXArray,
    frameLength: Int,
    hopLength: Int,
    nFrames: Int
) -> MLXArray {
    let batchSize = y.shape[0]

    // Create index array for gathering
    var indices = [Int32]()
    indices.reserveCapacity(nFrames * frameLength)

    for frame in 0..<nFrames {
        let start = frame * hopLength
        for i in 0..<frameLength {
            indices.append(Int32(start + i))
        }
    }

    let indicesArray = MLXArray(indices)

    // Gather along axis 1 for each batch
    let gathered = y.take(indicesArray, axis: 1)

    // Reshape to (batch, nFrames, frameLength)
    return gathered.reshaped([batchSize, nFrames, frameLength])
}

/// Simple overlap-add implementation.
private func overlapAddSimple(
    _ frames: MLXArray,
    hopLength: Int,
    window: MLXArray,
    outputLength: Int,
    batchSize: Int
) -> MLXArray {
    let nFrames = frames.shape[1]
    let nFFT = frames.shape[2]

    // Apply window to frames
    let windowedFrames = frames * window

    // Simple sum-based overlap-add (not normalized)
    // For a proper implementation, this would use scatter-add operations
    var outputData = [Float](repeating: 0, count: batchSize * outputLength)
    var windowSumData = [Float](repeating: 0, count: outputLength)

    let windowArray = window.asArray(Float.self)
    let framesData = windowedFrames.asArray(Float.self)

    for b in 0..<batchSize {
        for f in 0..<nFrames {
            let start = f * hopLength
            for i in 0..<nFFT {
                let outIdx = start + i
                if outIdx < outputLength {
                    let frameIdx = b * nFrames * nFFT + f * nFFT + i
                    outputData[b * outputLength + outIdx] += framesData[frameIdx]
                    if b == 0 {
                        windowSumData[outIdx] += windowArray[i] * windowArray[i]
                    }
                }
            }
        }
    }

    // Normalize by window sum
    let epsilon = STFTDefaults.windowSumEpsilon
    for b in 0..<batchSize {
        for i in 0..<outputLength {
            let norm = max(windowSumData[i], epsilon)
            outputData[b * outputLength + i] /= norm
        }
    }

    return MLXArray(outputData).reshaped([batchSize, outputLength])
}

// MARK: - Errors

/// Errors that can occur during STFT operations.
public enum STFTError: Error, LocalizedError {
    case invalidParameter(String)
    case signalTooShort(String)

    public var errorDescription: String? {
        switch self {
        case .invalidParameter(let message):
            return "Invalid STFT parameter: \(message)"
        case .signalTooShort(let message):
            return "Signal too short: \(message)"
        }
    }
}
