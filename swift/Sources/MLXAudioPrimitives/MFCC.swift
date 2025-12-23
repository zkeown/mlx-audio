// MFCC.swift
// Mel-frequency cepstral coefficients (MFCC) and DCT.
//
// Provides MFCC extraction with DCT computed via cached basis matrices.

import Accelerate
import Foundation
@preconcurrency import MLX

// MARK: - Configuration

/// Normalization mode for DCT.
public enum DCTNorm: String, Sendable {
    /// Orthonormal normalization.
    case ortho
}

/// Configuration for MFCC extraction.
public struct MFCCConfig: Sendable {
    /// Number of MFCCs to return. Default: 20.
    public var nMFCC: Int

    /// DCT type. Only type 2 is supported. Default: 2.
    public var dctType: Int

    /// DCT normalization. Default: .ortho.
    public var norm: DCTNorm?

    /// Liftering coefficient. If > 0, apply cepstral filtering. Default: 0.
    public var lifter: Int

    /// Creates a new MFCC configuration with default values.
    public init(
        nMFCC: Int = MFCCDefaults.nMFCC,
        dctType: Int = 2,
        norm: DCTNorm? = .ortho,
        lifter: Int = 0
    ) {
        self.nMFCC = nMFCC
        self.dctType = dctType
        self.norm = norm
        self.lifter = lifter
    }
}

// MARK: - DCT Cache

/// Thread-safe cache for DCT matrices using NSLock.
private final class DCTCache: @unchecked Sendable {
    private var cache: [String: MLXArray] = [:]
    private let maxSize: Int
    private let lock = NSLock()

    init(maxSize: Int = 32) {
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

private let dctCache = DCTCache()

// MARK: - DCT

/// Discrete Cosine Transform (DCT-II).
///
/// Computes the DCT using a pre-computed basis matrix.
///
/// - Parameters:
///   - x: Input signal.
///   - n: Number of output coefficients. Default: input size along axis.
///   - axis: Axis along which to compute DCT. Default: -1.
///   - norm: Normalization mode. Default: .ortho.
/// - Returns: DCT coefficients.
/// - Throws: `MFCCError` if type is not 2.
///
/// DCT-II formula: C[k, n] = cos(pi * k * (2n + 1) / (2N))
///
/// Ortho normalization:
/// - DC component (k=0): Scale by 1/sqrt(N)
/// - AC components (k>=1): Scale by sqrt(2/N)
public func dct(
    _ x: MLXArray,
    n: Int? = nil,
    axis: Int = -1,
    norm: DCTNorm? = .ortho
) throws -> MLXArray {
    // Get input size along axis
    let resolvedAxis = axis < 0 ? x.ndim + axis : axis
    let inputSize = x.shape[resolvedAxis]
    let outputSize = n ?? inputSize

    // Get cached DCT matrix
    let dctMatrix = try getDCTMatrix(nOut: outputSize, nIn: inputSize, norm: norm)

    // Apply DCT: output = x @ dct_matrix.T
    // We need to move the axis to the last position for matmul
    var input = x
    if resolvedAxis != x.ndim - 1 {
        // Move axis to last position
        var axes = Array(0..<x.ndim)
        axes.remove(at: resolvedAxis)
        axes.append(resolvedAxis)
        input = x.transposed(axes: axes)
    }

    // dctMatrix: (nOut, nIn)
    // input: (..., nIn)
    // output: (..., nOut)
    var result = MLX.matmul(input, dctMatrix.transposed())

    // Move axis back if needed
    if resolvedAxis != x.ndim - 1 {
        // Move last axis back to original position
        var axes = Array(0..<result.ndim)
        let lastAxis = axes.removeLast()
        axes.insert(lastAxis, at: resolvedAxis)
        result = result.transposed(axes: axes)
    }

    return result
}

/// Get or compute DCT-II basis matrix.
private func getDCTMatrix(nOut: Int, nIn: Int, norm: DCTNorm?) throws -> MLXArray {
    let cacheKey = "\(nOut)_\(nIn)_\(norm?.rawValue ?? "none")"

    if let cached = dctCache.get(key: cacheKey) {
        return cached
    }

    // Compute DCT matrix using double precision
    let matrix = computeDCTMatrix(nOut: nOut, nIn: nIn, norm: norm)

    dctCache.set(key: cacheKey, value: matrix)

    return matrix
}

/// Compute DCT-II basis matrix.
///
/// DCT-II formula: C[k, n] = cos(pi * k * (2n + 1) / (2N))
private func computeDCTMatrix(nOut: Int, nIn: Int, norm: DCTNorm?) -> MLXArray {
    var matrix = [Float](repeating: 0, count: nOut * nIn)

    let N = Double(nIn)

    for k in 0..<nOut {
        for n in 0..<nIn {
            // DCT-II formula
            let value = cos(Double.pi * Double(k) * (2.0 * Double(n) + 1.0) / (2.0 * N))
            matrix[k * nIn + n] = Float(value)
        }
    }

    // Apply ortho normalization
    if norm == .ortho {
        // DC component (k=0): Scale by 1/sqrt(N)
        let dcScale = Float(1.0 / sqrt(N))
        for n in 0..<nIn {
            matrix[n] *= dcScale
        }

        // AC components (k>=1): Scale by sqrt(2/N)
        let acScale = Float(sqrt(2.0 / N))
        for k in 1..<nOut {
            for n in 0..<nIn {
                matrix[k * nIn + n] *= acScale
            }
        }
    }

    return MLXArray(matrix).reshaped([nOut, nIn])
}

// MARK: - MFCC

/// Compute Mel-frequency cepstral coefficients (MFCCs).
///
/// MFCCs are computed from a power mel spectrogram by applying
/// the discrete cosine transform (DCT) to the log-power mel spectrogram.
///
/// - Parameters:
///   - y: Audio waveform. Shape: (samples,) or (batch, samples). Optional if S is provided.
///   - S: Pre-computed log-power mel spectrogram. If provided, y is ignored.
///   - sampleRate: Sample rate. Default: 22050.
///   - stftConfig: STFT configuration.
///   - melConfig: Mel configuration.
///   - mfccConfig: MFCC configuration.
/// - Returns: MFCC features.
///   - Shape: (nMFCC, nFrames) for 1D input.
///   - Shape: (batch, nMFCC, nFrames) for batched input.
/// - Throws: `MFCCError` if parameters are invalid.
///
/// Example:
/// ```swift
/// let mfccs = try mfcc(signal, sampleRate: 22050, mfccConfig: MFCCConfig(nMFCC: 13))
/// // mfccs.shape == [13, nFrames]
/// ```
public func mfcc(
    _ y: MLXArray? = nil,
    S: MLXArray? = nil,
    sampleRate: Int = MFCCDefaults.sampleRate,
    stftConfig: STFTConfig = STFTConfig(),
    melConfig: MelConfig = MelConfig(),
    mfccConfig: MFCCConfig = MFCCConfig()
) throws -> MLXArray {
    guard mfccConfig.nMFCC > 0 else {
        throw MFCCError.invalidParameter("nMFCC must be positive, got \(mfccConfig.nMFCC)")
    }

    guard mfccConfig.dctType == 2 else {
        throw MFCCError.invalidParameter("Only DCT type 2 is supported, got \(mfccConfig.dctType)")
    }

    // Track if S was provided directly (assumed to be log-power already)
    let sWasProvided = S != nil

    var spectrogram: MLXArray

    if let providedS = S {
        spectrogram = providedS
    } else if let audio = y {
        // Compute mel spectrogram from waveform
        var config = melConfig
        config.sampleRate = sampleRate
        spectrogram = try melspectrogram(
            audio,
            nFFT: stftConfig.nFFT,
            hopLength: stftConfig.hopLength,
            stftConfig: stftConfig,
            melConfig: config,
            power: 2.0  // Power spectrogram
        )
    } else {
        throw MFCCError.invalidParameter("Either y or S must be provided")
    }

    // Handle batched vs non-batched
    let isBatched = spectrogram.ndim == 3
    if !isBatched {
        spectrogram = spectrogram.expandedDimensions(axis: 0)
    }

    // Convert to log scale (dB) only if we computed the mel spectrogram
    var SDb: MLXArray
    if sWasProvided {
        SDb = spectrogram  // Already in log-power format
    } else {
        SDb = powerToDb(spectrogram, ref: 1.0, amin: 1e-10, topDb: 80.0)
    }

    // SDb shape: (batch, nMels, nFrames)
    // Apply DCT along the mel axis (axis=1)
    var M = try dct(SDb, n: mfccConfig.nMFCC, axis: 1, norm: mfccConfig.norm)

    // Apply liftering if specified
    if mfccConfig.lifter > 0 {
        let lifterCoeffs = computeLifterCoefficients(
            nMFCC: mfccConfig.nMFCC,
            lifter: mfccConfig.lifter
        )
        // lifterCoeffs shape: (nMFCC, 1) for broadcasting
        M = M * lifterCoeffs
    }

    // Remove batch dimension if input was not batched
    if !isBatched {
        M = M.squeezed(axis: 0)
    }

    return M
}

/// Compute liftering coefficients.
///
/// Liftering formula: lift[n] = 1 + (L/2) * sin(pi * (n+1) / L)
///
/// Liftering de-emphasizes higher-order coefficients to focus on formants
/// rather than pitch harmonics.
private func computeLifterCoefficients(nMFCC: Int, lifter: Int) -> MLXArray {
    var coeffs = [Float](repeating: 0, count: nMFCC)

    let L = Double(lifter)
    for n in 0..<nMFCC {
        coeffs[n] = Float(1.0 + (L / 2.0) * sin(Double.pi * Double(n + 1) / L))
    }

    // Shape: (nMFCC, 1) for broadcasting with (nMFCC, nFrames)
    return MLXArray(coeffs).reshaped([nMFCC, 1])
}

// MARK: - Delta Features

/// Compute delta (derivative) features.
///
/// Delta features capture temporal dynamics by computing local derivatives.
/// Uses simple finite differences (for librosa compatibility, consider using
/// Savitzky-Golay filtering).
///
/// - Parameters:
///   - data: Input feature matrix (e.g., MFCCs). Shape: (nFeatures, nFrames).
///   - width: Width of the delta window. Must be odd and >= 3. Default: 9.
///   - order: Order of the derivative. 1 for delta, 2 for delta-delta. Default: 1.
///   - axis: Axis along which to compute deltas (time axis). Default: -1.
/// - Returns: Delta features. Same shape as input.
/// - Throws: `MFCCError` if parameters are invalid.
///
/// Example:
/// ```swift
/// let mfccs = try mfcc(signal, sampleRate: 22050, mfccConfig: MFCCConfig(nMFCC: 13))
/// let delta1 = try delta(mfccs, width: 9, order: 1)
/// let delta2 = try delta(mfccs, width: 9, order: 2)
/// let features = MLX.concatenated([mfccs, delta1, delta2], axis: 0)
/// ```
public func delta(
    _ data: MLXArray,
    width: Int = 9,
    order: Int = 1,
    axis: Int = -1
) throws -> MLXArray {
    guard width >= 3 else {
        throw MFCCError.invalidParameter("width must be >= 3, got \(width)")
    }
    guard width % 2 == 1 else {
        throw MFCCError.invalidParameter("width must be odd, got \(width)")
    }
    guard order >= 1 else {
        throw MFCCError.invalidParameter("order must be >= 1, got \(order)")
    }

    let resolvedAxis = axis < 0 ? data.ndim + axis : axis

    guard data.shape[resolvedAxis] >= width else {
        throw MFCCError.invalidParameter(
            "data.shape[\(axis)]=\(data.shape[resolvedAxis]) must be >= width=\(width)"
        )
    }

    // Compute delta using regression weights
    // For order 1: weights = [-n, -(n-1), ..., -1, 0, 1, ..., n-1, n] / sum(i^2)
    let halfWidth = width / 2
    var weights = [Float](repeating: 0, count: width)

    var sumSquared: Float = 0
    for i in -halfWidth...halfWidth {
        sumSquared += Float(i * i)
    }

    for i in 0..<width {
        let offset = i - halfWidth
        weights[i] = Float(offset) / sumSquared
    }

    // Apply convolution for first-order delta
    var result = applyDeltaConvolution(data, weights: weights, axis: resolvedAxis)

    // For higher orders, apply recursively
    for _ in 1..<order {
        result = applyDeltaConvolution(result, weights: weights, axis: resolvedAxis)
    }

    return result
}

/// Apply delta convolution along an axis.
private func applyDeltaConvolution(
    _ data: MLXArray,
    weights: [Float],
    axis: Int
) -> MLXArray {
    let width = weights.count
    let halfWidth = width / 2
    let length = data.shape[axis]

    // Pad data for convolution
    // Use edge padding for border handling
    var padWidths = [IntOrPair](repeating: [0, 0], count: data.ndim)
    padWidths[axis] = [halfWidth, halfWidth]

    let padded = MLX.padded(data, widths: padWidths, mode: .edge)

    // Apply convolution using correlation
    // For each output position, compute weighted sum
    let weightsArray = MLXArray(weights)

    // Simple implementation: gather and weight
    var slices = [MLXArray]()
    for i in 0..<width {
        // Extract slice at offset i
        var sliceRanges = [any MLXArrayIndex](repeating: 0..., count: data.ndim)
        sliceRanges[axis] = i..<(i + length)

        let slice = padded[sliceRanges]
        slices.append(slice * weightsArray[i])
    }

    // Sum all weighted slices
    var result = slices[0]
    for i in 1..<slices.count {
        result = result + slices[i]
    }

    return result
}

// MARK: - Errors

/// Errors that can occur during MFCC operations.
public enum MFCCError: Error, LocalizedError {
    case invalidParameter(String)

    public var errorDescription: String? {
        switch self {
        case .invalidParameter(let message):
            return "Invalid MFCC parameter: \(message)"
        }
    }
}
