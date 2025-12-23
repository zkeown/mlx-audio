// Window.swift
// Window functions for audio signal processing.
//
// Provides window functions compatible with librosa/scipy conventions.
// Uses Accelerate framework for double-precision computation.

import Accelerate
import Foundation
@preconcurrency import MLX

// MARK: - Window Types

/// Supported window function types.
public enum WindowType: String, CaseIterable, Sendable {
    /// Hann window: w[k] = 0.5 - 0.5 * cos(2*pi*k/(n-1))
    case hann

    /// Hamming window: w[k] = 0.54 - 0.46 * cos(2*pi*k/(n-1))
    case hamming

    /// Blackman window: w[k] = 0.42 - 0.5*cos(...) + 0.08*cos(...)
    case blackman

    /// Bartlett (triangular) window: w[k] = 1 - |2*k/(n-1) - 1|
    case bartlett

    /// Rectangular (boxcar) window: all ones
    case rectangular

    // Aliases
    /// Alias for hann window
    case hanning

    /// Alias for bartlett window
    case triangular

    /// Alias for rectangular window
    case boxcar

    /// Alias for rectangular window
    case ones

    /// Returns the canonical window type (resolves aliases).
    var canonical: WindowType {
        switch self {
        case .hanning: return .hann
        case .triangular: return .bartlett
        case .boxcar, .ones: return .rectangular
        default: return self
        }
    }

    /// Returns the coefficients for generalized cosine windows.
    var cosineCoefficients: [Double]? {
        switch canonical {
        case .hann: return WindowCoefficients.hann
        case .hamming: return WindowCoefficients.hamming
        case .blackman: return WindowCoefficients.blackman
        default: return nil
        }
    }
}

// MARK: - Window Cache

/// Thread-safe cache for window functions using NSLock.
private final class WindowCache: @unchecked Sendable {
    private var cache: [String: MLXArray] = [:]
    private let maxSize: Int
    private let lock = NSLock()

    init(maxSize: Int = 128) {
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
        // Simple eviction: clear cache if full
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

/// Global window cache instance.
private let windowCache = WindowCache()

// MARK: - Public API

/// Get a window function.
///
/// Results are cached for repeated calls with identical parameters.
///
/// - Parameters:
///   - type: Window type (hann, hamming, blackman, bartlett, rectangular).
///   - length: Length of the window.
///   - periodic: If true, create a periodic window for FFT (DFT-even).
///               If false, create a symmetric window. Default: true.
/// - Returns: Window of shape (length,) with dtype float32.
///
/// Example:
/// ```swift
/// let window = getWindow(.hann, length: 2048)
/// // window.shape == [2048]
/// ```
public func getWindow(_ type: WindowType, length: Int, periodic: Bool = true) -> MLXArray {
    // Check cache first
    let cacheKey = "\(type.rawValue)_\(length)_\(periodic)"

    if let cached = windowCache.get(key: cacheKey) {
        return cached
    }

    // Compute window
    let window = computeWindow(type, length: length, periodic: periodic)

    // Cache the result
    windowCache.set(key: cacheKey, value: window)

    return window
}

/// Clears the window cache.
public func clearWindowCache() {
    windowCache.clear()
}

// MARK: - Window Computation

/// Compute a window function without caching.
private func computeWindow(_ type: WindowType, length: Int, periodic: Bool) -> MLXArray {
    // Edge case: length 1 should always return [1.0]
    if length == 1 {
        return MLXArray.ones([1], type: Float.self)
    }

    let canonical = type.canonical

    // For periodic (fftbins=True), compute n+1 points and drop the last
    let n = periodic ? length + 1 : length

    let window: MLXArray

    switch canonical {
    case .hann, .hamming, .blackman:
        if let coefficients = canonical.cosineCoefficients {
            window = generalizedCosineWindow(n: n, coefficients: coefficients)
        } else {
            // Fallback to rectangular
            window = MLXArray.ones([n], type: Float.self)
        }
    case .bartlett:
        window = bartlettWindow(n: n)
    case .rectangular:
        window = MLXArray.ones([n], type: Float.self)
    default:
        // Should not reach here after canonical resolution
        window = MLXArray.ones([n], type: Float.self)
    }

    // For periodic windows, drop the last sample
    if periodic && n > length {
        // Use slicing to get first 'length' elements
        return window[0..<length]
    }

    return window
}

/// Generalized cosine window with arbitrary coefficients.
///
/// w[k] = a0 - a1*cos(2*pi*k/(n-1)) + a2*cos(4*pi*k/(n-1)) - ...
///
/// Computed in Float64 for precision using Accelerate, then cast to Float32.
private func generalizedCosineWindow(n: Int, coefficients: [Double]) -> MLXArray {
    guard n > 1 else {
        return MLXArray.ones([n], type: Float.self)
    }

    // Use Accelerate for double-precision computation
    var window = [Double](repeating: coefficients[0], count: n)
    let denom = Double(n - 1)

    // Create index array [0, 1, 2, ..., n-1]
    var indices = [Double](repeating: 0, count: n)
    for i in 0..<n {
        indices[i] = Double(i)
    }

    // Add cosine terms with alternating signs
    for (i, coef) in coefficients.dropFirst().enumerated() {
        let termIndex = i + 1
        let sign: Double = (termIndex % 2 == 1) ? -1.0 : 1.0

        // Compute 2 * termIndex * pi * k / (n-1)
        var scaledIndices = [Double](repeating: 0, count: n)
        var scale = 2.0 * Double(termIndex) * Double.pi / denom
        vDSP_vsmulD(indices, 1, &scale, &scaledIndices, 1, vDSP_Length(n))

        // Compute cos of scaled indices
        var cosValues = [Double](repeating: 0, count: n)
        var count = Int32(n)
        vvcos(&cosValues, scaledIndices, &count)

        // Add sign * coef * cos to window
        var signedCoef = sign * coef
        vDSP_vsmaD(cosValues, 1, &signedCoef, window, 1, &window, 1, vDSP_Length(n))
    }

    // Clamp to non-negative for blackman (can have tiny negative values at endpoints)
    if coefficients.count > 2 {
        for i in 0..<n {
            window[i] = max(window[i], 0.0)
        }
    }

    // Convert to Float32 and create MLXArray
    let windowFloat32 = window.map { Float($0) }
    return MLXArray(windowFloat32)
}

/// Bartlett (triangular) window: w[k] = 1 - |2*k/(n-1) - 1|
///
/// Computed in Float64 for precision, then cast to Float32.
private func bartlettWindow(n: Int) -> MLXArray {
    guard n > 1 else {
        return MLXArray.ones([n], type: Float.self)
    }

    var window = [Float](repeating: 0, count: n)
    let denom = Double(n - 1)

    for k in 0..<n {
        let value = 1.0 - abs(2.0 * Double(k) / denom - 1.0)
        window[k] = Float(value)
    }

    return MLXArray(window)
}

// MARK: - Utility Functions

/// Check if a window satisfies the NOLA (Nonzero Overlap-Add) constraint.
///
/// The NOLA constraint ensures that ISTFT is invertible. It requires that
/// the sum of squared windows at every sample position is nonzero.
///
/// - Parameters:
///   - type: Window type.
///   - hopLength: Hop length.
///   - nFFT: FFT size (determines window length for string windows).
///   - tolerance: Tolerance for considering sum as nonzero. Default: 1e-10.
/// - Returns: True if NOLA constraint is satisfied.
///
/// Example:
/// ```swift
/// let isValid = checkNOLA(.hann, hopLength: 512, nFFT: 2048)
/// // isValid == true
/// ```
public func checkNOLA(
    _ type: WindowType,
    hopLength: Int,
    nFFT: Int,
    tolerance: Float = 1e-10
) -> Bool {
    let window = getWindow(type, length: nFFT, periodic: true)
    let windowArray = window.asArray(Float.self)

    // Sum squared windows for each position within a hop
    let step = hopLength
    let nBins = nFFT / step

    var binSums = [Float](repeating: 0, count: step)

    for ii in 0..<nBins {
        let start = ii * step
        let end = min((ii + 1) * step, nFFT)
        for j in start..<end {
            let posInBin = j - start
            binSums[posInBin] += windowArray[j] * windowArray[j]
        }
    }

    // Handle remainder if nFFT is not a multiple of hopLength
    if nFFT % step != 0 {
        let remainder = nFFT % step
        for j in (nFFT - remainder)..<nFFT {
            let posInBin = j - (nFFT - remainder)
            binSums[posInBin] += windowArray[j] * windowArray[j]
        }
    }

    // Check minimum bin sum
    let minSum = binSums.min() ?? 0
    return minSum > tolerance
}
