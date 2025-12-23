// ComplexArray.swift
// Type-safe complex array handling for FFT operations.
//
// Wraps separate real and imaginary MLX arrays, providing
// operations for magnitude, phase, and complex arithmetic.

import Foundation
@preconcurrency import MLX

/// A type-safe wrapper for complex arrays, storing real and imaginary parts separately.
///
/// MLX-Swift represents complex numbers with separate real and imaginary arrays.
/// This struct provides convenient operations for audio processing tasks like
/// computing magnitude spectrograms and phase information.
///
/// Example:
/// ```swift
/// let stftResult = stft(signal)
/// let magnitude = stftResult.magnitude()
/// let phase = stftResult.phase()
/// ```
public struct ComplexArray: @unchecked Sendable {
    /// The real component of the complex array.
    public let real: MLXArray

    /// The imaginary component of the complex array.
    public let imag: MLXArray

    /// The shape of the complex array (same for both real and imag).
    public var shape: [Int] { real.shape }

    /// The number of dimensions.
    public var ndim: Int { real.ndim }

    /// Creates a complex array from real and imaginary components.
    ///
    /// - Parameters:
    ///   - real: The real component array.
    ///   - imag: The imaginary component array.
    /// - Precondition: Both arrays must have the same shape.
    public init(real: MLXArray, imag: MLXArray) {
        precondition(
            real.shape == imag.shape,
            "Real and imaginary arrays must have the same shape"
        )
        self.real = real
        self.imag = imag
    }

    /// Creates a complex array with zero imaginary part.
    ///
    /// - Parameter real: The real component array.
    public init(real: MLXArray) {
        self.real = real
        self.imag = MLXArray.zeros(real.shape, type: Float.self)
    }

    /// Creates a complex array from polar coordinates (magnitude and phase).
    ///
    /// - Parameters:
    ///   - magnitude: The magnitude (absolute value) array.
    ///   - phase: The phase (angle in radians) array.
    /// - Returns: A complex array where real = magnitude * cos(phase), imag = magnitude * sin(phase).
    public static func fromPolar(magnitude: MLXArray, phase: MLXArray) -> ComplexArray {
        let real = magnitude * MLX.cos(phase)
        let imag = magnitude * MLX.sin(phase)
        return ComplexArray(real: real, imag: imag)
    }

    // MARK: - Basic Operations

    /// Computes the magnitude (absolute value) of the complex array.
    ///
    /// For a complex number z = a + bi, magnitude = sqrt(a^2 + b^2).
    ///
    /// - Returns: A real-valued array containing the magnitude at each position.
    public func magnitude() -> MLXArray {
        // sqrt(real^2 + imag^2)
        return MLX.sqrt(real * real + imag * imag)
    }

    /// Computes the phase (angle) of the complex array.
    ///
    /// For a complex number z = a + bi, phase = atan2(b, a).
    ///
    /// - Returns: A real-valued array containing the phase in radians at each position.
    public func phase() -> MLXArray {
        return atan2(imag, real)
    }

    /// Returns the complex conjugate.
    ///
    /// For a complex number z = a + bi, conjugate = a - bi.
    ///
    /// - Returns: A new complex array with negated imaginary part.
    public func conjugate() -> ComplexArray {
        return ComplexArray(real: real, imag: -imag)
    }

    /// Computes the squared magnitude (avoids sqrt for performance).
    ///
    /// - Returns: A real-valued array containing |z|^2 at each position.
    public func magnitudeSquared() -> MLXArray {
        return real * real + imag * imag
    }

    // MARK: - Indexing

    /// Subscript access for single index.
    public subscript(index: Int) -> ComplexArray {
        return ComplexArray(real: real[index], imag: imag[index])
    }

    /// Subscript access for range (first axis).
    public subscript(range: Range<Int>) -> ComplexArray {
        let indices = MLXArray(range.map { Int32($0) })
        return ComplexArray(real: real.take(indices, axis: 0), imag: imag.take(indices, axis: 0))
    }

    /// Slice the complex array on a specific axis.
    ///
    /// - Parameters:
    ///   - axis: The axis to slice on.
    ///   - range: The range to select.
    /// - Returns: A new complex array with the slice applied.
    public func slice(axis: Int, range: Range<Int>) -> ComplexArray {
        let realSliced = real.take(MLXArray(range.map { Int32($0) }), axis: axis)
        let imagSliced = imag.take(MLXArray(range.map { Int32($0) }), axis: axis)
        return ComplexArray(real: realSliced, imag: imagSliced)
    }

    // MARK: - Shape Operations

    /// Transpose the complex array.
    ///
    /// - Parameter axes: The permutation of axes. If nil, reverses all axes.
    /// - Returns: A new complex array with transposed axes.
    public func transposed(_ axes: [Int]? = nil) -> ComplexArray {
        if let axes = axes {
            return ComplexArray(
                real: real.transposed(axes: axes),
                imag: imag.transposed(axes: axes)
            )
        } else {
            return ComplexArray(
                real: real.transposed(),
                imag: imag.transposed()
            )
        }
    }

    /// Reshape the complex array.
    ///
    /// - Parameter shape: The new shape.
    /// - Returns: A new complex array with the specified shape.
    public func reshaped(_ shape: [Int]) -> ComplexArray {
        return ComplexArray(
            real: real.reshaped(shape),
            imag: imag.reshaped(shape)
        )
    }

    /// Add a dimension at the specified axis.
    ///
    /// - Parameter axis: The axis at which to add a dimension.
    /// - Returns: A new complex array with an additional dimension.
    public func expandedDimensions(axis: Int) -> ComplexArray {
        return ComplexArray(
            real: real.expandedDimensions(axis: axis),
            imag: imag.expandedDimensions(axis: axis)
        )
    }

    /// Remove a dimension at the specified axis.
    ///
    /// - Parameter axis: The axis to squeeze.
    /// - Returns: A new complex array with the dimension removed.
    public func squeezed(axis: Int? = nil) -> ComplexArray {
        if let axis = axis {
            return ComplexArray(
                real: real.squeezed(axis: axis),
                imag: imag.squeezed(axis: axis)
            )
        } else {
            return ComplexArray(
                real: real.squeezed(),
                imag: imag.squeezed()
            )
        }
    }
}

// MARK: - Arithmetic Operations

extension ComplexArray {
    /// Element-wise addition of two complex arrays.
    public static func + (lhs: ComplexArray, rhs: ComplexArray) -> ComplexArray {
        return ComplexArray(
            real: lhs.real + rhs.real,
            imag: lhs.imag + rhs.imag
        )
    }

    /// Element-wise subtraction of two complex arrays.
    public static func - (lhs: ComplexArray, rhs: ComplexArray) -> ComplexArray {
        return ComplexArray(
            real: lhs.real - rhs.real,
            imag: lhs.imag - rhs.imag
        )
    }

    /// Element-wise multiplication of two complex arrays.
    ///
    /// (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
    public static func * (lhs: ComplexArray, rhs: ComplexArray) -> ComplexArray {
        let real = lhs.real * rhs.real - lhs.imag * rhs.imag
        let imag = lhs.real * rhs.imag + lhs.imag * rhs.real
        return ComplexArray(real: real, imag: imag)
    }

    /// Element-wise division of two complex arrays.
    ///
    /// (a + bi) / (c + di) = ((ac + bd) + (bc - ad)i) / (c^2 + d^2)
    public static func / (lhs: ComplexArray, rhs: ComplexArray) -> ComplexArray {
        let denom = rhs.real * rhs.real + rhs.imag * rhs.imag
        let real = (lhs.real * rhs.real + lhs.imag * rhs.imag) / denom
        let imag = (lhs.imag * rhs.real - lhs.real * rhs.imag) / denom
        return ComplexArray(real: real, imag: imag)
    }

    /// Multiply complex array by a real scalar.
    public static func * (lhs: ComplexArray, rhs: MLXArray) -> ComplexArray {
        return ComplexArray(real: lhs.real * rhs, imag: lhs.imag * rhs)
    }

    /// Multiply complex array by a real scalar.
    public static func * (lhs: MLXArray, rhs: ComplexArray) -> ComplexArray {
        return ComplexArray(real: lhs * rhs.real, imag: lhs * rhs.imag)
    }

    /// Multiply complex array by a Float scalar.
    public static func * (lhs: ComplexArray, rhs: Float) -> ComplexArray {
        let scalar = MLXArray(rhs)
        return ComplexArray(real: lhs.real * scalar, imag: lhs.imag * scalar)
    }

    /// Divide complex array by a real scalar.
    public static func / (lhs: ComplexArray, rhs: MLXArray) -> ComplexArray {
        return ComplexArray(real: lhs.real / rhs, imag: lhs.imag / rhs)
    }

    /// Divide complex array by a Float scalar.
    public static func / (lhs: ComplexArray, rhs: Float) -> ComplexArray {
        let scalar = MLXArray(rhs)
        return ComplexArray(real: lhs.real / scalar, imag: lhs.imag / scalar)
    }

    /// Negate the complex array.
    public static prefix func - (array: ComplexArray) -> ComplexArray {
        return ComplexArray(real: -array.real, imag: -array.imag)
    }
}

// MARK: - Factory Methods

extension ComplexArray {
    /// Create a complex array of zeros.
    ///
    /// - Parameter shape: The shape of the array.
    /// - Returns: A complex array with all zeros.
    public static func zeros(_ shape: [Int]) -> ComplexArray {
        return ComplexArray(
            real: MLXArray.zeros(shape, type: Float.self),
            imag: MLXArray.zeros(shape, type: Float.self)
        )
    }

    /// Create a complex array of ones (real part only).
    ///
    /// - Parameter shape: The shape of the array.
    /// - Returns: A complex array with all ones in the real part.
    public static func ones(_ shape: [Int]) -> ComplexArray {
        return ComplexArray(
            real: MLXArray.ones(shape, type: Float.self),
            imag: MLXArray.zeros(shape, type: Float.self)
        )
    }
}

// MARK: - CustomStringConvertible

extension ComplexArray: CustomStringConvertible {
    public var description: String {
        return "ComplexArray(shape: \(shape), real: \(real), imag: \(imag))"
    }
}
