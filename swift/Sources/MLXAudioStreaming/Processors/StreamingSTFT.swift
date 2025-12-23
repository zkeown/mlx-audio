// StreamingSTFT.swift
// Streaming STFT processor for real-time spectral analysis.
//
// Accumulates audio samples and emits STFT frames when enough data is available.

import Foundation
@preconcurrency import MLX
import MLXAudioPrimitives

// MARK: - Streaming STFT Configuration

/// Configuration for streaming STFT processor.
public struct StreamingSTFTConfiguration: Sendable {
    /// FFT size
    public var nFFT: Int

    /// Hop length (samples between frames)
    public var hopLength: Int

    /// Window type
    public var window: WindowType

    /// Whether to return magnitude spectrogram (vs complex)
    public var returnMagnitude: Bool

    /// Whether to convert to decibels
    public var toDecibels: Bool

    /// Reference value for dB conversion
    public var dbReference: Float

    /// Default configuration for visualization
    public static let visualization = StreamingSTFTConfiguration(
        nFFT: 2048,
        hopLength: 512,
        window: .hann,
        returnMagnitude: true,
        toDecibels: true,
        dbReference: 1.0
    )

    /// Default configuration for processing
    public static let processing = StreamingSTFTConfiguration(
        nFFT: 2048,
        hopLength: 512,
        window: .hann,
        returnMagnitude: false,
        toDecibels: false,
        dbReference: 1.0
    )

    /// Configuration for Whisper-compatible mel spectrograms
    public static let whisper = StreamingSTFTConfiguration(
        nFFT: 400,
        hopLength: 160,
        window: .hann,
        returnMagnitude: true,
        toDecibels: false,
        dbReference: 1.0
    )

    public init(
        nFFT: Int = 2048,
        hopLength: Int = 512,
        window: WindowType = .hann,
        returnMagnitude: Bool = true,
        toDecibels: Bool = false,
        dbReference: Float = 1.0
    ) {
        self.nFFT = nFFT
        self.hopLength = hopLength
        self.window = window
        self.returnMagnitude = returnMagnitude
        self.toDecibels = toDecibels
        self.dbReference = dbReference
    }
}

// MARK: - Streaming STFT

/// Streaming STFT processor for real-time spectral analysis.
///
/// This actor accumulates audio samples and emits STFT frames when enough data
/// is available. It maintains state between process calls for continuous streaming.
///
/// Example:
/// ```swift
/// let stft = StreamingSTFT(configuration: .visualization)
///
/// // In processing loop
/// let spectrogram = try await stft.process(audioChunk)
/// // spectrogram shape: [freqBins, numFrames]
/// ```
public actor StreamingSTFT: StreamingProcessor {
    // MARK: - Properties

    private let configuration: StreamingSTFTConfiguration
    private let stftConfig: STFTConfig

    /// Sample buffer for accumulating input
    private var buffer: [Float] = []

    /// Pre-computed window function
    private var windowArray: MLXArray?

    /// Frame counter for tracking progress
    private var framePosition: Int = 0

    // MARK: - Initialization

    /// Creates a new streaming STFT processor.
    ///
    /// - Parameter configuration: STFT configuration
    public init(configuration: StreamingSTFTConfiguration = .visualization) {
        self.configuration = configuration
        self.stftConfig = STFTConfig(
            nFFT: configuration.nFFT,
            hopLength: configuration.hopLength,
            winLength: configuration.nFFT,
            window: configuration.window,
            center: false,  // No centering in streaming mode
            padMode: .constant
        )
    }

    // MARK: - StreamingProcessor Protocol

    /// Process an audio chunk and return STFT frames.
    ///
    /// Accumulates input samples and emits spectrogram frames when enough
    /// data is available. May return an empty array if not enough samples
    /// have accumulated.
    ///
    /// - Parameter input: Audio samples [samples] or [channels, samples]
    /// - Returns: Spectrogram frames [freqBins, numFrames] or empty if insufficient data
    public func process(_ input: MLXArray) async throws -> MLXArray {
        // Convert input to 1D float array
        let samples: [Float]
        if input.ndim == 1 {
            samples = input.asArray(Float.self)
        } else {
            // Multi-channel: take first channel or average
            let mono = MLX.mean(input, axis: 0)
            samples = mono.asArray(Float.self)
        }

        // Append to buffer
        buffer.append(contentsOf: samples)

        // Calculate how many frames we can emit
        let nFFT = configuration.nFFT
        let hopLength = configuration.hopLength

        guard buffer.count >= nFFT else {
            // Not enough samples yet
            return MLXArray.zeros([nFFT / 2 + 1, 0])
        }

        let numFrames = (buffer.count - nFFT) / hopLength + 1

        guard numFrames > 0 else {
            return MLXArray.zeros([nFFT / 2 + 1, 0])
        }

        // Get or create window
        if windowArray == nil {
            windowArray = window(configuration.window, length: nFFT)
        }
        let win = windowArray!

        // Process frames
        var frames: [MLXArray] = []

        for i in 0..<numFrames {
            let start = i * hopLength
            let end = start + nFFT

            // Extract frame
            let frameData = Array(buffer[start..<end])
            var frame = MLXArray(frameData)

            // Apply window
            frame = frame * win

            // Compute FFT (real-valued input -> complex output)
            let spectrum = MLX.FFT.rfft(frame)

            // Process output based on configuration
            var output: MLXArray
            if configuration.returnMagnitude {
                // Compute magnitude
                output = magnitude(ComplexArray(spectrum))

                if configuration.toDecibels {
                    // Convert to dB
                    let ref = MLXArray(configuration.dbReference)
                    let minVal = MLXArray(Float(1e-10))
                    output = 20 * MLX.log10(MLX.maximum(output, minVal) / ref)
                }
            } else {
                output = spectrum
            }

            frames.append(output)
        }

        // Remove processed samples from buffer (keep overlap)
        let samplesConsumed = numFrames * hopLength
        buffer.removeFirst(samplesConsumed)

        framePosition += numFrames

        // Stack frames: [freqBins, numFrames]
        if frames.isEmpty {
            return MLXArray.zeros([nFFT / 2 + 1, 0])
        }

        let stacked = MLX.stacked(frames, axis: 1)
        return stacked
    }

    /// Reset processor state.
    ///
    /// Clears the sample buffer and resets frame position.
    public func reset() async {
        buffer.removeAll(keepingCapacity: true)
        framePosition = 0
    }

    // MARK: - State Queries

    /// Number of samples currently buffered.
    public var bufferedSamples: Int {
        buffer.count
    }

    /// Number of frames processed since start/reset.
    public var framesProcessed: Int {
        framePosition
    }

    /// Whether enough samples are buffered to emit a frame.
    public var canEmitFrame: Bool {
        buffer.count >= configuration.nFFT
    }

    /// Number of frequency bins in output.
    public var frequencyBins: Int {
        configuration.nFFT / 2 + 1
    }

    /// Frequency resolution in Hz (given sample rate).
    public func frequencyResolution(sampleRate: Int) -> Double {
        Double(sampleRate) / Double(configuration.nFFT)
    }

    /// Time resolution in seconds.
    public func timeResolution(sampleRate: Int) -> Double {
        Double(configuration.hopLength) / Double(sampleRate)
    }
}

// MARK: - Streaming Mel Spectrogram

/// Streaming mel spectrogram processor.
///
/// Extends StreamingSTFT to output mel-scale spectrograms suitable for
/// speech recognition and audio classification.
public actor StreamingMelSpectrogram: StreamingProcessor {
    // MARK: - Properties

    private let stft: StreamingSTFT
    private let sampleRate: Int
    private let nMels: Int
    private var melFilterbank: MLXArray?

    // MARK: - Initialization

    /// Creates a new streaming mel spectrogram processor.
    ///
    /// - Parameters:
    ///   - sampleRate: Audio sample rate in Hz
    ///   - nFFT: FFT size
    ///   - hopLength: Hop length
    ///   - nMels: Number of mel bands
    public init(
        sampleRate: Int = 16000,
        nFFT: Int = 400,
        hopLength: Int = 160,
        nMels: Int = 80
    ) {
        self.sampleRate = sampleRate
        self.nMels = nMels

        let config = StreamingSTFTConfiguration(
            nFFT: nFFT,
            hopLength: hopLength,
            window: .hann,
            returnMagnitude: true,
            toDecibels: false
        )
        self.stft = StreamingSTFT(configuration: config)
    }

    /// Process audio and return mel spectrogram frames.
    public func process(_ input: MLXArray) async throws -> MLXArray {
        // Get magnitude spectrogram from STFT
        let spectrogram = try await stft.process(input)

        guard spectrogram.shape[1] > 0 else {
            return MLXArray.zeros([nMels, 0])
        }

        // Get or create mel filterbank
        if melFilterbank == nil {
            let config = MelConfig(
                sampleRate: sampleRate,
                nFFT: await stft.frequencyBins * 2 - 2,
                nMels: nMels
            )
            melFilterbank = try melFilterbank(config: config)
        }

        let filterbank = melFilterbank!

        // Apply mel filterbank: [nMels, freqBins] @ [freqBins, nFrames] = [nMels, nFrames]
        let powerSpec = spectrogram .* spectrogram  // Power spectrogram
        let melSpec = MLX.matmul(filterbank, powerSpec)

        // Convert to log scale
        let logMelSpec = MLX.log(MLX.maximum(melSpec, MLXArray(Float(1e-10))))

        return logMelSpec
    }

    /// Reset processor state.
    public func reset() async {
        await stft.reset()
    }

    /// Number of frames processed.
    public var framesProcessed: Int {
        get async {
            await stft.framesProcessed
        }
    }
}
