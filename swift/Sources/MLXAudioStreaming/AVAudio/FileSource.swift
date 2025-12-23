// FileSource.swift
// Audio source that reads from audio files.
//
// Uses AVAudioFile for reading WAV, AIFF, CAF, M4A, and other formats.

import AVFoundation
import Foundation
@preconcurrency import MLX

// MARK: - File Source Configuration

/// Configuration for file audio source.
public struct FileSourceConfiguration: Sendable {
    /// Number of frames to read per chunk
    public var chunkFrameCount: Int

    /// Whether to loop the file
    public var loop: Bool

    /// Starting sample position (0 = beginning)
    public var startFrame: Int

    /// Ending sample position (nil = end of file)
    public var endFrame: Int?

    /// Default configuration
    public static let `default` = FileSourceConfiguration(
        chunkFrameCount: 4096,
        loop: false,
        startFrame: 0,
        endFrame: nil
    )

    /// Looping configuration
    public static let looping = FileSourceConfiguration(
        chunkFrameCount: 4096,
        loop: true,
        startFrame: 0,
        endFrame: nil
    )

    public init(
        chunkFrameCount: Int = 4096,
        loop: Bool = false,
        startFrame: Int = 0,
        endFrame: Int? = nil
    ) {
        self.chunkFrameCount = chunkFrameCount
        self.loop = loop
        self.startFrame = startFrame
        self.endFrame = endFrame
    }
}

// MARK: - File Source Errors

/// Errors that can occur during file reading.
public enum FileSourceError: Error, LocalizedError {
    /// File not found
    case fileNotFound(URL)
    /// Failed to open file
    case openFailed(String)
    /// Failed to read from file
    case readFailed(String)
    /// Invalid seek position
    case invalidSeekPosition(Int)
    /// Not started
    case notStarted

    public var errorDescription: String? {
        switch self {
        case .fileNotFound(let url):
            return "Audio file not found: \(url.lastPathComponent)"
        case .openFailed(let message):
            return "Failed to open audio file: \(message)"
        case .readFailed(let message):
            return "Failed to read from audio file: \(message)"
        case .invalidSeekPosition(let position):
            return "Invalid seek position: \(position)"
        case .notStarted:
            return "File source not started"
        }
    }
}

// MARK: - File Source

/// Audio source that reads from audio files.
///
/// This actor reads audio from files in various formats (WAV, AIFF, CAF, M4A, etc.)
/// and provides it as MLXArrays for processing.
///
/// Example:
/// ```swift
/// let source = FileSource(url: audioFileURL)
/// try await source.start()
///
/// while let audio = try await source.read(count: 1024) {
///     // Process audio (shape: [channels, samples])
/// }
///
/// try await source.stop()
/// ```
public actor FileSource: @preconcurrency AudioSource {
    // MARK: - AudioStream Protocol

    public nonisolated var sampleRate: Int {
        _sampleRate
    }

    public nonisolated var channels: Int {
        _channels
    }

    public var isActive: Bool {
        _isActive
    }

    // MARK: - Properties

    private let url: URL
    private let configuration: FileSourceConfiguration

    private var audioFile: AVAudioFile?
    private var processingFormat: AVAudioFormat?
    private var readBuffer: AVAudioPCMBuffer?

    private nonisolated(unsafe) var _sampleRate: Int = 0
    private nonisolated(unsafe) var _channels: Int = 0
    private var _totalFrames: Int = 0
    private var _currentPosition: Int = 0
    private var _isActive = false

    // MARK: - Initialization

    /// Creates a new file source for the specified URL.
    ///
    /// - Parameters:
    ///   - url: URL of the audio file
    ///   - configuration: Source configuration
    public init(url: URL, configuration: FileSourceConfiguration = .default) {
        self.url = url
        self.configuration = configuration
    }

    // MARK: - AudioSource Protocol

    /// Start reading from the file.
    ///
    /// Opens the audio file and prepares for reading.
    ///
    /// - Throws: FileSourceError if file cannot be opened
    public func start() async throws {
        guard !_isActive else { return }

        // Check file exists
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw FileSourceError.fileNotFound(url)
        }

        // Open file
        do {
            let file = try AVAudioFile(forReading: url)
            audioFile = file

            // Get format info
            let format = file.processingFormat
            _sampleRate = Int(format.sampleRate)
            _channels = Int(format.channelCount)
            _totalFrames = Int(file.length)

            // Create processing format (standard float)
            guard let procFormat = AVAudioFormat(
                standardFormatWithSampleRate: format.sampleRate,
                channels: format.channelCount
            ) else {
                throw FileSourceError.openFailed("Failed to create processing format")
            }
            processingFormat = procFormat

            // Pre-allocate read buffer
            guard let buffer = AVAudioPCMBuffer(
                pcmFormat: procFormat,
                frameCapacity: AVAudioFrameCount(configuration.chunkFrameCount)
            ) else {
                throw FileSourceError.openFailed("Failed to allocate read buffer")
            }
            readBuffer = buffer

            // Seek to start position
            let startPosition = configuration.startFrame
            if startPosition > 0 && startPosition < _totalFrames {
                file.framePosition = AVAudioFramePosition(startPosition)
            }
            _currentPosition = Int(file.framePosition)

            _isActive = true

        } catch let error as FileSourceError {
            throw error
        } catch {
            throw FileSourceError.openFailed(error.localizedDescription)
        }
    }

    /// Stop reading from the file.
    public func stop() async throws {
        audioFile = nil
        readBuffer = nil
        processingFormat = nil
        _isActive = false
    }

    /// Read samples from the file.
    ///
    /// - Parameter count: Number of frames to read
    /// - Returns: Audio samples as MLXArray [channels, samples], or nil at end of file
    public func read(count: Int) async throws -> MLXArray? {
        guard _isActive else {
            throw FileSourceError.notStarted
        }

        guard let file = audioFile,
              let buffer = readBuffer,
              let format = processingFormat
        else {
            return nil
        }

        // Check for end of file
        let endFrame = configuration.endFrame ?? _totalFrames
        if _currentPosition >= endFrame {
            if configuration.loop {
                // Loop back to start
                file.framePosition = AVAudioFramePosition(configuration.startFrame)
                _currentPosition = configuration.startFrame
            } else {
                return nil  // EOF
            }
        }

        // Calculate how many frames to read
        let remainingFrames = endFrame - _currentPosition
        let framesToRead = min(count, remainingFrames)

        // Read from file
        buffer.frameLength = 0  // Reset
        do {
            try file.read(into: buffer, frameCount: AVAudioFrameCount(framesToRead))
        } catch {
            throw FileSourceError.readFailed(error.localizedDescription)
        }

        let framesRead = Int(buffer.frameLength)
        guard framesRead > 0 else {
            if configuration.loop {
                file.framePosition = AVAudioFramePosition(configuration.startFrame)
                _currentPosition = configuration.startFrame
                return try await read(count: count)  // Retry
            }
            return nil
        }

        _currentPosition += framesRead

        // Convert to MLXArray
        guard let channelData = buffer.floatChannelData else {
            return nil
        }

        let channelCount = Int(format.channelCount)

        // Build array: [channels, samples]
        var result = [[Float]](repeating: [], count: channelCount)
        for c in 0..<channelCount {
            result[c] = Array(UnsafeBufferPointer(start: channelData[c], count: framesRead))
        }

        let flat = result.flatMap { $0 }
        return MLXArray(flat).reshaped([channelCount, framesRead])
    }

    // MARK: - Seeking

    /// Seek to a specific frame position.
    ///
    /// - Parameter frame: Target frame position
    /// - Throws: FileSourceError if position is invalid
    public func seek(to frame: Int) throws {
        guard let file = audioFile else {
            throw FileSourceError.notStarted
        }

        guard frame >= 0 && frame < _totalFrames else {
            throw FileSourceError.invalidSeekPosition(frame)
        }

        file.framePosition = AVAudioFramePosition(frame)
        _currentPosition = frame
    }

    /// Seek to a specific time position.
    ///
    /// - Parameter time: Target time in seconds
    /// - Throws: FileSourceError if position is invalid
    public func seek(to time: TimeInterval) throws {
        let frame = Int(time * Double(_sampleRate))
        try seek(to: frame)
    }

    // MARK: - Properties

    /// Total number of frames in the file.
    public var totalFrames: Int {
        _totalFrames
    }

    /// Current read position in frames.
    public var currentPosition: Int {
        _currentPosition
    }

    /// Total duration of the file in seconds.
    public var duration: TimeInterval {
        guard _sampleRate > 0 else { return 0 }
        return Double(_totalFrames) / Double(_sampleRate)
    }

    /// Current position in seconds.
    public var currentTime: TimeInterval {
        guard _sampleRate > 0 else { return 0 }
        return Double(_currentPosition) / Double(_sampleRate)
    }

    /// Remaining frames to read.
    public var remaining: Int {
        let endFrame = configuration.endFrame ?? _totalFrames
        return max(0, endFrame - _currentPosition)
    }

    /// Whether end of file has been reached.
    public var isEOF: Bool {
        let endFrame = configuration.endFrame ?? _totalFrames
        return _currentPosition >= endFrame && !configuration.loop
    }

    /// File URL.
    public var fileURL: URL {
        url
    }
}
