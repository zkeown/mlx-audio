// FileSink.swift
// Audio sink that writes to audio files.
//
// Uses AVAudioFile for writing WAV, AIFF, CAF, and other formats.

import AVFoundation
import Foundation
@preconcurrency import MLX

// MARK: - Audio File Format

/// Audio file format for output.
public enum AudioFileFormat: String, Sendable, CaseIterable {
    /// WAV format (uncompressed PCM)
    case wav
    /// AIFF format (uncompressed PCM)
    case aiff
    /// Core Audio Format
    case caf
    /// AAC in M4A container
    case m4a

    /// File extension for this format
    public var fileExtension: String {
        rawValue
    }

    /// AVAudioFile settings for this format
    fileprivate var fileType: AudioFileTypeID {
        switch self {
        case .wav:
            return kAudioFileWAVEType
        case .aiff:
            return kAudioFileAIFFType
        case .caf:
            return kAudioFileCAFType
        case .m4a:
            return kAudioFileM4AType
        }
    }
}

// MARK: - File Sink Configuration

/// Configuration for file audio sink.
public struct FileSinkConfiguration: Sendable {
    /// Audio file format
    public var format: AudioFileFormat

    /// Bit depth for PCM formats (16 or 24)
    public var bitDepth: Int

    /// Index of stem to extract (for multi-stem output)
    public var stemIndex: Int?

    /// Default WAV configuration
    public static let wav = FileSinkConfiguration(
        format: .wav,
        bitDepth: 16,
        stemIndex: nil
    )

    /// High-quality WAV (24-bit)
    public static let wavHighQuality = FileSinkConfiguration(
        format: .wav,
        bitDepth: 24,
        stemIndex: nil
    )

    /// AAC in M4A container
    public static let m4a = FileSinkConfiguration(
        format: .m4a,
        bitDepth: 16,
        stemIndex: nil
    )

    public init(
        format: AudioFileFormat = .wav,
        bitDepth: Int = 16,
        stemIndex: Int? = nil
    ) {
        self.format = format
        self.bitDepth = bitDepth
        self.stemIndex = stemIndex
    }
}

// MARK: - File Sink Errors

/// Errors that can occur during file writing.
public enum FileSinkError: Error, LocalizedError {
    /// Failed to create file
    case createFailed(String)
    /// Failed to write to file
    case writeFailed(String)
    /// Invalid audio format
    case invalidFormat(String)
    /// Not started
    case notStarted
    /// Already running
    case alreadyRunning

    public var errorDescription: String? {
        switch self {
        case .createFailed(let message):
            return "Failed to create audio file: \(message)"
        case .writeFailed(let message):
            return "Failed to write to audio file: \(message)"
        case .invalidFormat(let message):
            return "Invalid audio format: \(message)"
        case .notStarted:
            return "File sink not started"
        case .alreadyRunning:
            return "File sink already running"
        }
    }
}

// MARK: - File Sink

/// Audio sink that writes to audio files.
///
/// This actor writes audio to files in various formats (WAV, AIFF, CAF, M4A).
///
/// Example:
/// ```swift
/// let sink = FileSink(url: outputURL, sampleRate: 44100, channels: 2)
/// try await sink.start()
///
/// // Write audio samples (shape: [channels, samples])
/// try await sink.write(audioData)
///
/// try await sink.stop()  // Closes and finalizes file
/// ```
public actor FileSink: @preconcurrency AudioSink {
    // MARK: - AudioStream Protocol

    public nonisolated let sampleRate: Int
    public nonisolated let channels: Int

    public var isActive: Bool {
        _isActive
    }

    // MARK: - Properties

    private let url: URL
    private let configuration: FileSinkConfiguration

    private var audioFile: AVAudioFile?
    private var processingFormat: AVAudioFormat?
    private var writeBuffer: AVAudioPCMBuffer?

    private var _isActive = false
    private var _framesWritten: Int = 0

    // MARK: - Initialization

    /// Creates a new file sink for the specified URL.
    ///
    /// - Parameters:
    ///   - url: URL where audio file will be written
    ///   - sampleRate: Sample rate in Hz
    ///   - channels: Number of audio channels
    ///   - configuration: Sink configuration
    public init(
        url: URL,
        sampleRate: Int,
        channels: Int,
        configuration: FileSinkConfiguration = .wav
    ) {
        self.url = url
        self.sampleRate = sampleRate
        self.channels = channels
        self.configuration = configuration
    }

    // MARK: - AudioSink Protocol

    /// Start writing to the file.
    ///
    /// Creates the audio file and prepares for writing.
    ///
    /// - Throws: FileSinkError if file cannot be created
    public func start() async throws {
        guard !_isActive else {
            throw FileSinkError.alreadyRunning
        }

        // Create directory if needed
        let directory = url.deletingLastPathComponent()
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

        // Create processing format
        guard let format = AVAudioFormat(
            standardFormatWithSampleRate: Double(sampleRate),
            channels: AVAudioChannelCount(channels)
        ) else {
            throw FileSinkError.invalidFormat("Failed to create audio format")
        }
        processingFormat = format

        // Create file settings based on format
        var settings: [String: Any] = [
            AVFormatIDKey: configuration.format == .m4a ? kAudioFormatMPEG4AAC : kAudioFormatLinearPCM,
            AVSampleRateKey: Double(sampleRate),
            AVNumberOfChannelsKey: channels,
        ]

        if configuration.format != .m4a {
            // PCM settings
            settings[AVLinearPCMBitDepthKey] = configuration.bitDepth
            settings[AVLinearPCMIsFloatKey] = false
            settings[AVLinearPCMIsBigEndianKey] = configuration.format == .aiff
            settings[AVLinearPCMIsNonInterleaved] = false
        } else {
            // AAC settings
            settings[AVEncoderAudioQualityKey] = AVAudioQuality.high.rawValue
            settings[AVEncoderBitRateKey] = 128000
        }

        // Create file
        do {
            let file = try AVAudioFile(
                forWriting: url,
                settings: settings,
                commonFormat: .pcmFormatFloat32,
                interleaved: false
            )
            audioFile = file
        } catch {
            throw FileSinkError.createFailed(error.localizedDescription)
        }

        // Pre-allocate write buffer (16K frames)
        guard let buffer = AVAudioPCMBuffer(
            pcmFormat: format,
            frameCapacity: 16384
        ) else {
            throw FileSinkError.createFailed("Failed to allocate write buffer")
        }
        writeBuffer = buffer

        _isActive = true
        _framesWritten = 0
    }

    /// Stop writing and close the file.
    public func stop() async throws {
        guard _isActive else {
            throw FileSinkError.notStarted
        }

        // File is automatically closed when AVAudioFile is deallocated
        audioFile = nil
        writeBuffer = nil
        processingFormat = nil
        _isActive = false
    }

    /// Write samples to the file.
    ///
    /// - Parameter samples: Audio samples as MLXArray [channels, samples] or [stems, channels, samples]
    public func write(_ samples: MLXArray) async throws {
        guard _isActive else {
            throw FileSinkError.notStarted
        }

        guard let file = audioFile,
              let buffer = writeBuffer,
              let format = processingFormat
        else {
            return
        }

        // Handle different input shapes
        let shape = samples.shape
        var audioToWrite: MLXArray

        if shape.count == 3 {
            // [stems, channels, samples] - extract specified stem
            let stemIndex = configuration.stemIndex ?? 0
            audioToWrite = samples[stemIndex]
        } else {
            // [channels, samples] or [samples]
            audioToWrite = samples
        }

        // Ensure 2D [channels, samples]
        if audioToWrite.ndim == 1 {
            audioToWrite = audioToWrite.reshaped([1, audioToWrite.shape[0]])
        }

        let inputChannels = audioToWrite.shape[0]
        let frameCount = audioToWrite.shape[1]

        // Get float data
        let floatData = audioToWrite.asArray(Float.self)

        // Write in chunks (buffer may be smaller than input)
        var offset = 0
        while offset < frameCount {
            let chunkSize = min(Int(buffer.frameCapacity), frameCount - offset)
            buffer.frameLength = AVAudioFrameCount(chunkSize)

            if let channelData = buffer.floatChannelData {
                let outputChannels = Int(format.channelCount)

                for c in 0..<outputChannels {
                    let inputChannel = min(c, inputChannels - 1)
                    for i in 0..<chunkSize {
                        let sourceIndex = inputChannel * frameCount + offset + i
                        channelData[c][i] = floatData[sourceIndex]
                    }
                }
            }

            do {
                try file.write(from: buffer)
                _framesWritten += chunkSize
            } catch {
                throw FileSinkError.writeFailed(error.localizedDescription)
            }

            offset += chunkSize
        }
    }

    // MARK: - Properties

    /// Total frames written to file.
    public var framesWritten: Int {
        _framesWritten
    }

    /// Duration written in seconds.
    public var durationWritten: TimeInterval {
        Double(_framesWritten) / Double(sampleRate)
    }

    /// Output file URL.
    public var fileURL: URL {
        url
    }

    /// Output file format.
    public var fileFormat: AudioFileFormat {
        configuration.format
    }
}
