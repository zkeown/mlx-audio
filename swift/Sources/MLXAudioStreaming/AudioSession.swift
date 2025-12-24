// AudioSession.swift
// Audio session management for iOS/macOS.
//
// Handles AVAudioSession configuration, interruptions, and route changes.

#if os(iOS) || os(tvOS) || os(visionOS)
import AVFoundation
#endif
import Foundation

// MARK: - Audio Session Configuration

/// Configuration for audio session setup.
public struct AudioSessionConfiguration: Sendable {
    /// Audio session category.
    public enum Category: Sendable {
        /// Playback only (speaker output)
        case playback
        /// Recording only (microphone input)
        case record
        /// Both playback and recording
        case playAndRecord
    }

    /// Session category
    public var category: Category

    /// Preferred sample rate in Hz
    public var preferredSampleRate: Double

    /// Preferred I/O buffer duration in seconds (lower = less latency)
    public var preferredBufferDuration: TimeInterval

    /// Allow Bluetooth audio devices
    public var allowBluetooth: Bool

    /// Allow AirPlay
    public var allowAirPlay: Bool

    /// Mix with other audio apps
    public var mixWithOthers: Bool

    /// Default playback-only configuration
    public static let playback = AudioSessionConfiguration(
        category: .playback,
        preferredSampleRate: 44100,
        preferredBufferDuration: 0.005,
        allowBluetooth: true,
        allowAirPlay: true,
        mixWithOthers: false
    )

    /// Default recording configuration
    public static let record = AudioSessionConfiguration(
        category: .record,
        preferredSampleRate: 44100,
        preferredBufferDuration: 0.005,
        allowBluetooth: false,
        allowAirPlay: false,
        mixWithOthers: false
    )

    /// Default play and record configuration (for audio processing)
    public static let playAndRecord = AudioSessionConfiguration(
        category: .playAndRecord,
        preferredSampleRate: 44100,
        preferredBufferDuration: 0.005,
        allowBluetooth: true,
        allowAirPlay: false,
        mixWithOthers: false
    )

    public init(
        category: Category = .playAndRecord,
        preferredSampleRate: Double = 44100,
        preferredBufferDuration: TimeInterval = 0.005,
        allowBluetooth: Bool = true,
        allowAirPlay: Bool = false,
        mixWithOthers: Bool = false
    ) {
        self.category = category
        self.preferredSampleRate = preferredSampleRate
        self.preferredBufferDuration = preferredBufferDuration
        self.allowBluetooth = allowBluetooth
        self.allowAirPlay = allowAirPlay
        self.mixWithOthers = mixWithOthers
    }
}

// MARK: - Audio Session Events

/// Audio session interruption event.
public enum AudioSessionInterruption: Sendable {
    /// Interruption began (e.g., phone call)
    case began
    /// Interruption ended, with flag indicating if playback should resume
    case ended(shouldResume: Bool)
}

/// Audio route change event.
public enum AudioRouteChange: Sendable {
    /// New audio device became available
    case newDeviceAvailable
    /// Previously used device became unavailable
    case oldDeviceUnavailable
    /// Category changed
    case categoryChanged
    /// Override changed
    case overrideChanged
    /// Unknown change
    case unknown
}

// MARK: - Audio Session Errors

/// Errors that can occur during audio session operations.
public enum AudioSessionError: Error, LocalizedError {
    /// Failed to configure audio session
    case configurationFailed(String)
    /// Failed to activate audio session
    case activationFailed(String)
    /// Microphone permission denied
    case microphonePermissionDenied
    /// Audio session not available on this platform
    case notAvailable

    public var errorDescription: String? {
        switch self {
        case .configurationFailed(let message):
            return "Audio session configuration failed: \(message)"
        case .activationFailed(let message):
            return "Audio session activation failed: \(message)"
        case .microphonePermissionDenied:
            return "Microphone permission denied"
        case .notAvailable:
            return "Audio session not available on this platform"
        }
    }
}

// MARK: - Audio Session Manager

/// Manages audio session configuration and lifecycle.
///
/// This actor provides a thread-safe interface for configuring AVAudioSession on iOS
/// and handling interruptions and route changes.
///
/// Example:
/// ```swift
/// let session = AudioSessionManager.shared
/// try await session.configure(.playAndRecord)
/// try await session.activate()
///
/// // Listen for interruptions
/// for await interruption in session.interruptions {
///     switch interruption {
///     case .began:
///         // Pause audio
///     case .ended(let shouldResume):
///         if shouldResume {
///             // Resume audio
///         }
///     }
/// }
/// ```
public actor AudioSessionManager {
    // MARK: - Shared Instance

    /// Shared audio session manager.
    public static let shared = AudioSessionManager()

    // MARK: - Properties

    /// Whether the session has been configured
    public private(set) var isConfigured = false

    /// Whether the session is currently active
    public private(set) var isActive = false

    /// Current configuration
    public private(set) var currentConfiguration: AudioSessionConfiguration?

    #if os(iOS) || os(tvOS) || os(visionOS)
    /// Notification observers
    private var interruptionObserver: NSObjectProtocol?
    private var routeChangeObserver: NSObjectProtocol?

    /// Continuation for interruption events
    private var interruptionContinuation: AsyncStream<AudioSessionInterruption>.Continuation?

    /// Continuation for route change events
    private var routeChangeContinuation: AsyncStream<AudioRouteChange>.Continuation?

    /// Stream of interruption events (created once, reused for all subscribers).
    private var _interruptions: AsyncStream<AudioSessionInterruption>?

    /// Stream of route change events (created once, reused for all subscribers).
    private var _routeChanges: AsyncStream<AudioRouteChange>?
    #endif

    // MARK: - Initialization

    private init() {
        #if os(iOS) || os(tvOS) || os(visionOS)
        // Create streams once during initialization to avoid recreating on each access
        _interruptions = AsyncStream { [weak self] continuation in
            Task { @MainActor in
                // Note: We need to set this after the actor is initialized,
                // but the stream creation happens synchronously. We use Task
                // to defer the assignment until after init completes.
                await self?.setInterruptionContinuation(continuation)
            }
        }
        _routeChanges = AsyncStream { [weak self] continuation in
            Task { @MainActor in
                await self?.setRouteChangeContinuation(continuation)
            }
        }
        #endif
    }

    #if os(iOS) || os(tvOS) || os(visionOS)
    /// Set the interruption continuation (called after init via Task).
    private func setInterruptionContinuation(_ continuation: AsyncStream<AudioSessionInterruption>.Continuation) {
        self.interruptionContinuation = continuation
    }

    /// Set the route change continuation (called after init via Task).
    private func setRouteChangeContinuation(_ continuation: AsyncStream<AudioRouteChange>.Continuation) {
        self.routeChangeContinuation = continuation
    }
    #endif

    // MARK: - Configuration

    /// Configure the audio session with the specified settings.
    ///
    /// - Parameter configuration: The audio session configuration
    /// - Throws: AudioSessionError if configuration fails
    public func configure(_ configuration: AudioSessionConfiguration) async throws {
        #if os(iOS) || os(tvOS) || os(visionOS)
        let session = AVAudioSession.sharedInstance()

        do {
            // Build category options
            var options: AVAudioSession.CategoryOptions = []
            if configuration.allowBluetooth {
                options.insert(.allowBluetooth)
                options.insert(.allowBluetoothA2DP)
            }
            if configuration.allowAirPlay {
                options.insert(.allowAirPlay)
            }
            if configuration.mixWithOthers {
                options.insert(.mixWithOthers)
            }
            if configuration.category == .playAndRecord {
                options.insert(.defaultToSpeaker)
            }

            // Set category
            let avCategory: AVAudioSession.Category
            switch configuration.category {
            case .playback:
                avCategory = .playback
            case .record:
                avCategory = .record
            case .playAndRecord:
                avCategory = .playAndRecord
            }

            try session.setCategory(avCategory, mode: .default, options: options)

            // Set preferred sample rate
            try session.setPreferredSampleRate(configuration.preferredSampleRate)

            // Set preferred buffer duration
            try session.setPreferredIOBufferDuration(configuration.preferredBufferDuration)

            currentConfiguration = configuration
            isConfigured = true

            // Set up notification observers
            setupNotificationObservers()

        } catch {
            throw AudioSessionError.configurationFailed(error.localizedDescription)
        }
        #else
        // macOS doesn't use AVAudioSession, but we still track configuration
        currentConfiguration = configuration
        isConfigured = true
        #endif
    }

    /// Activate the audio session.
    ///
    /// - Throws: AudioSessionError if activation fails
    public func activate() async throws {
        #if os(iOS) || os(tvOS) || os(visionOS)
        let session = AVAudioSession.sharedInstance()

        do {
            try session.setActive(true, options: [])
            isActive = true
        } catch {
            throw AudioSessionError.activationFailed(error.localizedDescription)
        }
        #else
        isActive = true
        #endif
    }

    /// Deactivate the audio session.
    ///
    /// - Parameter notifyOthers: Whether to notify other apps that audio is stopping
    public func deactivate(notifyOthers: Bool = true) async throws {
        #if os(iOS) || os(tvOS) || os(visionOS)
        let session = AVAudioSession.sharedInstance()

        do {
            let options: AVAudioSession.SetActiveOptions = notifyOthers ? .notifyOthersOnDeactivation : []
            try session.setActive(false, options: options)
            isActive = false
        } catch {
            throw AudioSessionError.activationFailed(error.localizedDescription)
        }
        #else
        isActive = false
        #endif
    }

    // MARK: - Hardware Properties

    /// Current hardware sample rate.
    public var hardwareSampleRate: Double {
        #if os(iOS) || os(tvOS) || os(visionOS)
        return AVAudioSession.sharedInstance().sampleRate
        #else
        return 44100  // Default for macOS
        #endif
    }

    /// Current I/O buffer duration.
    public var ioBufferDuration: TimeInterval {
        #if os(iOS) || os(tvOS) || os(visionOS)
        return AVAudioSession.sharedInstance().ioBufferDuration
        #else
        return 0.005  // Default for macOS
        #endif
    }

    /// Current input latency.
    public var inputLatency: TimeInterval {
        #if os(iOS) || os(tvOS) || os(visionOS)
        return AVAudioSession.sharedInstance().inputLatency
        #else
        return 0
        #endif
    }

    /// Current output latency.
    public var outputLatency: TimeInterval {
        #if os(iOS) || os(tvOS) || os(visionOS)
        return AVAudioSession.sharedInstance().outputLatency
        #else
        return 0
        #endif
    }

    // MARK: - Permissions

    /// Request microphone permission.
    ///
    /// - Returns: Whether permission was granted
    public func requestMicrophonePermission() async -> Bool {
        #if os(iOS) || os(tvOS) || os(visionOS)
        return await withCheckedContinuation { continuation in
            AVAudioApplication.requestRecordPermission { granted in
                continuation.resume(returning: granted)
            }
        }
        #elseif os(macOS)
        // macOS 10.14+ requires permission
        if #available(macOS 10.14, *) {
            switch AVCaptureDevice.authorizationStatus(for: .audio) {
            case .authorized:
                return true
            case .notDetermined:
                return await withCheckedContinuation { continuation in
                    AVCaptureDevice.requestAccess(for: .audio) { granted in
                        continuation.resume(returning: granted)
                    }
                }
            default:
                return false
            }
        } else {
            return true
        }
        #else
        return true
        #endif
    }

    /// Check if microphone permission is granted.
    public var hasMicrophonePermission: Bool {
        #if os(iOS) || os(tvOS) || os(visionOS)
        return AVAudioApplication.shared.recordPermission == .granted
        #elseif os(macOS)
        if #available(macOS 10.14, *) {
            return AVCaptureDevice.authorizationStatus(for: .audio) == .authorized
        } else {
            return true
        }
        #else
        return true
        #endif
    }

    // MARK: - Event Streams

    #if os(iOS) || os(tvOS) || os(visionOS)
    /// Stream of interruption events.
    ///
    /// Returns the same stream instance on each access. Multiple subscribers
    /// will receive the same events.
    public var interruptions: AsyncStream<AudioSessionInterruption> {
        guard let stream = _interruptions else {
            // Fallback for edge case where stream creation failed
            return AsyncStream { _ in }
        }
        return stream
    }

    /// Stream of route change events.
    ///
    /// Returns the same stream instance on each access. Multiple subscribers
    /// will receive the same events.
    public var routeChanges: AsyncStream<AudioRouteChange> {
        guard let stream = _routeChanges else {
            // Fallback for edge case where stream creation failed
            return AsyncStream { _ in }
        }
        return stream
    }

    // MARK: - Private

    private func setupNotificationObservers() {
        let center = NotificationCenter.default

        // Remove existing observers
        if let observer = interruptionObserver {
            center.removeObserver(observer)
        }
        if let observer = routeChangeObserver {
            center.removeObserver(observer)
        }

        // Interruption observer
        interruptionObserver = center.addObserver(
            forName: AVAudioSession.interruptionNotification,
            object: nil,
            queue: .main
        ) { [weak self] notification in
            guard let self = self else { return }

            guard let userInfo = notification.userInfo,
                  let typeValue = userInfo[AVAudioSessionInterruptionTypeKey] as? UInt,
                  let type = AVAudioSession.InterruptionType(rawValue: typeValue)
            else { return }

            Task {
                await self.handleInterruption(type: type, userInfo: userInfo)
            }
        }

        // Route change observer
        routeChangeObserver = center.addObserver(
            forName: AVAudioSession.routeChangeNotification,
            object: nil,
            queue: .main
        ) { [weak self] notification in
            guard let self = self else { return }

            guard let userInfo = notification.userInfo,
                  let reasonValue = userInfo[AVAudioSessionRouteChangeReasonKey] as? UInt,
                  let reason = AVAudioSession.RouteChangeReason(rawValue: reasonValue)
            else { return }

            Task {
                await self.handleRouteChange(reason: reason)
            }
        }
    }

    private func handleInterruption(type: AVAudioSession.InterruptionType, userInfo: [AnyHashable: Any]) {
        switch type {
        case .began:
            interruptionContinuation?.yield(.began)

        case .ended:
            var shouldResume = false
            if let optionsValue = userInfo[AVAudioSessionInterruptionOptionKey] as? UInt {
                let options = AVAudioSession.InterruptionOptions(rawValue: optionsValue)
                shouldResume = options.contains(.shouldResume)
            }
            interruptionContinuation?.yield(.ended(shouldResume: shouldResume))

        @unknown default:
            break
        }
    }

    private func handleRouteChange(reason: AVAudioSession.RouteChangeReason) {
        let event: AudioRouteChange
        switch reason {
        case .newDeviceAvailable:
            event = .newDeviceAvailable
        case .oldDeviceUnavailable:
            event = .oldDeviceUnavailable
        case .categoryChange:
            event = .categoryChanged
        case .override:
            event = .overrideChanged
        default:
            event = .unknown
        }
        routeChangeContinuation?.yield(event)
    }
    #endif
}

// MARK: - Platform Compatibility

#if os(macOS)
import AVFoundation

extension AudioSessionManager {
    /// Stream of interruption events (empty on macOS).
    public var interruptions: AsyncStream<AudioSessionInterruption> {
        AsyncStream { _ in }
    }

    /// Stream of route change events (empty on macOS).
    public var routeChanges: AsyncStream<AudioRouteChange> {
        AsyncStream { _ in }
    }
}
#endif
