// AudioPlayerView.swift
// Audio playback controls component.

import SwiftUI
import AVFoundation
#if canImport(UIKit)
import UIKit
#elseif canImport(AppKit)
import AppKit
#endif

/// Simple audio player view with play/pause and seek.
struct AudioPlayerView: View {
    @StateObject private var player = AudioPlayerController()
    let url: URL?
    var showWaveform: Bool = false
    var waveformData: [Float]? = nil

    var body: some View {
        VStack(spacing: 12) {
            // Waveform with position (optional)
            if showWaveform, let waveform = waveformData {
                WaveformWithPositionView(
                    samples: waveform,
                    position: player.progress
                )
                .frame(height: 60)
                .onTapGesture { location in
                    // Seek on tap
                    // Would need geometry reader for accurate positioning
                }
            }

            // Controls
            HStack(spacing: 20) {
                // Play/Pause button
                Button {
                    if player.isPlaying {
                        player.pause()
                    } else if let url = url {
                        player.play(url: url)
                    }
                } label: {
                    Image(systemName: player.isPlaying ? "pause.circle.fill" : "play.circle.fill")
                        .font(.system(size: 44))
                        .foregroundColor(.accentColor)
                }
                .disabled(url == nil)

                // Time display
                VStack(alignment: .leading, spacing: 2) {
                    // Progress bar
                    ProgressView(value: player.progress)
                        .progressViewStyle(.linear)

                    // Time labels
                    HStack {
                        Text(formatTime(player.currentTime))
                            .font(.caption)
                            .monospacedDigit()

                        Spacer()

                        Text(formatTime(player.duration))
                            .font(.caption)
                            .monospacedDigit()
                            .foregroundColor(.secondary)
                    }
                }
            }
        }
        .padding()
        .background(Color.systemBackground)
        .cornerRadius(12)
        .onDisappear {
            player.stop()
        }
    }

    private func formatTime(_ time: TimeInterval) -> String {
        let minutes = Int(time) / 60
        let seconds = Int(time) % 60
        return String(format: "%d:%02d", minutes, seconds)
    }
}

/// Controller for audio playback.
@MainActor
class AudioPlayerController: ObservableObject {
    @Published var isPlaying = false
    @Published var currentTime: TimeInterval = 0
    @Published var duration: TimeInterval = 0

    var progress: Double {
        duration > 0 ? currentTime / duration : 0
    }

    private var player: AVAudioPlayer?
    private var timer: Timer?

    func play(url: URL) {
        stop()

        do {
            // Configure audio session
            #if os(iOS)
            try AVAudioSession.sharedInstance().setCategory(.playback, mode: .default)
            try AVAudioSession.sharedInstance().setActive(true)
            #endif

            player = try AVAudioPlayer(contentsOf: url)
            player?.prepareToPlay()
            duration = player?.duration ?? 0

            player?.play()
            isPlaying = true

            startTimer()

        } catch {
            print("Error playing audio: \(error)")
        }
    }

    func pause() {
        player?.pause()
        isPlaying = false
        stopTimer()
    }

    func stop() {
        player?.stop()
        player = nil
        isPlaying = false
        currentTime = 0
        stopTimer()
    }

    func seek(to progress: Double) {
        guard let player = player else { return }
        let time = duration * progress
        player.currentTime = time
        currentTime = time
    }

    private func startTimer() {
        timer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { [weak self] _ in
            Task { @MainActor [weak self] in
                guard let self = self else { return }
                self.currentTime = self.player?.currentTime ?? 0

                // Check if playback finished
                if let player = self.player, !player.isPlaying && self.isPlaying {
                    self.isPlaying = false
                    self.currentTime = 0
                }
            }
        }
    }

    private func stopTimer() {
        timer?.invalidate()
        timer = nil
    }
}

/// Multi-stem player for separated audio.
struct StemPlayerView: View {
    let stems: [(name: String, url: URL, color: Color)]
    @State private var selectedStem: String?
    @State private var soloStem: String?

    var body: some View {
        VStack(spacing: 12) {
            // Stem buttons
            HStack(spacing: 8) {
                ForEach(stems, id: \.name) { stem in
                    StemButton(
                        name: stem.name,
                        color: stem.color,
                        isSelected: selectedStem == stem.name,
                        isSolo: soloStem == stem.name
                    ) {
                        selectedStem = stem.name
                    } onSolo: {
                        soloStem = soloStem == stem.name ? nil : stem.name
                    }
                }
            }

            // Player for selected stem
            if let selected = selectedStem,
               let stem = stems.first(where: { $0.name == selected }) {
                AudioPlayerView(url: stem.url)
            }
        }
    }
}

/// Individual stem button.
struct StemButton: View {
    let name: String
    let color: Color
    let isSelected: Bool
    let isSolo: Bool
    let onSelect: () -> Void
    let onSolo: () -> Void

    var body: some View {
        VStack(spacing: 4) {
            Button(action: onSelect) {
                Text(name)
                    .font(.caption)
                    .fontWeight(isSelected ? .bold : .regular)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                    .background(isSelected ? color : color.opacity(0.2))
                    .foregroundColor(isSelected ? .white : color)
                    .cornerRadius(8)
            }

            Button(action: onSolo) {
                Text("S")
                    .font(.caption2)
                    .fontWeight(.bold)
                    .padding(4)
                    .background(isSolo ? .yellow : Color.systemGray5)
                    .foregroundColor(isSolo ? .black : .secondary)
                    .cornerRadius(4)
            }
        }
    }
}

// MARK: - Preview

#Preview {
    VStack(spacing: 20) {
        AudioPlayerView(url: nil)

        StemPlayerView(stems: [
            ("Drums", URL(fileURLWithPath: "/tmp/drums.wav"), .orange),
            ("Bass", URL(fileURLWithPath: "/tmp/bass.wav"), .purple),
            ("Other", URL(fileURLWithPath: "/tmp/other.wav"), .green),
            ("Vocals", URL(fileURLWithPath: "/tmp/vocals.wav"), .blue),
        ])
    }
    .padding()
}
