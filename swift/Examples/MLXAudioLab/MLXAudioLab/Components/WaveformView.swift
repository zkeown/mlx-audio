// WaveformView.swift
// Audio waveform visualization component.

import SwiftUI
#if canImport(UIKit)
import UIKit
#elseif canImport(AppKit)
import AppKit
#endif

// Cross-platform color helpers
extension Color {
    static var systemGray6: Color {
        #if canImport(UIKit)
        return Color(UIColor.systemGray6)
        #else
        return Color(NSColor.controlBackgroundColor)
        #endif
    }

    static var systemGray5: Color {
        #if canImport(UIKit)
        return Color(UIColor.systemGray5)
        #else
        return Color(NSColor.controlColor)
        #endif
    }

    static var systemBackground: Color {
        #if canImport(UIKit)
        return Color(UIColor.systemBackground)
        #else
        return Color(NSColor.windowBackgroundColor)
        #endif
    }
}

/// Displays an audio waveform visualization.
struct WaveformView: View {
    let samples: [Float]
    var color: Color = .blue
    var backgroundColor: Color = .systemGray6

    var body: some View {
        GeometryReader { geometry in
            Canvas { context, size in
                guard !samples.isEmpty else { return }

                let midY = size.height / 2
                let stepX = size.width / CGFloat(samples.count)

                // Draw waveform as filled shape
                var path = Path()
                path.move(to: CGPoint(x: 0, y: midY))

                // Upper half
                for (index, sample) in samples.enumerated() {
                    let x = CGFloat(index) * stepX
                    let y = midY - CGFloat(sample) * midY * 0.9
                    path.addLine(to: CGPoint(x: x, y: y))
                }

                // Lower half (mirror)
                for (index, sample) in samples.enumerated().reversed() {
                    let x = CGFloat(index) * stepX
                    let y = midY + CGFloat(sample) * midY * 0.9
                    path.addLine(to: CGPoint(x: x, y: y))
                }

                path.closeSubpath()

                context.fill(path, with: .color(color.opacity(0.6)))
                context.stroke(path, with: .color(color), lineWidth: 1)

                // Center line
                var centerLine = Path()
                centerLine.move(to: CGPoint(x: 0, y: midY))
                centerLine.addLine(to: CGPoint(x: size.width, y: midY))
                context.stroke(centerLine, with: .color(.gray.opacity(0.3)), lineWidth: 0.5)
            }
        }
        .background(backgroundColor)
        .cornerRadius(8)
    }
}

/// Waveform view with playback position indicator.
struct WaveformWithPositionView: View {
    let samples: [Float]
    let position: Double  // 0.0 to 1.0
    var color: Color = .blue
    var positionColor: Color = .red

    var body: some View {
        ZStack {
            WaveformView(samples: samples, color: color)

            // Position indicator
            GeometryReader { geometry in
                Rectangle()
                    .fill(positionColor)
                    .frame(width: 2)
                    .offset(x: geometry.size.width * CGFloat(position) - 1)
            }
        }
    }
}

/// Multi-stem waveform display.
struct StemWaveformsView: View {
    let stems: [(name: String, samples: [Float], color: Color)]
    var selectedStem: String?
    var onStemTap: ((String) -> Void)?

    var body: some View {
        VStack(spacing: 8) {
            ForEach(stems, id: \.name) { stem in
                HStack(spacing: 12) {
                    // Stem label
                    Text(stem.name)
                        .font(.caption)
                        .fontWeight(.medium)
                        .frame(width: 60, alignment: .leading)
                        .foregroundColor(selectedStem == stem.name ? stem.color : .secondary)

                    // Waveform
                    WaveformView(
                        samples: stem.samples,
                        color: selectedStem == stem.name ? stem.color : stem.color.opacity(0.5)
                    )
                    .frame(height: 40)
                }
                .contentShape(Rectangle())
                .onTapGesture {
                    onStemTap?(stem.name)
                }
            }
        }
    }
}

// MARK: - Preview

#Preview {
    VStack(spacing: 20) {
        WaveformView(samples: generateSampleWaveform())
            .frame(height: 80)

        WaveformWithPositionView(
            samples: generateSampleWaveform(),
            position: 0.3
        )
        .frame(height: 80)

        StemWaveformsView(
            stems: [
                ("Drums", generateSampleWaveform(), .orange),
                ("Bass", generateSampleWaveform(), .purple),
                ("Other", generateSampleWaveform(), .green),
                ("Vocals", generateSampleWaveform(), .blue),
            ],
            selectedStem: "Vocals"
        )
    }
    .padding()
}

/// Generate sample waveform data for previews.
private func generateSampleWaveform(points: Int = 200) -> [Float] {
    (0..<points).map { i in
        let t = Float(i) / Float(points) * .pi * 8
        return abs(sin(t) * cos(t * 0.3)) * Float.random(in: 0.5...1.0)
    }
}
