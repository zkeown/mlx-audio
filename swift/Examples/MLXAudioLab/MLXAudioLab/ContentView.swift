// ContentView.swift
// Main tab container for the MLX Audio Lab demo app.

import SwiftUI

/// Available tabs in the app.
enum AppTab: String, CaseIterable, Identifiable {
    case separate = "Separate"
    case transcribe = "Transcribe"
    case generate = "Generate"
    case embed = "Embed"
    case live = "Live"
    case banquet = "Banquet"

    var id: String { rawValue }

    var icon: String {
        switch self {
        case .separate: return "waveform.path"
        case .transcribe: return "text.bubble"
        case .generate: return "music.note"
        case .embed: return "tag"
        case .live: return "mic.fill"
        case .banquet: return "arrow.triangle.branch"
        }
    }

    var description: String {
        switch self {
        case .separate: return "Source Separation"
        case .transcribe: return "Speech to Text"
        case .generate: return "Text to Music"
        case .embed: return "Audio Classification"
        case .live: return "Real-time Processing"
        case .banquet: return "Query-based Separation"
        }
    }
}

struct ContentView: View {
    @EnvironmentObject var modelManager: ModelManager
    @State private var selectedTab: AppTab = .separate

    var body: some View {
        TabView(selection: $selectedTab) {
            SeparateView()
                .tabItem {
                    Label(AppTab.separate.rawValue, systemImage: AppTab.separate.icon)
                }
                .tag(AppTab.separate)

            TranscribeView()
                .tabItem {
                    Label(AppTab.transcribe.rawValue, systemImage: AppTab.transcribe.icon)
                }
                .tag(AppTab.transcribe)

            GenerateView()
                .tabItem {
                    Label(AppTab.generate.rawValue, systemImage: AppTab.generate.icon)
                }
                .tag(AppTab.generate)

            EmbedView()
                .tabItem {
                    Label(AppTab.embed.rawValue, systemImage: AppTab.embed.icon)
                }
                .tag(AppTab.embed)

            LiveView()
                .tabItem {
                    Label(AppTab.live.rawValue, systemImage: AppTab.live.icon)
                }
                .tag(AppTab.live)

            BanquetView()
                .tabItem {
                    Label(AppTab.banquet.rawValue, systemImage: AppTab.banquet.icon)
                }
                .tag(AppTab.banquet)
        }
        .onChange(of: selectedTab) { _, newTab in
            Task {
                await modelManager.onTabChange(to: newTab)
            }
        }
    }
}

#Preview {
    ContentView()
        .environmentObject(ModelManager())
}
