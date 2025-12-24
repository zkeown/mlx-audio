// MLXAudioLabApp.swift
// Entry point for the MLX Audio Lab demo app.

import SwiftUI

@main
struct MLXAudioLabApp: App {
    @StateObject private var modelManager = ModelManager()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(modelManager)
        }
    }
}
