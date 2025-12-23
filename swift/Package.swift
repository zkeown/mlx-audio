// swift-tools-version: 6.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "MLXAudio",
    platforms: [
        .iOS(.v17),
        .macOS(.v14),
    ],
    products: [
        // High-level API for audio ML tasks
        .library(name: "MLXAudio", targets: ["MLXAudio"]),
        // Core DSP primitives (STFT, mel, MFCC, etc.)
        .library(name: "MLXAudioPrimitives", targets: ["MLXAudioPrimitives"]),
        // Pre-built models (Whisper, MusicGen, HTDemucs, CLAP)
        .library(name: "MLXAudioModels", targets: ["MLXAudioModels"]),
        // Real-time audio streaming
        .library(name: "MLXAudioStreaming", targets: ["MLXAudioStreaming"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.10.0"),
        .package(url: "https://github.com/apple/swift-atomics", from: "1.2.0"),
    ],
    targets: [
        // Core DSP primitives - no dependencies except MLX
        .target(
            name: "MLXAudioPrimitives",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
            ]
        ),

        // Model implementations - depends on primitives
        .target(
            name: "MLXAudioModels",
            dependencies: [
                "MLXAudioPrimitives",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ]
        ),

        // High-level API - depends on primitives and models
        .target(
            name: "MLXAudio",
            dependencies: [
                "MLXAudioPrimitives",
                "MLXAudioModels",
                .product(name: "MLX", package: "mlx-swift"),
            ]
        ),

        // Streaming support - depends on primitives
        .target(
            name: "MLXAudioStreaming",
            dependencies: [
                "MLXAudioPrimitives",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "Atomics", package: "swift-atomics"),
            ],
            linkerSettings: [
                .linkedFramework("AVFoundation"),
                .linkedFramework("Accelerate"),
            ]
        ),

        // Tests
        .testTarget(
            name: "MLXAudioPrimitivesTests",
            dependencies: ["MLXAudioPrimitives"]
        ),
        .testTarget(
            name: "MLXAudioModelsTests",
            dependencies: [
                "MLXAudioModels",
                "MLXAudioPrimitives",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ]
        ),
        .testTarget(
            name: "MLXAudioStreamingTests",
            dependencies: [
                "MLXAudioStreaming",
                "MLXAudioPrimitives",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "Atomics", package: "swift-atomics"),
            ]
        ),
    ]
)
