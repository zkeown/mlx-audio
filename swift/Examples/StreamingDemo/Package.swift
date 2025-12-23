// swift-tools-version: 6.0
// StreamingDemo - Example app for MLXAudio streaming

import PackageDescription

let package = Package(
    name: "StreamingDemo",
    platforms: [
        .iOS(.v17),
        .macOS(.v14),
    ],
    products: [
        .executable(name: "StreamingDemo", targets: ["StreamingDemo"]),
    ],
    dependencies: [
        // Local MLXAudio package
        .package(path: "../.."),
    ],
    targets: [
        .executableTarget(
            name: "StreamingDemo",
            dependencies: [
                .product(name: "MLXAudioStreaming", package: "MLXAudio"),
                .product(name: "MLXAudioPrimitives", package: "MLXAudio"),
            ],
            path: "StreamingDemo"
        ),
    ]
)
