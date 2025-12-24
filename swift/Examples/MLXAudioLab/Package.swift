// swift-tools-version: 6.0
// MLXAudioLab - Demo app showcasing all MLXAudio capabilities

import PackageDescription

let package = Package(
    name: "MLXAudioLab",
    platforms: [
        .iOS(.v17),
        .macOS(.v14),
    ],
    products: [
        .executable(name: "MLXAudioLab", targets: ["MLXAudioLab"]),
    ],
    dependencies: [
        // Local MLXAudio package
        .package(path: "../.."),
    ],
    targets: [
        .executableTarget(
            name: "MLXAudioLab",
            dependencies: [
                .product(name: "MLXAudioModels", package: "swift"),
                .product(name: "MLXAudioStreaming", package: "swift"),
                .product(name: "MLXAudioPrimitives", package: "swift"),
            ],
            path: "MLXAudioLab",
            linkerSettings: [
                .linkedFramework("AVFoundation"),
                .linkedFramework("UniformTypeIdentifiers"),
            ]
        ),
    ]
)
