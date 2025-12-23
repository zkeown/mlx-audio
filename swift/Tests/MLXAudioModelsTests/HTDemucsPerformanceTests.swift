// HTDemucsPerformanceTests.swift
// Performance benchmarks for HTDemucs.

import XCTest
@testable import MLXAudioModels
import MLX

final class HTDemucsPerformanceTests: XCTestCase {

    /// Benchmark individual encoder layer (tiny memory footprint).
    func testEncoderLayerPerformance() {
        let encoder = HEncLayer(
            chin: 4,
            chout: 48,
            kernelSize: 8,
            stride: 4,
            freq: true,
            dconvDepth: 2,
            dconvCompress: 8
        )

        // Input in NHWC format: [B, F, T, C]
        let input = MLXRandom.normal([1, 512, 216, 4])
        eval(input)

        // Warmup
        for _ in 0..<3 {
            let _ = encoder(input)
            eval()
        }

        // Measure
        measure {
            let _ = encoder(input)
            eval()
        }
    }

    /// Benchmark individual decoder layer.
    func testDecoderLayerPerformance() {
        let decoder = HDecLayer(
            chin: 48,
            chout: 4,
            kernelSize: 8,
            stride: 4,
            freq: true,
            dconvDepth: 2,
            dconvCompress: 8,
            last: true
        )

        // Input in NHWC format: [B, F, T, C]
        let input = MLXRandom.normal([1, 128, 54, 48])
        let skip = MLXRandom.normal([1, 128, 54, 48])
        eval(input, skip)

        // Warmup
        for _ in 0..<3 {
            let _ = decoder(input, skip: skip, length: 216)
            eval()
        }

        // Measure
        measure {
            let _ = decoder(input, skip: skip, length: 216)
            eval()
        }
    }
}
