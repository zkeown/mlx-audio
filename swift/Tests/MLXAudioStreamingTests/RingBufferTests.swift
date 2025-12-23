// RingBufferTests.swift
// Tests for the lock-free AudioRingBuffer.
//
// Includes correctness tests, edge case tests, and thread-safety tests.

import XCTest
@testable import MLXAudioStreaming

final class RingBufferTests: XCTestCase {

    // MARK: - Basic Functionality Tests

    func testInitialization() {
        let buffer = AudioRingBuffer(capacity: 1024, channels: 2)

        XCTAssertEqual(buffer.channels, 2)
        XCTAssertTrue(buffer.isEmpty)
        XCTAssertFalse(buffer.isFull)
        XCTAssertEqual(buffer.available, 0)
        XCTAssertGreaterThan(buffer.space, 0)
    }

    func testCapacityRoundsUpToPowerOf2() {
        // Requesting 1000 should give 1024
        let buffer = AudioRingBuffer(capacity: 1000, channels: 1)
        XCTAssertEqual(buffer.capacity, 1024)

        // Requesting 2048 should stay 2048
        let buffer2 = AudioRingBuffer(capacity: 2048, channels: 1)
        XCTAssertEqual(buffer2.capacity, 2048)

        // Requesting 100 should give 128
        let buffer3 = AudioRingBuffer(capacity: 100, channels: 1)
        XCTAssertEqual(buffer3.capacity, 128)
    }

    func testWriteAndReadMono() {
        let buffer = AudioRingBuffer(capacity: 256, channels: 1)

        let samples: [Float] = [1.0, 2.0, 3.0, 4.0]
        let written = buffer.write(samples)

        XCTAssertEqual(written, 4)
        XCTAssertEqual(buffer.available, 4)

        let read = buffer.read(count: 4)
        XCTAssertNotNil(read)
        XCTAssertEqual(read!, samples)
        XCTAssertEqual(buffer.available, 0)
    }

    func testWriteAndReadStereo() {
        let buffer = AudioRingBuffer(capacity: 256, channels: 2)

        // Interleaved stereo: [L0, R0, L1, R1, ...]
        let samples: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]  // 3 stereo frames
        let written = buffer.write(samples)

        XCTAssertEqual(written, 3)  // 3 frames
        XCTAssertEqual(buffer.available, 3)

        let read = buffer.read(count: 3)
        XCTAssertNotNil(read)
        XCTAssertEqual(read!, samples)
    }

    // MARK: - Edge Case Tests

    func testReadEmptyBuffer() {
        let buffer = AudioRingBuffer(capacity: 256, channels: 1)

        let read = buffer.read(count: 10)
        XCTAssertNil(read)
    }

    func testReadInsufficientData() {
        let buffer = AudioRingBuffer(capacity: 256, channels: 1)

        let samples: [Float] = [1.0, 2.0, 3.0]
        _ = buffer.write(samples)

        // Try to read more than available
        let read = buffer.read(count: 10)
        XCTAssertNil(read)

        // Original data should still be there
        XCTAssertEqual(buffer.available, 3)
    }

    func testReadAvailable() {
        let buffer = AudioRingBuffer(capacity: 256, channels: 1)

        let samples: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0]
        _ = buffer.write(samples)

        let read = buffer.readAvailable()
        XCTAssertEqual(read.count, 5)
        XCTAssertEqual(read, samples)
        XCTAssertTrue(buffer.isEmpty)
    }

    func testPeekDoesNotConsume() {
        let buffer = AudioRingBuffer(capacity: 256, channels: 1)

        let samples: [Float] = [1.0, 2.0, 3.0]
        _ = buffer.write(samples)

        let peeked = buffer.peek(count: 2)
        XCTAssertNotNil(peeked)
        XCTAssertEqual(peeked!, [1.0, 2.0])

        // Data should still be available
        XCTAssertEqual(buffer.available, 3)

        // Can still read all data
        let read = buffer.read(count: 3)
        XCTAssertEqual(read!, samples)
    }

    func testWriteToFullBuffer() {
        let buffer = AudioRingBuffer(capacity: 8, channels: 1)

        // Fill buffer (capacity - 1 because of circular buffer design)
        let samples = [Float](repeating: 1.0, count: 7)
        let written1 = buffer.write(samples)
        XCTAssertEqual(written1, 7)

        // Try to write more - should drop
        let moreSamples: [Float] = [2.0, 3.0, 4.0]
        let written2 = buffer.write(moreSamples)
        XCTAssertEqual(written2, 0)

        // Check dropped count
        XCTAssertEqual(buffer.statisticsDropped, 3)
    }

    func testReset() {
        let buffer = AudioRingBuffer(capacity: 256, channels: 1)

        let samples: [Float] = [1.0, 2.0, 3.0]
        _ = buffer.write(samples)
        XCTAssertFalse(buffer.isEmpty)

        buffer.reset()
        XCTAssertTrue(buffer.isEmpty)
        XCTAssertEqual(buffer.available, 0)
    }

    // MARK: - Wrap-Around Tests

    func testWrapAroundWrite() {
        let buffer = AudioRingBuffer(capacity: 8, channels: 1)

        // Write near end
        let first: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0]
        _ = buffer.write(first)

        // Read some to advance read pointer
        _ = buffer.read(count: 4)

        // Write more - should wrap around
        let second: [Float] = [6.0, 7.0, 8.0, 9.0, 10.0]
        let written = buffer.write(second)
        XCTAssertEqual(written, 5)

        // Read all - should get correct data across wrap
        let read = buffer.readAvailable()
        XCTAssertEqual(read, [5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    }

    func testWrapAroundRead() {
        let buffer = AudioRingBuffer(capacity: 8, channels: 1)

        // Fill, read, fill again to create wrap-around scenario
        _ = buffer.write([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        _ = buffer.read(count: 5)
        _ = buffer.write([7.0, 8.0, 9.0, 10.0])

        // Read across wrap-around boundary
        let read = buffer.read(count: 5)
        XCTAssertNotNil(read)
        XCTAssertEqual(read!, [6.0, 7.0, 8.0, 9.0, 10.0])
    }

    // MARK: - Statistics Tests

    func testStatisticsTracking() {
        let buffer = AudioRingBuffer(capacity: 64, channels: 1)

        // Write some data
        let samples = [Float](repeating: 1.0, count: 10)
        _ = buffer.write(samples)

        XCTAssertEqual(buffer.statisticsTotalWritten, 10)

        // Read some data
        _ = buffer.read(count: 5)
        XCTAssertEqual(buffer.statisticsTotalRead, 5)

        // Statistics snapshot
        let stats = buffer.statistics
        XCTAssertEqual(stats.totalWritten, 10)
        XCTAssertEqual(stats.totalRead, 5)
        XCTAssertEqual(stats.available, 5)
    }

    func testFillLevel() {
        // Use power-of-2 capacity to get exact fill level
        let buffer = AudioRingBuffer(capacity: 128, channels: 1)

        XCTAssertEqual(buffer.fillLevel, 0.0, accuracy: 0.01)

        _ = buffer.write([Float](repeating: 1.0, count: 64))
        XCTAssertEqual(buffer.fillLevel, 0.5, accuracy: 0.01)  // 64/128 = 50%
    }

    // MARK: - Thread Safety Tests

    func testConcurrentWriteAndRead() async {
        let buffer = AudioRingBuffer(capacity: 8192, channels: 1)
        let iterations = 1000
        let chunkSize = 64

        // Producer task
        let producer = Task {
            for i in 0..<iterations {
                let samples = [Float](repeating: Float(i % 100), count: chunkSize)
                while buffer.write(samples) == 0 {
                    // Buffer full, yield
                    await Task.yield()
                }
            }
        }

        // Consumer task
        let consumer = Task {
            var totalRead = 0
            while totalRead < iterations * chunkSize {
                if let samples = buffer.read(count: chunkSize) {
                    totalRead += samples.count / buffer.channels
                } else {
                    await Task.yield()
                }
            }
            return totalRead
        }

        await producer.value
        let totalRead = await consumer.value

        XCTAssertEqual(totalRead, iterations * chunkSize)
    }

    func testHighThroughput() async {
        let buffer = AudioRingBuffer(capacity: 16384, channels: 2)
        let totalFrames = 100_000
        let chunkSize = 256

        let startTime = Date()

        // Simulate audio capture rate (~44100 samples/sec)
        let writeTask = Task {
            var written = 0
            while written < totalFrames {
                let samples = [Float](repeating: 0.5, count: chunkSize * buffer.channels)
                let count = buffer.write(samples)
                written += count
                if count == 0 {
                    await Task.yield()
                }
            }
            return written
        }

        // Consumer reads at similar rate
        let readTask = Task {
            var read = 0
            while read < totalFrames {
                if buffer.read(count: chunkSize) != nil {
                    read += chunkSize
                } else {
                    await Task.yield()
                }
            }
            return read
        }

        let written = await writeTask.value
        let read = await readTask.value

        let elapsed = Date().timeIntervalSince(startTime)

        XCTAssertEqual(written, totalFrames)
        XCTAssertEqual(read, totalFrames)

        // Should complete in reasonable time (< 1 second for 100k frames)
        XCTAssertLessThan(elapsed, 1.0, "High throughput test took too long: \(elapsed)s")
    }

    // MARK: - UnsafeBufferPointer API Tests

    func testWriteWithUnsafeBufferPointer() {
        let buffer = AudioRingBuffer(capacity: 256, channels: 1)

        let samples: [Float] = [1.0, 2.0, 3.0, 4.0]
        let written = samples.withUnsafeBufferPointer { ptr in
            buffer.write(ptr)
        }

        XCTAssertEqual(written, 4)
        XCTAssertEqual(buffer.available, 4)

        let read = buffer.read(count: 4)
        XCTAssertEqual(read!, samples)
    }

    // MARK: - Zero-Length Tests

    func testWriteEmptyArray() {
        let buffer = AudioRingBuffer(capacity: 256, channels: 1)

        let written = buffer.write([])
        XCTAssertEqual(written, 0)
        XCTAssertEqual(buffer.available, 0)
    }

    func testReadZeroCount() {
        let buffer = AudioRingBuffer(capacity: 256, channels: 1)
        _ = buffer.write([1.0, 2.0, 3.0])

        let read = buffer.read(count: 0)
        XCTAssertNotNil(read)
        XCTAssertEqual(read!, [])
        XCTAssertEqual(buffer.available, 3)  // Nothing consumed
    }
}
