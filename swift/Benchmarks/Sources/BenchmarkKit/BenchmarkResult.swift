// BenchmarkResult.swift
// Core data structures for benchmark results.

import Foundation

/// Result of a single benchmark run.
public struct BenchmarkResult: Codable, Sendable {
    public let name: String
    public let meanTimeMs: Double
    public let stdTimeMs: Double
    public let minTimeMs: Double
    public let maxTimeMs: Double
    public let throughput: Double
    public let peakMemoryMB: Double
    public let gpuUtilization: Double?
    public let realtimeFactor: Double?
    public let iterations: Int
    public let params: [String: AnyCodableValue]
    public let timestamp: Date
    public var pythonBaselineMs: Double?

    public var speedupVsPython: Double? {
        guard let baseline = pythonBaselineMs, baseline > 0 else { return nil }
        return baseline / meanTimeMs
    }

    public init(
        name: String,
        meanTimeMs: Double,
        stdTimeMs: Double,
        minTimeMs: Double,
        maxTimeMs: Double,
        throughput: Double,
        peakMemoryMB: Double,
        gpuUtilization: Double? = nil,
        realtimeFactor: Double? = nil,
        iterations: Int,
        params: [String: AnyCodableValue] = [:],
        timestamp: Date = Date(),
        pythonBaselineMs: Double? = nil
    ) {
        self.name = name
        self.meanTimeMs = meanTimeMs
        self.stdTimeMs = stdTimeMs
        self.minTimeMs = minTimeMs
        self.maxTimeMs = maxTimeMs
        self.throughput = throughput
        self.peakMemoryMB = peakMemoryMB
        self.gpuUtilization = gpuUtilization
        self.realtimeFactor = realtimeFactor
        self.iterations = iterations
        self.params = params
        self.timestamp = timestamp
        self.pythonBaselineMs = pythonBaselineMs
    }
}

/// Collection of benchmark results with device metadata.
public struct BenchmarkSuite: Codable, Sendable {
    public let name: String
    public let device: DeviceInfo
    public let timestamp: Date
    public let results: [BenchmarkResult]
    public let metadata: [String: String]

    public init(
        name: String,
        device: DeviceInfo,
        timestamp: Date = Date(),
        results: [BenchmarkResult],
        metadata: [String: String] = [:]
    ) {
        self.name = name
        self.device = device
        self.timestamp = timestamp
        self.results = results
        self.metadata = metadata
    }
}

/// Device information for benchmark context.
public struct DeviceInfo: Codable, Sendable {
    public let chip: String
    public let coreCount: Int
    public let gpuCores: Int
    public let unifiedMemoryGB: Int
    public let osVersion: String
    public let mlxVersion: String

    public init(
        chip: String,
        coreCount: Int,
        gpuCores: Int,
        unifiedMemoryGB: Int,
        osVersion: String,
        mlxVersion: String
    ) {
        self.chip = chip
        self.coreCount = coreCount
        self.gpuCores = gpuCores
        self.unifiedMemoryGB = unifiedMemoryGB
        self.osVersion = osVersion
        self.mlxVersion = mlxVersion
    }

    /// Get current device information.
    public static func current() -> DeviceInfo {
        var size = 0
        sysctlbyname("machdep.cpu.brand_string", nil, &size, nil, 0)
        var cpuBrand = [CChar](repeating: 0, count: size)
        sysctlbyname("machdep.cpu.brand_string", &cpuBrand, &size, nil, 0)
        let chip = String(cString: cpuBrand)

        var coreCount: Int32 = 0
        size = MemoryLayout<Int32>.size
        sysctlbyname("hw.ncpu", &coreCount, &size, nil, 0)

        var memSize: UInt64 = 0
        size = MemoryLayout<UInt64>.size
        sysctlbyname("hw.memsize", &memSize, &size, nil, 0)
        let memoryGB = Int(memSize / (1024 * 1024 * 1024))

        let osVersion = ProcessInfo.processInfo.operatingSystemVersionString

        return DeviceInfo(
            chip: chip,
            coreCount: Int(coreCount),
            gpuCores: 0,  // Not easily accessible without Metal query
            unifiedMemoryGB: memoryGB,
            osVersion: osVersion,
            mlxVersion: "0.10.0+"  // Would need to query MLX
        )
    }
}

/// Type-erased codable value for benchmark parameters.
public enum AnyCodableValue: Codable, Sendable {
    case int(Int)
    case double(Double)
    case string(String)
    case bool(Bool)

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let value = try? container.decode(Int.self) {
            self = .int(value)
        } else if let value = try? container.decode(Double.self) {
            self = .double(value)
        } else if let value = try? container.decode(Bool.self) {
            self = .bool(value)
        } else if let value = try? container.decode(String.self) {
            self = .string(value)
        } else {
            throw DecodingError.typeMismatch(
                AnyCodableValue.self,
                DecodingError.Context(
                    codingPath: decoder.codingPath,
                    debugDescription: "Unsupported type"
                )
            )
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .int(let value): try container.encode(value)
        case .double(let value): try container.encode(value)
        case .string(let value): try container.encode(value)
        case .bool(let value): try container.encode(value)
        }
    }
}

extension AnyCodableValue: ExpressibleByIntegerLiteral {
    public init(integerLiteral value: Int) {
        self = .int(value)
    }
}

extension AnyCodableValue: ExpressibleByFloatLiteral {
    public init(floatLiteral value: Double) {
        self = .double(value)
    }
}

extension AnyCodableValue: ExpressibleByStringLiteral {
    public init(stringLiteral value: String) {
        self = .string(value)
    }
}

extension AnyCodableValue: ExpressibleByBooleanLiteral {
    public init(booleanLiteral value: Bool) {
        self = .bool(value)
    }
}
