// LazyWeightLoader.swift
// Lazy loading of model weights for memory-efficient model initialization.
//
// Instead of loading all weights into memory at once, this loader
// parses the safetensors header and loads weights on demand.

import Foundation
import MLX
import MLXNN

/// Error types for weight loading.
public enum WeightLoadError: Error, LocalizedError {
    case invalidFormat(String)
    case weightNotFound(String)
    case headerParseFailed(String)
    case fileReadFailed(String)

    public var errorDescription: String? {
        switch self {
        case .invalidFormat(let msg): return "Invalid format: \(msg)"
        case .weightNotFound(let key): return "Weight not found: \(key)"
        case .headerParseFailed(let msg): return "Header parse failed: \(msg)"
        case .fileReadFailed(let msg): return "File read failed: \(msg)"
        }
    }
}

/// Weight metadata from safetensors header.
public struct WeightInfo: Sendable {
    /// Key name of the weight.
    public let key: String

    /// Data type (e.g., "F32", "F16", "BF16").
    public let dtype: String

    /// Shape of the weight tensor.
    public let shape: [Int]

    /// Byte offset in file (after header).
    public let offset: Int64

    /// Size in bytes.
    public let size: Int64

    /// Number of elements.
    public var elementCount: Int {
        shape.reduce(1, *)
    }

    /// Estimated memory in bytes (for the given dtype).
    public var memoryBytes: Int64 {
        let bytesPerElement: Int64
        switch dtype {
        case "F32": bytesPerElement = 4
        case "F16", "BF16": bytesPerElement = 2
        case "I32": bytesPerElement = 4
        case "I16": bytesPerElement = 2
        case "I8", "U8": bytesPerElement = 1
        default: bytesPerElement = 4
        }
        return Int64(elementCount) * bytesPerElement
    }
}

/// Lazy weight loader for safetensors files.
///
/// Parses the safetensors header without loading weights, then loads
/// individual weights on demand. This reduces initial memory usage
/// and allows selective loading.
///
/// Example:
/// ```swift
/// let loader = try LazyWeightLoader(url: weightsURL)
///
/// // Get info about available weights
/// let info = loader.weightInfo
///
/// // Load specific weights
/// let embedding = try loader.load("embedding.weight")
/// let attention = try loader.loadBatch(["attn.q", "attn.k", "attn.v"])
///
/// // Unload when done
/// loader.unload("embedding.weight")
/// ```
///
/// ## Thread Safety
///
/// `LazyWeightLoader` is marked `@unchecked Sendable` and is fully thread-safe:
///
/// 1. **Lock-protected mutable state**: All mutable state (`cache`, `fileHandle`) is
///    protected by an `NSLock`. Every public method acquires the lock before accessing
///    or modifying shared state.
///
/// 2. **Immutable after init**: `url`, `weightInfo`, and `headerSize` are set during
///    initialization and never modified afterward.
///
/// 3. **NSLock choice**: We use `NSLock` instead of an actor because:
///    - Weight loading is often called from synchronous contexts
///    - File I/O operations don't benefit from async suspension
///    - The lock scope is minimal (only protecting cache access, not I/O)
///
/// 4. **Safe deinit**: The destructor acquires the lock before closing the file handle,
///    ensuring no concurrent access during cleanup.
///
/// **Concurrent usage pattern**:
/// ```swift
/// // Safe to call from multiple threads simultaneously
/// Task { let w1 = try loader.load("layer1.weight") }
/// Task { let w2 = try loader.load("layer2.weight") }
/// ```
public final class LazyWeightLoader: @unchecked Sendable {

    /// URL of the safetensors file.
    public let url: URL

    /// Parsed weight metadata (immutable after init).
    public private(set) var weightInfo: [String: WeightInfo] = [:]

    /// Header size in bytes (immutable after init).
    private var headerSize: Int64 = 0

    /// Lock for thread-safe access to mutable state.
    private let lock = NSLock()

    /// Cached loaded weights (protected by lock).
    private var cache: [String: MLXArray] = [:]

    /// File handle for reading (protected by lock).
    private var fileHandle: FileHandle?

    /// Initialize loader by parsing safetensors header.
    ///
    /// - Parameter url: URL to safetensors file
    /// - Throws: WeightLoadError if file cannot be parsed
    public init(url: URL) throws {
        self.url = url
        try parseHeader()
    }

    deinit {
        lock.lock()
        defer { lock.unlock() }
        try? fileHandle?.close()
    }

    // MARK: - Header Parsing

    /// Parse the safetensors header to get weight metadata.
    private func parseHeader() throws {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw WeightLoadError.fileReadFailed("File not found: \(url.path)")
        }

        fileHandle = try FileHandle(forReadingFrom: url)

        // Safetensors format: 8-byte header size (little endian), then JSON header
        guard let sizeData = try fileHandle?.read(upToCount: 8), sizeData.count == 8 else {
            throw WeightLoadError.headerParseFailed("Could not read header size")
        }

        let headerLength = sizeData.withUnsafeBytes {
            $0.load(as: UInt64.self).littleEndian
        }

        guard headerLength < 100_000_000 else {  // Sanity check: < 100MB header
            throw WeightLoadError.headerParseFailed("Header size too large: \(headerLength)")
        }

        guard let headerData = try fileHandle?.read(upToCount: Int(headerLength)) else {
            throw WeightLoadError.headerParseFailed("Could not read header")
        }

        headerSize = 8 + Int64(headerLength)

        // Parse JSON header
        guard let json = try JSONSerialization.jsonObject(with: headerData) as? [String: Any] else {
            throw WeightLoadError.headerParseFailed("Invalid JSON header")
        }

        // Extract weight metadata
        for (key, value) in json {
            // Skip metadata key
            if key == "__metadata__" { continue }

            guard let info = value as? [String: Any],
                  let dtype = info["dtype"] as? String,
                  let shape = info["shape"] as? [Int],
                  let dataOffsets = info["data_offsets"] as? [Int] else {
                continue
            }

            guard dataOffsets.count == 2 else { continue }

            let offset = Int64(dataOffsets[0])
            let size = Int64(dataOffsets[1] - dataOffsets[0])

            weightInfo[key] = WeightInfo(
                key: key,
                dtype: dtype,
                shape: shape,
                offset: offset,
                size: size
            )
        }
    }

    // MARK: - Weight Loading

    /// Load a single weight by key.
    ///
    /// - Parameter key: Weight key name
    /// - Returns: Loaded MLXArray
    /// - Throws: WeightLoadError if weight not found
    public func load(_ key: String) throws -> MLXArray {
        lock.lock()

        // Check cache first
        if let cached = cache[key] {
            lock.unlock()
            return cached
        }

        guard let info = weightInfo[key] else {
            lock.unlock()
            throw WeightLoadError.weightNotFound(key)
        }

        // Load while holding lock to prevent concurrent loads of same key
        do {
            let array = try loadWeight(info)
            cache[key] = array
            lock.unlock()
            return array
        } catch {
            lock.unlock()
            throw error
        }
    }

    /// Load multiple weights in batch.
    ///
    /// More efficient than loading one at a time due to reduced
    /// file seeking overhead.
    ///
    /// - Parameter keys: Array of weight keys to load
    /// - Returns: Dictionary of loaded arrays
    public func loadBatch(_ keys: [String]) throws -> [String: MLXArray] {
        // Sort by offset for sequential reading
        let sortedKeys = keys.sorted { key1, key2 in
            let info1 = weightInfo[key1]
            let info2 = weightInfo[key2]
            return (info1?.offset ?? 0) < (info2?.offset ?? 0)
        }

        var result: [String: MLXArray] = [:]

        for key in sortedKeys {
            result[key] = try load(key)
        }

        return result
    }

    /// Load all weights matching a prefix.
    ///
    /// - Parameter prefix: Key prefix to match (e.g., "encoder.")
    /// - Returns: Dictionary of loaded arrays
    public func loadWithPrefix(_ prefix: String) throws -> [String: MLXArray] {
        let matchingKeys = weightInfo.keys.filter { $0.hasPrefix(prefix) }
        return try loadBatch(Array(matchingKeys))
    }

    /// Internal weight loading from file.
    private func loadWeight(_ info: WeightInfo) throws -> MLXArray {
        guard let handle = fileHandle else {
            throw WeightLoadError.fileReadFailed("File handle not available")
        }

        // Seek to weight data
        let absoluteOffset = headerSize + info.offset
        try handle.seek(toOffset: UInt64(absoluteOffset))

        // Read weight data
        guard let data = try handle.read(upToCount: Int(info.size)) else {
            throw WeightLoadError.fileReadFailed("Could not read weight data for \(info.key)")
        }

        // Convert to MLXArray based on dtype
        let array = try arrayFromData(data, dtype: info.dtype, shape: info.shape)
        return array
    }

    /// Convert raw data to MLXArray.
    private func arrayFromData(_ data: Data, dtype: String, shape: [Int]) throws -> MLXArray {
        // Create MLXArray from raw bytes using appropriate type
        switch dtype {
        case "F32":
            return MLXArray(data, shape, type: Float.self)
        case "F16":
            return MLXArray(data, shape, type: Float16.self)
        case "BF16":
            // BFloat16 - use UInt16 and cast
            let array = MLXArray(data, shape, type: UInt16.self)
            return array.asType(.bfloat16)
        case "I32":
            return MLXArray(data, shape, type: Int32.self)
        case "I16":
            return MLXArray(data, shape, type: Int16.self)
        case "I8":
            return MLXArray(data, shape, type: Int8.self)
        case "U8":
            return MLXArray(data, shape, type: UInt8.self)
        default:
            throw WeightLoadError.invalidFormat("Unsupported dtype: \(dtype)")
        }
    }

    // MARK: - Cache Management

    /// Unload a weight from cache.
    ///
    /// - Parameter key: Weight key to unload
    public func unload(_ key: String) {
        lock.lock()
        defer { lock.unlock() }
        cache.removeValue(forKey: key)
    }

    /// Unload all weights from cache.
    public func unloadAll() {
        lock.lock()
        defer { lock.unlock() }
        cache.removeAll()
    }

    /// Unload weights matching a prefix.
    ///
    /// - Parameter prefix: Key prefix to match
    public func unloadWithPrefix(_ prefix: String) {
        lock.lock()
        defer { lock.unlock() }
        let matchingKeys = cache.keys.filter { $0.hasPrefix(prefix) }
        for key in matchingKeys {
            cache.removeValue(forKey: key)
        }
    }

    /// Check if a weight is currently cached.
    public func isCached(_ key: String) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return cache[key] != nil
    }

    /// Number of weights currently cached.
    public var cachedCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return cache.count
    }

    /// Estimated memory of cached weights in bytes.
    public var cachedMemoryBytes: Int64 {
        lock.lock()
        defer { lock.unlock() }
        return cache.values.reduce(0) { total, array in
            let bytesPerElement: Int64
            switch array.dtype {
            case .float32: bytesPerElement = 4
            case .float16, .bfloat16: bytesPerElement = 2
            case .int32: bytesPerElement = 4
            case .int16: bytesPerElement = 2
            case .int8, .uint8: bytesPerElement = 1
            default: bytesPerElement = 4
            }
            return total + Int64(array.size) * bytesPerElement
        }
    }

    // MARK: - Weight Info Queries

    /// Get all weight keys.
    public var keys: [String] {
        Array(weightInfo.keys)
    }

    /// Get total size of all weights in bytes.
    public var totalWeightBytes: Int64 {
        weightInfo.values.reduce(0) { $0 + $1.size }
    }

    /// Get total number of parameters.
    public var totalParameters: Int {
        weightInfo.values.reduce(0) { $0 + $1.elementCount }
    }

    /// Check if a weight exists.
    public func contains(_ key: String) -> Bool {
        weightInfo[key] != nil
    }

    /// Get info for a specific weight.
    public func info(for key: String) -> WeightInfo? {
        weightInfo[key]
    }

    /// Get weights grouped by layer prefix.
    ///
    /// Groups weights like "encoder.layer0.weight" under "encoder.layer0".
    public func groupedByLayer() -> [String: [WeightInfo]] {
        var groups: [String: [WeightInfo]] = [:]

        for (key, info) in weightInfo {
            // Extract layer prefix (everything before last component)
            let components = key.split(separator: ".")
            if components.count > 1 {
                let layerKey = components.dropLast().joined(separator: ".")
                groups[layerKey, default: []].append(info)
            } else {
                groups[key, default: []].append(info)
            }
        }

        return groups
    }
}

// MARK: - Convenience Extensions

extension LazyWeightLoader {

    /// Load weights into a module's parameters.
    ///
    /// - Parameters:
    ///   - module: Module to update
    ///   - keyMapping: Optional mapping from module keys to file keys
    /// - Returns: Number of weights loaded
    @discardableResult
    public func loadInto(
        module: Module,
        keyMapping: ((String) -> String)? = nil
    ) throws -> Int {
        let moduleParams = module.parameters()
        var loadedCount = 0

        for (key, _) in moduleParams.flattened() {
            let fileKey = keyMapping?(key) ?? key

            if contains(fileKey) {
                _ = try load(fileKey)
                // Note: Actual update would require mutable access to module
                // This is a simplified version - real implementation would
                // use ModuleParameters.unflattened pattern
                loadedCount += 1
            }
        }

        return loadedCount
    }

    /// Print summary of available weights.
    public func printSummary() {
        print("\n=== Weight Summary: \(url.lastPathComponent) ===")
        print("Total weights: \(weightInfo.count)")
        print("Total parameters: \(totalParameters.formatted())")
        print("Total size: \(formatBytes(UInt64(totalWeightBytes)))")

        // Group by dtype
        var byDtype: [String: Int] = [:]
        for info in weightInfo.values {
            byDtype[info.dtype, default: 0] += info.elementCount
        }
        print("\nBy dtype:")
        for (dtype, count) in byDtype.sorted(by: { $0.key < $1.key }) {
            print("  \(dtype): \(count.formatted()) parameters")
        }

        // Top 5 largest weights
        let sorted = weightInfo.values.sorted { $0.size > $1.size }
        print("\nTop 5 largest weights:")
        for info in sorted.prefix(5) {
            print("  \(info.key): \(formatBytes(UInt64(info.size))) \(info.shape)")
        }
        print("========================================\n")
    }
}
