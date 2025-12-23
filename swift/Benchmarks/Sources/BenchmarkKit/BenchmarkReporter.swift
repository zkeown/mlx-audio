// BenchmarkReporter.swift
// Output formatting and comparison utilities.

import Foundation

/// Benchmark output and comparison utilities.
public enum BenchmarkReporter {

    /// Save benchmark suite to JSON file.
    public static func saveJSON(_ suite: BenchmarkSuite, to path: String) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601

        let data = try encoder.encode(suite)
        let url = URL(fileURLWithPath: path)
        try data.write(to: url)
    }

    /// Load benchmark suite from JSON file.
    public static func loadJSON(from path: String) throws -> BenchmarkSuite {
        let url = URL(fileURLWithPath: path)
        let data = try Data(contentsOf: url)

        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601

        return try decoder.decode(BenchmarkSuite.self, from: data)
    }

    /// Print summary table to console.
    public static func printSummary(_ suite: BenchmarkSuite) {
        print("\n" + String(repeating: "=", count: 80))
        print("BENCHMARK RESULTS: \(suite.name)")
        print("Device: \(suite.device.chip)")
        print("Memory: \(suite.device.unifiedMemoryGB) GB")
        print("OS: \(suite.device.osVersion)")
        print("Date: \(formatDate(suite.timestamp))")
        print(String(repeating: "=", count: 80))
        print()

        // Group results by model
        let grouped = Dictionary(grouping: suite.results) { result -> String in
            let parts = result.name.split(separator: "_")
            return String(parts.first ?? "unknown")
        }

        for (model, results) in grouped.sorted(by: { $0.key < $1.key }) {
            print("[\(model.uppercased())]")
            print(String(repeating: "-", count: 80))
            print(
                String(
                    format: "%-40s %10s %10s %10s %10s",
                    "Benchmark", "Mean (ms)", "Std (ms)", "RTF", "Memory"
                )
            )
            print(String(repeating: "-", count: 80))

            for result in results {
                let rtfStr = result.realtimeFactor.map { String(format: "%.2fx", $0) } ?? "-"
                let memStr = String(format: "%.1f MB", result.peakMemoryMB)
                print(
                    String(
                        format: "%-40s %10.2f %10.2f %10s %10s",
                        truncate(result.name, to: 40),
                        result.meanTimeMs,
                        result.stdTimeMs,
                        rtfStr,
                        memStr
                    )
                )
            }
            print()
        }

        // Performance targets
        printTargetStatus(suite)
    }

    /// Print comparison between current and baseline.
    public static func compare(current: BenchmarkSuite, baselinePath: String) throws {
        let baseline = try loadJSON(from: baselinePath)
        print("\n" + String(repeating: "=", count: 80))
        print("PERFORMANCE COMPARISON")
        print("Current: \(formatDate(current.timestamp))")
        print("Baseline: \(formatDate(baseline.timestamp))")
        print(String(repeating: "=", count: 80))
        print()

        print(
            String(
                format: "%-40s %12s %12s %12s",
                "Benchmark", "Current", "Baseline", "Change"
            )
        )
        print(String(repeating: "-", count: 80))

        let baselineMap = Dictionary(uniqueKeysWithValues: baseline.results.map { ($0.name, $0) })

        for result in current.results {
            guard let baseResult = baselineMap[result.name] else {
                print(
                    String(
                        format: "%-40s %12.2f %12s %12s",
                        truncate(result.name, to: 40),
                        result.meanTimeMs,
                        "NEW",
                        "-"
                    )
                )
                continue
            }

            let change = (result.meanTimeMs - baseResult.meanTimeMs) / baseResult.meanTimeMs * 100
            let changeStr: String
            let emoji: String

            if change < -5 {
                changeStr = String(format: "%.1f%%", change)
                emoji = "faster"
            } else if change > 5 {
                changeStr = String(format: "+%.1f%%", change)
                emoji = "slower"
            } else {
                changeStr = String(format: "%.1f%%", change)
                emoji = "~same"
            }

            print(
                String(
                    format: "%-40s %12.2f %12.2f %12s",
                    truncate(result.name, to: 40),
                    result.meanTimeMs,
                    baseResult.meanTimeMs,
                    "\(changeStr) (\(emoji))"
                )
            )
        }
        print()
    }

    /// Print target performance status.
    private static func printTargetStatus(_ suite: BenchmarkSuite) {
        print("PERFORMANCE TARGETS")
        print(String(repeating: "-", count: 80))

        let targets: [(model: String, benchmark: String, targetMs: Double, description: String)] = [
            ("htdemucs", "htdemucs_full_180s", 5000, "3min song < 5s"),
            ("whisper", "whisper_transcribe_30s", 3000, "30s audio < 3s"),
            ("clap", "clap_audio_embed_10s", 100, "10s embed < 100ms"),
            ("musicgen", "musicgen_generate_10s", 15000, "10s generation < 15s"),
            ("encodec", "encodec_decode_5s", 50, "5s decode < 50ms"),
        ]

        for target in targets {
            let result = suite.results.first { $0.name == target.benchmark }
            let status: String
            let emoji: String

            if let r = result {
                if r.meanTimeMs <= target.targetMs {
                    status = String(format: "%.0fms (target: %.0fms)", r.meanTimeMs, target.targetMs)
                    emoji = "PASS"
                } else {
                    status = String(format: "%.0fms (target: %.0fms)", r.meanTimeMs, target.targetMs)
                    emoji = "FAIL"
                }
            } else {
                status = "Not run"
                emoji = "SKIP"
            }

            print(
                String(
                    format: "%-12s %-30s %s [%s]",
                    target.model.uppercased(),
                    target.description,
                    status,
                    emoji
                )
            )
        }
        print()
    }

    private static func formatDate(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.dateStyle = .medium
        formatter.timeStyle = .short
        return formatter.string(from: date)
    }

    private static func truncate(_ string: String, to length: Int) -> String {
        if string.count <= length {
            return string
        }
        return String(string.prefix(length - 3)) + "..."
    }
}

/// CSV export for spreadsheet analysis.
extension BenchmarkReporter {
    public static func saveCSV(_ suite: BenchmarkSuite, to path: String) throws {
        var csv = "name,mean_ms,std_ms,min_ms,max_ms,throughput,memory_mb,rtf,iterations\n"

        for result in suite.results {
            let rtf = result.realtimeFactor.map { String($0) } ?? ""
            csv +=
                "\(result.name),\(result.meanTimeMs),\(result.stdTimeMs),"
            csv +=
                "\(result.minTimeMs),\(result.maxTimeMs),\(result.throughput),"
            csv += "\(result.peakMemoryMB),\(rtf),\(result.iterations)\n"
        }

        let url = URL(fileURLWithPath: path)
        try csv.write(to: url, atomically: true, encoding: .utf8)
    }
}
