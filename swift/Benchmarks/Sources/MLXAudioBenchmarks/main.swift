// main.swift
// CLI entry point for mlx-audio benchmarks.

import Foundation
import MLX
import BenchmarkKit
import ModelBenchmarks

// MARK: - Main Entry Point

@main
struct MLXAudioBenchmarks {

    // MARK: - CLI Arguments

    struct CLI {
        var output: String?
        var compare: String?
        var quick: Bool = false
        var model: String?
        var warmup: Int = 3
        var iterations: Int = 10
        var verbose: Bool = false
        var listModels: Bool = false
        var checkTargets: Bool = false
        var showHelp: Bool = false
        var parseError: String?

        static func parseArguments() -> CLI {
            var cli = CLI()
            var args = Array(CommandLine.arguments.dropFirst())

            while !args.isEmpty {
                let arg = args.removeFirst()

                switch arg {
                case "-o", "--output":
                    cli.output = args.isEmpty ? nil : args.removeFirst()
                case "-c", "--compare":
                    cli.compare = args.isEmpty ? nil : args.removeFirst()
                case "-q", "--quick":
                    cli.quick = true
                case "-m", "--model":
                    cli.model = args.isEmpty ? nil : args.removeFirst()
                case "-w", "--warmup":
                    cli.warmup = Int(args.isEmpty ? "3" : args.removeFirst()) ?? 3
                case "-i", "--iterations":
                    cli.iterations = Int(args.isEmpty ? "10" : args.removeFirst()) ?? 10
                case "-v", "--verbose":
                    cli.verbose = true
                case "-l", "--list":
                    cli.listModels = true
                case "-t", "--targets":
                    cli.checkTargets = true
                case "-h", "--help":
                    cli.showHelp = true
                default:
                    cli.parseError = "Unknown argument: \(arg)"
                }
            }

            return cli
        }
    }

    // MARK: - Help Text

    static func printUsage() {
        print(
            """
            mlx-audio-benchmarks - Performance benchmarks for Swift mlx-audio models

            USAGE:
                swift run MLXAudioBenchmarks [OPTIONS]

            OPTIONS:
                -o, --output <path>      Save results to JSON file
                -c, --compare <path>     Compare with baseline JSON
                -q, --quick              Quick mode (fewer iterations)
                -m, --model <name>       Run only specific model (htdemucs, whisper, clap, musicgen, encodec)
                -w, --warmup <n>         Number of warmup iterations (default: 3)
                -i, --iterations <n>     Number of measurement iterations (default: 10)
                -v, --verbose            Verbose output
                -l, --list               List available benchmarks
                -t, --targets            Check performance targets
                -h, --help               Show this help

            EXAMPLES:
                swift run MLXAudioBenchmarks --output results.json
                swift run MLXAudioBenchmarks --quick --model htdemucs
                swift run MLXAudioBenchmarks --compare baseline.json
                swift run MLXAudioBenchmarks --targets
            """
        )
    }

    // MARK: - Benchmark List

    static func listBenchmarks() {
        print(
            """
            Available Benchmarks
            ====================

            HTDemucs (htdemucs):
              - htdemucs_encoder_layer     Single encoder layer
              - htdemucs_decoder_layer     Single decoder layer
              - htdemucs_full_10s          Full model (10s audio)
              - htdemucs_full_30s          Full model (30s audio)
              - htdemucs_full_60s          Full model (60s audio)
              - htdemucs_full_180s         Full model (3min audio) [TARGET]

            Whisper (whisper):
              - whisper_encoder            Audio encoder only
              - whisper_transcribe_10s     Full transcription (10s)
              - whisper_transcribe_30s     Full transcription (30s) [TARGET]
              - whisper_transcribe_60s     Full transcription (60s)

            CLAP (clap):
              - clap_audio_embed_1s        Audio embedding (1s)
              - clap_audio_embed_5s        Audio embedding (5s)
              - clap_audio_embed_10s       Audio embedding (10s) [TARGET]
              - clap_audio_embed_30s       Audio embedding (30s)
              - clap_batch_audio_embed_*   Batch encoding

            MusicGen (musicgen):
              - musicgen_decoder_step      Single decoder step
              - musicgen_generate_5s       Generation (5s output)
              - musicgen_generate_10s      Generation (10s output) [TARGET]
              - musicgen_generate_20s      Generation (20s output)

            EnCodec (encodec):
              - encodec_encode_*           Encoding at various durations
              - encodec_decode_5s          Decoding (5s) [TARGET]
              - encodec_roundtrip_*        Full encode/decode cycle
              - encodec_batch_decode_*     Batch decoding
            """
        )
    }

    // MARK: - Benchmark Runner

    static func runBenchmarks(cli: CLI) async throws {
        let config = BenchmarkConfig(
            warmup: cli.quick ? 1 : cli.warmup,
            iterations: cli.quick ? 3 : cli.iterations,
            verbose: cli.verbose
        )

        var allResults: [BenchmarkResult] = []

        // Run selected benchmarks
        if cli.model == nil || cli.model == "htdemucs" {
            print("Running HTDemucs benchmarks...")
            let results = try HTDemucsBenchmarks.runAll(config: config)
            allResults.append(contentsOf: results)
            print("  Completed \(results.count) benchmarks")
        }

        if cli.model == nil || cli.model == "whisper" {
            print("Running Whisper benchmarks...")
            let results = try WhisperBenchmarks.runAll(config: config)
            allResults.append(contentsOf: results)
            print("  Completed \(results.count) benchmarks")
        }

        if cli.model == nil || cli.model == "clap" {
            print("Running CLAP benchmarks...")
            let results = try CLAPBenchmarks.runAll(config: config)
            allResults.append(contentsOf: results)
            print("  Completed \(results.count) benchmarks")
        }

        if cli.model == nil || cli.model == "musicgen" {
            print("Running MusicGen benchmarks...")
            let results = try MusicGenBenchmarks.runAll(config: config)
            allResults.append(contentsOf: results)
            print("  Completed \(results.count) benchmarks")
        }

        if cli.model == nil || cli.model == "encodec" {
            print("Running EnCodec benchmarks...")
            let results = try EnCodecBenchmarks.runAll(config: config)
            allResults.append(contentsOf: results)
            print("  Completed \(results.count) benchmarks")
        }

        // Create suite
        let suite = BenchmarkSuite(
            name: "mlx-audio-swift",
            device: DeviceInfo.current(),
            results: allResults,
            metadata: [
                "version": "1.0.0",
                "config": cli.quick ? "quick" : "standard",
            ]
        )

        // Save results
        if let outputPath = cli.output {
            try BenchmarkReporter.saveJSON(suite, to: outputPath)
            print("\nResults saved to: \(outputPath)")

            // Also save CSV
            let csvPath = outputPath.replacingOccurrences(of: ".json", with: ".csv")
            try BenchmarkReporter.saveCSV(suite, to: csvPath)
            print("CSV saved to: \(csvPath)")
        }

        // Compare with baseline
        if let comparePath = cli.compare {
            try BenchmarkReporter.compare(current: suite, baselinePath: comparePath)
        }

        // Print summary
        BenchmarkReporter.printSummary(suite)
    }

    // MARK: - Target Checking

    static func checkTargets() throws {
        print("\nChecking Performance Targets")
        print(String(repeating: "=", count: 60))

        print("\n[HTDemucs] 3min song < 5s...")
        let htdemucs = try HTDemucsBenchmarks.checkTargetPerformance()
        print(
            "  Result: \(String(format: "%.0fms", htdemucs.timeMs)) [\(htdemucs.passed ? "PASS" : "FAIL")]"
        )

        print("\n[Whisper] 30s audio < 3s...")
        let whisper = try WhisperBenchmarks.checkTargetPerformance()
        print(
            "  Result: \(String(format: "%.0fms", whisper.timeMs)) [\(whisper.passed ? "PASS" : "FAIL")]"
        )

        print("\n[CLAP] 10s embed < 100ms...")
        let clap = try CLAPBenchmarks.checkTargetPerformance()
        print(
            "  Result: \(String(format: "%.0fms", clap.timeMs)) [\(clap.passed ? "PASS" : "FAIL")]")

        print("\n[MusicGen] 10s generation < 15s...")
        let musicgen = try MusicGenBenchmarks.checkTargetPerformance()
        print(
            "  Result: \(String(format: "%.0fms", musicgen.timeMs)) [\(musicgen.passed ? "PASS" : "FAIL")]"
        )

        print("\n[EnCodec] 5s decode < 50ms...")
        let encodec = try EnCodecBenchmarks.checkTargetPerformance()
        print(
            "  Result: \(String(format: "%.0fms", encodec.timeMs)) [\(encodec.passed ? "PASS" : "FAIL")]"
        )

        print()
        let allPassed =
            htdemucs.passed && whisper.passed && clap.passed && musicgen.passed && encodec.passed
        print("Overall: \(allPassed ? "ALL TARGETS MET" : "SOME TARGETS NOT MET")")
    }

    // MARK: - Main

    static func main() async {
        let cli = CLI.parseArguments()

        if let error = cli.parseError {
            print(error)
            printUsage()
            Darwin.exit(1)
        }

        if cli.showHelp {
            printUsage()
            return
        }

        if cli.listModels {
            listBenchmarks()
            return
        }

        do {
            if cli.checkTargets {
                try checkTargets()
            } else {
                try await runBenchmarks(cli: cli)
            }
        } catch {
            print("Error: \(error)")
            Darwin.exit(1)
        }
    }
}
