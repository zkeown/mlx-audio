#!/usr/bin/env python3
"""
Run mlx-audio benchmarks and compare results.

Usage:
    # Run full benchmark suite and save baseline
    python -m benchmarks.run_benchmarks --output baseline.json

    # Run after optimization and save results
    python -m benchmarks.run_benchmarks --output optimized.json

    # Compare two benchmark runs
    python -m benchmarks.run_benchmarks --compare baseline.json optimized.json

    # Quick run with fewer iterations
    python -m benchmarks.run_benchmarks --quick --output quick_test.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def run_benchmarks(quick: bool = False) -> dict[str, Any]:
    """Run all benchmarks and return results."""
    from benchmarks.bench_primitives import BenchPrimitives
    from benchmarks.bench_streaming import BenchStreaming

    results: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "primitives": [],
        "streaming": [],
    }

    # Configure based on quick mode
    if quick:
        durations = [1.0, 5.0]
        batch_sizes = [1]
        iterations = 5
        buffer_sizes = [16000]
        chunk_sizes = [512, 1024]
        stream_iterations = 200
    else:
        durations = [1.0, 5.0, 10.0, 30.0]
        batch_sizes = [1, 4, 8]
        iterations = 10
        buffer_sizes = [8000, 16000, 48000]
        chunk_sizes = [256, 512, 1024, 2048]
        stream_iterations = 1000

    print("Running primitive benchmarks...")
    prim_bench = BenchPrimitives()
    prim_bench.run_all(durations=durations, batch_sizes=batch_sizes, iterations=iterations)
    results["primitives"] = [r.to_dict() for r in prim_bench.results]
    print(prim_bench.summary())

    print("\nRunning streaming benchmarks...")
    stream_bench = BenchStreaming()
    stream_bench.run_all(
        buffer_sizes=buffer_sizes, chunk_sizes=chunk_sizes, iterations=stream_iterations
    )
    results["streaming"] = [r.to_dict() for r in stream_bench.results]
    print(stream_bench.summary())

    return results


def compare_results(baseline_path: str, optimized_path: str) -> None:
    """Compare two benchmark result files and show improvements."""
    with open(baseline_path) as f:
        baseline = json.load(f)
    with open(optimized_path) as f:
        optimized = json.load(f)

    print("=" * 80)
    print("BENCHMARK COMPARISON")
    print(f"Baseline: {baseline_path} ({baseline['timestamp']})")
    print(f"Optimized: {optimized_path} ({optimized['timestamp']})")
    print("=" * 80)

    # Build lookup tables
    def build_lookup(results: list[dict]) -> dict[str, dict]:
        lookup = {}
        for r in results:
            key = (r["name"], json.dumps(r["params"], sort_keys=True))
            lookup[key] = r
        return lookup

    for category in ["primitives", "streaming"]:
        if category not in baseline or category not in optimized:
            continue

        print(f"\n{category.upper()}")
        print("-" * 60)

        base_lookup = build_lookup(baseline[category])
        opt_lookup = build_lookup(optimized[category])

        improvements = []

        for key, base_result in base_lookup.items():
            if key not in opt_lookup:
                continue

            opt_result = opt_lookup[key]
            name, params_json = key

            base_time = base_result["mean_time_ms"]
            opt_time = opt_result["mean_time_ms"]

            if base_time > 0:
                speedup = base_time / opt_time
                improvement_pct = (base_time - opt_time) / base_time * 100
            else:
                speedup = 1.0
                improvement_pct = 0.0

            improvements.append(
                {
                    "name": name,
                    "params": json.loads(params_json),
                    "base_time_ms": base_time,
                    "opt_time_ms": opt_time,
                    "speedup": speedup,
                    "improvement_pct": improvement_pct,
                }
            )

        # Sort by improvement
        improvements.sort(key=lambda x: x["improvement_pct"], reverse=True)

        # Print results
        print(f"{'Benchmark':<25} {'Base(ms)':>10} {'Opt(ms)':>10} {'Speedup':>10} {'Improve':>10}")
        print("-" * 65)

        for imp in improvements:
            name = imp["name"]
            params = imp["params"]
            # Add key param info to name
            if "duration_sec" in params:
                name = f"{name}[{params['duration_sec']}s]"
            elif "buffer_size" in params:
                name = f"{name}[{params['buffer_size']}]"

            color = ""
            reset = ""
            if imp["improvement_pct"] > 5:
                color = "\033[92m"  # Green
                reset = "\033[0m"
            elif imp["improvement_pct"] < -5:
                color = "\033[91m"  # Red
                reset = "\033[0m"

            print(
                f"{color}{name:<25} {imp['base_time_ms']:>10.2f} {imp['opt_time_ms']:>10.2f} "
                f"{imp['speedup']:>9.2f}x {imp['improvement_pct']:>9.1f}%{reset}"
            )

        # Summary
        if improvements:
            avg_speedup = sum(i["speedup"] for i in improvements) / len(improvements)
            avg_improvement = sum(i["improvement_pct"] for i in improvements) / len(improvements)
            print("-" * 65)
            print(f"{'AVERAGE':<25} {'':<10} {'':<10} {avg_speedup:>9.2f}x {avg_improvement:>9.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Run mlx-audio benchmarks and compare results."
    )
    parser.add_argument(
        "--output", "-o", type=str, help="Output file for benchmark results (JSON)"
    )
    parser.add_argument(
        "--compare",
        "-c",
        nargs=2,
        metavar=("BASELINE", "OPTIMIZED"),
        help="Compare two benchmark result files",
    )
    parser.add_argument(
        "--quick", "-q", action="store_true", help="Quick run with fewer iterations"
    )

    args = parser.parse_args()

    if args.compare:
        compare_results(args.compare[0], args.compare[1])
    else:
        results = run_benchmarks(quick=args.quick)

        if args.output:
            output_path = Path(args.output)
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {output_path}")
        else:
            print("\nNo output file specified. Use --output to save results.")


if __name__ == "__main__":
    main()
