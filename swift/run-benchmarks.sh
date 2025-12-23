#!/bin/bash
# run-benchmarks.sh - Run Swift benchmarks with proper Metal library support
#
# Usage:
#   ./run-benchmarks.sh                    # Run all benchmarks
#   ./run-benchmarks.sh --quick            # Quick mode (fewer iterations)
#   ./run-benchmarks.sh --model htdemucs   # Single model
#   ./run-benchmarks.sh --output out.json  # Save to JSON

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Note: We use xcodebuild to build because MLX-Swift requires Metal shaders
# bundled in mlx-swift_Cmlx.bundle, which SwiftPM's command line runner
# doesn't properly locate.

echo "Building benchmarks in Release mode..."

# Build in release mode for accurate performance measurements
xcodebuild build \
    -scheme MLXAudioBenchmarks \
    -configuration Release \
    -destination 'platform=macOS' \
    2>&1 | grep -E "(BUILD SUCCEEDED|BUILD FAILED|error:)" || true

# Get the DerivedData path
DERIVED_DATA="$HOME/Library/Developer/Xcode/DerivedData"

# Find the most recent build
BUILD_DIR=$(find "$DERIVED_DATA" -name "swift-*" -type d -maxdepth 1 2>/dev/null | head -1)

if [ -z "$BUILD_DIR" ]; then
    echo "Could not find DerivedData build directory"
    exit 1
fi

EXECUTABLE="$BUILD_DIR/Build/Products/Release/MLXAudioBenchmarks"

if [ ! -f "$EXECUTABLE" ]; then
    echo "Executable not found at: $EXECUTABLE"
    echo "Looking for alternative paths..."
    find "$BUILD_DIR/Build/Products" -name "MLXAudioBenchmarks" -type f 2>/dev/null
    exit 1
fi

echo "Running benchmarks..."
echo ""

# Run the benchmark executable with all passed arguments
"$EXECUTABLE" "$@"
