#!/bin/bash
# test.sh - Run Swift tests with proper Metal library support
#
# Usage:
#   ./test.sh              # Run all tests
#   ./test.sh --coverage   # Run with code coverage
#   ./test.sh --quiet      # Minimal output

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Parse arguments
COVERAGE=""
QUIET=""
for arg in "$@"; do
    case $arg in
        --coverage)
            COVERAGE="-enableCodeCoverage YES"
            ;;
        --quiet)
            QUIET="-quiet"
            ;;
    esac
done

# Note: We use xcodebuild instead of `swift test` because MLX-Swift
# requires Metal shaders bundled in mlx-swift_Cmlx.bundle, which
# SwiftPM's test runner doesn't properly locate from command line.

echo "Building and running tests..."

xcodebuild test \
    -scheme MLXAudio-Package \
    -destination 'platform=macOS' \
    $COVERAGE \
    $QUIET \
    2>&1 | grep -E "(Test Case.*passed|Test Case.*failed|Executed|TEST SUCCEEDED|TEST FAILED|error:)" || true

# Capture exit code from xcodebuild (not grep)
RESULT=${PIPESTATUS[0]}

if [ $RESULT -eq 0 ]; then
    echo ""
    echo "✅ All tests passed"
else
    echo ""
    echo "❌ Tests failed"
    exit 1
fi
