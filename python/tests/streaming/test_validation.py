"""Validation tests using MUSDB18-HQ dataset.

These tests validate:
1. Streaming output matches batch processing output
2. MLX output matches PyTorch demucs output
3. Separation quality meets expected SDR thresholds

Requires:
- MUSDB18-HQ dataset at /Users/zakkeown/datasets/MUSDB18HQ/
- demucs package for PyTorch comparison (pip install demucs)
- soundfile for audio I/O (pip install soundfile)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import mlx.core as mx

# Dataset path
MUSDB_PATH = Path("/Users/zakkeown/datasets/MUSDB18HQ")
TEST_TRACKS_PATH = MUSDB_PATH / "test"

# Check if dataset is available
HAS_MUSDB = TEST_TRACKS_PATH.exists()

# Check if demucs is available
try:
    from demucs.pretrained import get_model as _get_model  # noqa: F401

    HAS_DEMUCS = True
except ImportError:
    HAS_DEMUCS = False

# Check if soundfile is available
try:
    import soundfile as _sf  # noqa: F401

    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False


def get_test_tracks() -> list[Path]:
    """Get list of available test tracks."""
    if not HAS_MUSDB:
        return []
    return sorted(TEST_TRACKS_PATH.iterdir())[:5]  # First 5 tracks for speed


def get_full_stem_tracks() -> list[Path]:
    """Get tracks that have all stems with significant energy.

    Some MUSDB18 tracks have silent stems (no bass or vocals).
    This returns tracks suitable for full separation testing.
    """
    if not HAS_MUSDB:
        return []

    # Known good tracks with all 4 stems having significant energy
    good_tracks = [
        "Al James - Schoolboy Facination",
    ]

    tracks = []
    for name in good_tracks:
        path = TEST_TRACKS_PATH / name
        if path.exists():
            tracks.append(path)

    return tracks


def stem_has_energy(audio: np.ndarray, threshold: float = 1e-5) -> bool:
    """Check if a stem has significant energy."""
    return float(np.mean(audio**2)) > threshold


def load_track(
    track_path: Path, duration_seconds: float = 10.0
) -> tuple[np.ndarray, dict[str, np.ndarray], int]:
    """Load a track and its stems.

    Args:
        track_path: Path to track directory
        duration_seconds: How many seconds to load

    Returns:
        (mixture, stems_dict, sample_rate)
    """
    import soundfile as sf

    mixture, sr = sf.read(track_path / "mixture.wav")
    samples = int(duration_seconds * sr)
    mixture = mixture[:samples].T.astype(np.float32)  # [channels, samples]

    stems = {}
    for stem_name in ["drums", "bass", "other", "vocals"]:
        audio, _ = sf.read(track_path / f"{stem_name}.wav")
        stems[stem_name] = audio[:samples].T.astype(np.float32)

    return mixture, stems, sr


@pytest.fixture(scope="module")
def mlx_model():
    """Load MLX HTDemucs model."""
    from mlx_audio.models.demucs import HTDemucs

    model_path = Path.home() / ".cache/mlx_audio/models/htdemucs_ft"
    if not model_path.exists():
        pytest.skip("MLX HTDemucs model not available")

    return HTDemucs.from_pretrained(str(model_path))


@pytest.fixture(scope="module")
def pt_model():
    """Load PyTorch HTDemucs model."""
    if not HAS_DEMUCS:
        pytest.skip("demucs package not available")

    from demucs.pretrained import get_model

    bag = get_model("htdemucs_ft")
    model = bag.models[0]
    model.eval()
    return model


@pytest.fixture
def sample_track():
    """Load a sample track from MUSDB18-HQ with all stems."""
    if not HAS_MUSDB or not HAS_SOUNDFILE:
        pytest.skip("MUSDB18-HQ or soundfile not available")

    tracks = get_full_stem_tracks()
    if not tracks:
        pytest.skip("No full-stem test tracks found")

    return load_track(tracks[0], duration_seconds=10.0)


class TestMetrics:
    """Test the metrics module."""

    def test_si_sdr_perfect(self):
        """SI-SDR of identical signals should be very high."""
        from mlx_audio.streaming.metrics import si_sdr

        signal = np.random.randn(44100).astype(np.float32)
        score = si_sdr(signal, signal)

        # Perfect match should give very high SI-SDR
        assert score > 100, f"SI-SDR identical should be >100 dB, got {score}"

    def test_si_sdr_scaled(self):
        """SI-SDR should be invariant to scaling."""
        from mlx_audio.streaming.metrics import si_sdr

        signal = np.random.randn(44100).astype(np.float32)
        scaled = signal * 2.0

        score = si_sdr(scaled, signal)

        # Scale-invariant, so should still be very high
        assert score > 100, f"SI-SDR should be scale-invariant, got {score}"

    def test_si_sdr_noise(self):
        """SI-SDR with noise should be finite."""
        from mlx_audio.streaming.metrics import si_sdr

        t = np.arange(44100) / 44100
        signal = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        noisy = signal + 0.1 * np.random.randn(44100).astype(np.float32)

        score = si_sdr(noisy, signal)

        # Should be positive but not infinite
        assert 0 < score < 50, f"SI-SDR with noise: {score}"

    def test_sdr_computation(self):
        """Test basic SDR computation."""
        from mlx_audio.streaming.metrics import sdr

        signal = np.random.randn(44100).astype(np.float32)
        score = sdr(signal, signal)

        assert score > 100, f"SDR identical should be >100 dB, got {score}"

    def test_correlation(self):
        """Test correlation computation."""
        from mlx_audio.streaming.metrics import correlation

        signal = np.random.randn(44100).astype(np.float32)

        # Perfect correlation
        assert abs(correlation(signal, signal) - 1.0) < 1e-6

        # Anti-correlation
        assert abs(correlation(signal, -signal) + 1.0) < 1e-6

    def test_separation_metrics(self):
        """Test SeparationMetrics class."""
        from mlx_audio.streaming.metrics import SeparationMetrics

        metrics = SeparationMetrics()

        # Create fake estimates and references
        np.random.seed(42)
        references = np.random.randn(4, 2, 44100).astype(np.float32)
        noise = 0.01 * np.random.randn(4, 2, 44100).astype(np.float32)
        estimates = references + noise

        results = metrics.evaluate(estimates, references)

        assert "drums" in results
        assert "vocals" in results
        assert "mean" in results
        assert "si_sdr" in results["drums"]

        # High quality since we added only small noise
        assert results["mean"]["si_sdr"] > 30


@pytest.mark.skipif(
    not HAS_MUSDB or not HAS_SOUNDFILE,
    reason="MUSDB18-HQ or soundfile not available",
)
class TestMusdbValidation:
    """Validation tests using MUSDB18-HQ."""

    def test_dataset_accessible(self):
        """Verify MUSDB18-HQ dataset is accessible."""
        tracks = get_test_tracks()
        assert len(tracks) > 0, "No test tracks found"

        # Check structure
        track = tracks[0]
        assert (track / "mixture.wav").exists()
        assert (track / "vocals.wav").exists()
        assert (track / "drums.wav").exists()
        assert (track / "bass.wav").exists()
        assert (track / "other.wav").exists()

    def test_load_track(self):
        """Test loading a track."""
        tracks = get_test_tracks()
        mixture, stems, sr = load_track(tracks[0], duration_seconds=5.0)

        assert sr == 44100
        assert mixture.shape[0] == 2  # Stereo
        assert mixture.shape[1] == 5 * 44100  # 5 seconds
        assert len(stems) == 4
        assert all(s.shape == mixture.shape for s in stems.values())


@pytest.mark.skipif(
    not HAS_MUSDB or not HAS_SOUNDFILE,
    reason="MUSDB18-HQ or soundfile not available",
)
class TestBatchProcessing:
    """Test batch processing against ground truth."""

    def test_batch_separation_quality(self, mlx_model, sample_track):
        """Test that batch separation produces reasonable quality."""
        from mlx_audio.streaming.metrics import SeparationMetrics

        mixture, gt_stems, sr = sample_track

        # Run batch inference
        mlx_audio = mx.array(mixture[np.newaxis, :, :])  # [1, 2, samples]
        mx.eval(mlx_audio)

        output = mlx_model(mlx_audio)
        mx.eval(output)

        estimates = np.array(output)[0]  # [4, 2, samples]

        # Convert ground truth to array
        stem_order = ["drums", "bass", "other", "vocals"]
        gt_array = np.stack([gt_stems[s] for s in stem_order])

        # Evaluate
        metrics = SeparationMetrics()
        results = metrics.evaluate(estimates, gt_array)

        # Expected minimum quality thresholds
        # HTDemucs achieves ~9 dB SDR on MUSDB18 with chunked inference
        # Direct model call may be slightly lower; accept > 3.0 dB
        mean_si_sdr = results["mean"]["si_sdr"]
        mean_corr = results["mean"]["correlation"]
        assert mean_si_sdr > 3.0, f"Mean SI-SDR too low: {mean_si_sdr}"
        assert mean_corr > 0.7, f"Mean correlation too low: {mean_corr}"

        # Print detailed results
        print("\nBatch Separation Results:")
        for stem in ["drums", "bass", "other", "vocals"]:
            si = results[stem]["si_sdr"]
            corr = results[stem]["correlation"]
            print(f"  {stem}: SI-SDR={si:.2f} dB, corr={corr:.3f}")
        print(f"  mean: SI-SDR={mean_si_sdr:.2f} dB")


@pytest.mark.skipif(
    not HAS_MUSDB or not HAS_SOUNDFILE,
    reason="MUSDB18-HQ or soundfile not available",
)
class TestStreamingVsBatch:
    """Test that streaming output matches batch output."""

    def test_streaming_matches_batch(self, mlx_model, sample_track):
        """Streaming separation should match batch separation."""
        from mlx_audio.streaming import HTDemucsStreamProcessor
        from mlx_audio.streaming.metrics import SeparationMetrics

        mixture, gt_stems, sr = sample_track

        # Batch processing
        mlx_audio = mx.array(mixture[np.newaxis, :, :])
        mx.eval(mlx_audio)
        batch_output = mlx_model(mlx_audio)
        mx.eval(batch_output)
        batch_result = np.array(batch_output)[0]

        # Streaming processing
        processor = HTDemucsStreamProcessor(
            mlx_model, segment=6.0, overlap=0.25
        )
        ctx = processor.initialize_context(sr)

        chunk_size = processor.get_chunk_size()
        overlap_size = processor.get_overlap_size()
        stride = chunk_size - overlap_size

        # Process in chunks
        streaming_outputs = []
        position = 0
        total_samples = mixture.shape[-1]

        while position < total_samples:
            end = min(position + chunk_size, total_samples)
            chunk = mx.array(mixture[:, position:end])

            # Pad if needed
            if chunk.shape[-1] < chunk_size:
                pad_size = chunk_size - chunk.shape[-1]
                chunk = mx.pad(chunk, [(0, 0), (0, pad_size)])

            output = processor.process_chunk(chunk, ctx)
            mx.eval(output)

            if output is not None:
                streaming_outputs.append(np.array(output))

            position += stride

        # Finalize
        final = processor.finalize(ctx)
        if final is not None:
            mx.eval(final)
            final_np = np.array(final)
            # Finalize may have batch dim removed, add it back
            if final_np.ndim == 3:  # [S, C, T]
                streaming_outputs.append(final_np)

        # Concatenate streaming outputs
        if streaming_outputs:
            streaming_result = np.concatenate(streaming_outputs, axis=-1)
            # Trim to match batch length
            streaming_result = streaming_result[..., :total_samples]
        else:
            pytest.fail("No streaming output produced")

        # Compare streaming vs batch
        stem_order = ["drums", "bass", "other", "vocals"]
        gt_array = np.stack([gt_stems[s] for s in stem_order])

        metrics = SeparationMetrics()
        comparison = metrics.compare(streaming_result, batch_result, gt_array)

        # Streaming should be close to batch
        # Allow some difference due to different overlap-add accumulation
        si_sdr_diff = abs(comparison["diff"]["mean"]["si_sdr"])
        assert si_sdr_diff < 2.0, f"Stream/batch diff too large: {si_sdr_diff}"

        stream_si = comparison["a"]["mean"]["si_sdr"]
        batch_si = comparison["b"]["mean"]["si_sdr"]
        diff_si = comparison["diff"]["mean"]["si_sdr"]
        print("\nStreaming vs Batch Comparison:")
        print(f"  Streaming mean SI-SDR: {stream_si:.2f} dB")
        print(f"  Batch mean SI-SDR: {batch_si:.2f} dB")
        print(f"  Difference: {diff_si:.2f} dB")


@pytest.mark.skipif(not HAS_DEMUCS, reason="demucs package not available")
@pytest.mark.skipif(
    not HAS_MUSDB or not HAS_SOUNDFILE,
    reason="MUSDB18-HQ or soundfile not available",
)
class TestPyTorchParity:
    """Test MLX output matches PyTorch demucs output."""

    def test_mlx_matches_pytorch(self, mlx_model, pt_model, sample_track):
        """MLX apply_model output should match PyTorch apply_model output."""
        import torch
        from demucs.apply import apply_model as pt_apply_model

        from mlx_audio.models.demucs.inference import (
            apply_model as mlx_apply_model,
        )

        mixture, gt_stems, sr = sample_track

        # PyTorch inference using apply_model (handles chunking)
        pt_audio = torch.from_numpy(mixture[np.newaxis, :, :])
        with torch.no_grad():
            pt_output = pt_apply_model(pt_model, pt_audio, overlap=0.25)[0]
        pt_result = pt_output.numpy()

        # MLX inference using apply_model (same chunking strategy)
        mlx_output = mlx_apply_model(
            mlx_model, mx.array(mixture), overlap=0.25
        )
        mx.eval(mlx_output)
        mlx_result = np.array(mlx_output)

        # Compare correlations - both use chunked inference with same overlap
        stems = ["drums", "bass", "other", "vocals"]
        print("\nMLX apply_model vs PyTorch apply_model:")
        for i, stem in enumerate(stems):
            flat_pt = pt_result[i].flatten()
            flat_mlx = mlx_result[i].flatten()
            stem_corr = np.corrcoef(flat_pt, flat_mlx)[0, 1]
            print(f"  {stem}: corr={stem_corr:.6f}")
            # High correlation expected with same chunking approach
            assert stem_corr > 0.95, f"{stem} corr: {stem_corr}"

    def test_streaming_matches_pytorch(
        self, mlx_model, pt_model, sample_track
    ):
        """MLX streaming output should match PyTorch apply_model."""
        import torch
        from demucs.apply import apply_model as pt_apply_model

        from mlx_audio.streaming import HTDemucsStreamProcessor
        from mlx_audio.streaming.metrics import correlation

        mixture, _, sr = sample_track

        # PyTorch reference using apply_model
        pt_audio = torch.from_numpy(mixture[np.newaxis, :, :])
        with torch.no_grad():
            pt_output = pt_apply_model(pt_model, pt_audio, overlap=0.25)[0]
        pt_result = pt_output.numpy()

        # MLX streaming
        processor = HTDemucsStreamProcessor(
            mlx_model, segment=6.0, overlap=0.25
        )
        ctx = processor.initialize_context(sr)

        chunk_size = processor.get_chunk_size()
        overlap_size = processor.get_overlap_size()
        stride = chunk_size - overlap_size

        outputs = []
        position = 0
        total_samples = mixture.shape[-1]

        while position < total_samples:
            end = min(position + chunk_size, total_samples)
            chunk = mx.array(mixture[:, position:end])

            if chunk.shape[-1] < chunk_size:
                pad = chunk_size - chunk.shape[-1]
                chunk = mx.pad(chunk, [(0, 0), (0, pad)])

            output = processor.process_chunk(chunk, ctx)
            mx.eval(output)
            if output is not None:
                outputs.append(np.array(output))

            position += stride

        final = processor.finalize(ctx)
        if final is not None:
            mx.eval(final)
            final_np = np.array(final)
            if final_np.ndim == 3:
                outputs.append(final_np)

        streaming_result = np.concatenate(outputs, axis=-1)
        streaming_result = streaming_result[..., :total_samples]

        # Compare correlations (streaming vs PyTorch apply_model)
        stems = ["drums", "bass", "other", "vocals"]
        print("\nStreaming MLX vs PyTorch apply_model:")
        for i, stem in enumerate(stems):
            corr = correlation(streaming_result[i], pt_result[i])
            print(f"  {stem}: corr={corr:.4f}")
            # Good correlation despite different implementations
            assert corr > 0.8, f"{stem} corr too low: {corr}"


@pytest.mark.skipif(
    not HAS_MUSDB or not HAS_SOUNDFILE,
    reason="MUSDB18-HQ or soundfile not available",
)
class TestMultiTrackValidation:
    """Test across multiple tracks for robustness."""

    def test_separation_quality_full_stem_tracks(self, mlx_model):
        """Test separation quality on tracks with all stems present."""
        from mlx_audio.streaming.metrics import SeparationMetrics

        tracks = get_full_stem_tracks()
        if not tracks:
            pytest.skip("No full-stem tracks available")

        metrics = SeparationMetrics()
        all_results = []

        for track in tracks:
            mixture, gt_stems, sr = load_track(track, duration_seconds=10.0)

            # Run inference
            mlx_audio = mx.array(mixture[np.newaxis, :, :])
            mx.eval(mlx_audio)
            output = mlx_model(mlx_audio)
            mx.eval(output)

            estimates = np.array(output)[0]
            stem_order = ["drums", "bass", "other", "vocals"]
            gt_array = np.stack([gt_stems[s] for s in stem_order])

            results = metrics.evaluate(estimates, gt_array)
            all_results.append(results)

            print(f"\n{track.name}:")
            print(f"  Mean SI-SDR: {results['mean']['si_sdr']:.2f} dB")

            # Per-track threshold
            assert results["mean"]["si_sdr"] > 2.0, (
                f"Track '{track.name}' SI-SDR too low"
            )

        # Average across tracks should be reasonable
        avg_si_sdr = np.mean([r["mean"]["si_sdr"] for r in all_results])
        n = len(tracks)
        print(f"\nAverage SI-SDR across {n} tracks: {avg_si_sdr:.2f}")
        assert avg_si_sdr > 3.0, f"Average SI-SDR too low: {avg_si_sdr}"


BAG_MODEL_PATH = Path.home() / ".cache/mlx_audio/models/htdemucs_ft_bag"
HAS_BAG_MODEL = BAG_MODEL_PATH.exists()


@pytest.fixture(scope="module")
def mlx_bag_model():
    """Load MLX BagOfModels."""
    from mlx_audio.models.demucs import BagOfModels

    return BagOfModels.from_pretrained(BAG_MODEL_PATH)


@pytest.fixture(scope="module")
def pt_bag_model():
    """Load PyTorch BagOfModels."""
    from demucs.pretrained import get_model

    return get_model("htdemucs_ft")


@pytest.mark.skipif(not HAS_BAG_MODEL, reason="BagOfModels not converted")
@pytest.mark.skipif(not HAS_DEMUCS, reason="demucs package not available")
@pytest.mark.skipif(
    not HAS_MUSDB or not HAS_SOUNDFILE,
    reason="MUSDB18-HQ or soundfile not available",
)
class TestBagOfModels:
    """Test BagOfModels ensemble matches PyTorch and improves SI-SDR."""

    def test_bag_matches_pytorch(
        self, mlx_bag_model, pt_bag_model, sample_track
    ):
        """MLX BagOfModels should match PyTorch BagOfModels output."""
        import torch
        from demucs.apply import apply_model as pt_apply_model

        from mlx_audio.models.demucs.inference import (
            apply_model as mlx_apply_model,
        )

        mixture, gt_stems, sr = sample_track

        # PyTorch BagOfModels inference
        pt_audio = torch.from_numpy(mixture[np.newaxis, :, :])
        with torch.no_grad():
            pt_output = pt_apply_model(pt_bag_model, pt_audio, overlap=0.25)[0]
        pt_result = pt_output.numpy()

        # MLX BagOfModels inference
        mlx_output = mlx_apply_model(
            mlx_bag_model, mx.array(mixture), overlap=0.25
        )
        mx.eval(mlx_output)
        mlx_result = np.array(mlx_output)

        # Compare per-stem SI-SDR
        from mlx_audio.streaming.metrics import si_sdr

        stems = ["drums", "bass", "other", "vocals"]
        print("\nBagOfModels SI-SDR Comparison:")
        print(f"{'Stem':<10} | {'PyTorch':>10} | {'MLX':>10} | {'Gap':>8}")
        print("-" * 50)

        for i, stem in enumerate(stems):
            gt = gt_stems[stem]
            pt_sdr = float(si_sdr(mx.array(pt_result[i]), mx.array(gt)))
            mlx_sdr = float(si_sdr(mx.array(mlx_result[i]), mx.array(gt)))
            gap = pt_sdr - mlx_sdr
            print(
                f"{stem:<10} | {pt_sdr:>10.2f} | {mlx_sdr:>10.2f} | {gap:>+8.2f}"
            )

            # MLX should be within 1 dB of PyTorch
            assert abs(gap) < 1.0, f"{stem} gap too large: {gap:.2f} dB"

        # Check numerical similarity
        mae = np.mean(np.abs(pt_result - mlx_result))
        print(f"\nMAE: {mae:.2e}")
        assert mae < 0.01, f"MAE too high: {mae}"

    def test_bag_improves_over_single_model(
        self, mlx_bag_model, mlx_model, sample_track
    ):
        """BagOfModels should improve SI-SDR over single model."""
        from mlx_audio.models.demucs.inference import (
            apply_model as mlx_apply_model,
        )
        from mlx_audio.streaming.metrics import si_sdr

        mixture, gt_stems, sr = sample_track

        # Single model inference
        single_output = mlx_apply_model(
            mlx_model, mx.array(mixture), overlap=0.25
        )
        mx.eval(single_output)
        single_result = np.array(single_output)

        # Bag model inference
        bag_output = mlx_apply_model(
            mlx_bag_model, mx.array(mixture), overlap=0.25
        )
        mx.eval(bag_output)
        bag_result = np.array(bag_output)

        # Compare SI-SDR
        stems = ["drums", "bass", "other", "vocals"]
        print("\nSingle Model vs BagOfModels SI-SDR:")
        print(f"{'Stem':<10} | {'Single':>10} | {'Bag':>10} | {'Gain':>8}")
        print("-" * 50)

        single_sdrs = []
        bag_sdrs = []

        for i, stem in enumerate(stems):
            gt = gt_stems[stem]
            s_sdr = float(si_sdr(mx.array(single_result[i]), mx.array(gt)))
            b_sdr = float(si_sdr(mx.array(bag_result[i]), mx.array(gt)))
            gain = b_sdr - s_sdr
            single_sdrs.append(s_sdr)
            bag_sdrs.append(b_sdr)
            print(
                f"{stem:<10} | {s_sdr:>10.2f} | {b_sdr:>10.2f} | {gain:>+8.2f}"
            )

        # Mean comparison
        single_mean = np.mean(single_sdrs)
        bag_mean = np.mean(bag_sdrs)
        gain = bag_mean - single_mean
        print("-" * 50)
        print(
            f"{'Mean':<10} | {single_mean:>10.2f} | {bag_mean:>10.2f} | {gain:>+8.2f}"
        )

        # BagOfModels should improve mean SI-SDR by at least 2 dB
        assert gain > 2.0, f"Bag improvement too small: {gain:.2f} dB"

    def test_bag_streaming_matches_apply_model(
        self, mlx_bag_model, sample_track
    ):
        """BagOfModels streaming should match apply_model (both use chunking)."""
        from mlx_audio.models.demucs.inference import (
            apply_model as mlx_apply_model,
        )
        from mlx_audio.streaming import HTDemucsStreamProcessor

        mixture, gt_stems, sr = sample_track

        # apply_model with BagOfModels (uses chunking with overlap-add)
        batch_output = mlx_apply_model(
            mlx_bag_model, mx.array(mixture), overlap=0.25
        )
        mx.eval(batch_output)
        batch_result = np.array(batch_output)

        # Streaming processing with BagOfModels
        processor = HTDemucsStreamProcessor(
            mlx_bag_model, segment=6.0, overlap=0.25
        )
        ctx = processor.initialize_context(sr)

        chunk_size = processor.get_chunk_size()
        stride = chunk_size - processor.get_overlap_size()

        outputs = []
        position = 0
        total_samples = mixture.shape[-1]

        while position < total_samples:
            end = min(position + chunk_size, total_samples)
            chunk = mx.array(mixture[:, position:end])

            if chunk.shape[-1] < chunk_size:
                pad = chunk_size - chunk.shape[-1]
                chunk = mx.pad(chunk, [(0, 0), (0, pad)])

            output = processor.process_chunk(chunk, ctx)
            mx.eval(output)
            if output is not None:
                outputs.append(np.array(output))

            position += stride

        final = processor.finalize(ctx)
        if final is not None:
            mx.eval(final)
            outputs.append(np.array(final))

        streaming_result = np.concatenate(outputs, axis=-1)
        streaming_result = streaming_result[..., :total_samples]

        # Compare streaming vs apply_model (both use chunking)
        stems = ["drums", "bass", "other", "vocals"]
        print("\nBagOfModels Streaming vs apply_model:")

        for i, stem in enumerate(stems):
            corr = np.corrcoef(
                streaming_result[i].flatten(),
                batch_result[i].flatten()
            )[0, 1]
            print(f"  {stem}: corr={corr:.6f}")
            # Good correlation expected with same chunking approach
            assert corr > 0.95, f"{stem} streaming/apply_model corr too low: {corr}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
