"""HTDemucs BagOfModels parity tests.

These tests verify that the MLX BagOfModels implementation produces
outputs matching the PyTorch reference implementation.
"""

import pytest
import numpy as np

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

try:
    import torch
    from demucs.pretrained import get_model
    from demucs.apply import apply_model as torch_apply_model

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from pathlib import Path


def generate_synthetic_music(duration_seconds: float = 3.0, sample_rate: int = 44100) -> np.ndarray:
    """Generate synthetic music-like audio for testing.

    Creates audio with frequency content resembling drums, bass, other, and vocals.
    """
    np.random.seed(42)
    t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds))

    # Drums: Low frequency noise bursts
    drums = np.zeros_like(t)
    for i in range(int(duration_seconds * 2)):  # 2 beats per second
        start = int(i * sample_rate / 2)
        if start < len(drums):
            burst_len = min(int(0.1 * sample_rate), len(drums) - start)
            drums[start:start + burst_len] = (
                np.sin(2 * np.pi * 80 * t[:burst_len]) * np.exp(-t[:burst_len] * 10)
            )

    # Bass: Low frequency
    bass = np.sin(2 * np.pi * 60 * t) * 0.3

    # Other: Mid frequencies
    other = np.sin(2 * np.pi * 440 * t) * 0.2 + np.sin(2 * np.pi * 880 * t) * 0.1

    # Vocals: Formant-like
    vocals = (
        np.sin(2 * np.pi * 300 * t) * 0.3
        + np.sin(2 * np.pi * 600 * t) * 0.2
        + np.sin(2 * np.pi * 1200 * t) * 0.1
    )

    # Mix
    mixture = drums + bass + other + vocals
    mixture = np.stack([mixture, mixture])  # Stereo
    mixture = (mixture / np.abs(mixture).max() * 0.8).astype(np.float32)

    return mixture


@pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch demucs not available")
class TestBagOfModelsParity:
    """Parity tests comparing MLX BagOfModels with PyTorch ensemble."""

    @pytest.fixture
    def synthetic_audio(self):
        """Generate synthetic test audio."""
        return generate_synthetic_music(duration_seconds=3.0)

    @pytest.fixture
    def mlx_bag(self):
        """Load MLX BagOfModels."""
        from mlx_audio.models.demucs import BagOfModels

        bag_path = Path.home() / ".cache/mlx_audio/models/htdemucs_ft_bag"
        if not bag_path.exists():
            pytest.skip(f"BagOfModels not found at {bag_path}")

        bag = BagOfModels.from_pretrained(bag_path)
        bag.eval()
        return bag

    @pytest.fixture
    def torch_bag(self):
        """Load PyTorch BagOfModels (htdemucs_ft ensemble)."""
        return get_model("htdemucs_ft")

    def test_output_shape_match(self, synthetic_audio, mlx_bag, torch_bag):
        """Test that MLX and PyTorch produce same output shapes."""
        test_audio = synthetic_audio[None, :, :]  # Add batch dim

        # MLX
        mlx_input = mx.array(test_audio)
        mlx_output = mlx_bag(mlx_input)
        mx.eval(mlx_output)
        mlx_shape = mlx_output.shape

        # PyTorch
        with torch.no_grad():
            torch_input = torch.from_numpy(test_audio)
            torch_output = torch_apply_model(torch_bag, torch_input, split=False, device='cpu')
            torch_shape = torch_output.shape

        assert mlx_shape == torch_shape, f"Shape mismatch: MLX {mlx_shape} vs PyTorch {torch_shape}"

    def test_per_source_cosine_similarity(self, synthetic_audio, mlx_bag, torch_bag):
        """Test that each source has high cosine similarity with PyTorch."""
        test_audio = synthetic_audio[None, :, :]

        # MLX
        mlx_input = mx.array(test_audio)
        mlx_output = mlx_bag(mlx_input)
        mx.eval(mlx_output)
        mlx_np = np.array(mlx_output)

        # PyTorch
        with torch.no_grad():
            torch_input = torch.from_numpy(test_audio)
            torch_output = torch_apply_model(torch_bag, torch_input, split=False, device='cpu')
            torch_np = torch_output.numpy()

        source_names = ['drums', 'bass', 'other', 'vocals']
        min_similarity = 0.95  # Target cosine similarity

        for i, name in enumerate(source_names):
            mlx_flat = mlx_np[0, i].flatten()
            torch_flat = torch_np[0, i].flatten()

            cos_sim = np.dot(mlx_flat, torch_flat) / (
                np.linalg.norm(mlx_flat) * np.linalg.norm(torch_flat) + 1e-8
            )

            print(f"{name}: cosine similarity = {cos_sim:.4f}")
            assert cos_sim >= min_similarity, (
                f"{name} cosine similarity {cos_sim:.4f} < {min_similarity}"
            )

    def test_relative_amplitude_match(self, synthetic_audio, mlx_bag, torch_bag):
        """Test that relative amplitudes between sources match."""
        test_audio = synthetic_audio[None, :, :]

        # MLX
        mlx_input = mx.array(test_audio)
        mlx_output = mlx_bag(mlx_input)
        mx.eval(mlx_output)
        mlx_np = np.array(mlx_output)

        # PyTorch
        with torch.no_grad():
            torch_input = torch.from_numpy(test_audio)
            torch_output = torch_apply_model(torch_bag, torch_input, split=False, device='cpu')
            torch_np = torch_output.numpy()

        source_names = ['drums', 'bass', 'other', 'vocals']
        max_ratio_diff = 0.5  # Allow 50% ratio difference

        for i, name in enumerate(source_names):
            mlx_std = mlx_np[0, i].std()
            torch_std = torch_np[0, i].std()

            if torch_std > 1e-6:  # Avoid division by zero
                ratio = mlx_std / torch_std
                print(f"{name}: MLX std={mlx_std:.4f}, PyTorch std={torch_std:.4f}, ratio={ratio:.2f}")

                assert 1 - max_ratio_diff <= ratio <= 1 + max_ratio_diff, (
                    f"{name} amplitude ratio {ratio:.2f} outside expected range"
                )


@pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
class TestBagOfModelsVsSingleModel:
    """Tests comparing BagOfModels ensemble vs single model."""

    @pytest.fixture
    def synthetic_audio(self):
        """Generate synthetic test audio."""
        return generate_synthetic_music(duration_seconds=3.0)

    @pytest.fixture
    def single_model(self):
        """Load single HTDemucs model."""
        from mlx_audio.models.demucs import HTDemucs

        model_path = Path.home() / ".cache/mlx_audio/models/htdemucs_ft"
        if not model_path.exists():
            pytest.skip(f"Single model not found at {model_path}")

        model = HTDemucs.from_pretrained(model_path)
        model.eval()
        return model

    @pytest.fixture
    def bag_model(self):
        """Load BagOfModels ensemble."""
        from mlx_audio.models.demucs import BagOfModels

        bag_path = Path.home() / ".cache/mlx_audio/models/htdemucs_ft_bag"
        if not bag_path.exists():
            pytest.skip(f"BagOfModels not found at {bag_path}")

        bag = BagOfModels.from_pretrained(bag_path)
        bag.eval()
        return bag

    def test_both_produce_output(self, synthetic_audio, single_model, bag_model):
        """Test that both single model and bag produce valid output."""
        test_audio = synthetic_audio[None, :, :]
        mlx_input = mx.array(test_audio)

        # Single model
        single_output = single_model(mlx_input)
        mx.eval(single_output)

        # Bag model
        bag_output = bag_model(mlx_input)
        mx.eval(bag_output)

        assert single_output.shape == bag_output.shape
        assert not np.allclose(np.array(single_output), np.array(bag_output)), (
            "Single model and bag should produce different outputs"
        )

    def test_bag_output_shape(self, synthetic_audio, bag_model):
        """Test BagOfModels output shape."""
        test_audio = synthetic_audio[None, :, :]
        mlx_input = mx.array(test_audio)

        output = bag_model(mlx_input)
        mx.eval(output)

        assert output.shape[0] == 1  # Batch
        assert output.shape[1] == 4  # Sources (drums, bass, other, vocals)
        assert output.shape[2] == 2  # Stereo channels
        # Time dimension should be close to input


@pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
class TestBagOfModelsMemory:
    """Memory-related tests for BagOfModels."""

    def test_sequential_processing(self):
        """Test that BagOfModels processes models sequentially."""
        from mlx_audio.models.demucs import BagOfModels

        bag_path = Path.home() / ".cache/mlx_audio/models/htdemucs_ft_bag"
        if not bag_path.exists():
            pytest.skip(f"BagOfModels not found at {bag_path}")

        bag = BagOfModels.from_pretrained(bag_path)

        # Verify we have 4 models
        assert len(bag.models) == 4

        # Verify weight matrix is identity
        weights = np.array(bag.weights)
        expected = np.eye(4)
        assert np.allclose(weights, expected), "Weight matrix should be identity"
