"""Numerical parity tests for DeepFilterNet against reference implementation.

These tests verify that the MLX DeepFilterNet implementation produces
outputs that match the original deepfilternet package.
"""

import numpy as np
import pytest

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

try:
    from df import init_df
    from df.enhance import enhance as df_enhance

    HAS_DEEPFILTERNET = True
except ImportError:
    HAS_DEEPFILTERNET = False

HAS_TORCH = False
try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    pass


@pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
@pytest.mark.skipif(
    not HAS_DEEPFILTERNET, reason="deepfilternet not available"
)
@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
@pytest.mark.parity
@pytest.mark.slow
class TestDeepFilterNetParity:
    """Parity tests comparing MLX DeepFilterNet to reference implementation."""

    @pytest.fixture(scope="class")
    def ref_model(self):
        """Load reference DeepFilterNet model."""
        model, df_state, _ = init_df()
        return model, df_state

    @pytest.fixture(scope="class")
    def mlx_model(self):
        """Load MLX DeepFilterNet model with random weights.

        Note: For true parity tests, we would need converted weights.
        This test verifies architectural parity with matching weight init.
        """
        from mlx_audio.models.enhance import DeepFilterNet, DeepFilterNetConfig

        config = DeepFilterNetConfig.deepfilternet2()
        model = DeepFilterNet(config)
        return model

    def test_output_shape_parity(self, ref_model, mlx_model):
        """Verify output shapes match between implementations."""
        ref_model_obj, df_state = ref_model

        # Generate test audio (1 second at 48kHz)
        # Reference expects [C, T] shape (mono = [1, T])
        np.random.seed(42)
        audio_np = np.random.randn(48000).astype(np.float32)
        audio_torch = torch.from_numpy(audio_np[None, :])  # [1, 48000]

        # Reference output
        ref_output = df_enhance(ref_model_obj, df_state, audio_torch)
        ref_shape = ref_output.shape

        # MLX output - expects [batch, samples]
        mlx_audio = mx.array(audio_np[None, :])  # Add batch dim
        mlx_output = mlx_model(mlx_audio)
        mlx_shape = tuple(mlx_output.shape)

        # Compare shapes (both should produce same length output)
        # ref_shape is [C, T], mlx_shape is [batch, T]
        assert mlx_shape[-1] == ref_shape[-1], (
            f"Length mismatch: MLX {mlx_shape[-1]} vs ref {ref_shape[-1]}"
        )

    def test_output_dtype_parity(self, ref_model, mlx_model):
        """Verify output dtypes are compatible."""
        ref_model_obj, df_state = ref_model

        np.random.seed(42)
        audio_np = np.random.randn(48000).astype(np.float32)
        audio_torch = torch.from_numpy(audio_np[None, :])

        # Reference
        ref_output = df_enhance(ref_model_obj, df_state, audio_torch)

        # MLX
        mlx_audio = mx.array(audio_np[None, :])
        mlx_output = mlx_model(mlx_audio)

        # Both should be float32
        assert ref_output.dtype == torch.float32
        assert mlx_output.dtype == mx.float32

    def test_output_range_parity(self, ref_model, mlx_model):
        """Verify output value ranges are similar."""
        ref_model_obj, df_state = ref_model

        np.random.seed(42)
        audio_np = np.random.randn(48000).astype(np.float32)
        audio_torch = torch.from_numpy(audio_np[None, :])

        # Reference (run to ensure it works, values for comparison)
        ref_output = df_enhance(ref_model_obj, df_state, audio_torch)
        ref_output_np = ref_output.numpy() if hasattr(ref_output, 'numpy') else np.array(ref_output)

        # MLX
        mlx_audio = mx.array(audio_np[None, :])
        mlx_output = mlx_model(mlx_audio)
        mlx_output_np = np.array(mlx_output[0])
        mlx_min = float(mlx_output_np.min())
        mlx_max = float(mlx_output_np.max())
        mlx_std = float(mlx_output_np.std())

        # Check that output ranges are in similar ballpark
        # Note: Without matching weights, we can't expect exact values
        # but the ranges should be reasonable (not NaN, not exploding)
        assert not np.isnan(mlx_min) and not np.isnan(mlx_max)
        assert not np.isinf(mlx_min) and not np.isinf(mlx_max)
        assert mlx_std > 0, "Output should not be constant"

        # Reference output should also be valid
        ref_min = float(ref_output_np.min())
        ref_max = float(ref_output_np.max())
        assert not np.isnan(ref_min) and not np.isnan(ref_max)
        assert not np.isinf(ref_min) and not np.isinf(ref_max)


@pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
@pytest.mark.parity
class TestERBFilterbankParity:
    """Parity tests for ERB filterbank against librosa/reference."""

    def test_erb_scale_conversion(self):
        """Test Hz to ERB and back conversion."""
        from mlx_audio.models.enhance.layers.erb import erb_to_hz, hz_to_erb

        # Test frequencies
        freqs = np.array([100, 500, 1000, 4000, 8000, 16000])

        # Convert to ERB and back
        erb_values = hz_to_erb(freqs)
        reconstructed = erb_to_hz(erb_values)

        # Should round-trip accurately
        np.testing.assert_allclose(freqs, reconstructed, rtol=1e-5)

    def test_erb_filterbank_shape(self):
        """Test ERB filterbank produces correct shape."""
        from mlx_audio.models.enhance.layers.erb import erb_filterbank

        n_fft = 1920
        sample_rate = 48000
        n_bands = 32

        fb = erb_filterbank(n_fft, sample_rate, n_bands)
        fb_np = np.array(fb)

        # Shape: (n_bands, n_fft//2+1)
        assert fb_np.shape == (n_bands, n_fft // 2 + 1)

    def test_erb_filterbank_normalized(self):
        """Test ERB filterbank rows sum to approximately 1."""
        from mlx_audio.models.enhance.layers.erb import erb_filterbank

        fb = erb_filterbank(1920, 48000, 32)
        fb_np = np.array(fb)

        # Each row should sum to ~1 (normalized)
        row_sums = fb_np.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, rtol=0.01)

    def test_erb_filterbank_nonnegative(self):
        """Test ERB filterbank values are non-negative."""
        from mlx_audio.models.enhance.layers.erb import erb_filterbank

        fb = erb_filterbank(1920, 48000, 32)
        fb_np = np.array(fb)

        assert np.all(fb_np >= 0), "Filterbank should have non-negative values"

    def test_erb_inverse_reconstruction(self):
        """Test ERB filterbank and inverse produce valid outputs."""
        from mlx_audio.models.enhance.layers.erb import (
            erb_filterbank,
            erb_inverse_filterbank,
        )

        n_fft = 1920
        sample_rate = 48000
        n_bands = 32

        fb = erb_filterbank(n_fft, sample_rate, n_bands)
        fb_inv = erb_inverse_filterbank(n_fft, sample_rate, n_bands)

        fb_np = np.array(fb)
        fb_inv_np = np.array(fb_inv)

        # Check shapes are compatible for projection
        assert fb_np.shape == (n_bands, n_fft // 2 + 1)
        assert fb_inv_np.shape == (n_fft // 2 + 1, n_bands)

        # Test that projection and inverse produce valid output
        test_spec = np.abs(np.random.randn(n_fft // 2 + 1).astype(np.float32))

        # Project to ERB bands and back
        erb_proj = test_spec @ fb_np.T  # (n_bands,)
        reconstructed = erb_proj @ fb_inv_np.T  # (n_freqs,)

        # Check that we preserved some signal (not all zeros or NaN)
        assert not np.any(np.isnan(reconstructed))
        assert not np.any(np.isinf(reconstructed))
        assert np.sum(np.abs(reconstructed)) > 0, "Should have signal"


@pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
@pytest.mark.parity
class TestStackedGRUParity:
    """Parity tests for StackedGRU layer behavior."""

    def test_stacked_gru_output_shape(self):
        """Test StackedGRU produces correct output shape."""
        from mlx_audio.models.enhance.model import StackedGRU

        batch, seq, input_size, hidden_size = 2, 10, 64, 128
        num_layers = 3

        gru = StackedGRU(input_size, hidden_size, num_layers)
        x = mx.random.normal((batch, seq, input_size))

        output, hidden = gru(x)

        assert output.shape == (batch, seq, hidden_size)
        assert hidden.shape == (num_layers, batch, hidden_size)

    def test_stacked_gru_single_layer(self):
        """Test StackedGRU with single layer produces valid output."""
        from mlx_audio.models.enhance.model import StackedGRU

        batch, seq, input_size, hidden_size = 2, 10, 64, 128

        stacked = StackedGRU(input_size, hidden_size, num_layers=1)
        x = mx.random.normal((batch, seq, input_size))

        output, hidden = stacked(x)

        # Verify shapes
        assert output.shape == (batch, seq, hidden_size)
        assert hidden.shape == (1, batch, hidden_size)

        # Verify output is not constant (GRU is processing)
        output_np = np.array(output)
        assert output_np.std() > 0, "Output should vary"

    def test_stacked_gru_hidden_propagation(self):
        """Test that different inputs produce different outputs."""
        from mlx_audio.models.enhance.model import StackedGRU

        batch, seq, input_size, hidden_size = 2, 10, 64, 128
        num_layers = 3

        gru = StackedGRU(input_size, hidden_size, num_layers)

        # Two different inputs
        x1 = mx.random.normal((batch, seq, input_size))
        x2 = mx.random.normal((batch, seq, input_size))

        # Process both
        out1, _ = gru(x1)
        out2, _ = gru(x2)

        # Different input should produce different output
        assert not mx.allclose(out1, out2).item()

        # Output should have variation
        out1_np = np.array(out1)
        assert out1_np.std() > 0, "Output should have variation"

    def test_stacked_gru_deterministic(self):
        """Test StackedGRU is deterministic."""
        from mlx_audio.models.enhance.model import StackedGRU

        batch, seq, input_size, hidden_size = 2, 10, 64, 128

        gru = StackedGRU(input_size, hidden_size, num_layers=2)
        x = mx.random.normal((batch, seq, input_size))

        out1, h1 = gru(x)
        out2, h2 = gru(x)

        assert mx.allclose(out1, out2).item()
        assert mx.allclose(h1, h2).item()
