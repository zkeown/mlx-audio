"""Tests for HTDemucs MLX implementation.

These tests validate that the MLX implementation matches PyTorch exactly.
Requires: demucs package installed (pip install demucs)
"""

import numpy as np
import pytest
import mlx.core as mx

# Check if demucs with pretrained module is available
try:
    from demucs.pretrained import get_model as _get_model  # noqa: F401
    HAS_DEMUCS = True
except ImportError:
    HAS_DEMUCS = False


@pytest.fixture(scope="module")
def pt_model():
    """Load PyTorch HTDemucs model."""
    if not HAS_DEMUCS:
        pytest.skip("demucs package not available")
    import torch  # noqa: F401
    from demucs.pretrained import get_model

    bag = get_model("htdemucs_ft")
    model = bag.models[0]
    model.eval()
    return model


@pytest.fixture(scope="module")
def mlx_model():
    """Load MLX HTDemucs model."""
    from mlx_audio.models.demucs import HTDemucs

    model = HTDemucs.from_pretrained(
        "/Users/zakkeown/.cache/mlx_audio/models/htdemucs_ft"
    )
    return model


@pytest.fixture
def random_audio():
    """Generate random audio for testing."""
    np.random.seed(42)
    # 5 seconds at 44.1kHz, stereo
    return np.random.randn(1, 2, 220500).astype(np.float32) * 0.1


class TestHTDemucsComponents:
    """Test individual model components."""

    def test_stft_matches(self, pt_model, mlx_model, random_audio):
        """Test STFT output matches PyTorch."""
        import torch

        pt_audio = torch.from_numpy(random_audio)
        mlx_audio = mx.array(random_audio)
        mx.eval(mlx_audio)

        with torch.no_grad():
            pt_spec = pt_model._spec(pt_audio)

        mlx_spec = mlx_model._compute_stft(mlx_audio)
        mx.eval(mlx_spec)

        # Compare shapes
        assert pt_spec.shape == mlx_spec.shape, (
            f"STFT shape mismatch: PT={pt_spec.shape}, MLX={mlx_spec.shape}"
        )

        # Compare values (allow small numerical difference)
        mae = np.mean(np.abs(pt_spec.numpy() - np.array(mlx_spec)))
        assert mae < 0.001, f"STFT MAE too high: {mae}"

    def test_cac_conversion_matches(self, pt_model, mlx_model, random_audio):
        """Test complex-to-real conversion matches PyTorch."""
        import torch

        pt_audio = torch.from_numpy(random_audio)
        mlx_audio = mx.array(random_audio)
        mx.eval(mlx_audio)

        with torch.no_grad():
            pt_spec = pt_model._spec(pt_audio)
            pt_mag = pt_model._magnitude(pt_spec)

        mlx_spec = mlx_model._compute_stft(mlx_audio)
        mx.eval(mlx_spec)

        # Replicate CAC conversion
        real_part = mx.real(mlx_spec)
        imag_part = mx.imag(mlx_spec)
        stacked = mx.stack([real_part, imag_part], axis=2)
        B, C, _, F, T = stacked.shape
        mlx_mag = stacked.reshape(B, C * 2, F, T)
        mx.eval(mlx_mag)

        mae = np.mean(np.abs(pt_mag.numpy() - np.array(mlx_mag)))
        assert mae < 0.001, f"CAC conversion MAE too high: {mae}"

    def test_freq_emb_matches(self, pt_model, mlx_model):
        """Test frequency embedding matches PyTorch."""
        import torch

        n_freqs = 512
        pt_frs = torch.arange(n_freqs)
        mlx_frs = mx.arange(n_freqs)

        with torch.no_grad():
            pt_emb = pt_model.freq_emb(pt_frs)

        mlx_emb = mlx_model.freq_emb(mlx_frs)
        mx.eval(mlx_emb)

        mae = np.mean(np.abs(pt_emb.numpy() - np.array(mlx_emb)))
        assert mae < 1e-6, f"freq_emb MAE too high: {mae}"

    def test_encoder_output_shapes(self, mlx_model, random_audio):
        """Test encoder produces correct output shapes."""
        mlx_audio = mx.array(random_audio)
        mx.eval(mlx_audio)

        # Prepare input
        spec = mlx_model._compute_stft(mlx_audio)
        real_part = mx.real(spec)
        imag_part = mx.imag(spec)
        stacked = mx.stack([real_part, imag_part], axis=2)
        B, C, _, F, T = stacked.shape
        mag = stacked.reshape(B, C * 2, F, T)

        mean = mx.mean(mag, axis=(1, 2, 3), keepdims=True)
        std = mx.std(mag, axis=(1, 2, 3), keepdims=True) + 1e-5
        x = (mag - mean) / std

        # Convert to NHWC format (encoders now use MLX-native format)
        # [B, C, F, T] -> [B, F, T, C]
        x = x.transpose(0, 2, 3, 1)
        mx.eval(x)

        # Expected channel progression: [48, 96, 192, 384]
        expected_channels = [48, 96, 192, 384]

        for i, enc in enumerate(mlx_model.encoder):
            x = enc(x)
            mx.eval(x)
            # In NHWC format, channels are at axis 3
            assert x.shape[3] == expected_channels[i], (
                f"Encoder layer {i} channel mismatch: "
                f"got {x.shape[3]}, expected {expected_channels[i]}"
            )


class TestHTDemucsFullModel:
    """Test full model forward pass."""

    def test_output_shape(self, mlx_model, random_audio):
        """Test output has correct shape."""
        mlx_audio = mx.array(random_audio)
        mx.eval(mlx_audio)

        output = mlx_model(mlx_audio)
        mx.eval(output)

        B, C, T = random_audio.shape
        S = 4  # num_sources

        assert output.shape == (B, S, C, T), (
            f"Output shape mismatch: got {output.shape}, "
            f"expected {(B, S, C, T)}"
        )

    def test_output_matches_pytorch(self, pt_model, mlx_model, random_audio):
        """Test full model output matches PyTorch."""
        import torch

        pt_audio = torch.from_numpy(random_audio)
        mlx_audio = mx.array(random_audio)
        mx.eval(mlx_audio)

        with torch.no_grad():
            pt_out = pt_model(pt_audio)

        mlx_out = mlx_model(mlx_audio)
        mx.eval(mlx_out)

        # Overall MAE
        mae = np.mean(np.abs(pt_out.numpy() - np.array(mlx_out)))
        assert mae < 0.001, f"Full model MAE too high: {mae}"

    def test_per_stem_matches_pytorch(self, pt_model, mlx_model, random_audio):
        """Test each stem output matches PyTorch."""
        import torch

        pt_audio = torch.from_numpy(random_audio)
        mlx_audio = mx.array(random_audio)
        mx.eval(mlx_audio)

        with torch.no_grad():
            pt_out = pt_model(pt_audio)

        mlx_out = mlx_model(mlx_audio)
        mx.eval(mlx_out)

        stems = ["drums", "bass", "other", "vocals"]
        for i, stem in enumerate(stems):
            pt_stem = pt_out[:, i].numpy()
            mlx_stem = np.array(mlx_out)[:, i]
            mae = np.mean(np.abs(pt_stem - mlx_stem))
            assert mae < 0.001, f"{stem} MAE too high: {mae}"

    def test_output_range_matches_pytorch(self, pt_model, mlx_model, random_audio):
        """Test output value range matches PyTorch."""
        import torch

        pt_audio = torch.from_numpy(random_audio)
        mlx_audio = mx.array(random_audio)
        mx.eval(mlx_audio)

        with torch.no_grad():
            pt_out = pt_model(pt_audio)

        mlx_out = mlx_model(mlx_audio)
        mx.eval(mlx_out)

        pt_range = (pt_out.min().item(), pt_out.max().item())
        mlx_range = (float(mlx_out.min()), float(mlx_out.max()))

        # Ranges should be very close
        assert abs(pt_range[0] - mlx_range[0]) < 0.01, (
            f"Min value mismatch: PT={pt_range[0]}, MLX={mlx_range[0]}"
        )
        assert abs(pt_range[1] - mlx_range[1]) < 0.01, (
            f"Max value mismatch: PT={pt_range[1]}, MLX={mlx_range[1]}"
        )


class TestHTDemucsVariableLengths:
    """Test model with different input lengths."""

    @pytest.mark.parametrize("duration_seconds", [1, 3, 5, 7])
    def test_different_lengths(
        self, pt_model, mlx_model, duration_seconds
    ):
        """Test model handles different input lengths."""
        import torch

        np.random.seed(42)
        samples = duration_seconds * 44100
        audio = np.random.randn(1, 2, samples).astype(np.float32) * 0.1

        pt_audio = torch.from_numpy(audio)
        mlx_audio = mx.array(audio)
        mx.eval(mlx_audio)

        with torch.no_grad():
            pt_out = pt_model(pt_audio)

        mlx_out = mlx_model(mlx_audio)
        mx.eval(mlx_out)

        # Check shapes match
        assert pt_out.shape == mlx_out.shape, (
            f"Shape mismatch for {duration_seconds}s: "
            f"PT={pt_out.shape}, MLX={mlx_out.shape}"
        )

        # Check values match
        mae = np.mean(np.abs(pt_out.numpy() - np.array(mlx_out)))
        assert mae < 0.001, f"MAE too high for {duration_seconds}s: {mae}"


class TestHTDemucsRealAudio:
    """Test with real audio from MUSDB18HQ if available."""

    @pytest.fixture
    def musdb_track(self):
        """Load a track from MUSDB18HQ if available."""
        from pathlib import Path
        import soundfile as sf

        track_path = Path(
            "/Users/zakkeown/datasets/musdb18hq/test/"
            "Al James - Schoolboy Facination"
        )

        if not track_path.exists():
            pytest.skip("MUSDB18HQ not available")

        mixture, sr = sf.read(track_path / "mixture.wav")
        # Take first 5 seconds
        samples = 5 * sr
        mixture = mixture[:samples].T[np.newaxis, :, :].astype(np.float32)

        # Load ground truth stems
        stems = {}
        for stem_name in ["drums", "bass", "other", "vocals"]:
            audio, _ = sf.read(track_path / f"{stem_name}.wav")
            stems[stem_name] = audio[:samples].T

        return mixture, stems

    def test_real_audio_matches_pytorch(
        self, pt_model, mlx_model, musdb_track
    ):
        """Test real audio separation matches PyTorch."""
        import torch

        mixture, _ = musdb_track

        pt_audio = torch.from_numpy(mixture)
        mlx_audio = mx.array(mixture)
        mx.eval(mlx_audio)

        with torch.no_grad():
            pt_out = pt_model(pt_audio)

        mlx_out = mlx_model(mlx_audio)
        mx.eval(mlx_out)

        mae = np.mean(np.abs(pt_out.numpy() - np.array(mlx_out)))
        assert mae < 0.0001, f"Real audio MAE too high: {mae}"

    def test_real_audio_correlation_matches(
        self, pt_model, mlx_model, musdb_track
    ):
        """Test correlation with ground truth matches between PT and MLX."""
        import torch

        mixture, gt_stems = musdb_track

        pt_audio = torch.from_numpy(mixture)
        mlx_audio = mx.array(mixture)
        mx.eval(mlx_audio)

        with torch.no_grad():
            pt_out = pt_model(pt_audio)

        mlx_out = mlx_model(mlx_audio)
        mx.eval(mlx_out)

        stems = ["drums", "bass", "other", "vocals"]
        for i, stem in enumerate(stems):
            gt = gt_stems[stem]
            pt_stem = pt_out[0, i].numpy()
            mlx_stem = np.array(mlx_out)[0, i]

            pt_corr = np.corrcoef(gt.flatten(), pt_stem.flatten())[0, 1]
            mlx_corr = np.corrcoef(gt.flatten(), mlx_stem.flatten())[0, 1]

            # Correlations should be nearly identical
            assert abs(pt_corr - mlx_corr) < 0.0001, (
                f"{stem} correlation mismatch: PT={pt_corr}, MLX={mlx_corr}"
            )


class TestHTDemucsConfig:
    """Test model configuration."""

    def test_config_loads(self, mlx_model):
        """Test config is loaded correctly."""
        config = mlx_model.config

        assert config.audio_channels == 2
        assert config.num_sources == 4
        assert config.samplerate == 44100
        assert config.nfft == 4096
        assert config.depth == 4

    def test_segment_matches_pytorch(self, pt_model, mlx_model):
        """Test segment configuration matches PyTorch."""
        pt_segment = float(pt_model.segment)
        mlx_segment = mlx_model.config.segment

        assert abs(pt_segment - mlx_segment) < 0.01, (
            f"Segment mismatch: PT={pt_segment}, MLX={mlx_segment}"
        )


class TestHTDemucsLayers:
    """Test individual layer implementations.

    NOTE: Layers now use MLX-native NHWC/NLC format internally.
    - Frequency branch: [B, F, T, C] (NHWC)
    - Time branch: [B, T, C] (NLC)
    """

    def test_encoder_layer_freq(self):
        """Test frequency encoder layer."""
        from mlx_audio.models.demucs.layers import HEncLayer

        layer = HEncLayer(4, 48, kernel_size=8, stride=4, freq=True)

        # Test input: [B, F, T, C] (NHWC format)
        x = mx.random.normal((1, 2048, 216, 4))
        mx.eval(x)

        out = layer(x)
        mx.eval(out)

        # Output should have correct shape [B, F', T', C']
        # F': 2048/4 = 512, C': 48
        assert out.shape[3] == 48, f"Wrong channels: {out.shape[3]}"
        assert out.shape[1] == 512, f"Wrong freq bins: {out.shape[1]}"

    def test_encoder_layer_time(self):
        """Test time encoder layer."""
        from mlx_audio.models.demucs.layers import HEncLayer

        layer = HEncLayer(2, 48, kernel_size=8, stride=4, freq=False)

        # Test input: [B, T, C] (NLC format)
        x = mx.random.normal((1, 220500, 2))
        mx.eval(x)

        out = layer(x)
        mx.eval(out)

        # Output should have correct shape [B, T', C']
        assert out.shape[2] == 48, f"Wrong channels: {out.shape[2]}"

    def test_decoder_layer_freq(self):
        """Test frequency decoder layer."""
        from mlx_audio.models.demucs.layers import HDecLayer

        layer = HDecLayer(384, 192, kernel_size=8, stride=4, freq=True)

        # Test input: [B, F, T, C] (NHWC format)
        x = mx.random.normal((1, 8, 216, 384))
        skip = mx.random.normal((1, 8, 216, 384))
        mx.eval(x, skip)

        out, pre = layer(x, skip, length=216)
        mx.eval(out)

        # Output should have correct shape [B, F', T', C']
        # F': 8*4 - 2*pad = 32, C': 192
        assert out.shape[3] == 192, f"Wrong channels: {out.shape[3]}"
        assert out.shape[1] == 32, f"Wrong freq bins: {out.shape[1]}"

    def test_decoder_layer_time(self):
        """Test time decoder layer."""
        from mlx_audio.models.demucs.layers import HDecLayer

        layer = HDecLayer(384, 192, kernel_size=8, stride=4, freq=False)

        # Test input: [B, T, C] (NLC format)
        x = mx.random.normal((1, 862, 384))
        skip = mx.random.normal((1, 862, 384))
        mx.eval(x, skip)

        out, pre = layer(x, skip, length=3446)
        mx.eval(out)

        # Output should have correct shape [B, T', C']
        assert out.shape[2] == 192, f"Wrong channels: {out.shape[2]}"
        assert out.shape[1] == 3446, f"Wrong time dim: {out.shape[1]}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
