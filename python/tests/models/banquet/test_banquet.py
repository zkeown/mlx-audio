"""Tests for Banquet query-based source separation model."""

from __future__ import annotations

import pytest
import mlx.core as mx


class TestBanquetConfig:
    """Test Banquet configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        from mlx_audio.models.banquet import BanquetConfig

        config = BanquetConfig()

        assert config.sample_rate == 44100
        assert config.n_fft == 2048
        assert config.hop_length == 512
        assert config.in_channel == 2
        assert config.n_bands == 64
        assert config.emb_dim == 128
        assert config.rnn_dim == 256
        assert config.n_sqm_modules == 12
        assert config.mlp_dim == 512
        assert config.bidirectional is True
        assert config.rnn_type == "LSTM"
        assert config.complex_mask is True
        assert config.cond_emb_dim == 768

    def test_freq_bins_computed(self):
        """Test computed frequency bins property."""
        from mlx_audio.models.banquet import BanquetConfig

        config = BanquetConfig(n_fft=2048)
        assert config.freq_bins == 1025

        config = BanquetConfig(n_fft=4096)
        assert config.freq_bins == 2049


class TestPaSSTConfig:
    """Test PaSST configuration."""

    def test_default_config(self):
        """Test default PaSST configuration values."""
        from mlx_audio.models.banquet import PaSSTConfig

        config = PaSSTConfig()

        assert config.sample_rate == 32000
        assert config.n_mels == 128
        assert config.n_fft == 1024
        assert config.hop_length == 320
        assert config.embed_dim == 768
        assert config.num_heads == 12
        assert config.num_layers == 12
        assert config.mlp_ratio == 4.0


class TestMusicalBandsplit:
    """Test musical band splitting specification."""

    def test_band_specs_generation(self):
        """Test that band specs are generated correctly."""
        from mlx_audio.models.banquet import MusicalBandsplitSpecification

        spec = MusicalBandsplitSpecification(nfft=2048, fs=44100, n_bands=64)
        band_specs = spec.get_band_specs()

        assert len(band_specs) == 64
        # First band should start at 0
        assert band_specs[0][0] == 0
        # All bands should have positive width
        for start, end in band_specs:
            assert end > start

    def test_freq_weights_generation(self):
        """Test that frequency weights are generated."""
        from mlx_audio.models.banquet import MusicalBandsplitSpecification

        spec = MusicalBandsplitSpecification(nfft=2048, fs=44100, n_bands=64)
        freq_weights = spec.get_freq_weights()

        assert len(freq_weights) == 64
        # Each weight should be an array
        for w in freq_weights:
            assert isinstance(w, mx.array)
            assert w.ndim == 1


class TestBandSplit:
    """Test BandSplit module."""

    def test_bandsplit_output_shape(self):
        """Test BandSplit output dimensions."""
        from mlx_audio.models.banquet import BandSplitModule, BanquetConfig
        from mlx_audio.models.banquet.utils import MusicalBandsplitSpecification

        config = BanquetConfig()
        spec = MusicalBandsplitSpecification(
            nfft=config.n_fft,
            fs=config.sample_rate,
            n_bands=config.n_bands,
        )

        band_split = BandSplitModule(
            band_specs=spec.get_band_specs(),
            in_channel=config.in_channel,
            emb_dim=config.emb_dim,
        )

        # Input: [batch, channels, freq, time] complex
        batch, channels, freq, time = 2, 2, 1025, 100
        x = mx.zeros((batch, channels, freq, time)) + 1j * mx.zeros((batch, channels, freq, time))

        # Output: [batch, n_bands, time, emb_dim]
        output = band_split(x)

        assert output.shape == (batch, 64, time, config.emb_dim)


class TestSeqBandModelling:
    """Test SeqBandModelling module."""

    def test_seqband_output_shape(self):
        """Test SeqBandModelling output dimensions."""
        from mlx_audio.models.banquet import SeqBandModellingModule

        seq_band = SeqBandModellingModule(
            n_modules=2,  # Use fewer modules for faster test
            emb_dim=128,
            rnn_dim=256,
            bidirectional=True,
        )

        # Input: [batch, n_bands, time, emb_dim]
        batch, n_bands, time, emb_dim = 2, 64, 50, 128
        x = mx.zeros((batch, n_bands, time, emb_dim))

        output = seq_band(x)

        # Output should have same shape
        assert output.shape == (batch, n_bands, time, emb_dim)


class TestFiLM:
    """Test FiLM conditioning module."""

    def test_film_output_shape(self):
        """Test FiLM output dimensions."""
        from mlx_audio.models.banquet import FiLM

        film = FiLM(
            cond_embedding_dim=768,
            channels=128,
            additive=True,
            multiplicative=True,
            depth=2,
        )

        # Input: [batch, channels, n_bands, time]
        batch, channels, n_bands, time = 2, 128, 64, 50
        x = mx.zeros((batch, channels, n_bands, time))
        w = mx.zeros((batch, 768))

        output = film(x, w)

        assert output.shape == x.shape


class TestMaskEstimation:
    """Test mask estimation module."""

    def test_mask_estimation_output_shape(self):
        """Test OverlappingMaskEstimationModule output dimensions."""
        from mlx_audio.models.banquet import OverlappingMaskEstimationModule, BanquetConfig
        from mlx_audio.models.banquet.utils import MusicalBandsplitSpecification

        config = BanquetConfig()
        spec = MusicalBandsplitSpecification(
            nfft=config.n_fft,
            fs=config.sample_rate,
            n_bands=config.n_bands,
        )

        mask_estim = OverlappingMaskEstimationModule(
            in_channel=config.in_channel,
            band_specs=spec.get_band_specs(),
            freq_weights=spec.get_freq_weights(),
            n_freq=config.freq_bins,
            emb_dim=config.emb_dim,
            mlp_dim=config.mlp_dim,
            complex_mask=True,
        )

        # Input: [batch, n_bands, time, emb_dim]
        batch, n_bands, time, emb_dim = 2, 64, 50, 128
        x = mx.zeros((batch, n_bands, time, emb_dim))

        output = mask_estim(x)

        # Output: [batch, in_channel, freq, time, 2] for complex mask
        assert output.shape == (batch, config.in_channel, config.freq_bins, time, 2)


class TestPaSST:
    """Test PaSST query encoder."""

    def test_passt_output_shape(self):
        """Test PaSST output dimensions."""
        from mlx_audio.models.banquet import PaSST, PaSSTConfig

        config = PaSSTConfig()
        passt = PaSST(config)

        # Input: mel spectrogram [batch, 1, 128, 998]
        batch = 2
        x = mx.zeros((batch, 1, 128, 998))

        output = passt(x)

        # Output: [batch, 768]
        assert output.shape == (batch, 768)


class TestBanquetModel:
    """Test full Banquet model."""

    @pytest.mark.slow
    def test_banquet_model_instantiation(self):
        """Test that Banquet model can be instantiated."""
        from mlx_audio.models.banquet import Banquet, BanquetConfig, PaSSTConfig

        config = BanquetConfig(n_sqm_modules=2)  # Fewer modules for faster test
        passt_config = PaSSTConfig(num_layers=2)  # Fewer layers for faster test

        model = Banquet(config=config, passt_config=passt_config)

        assert model is not None
        assert model.config == config

    @pytest.mark.slow
    def test_banquet_forward_pass(self):
        """Test Banquet forward pass with dummy data."""
        from mlx_audio.models.banquet import Banquet, BanquetConfig, PaSSTConfig

        # Use minimal configuration for faster test
        config = BanquetConfig(n_sqm_modules=1)
        passt_config = PaSSTConfig(num_layers=1)

        model = Banquet(config=config, passt_config=passt_config)

        # Dummy inputs
        batch = 1
        mixture = mx.zeros((batch, 2, 44100))  # 1 second stereo audio
        query_embedding = mx.zeros((batch, 768))

        # Forward pass
        output = model(mixture, query_embedding)

        # Check output structure
        assert output.audio.shape[0] == batch
        assert output.audio.shape[1] == 2  # stereo
        assert output.spectrogram is not None
        assert output.mask is not None


class TestInference:
    """Test inference utilities."""

    def test_weight_window_creation(self):
        """Test triangular weight window creation."""
        from mlx_audio.models.banquet.inference import _create_weight_window

        window = _create_weight_window(100)

        assert window.shape == (100,)
        # Peak should be in the middle
        assert mx.argmax(window).item() in [49, 50]
        # Values should be in [0, 1]
        assert mx.min(window).item() >= 0
        assert mx.max(window).item() <= 1
