"""Tests for diarization configuration."""

import pytest

from mlx_audio.models.diarization import EcapaTDNNConfig, DiarizationConfig


class TestEcapaTDNNConfig:
    """Tests for ECAPA-TDNN configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = EcapaTDNNConfig()

        assert config.input_dim == 80  # Mel bands
        assert config.lin_neurons == 192  # Embedding dimension
        assert config.attention_channels == 128
        assert config.res2net_scale == 8
        assert config.se_channels == 128
        assert config.global_context is True

    def test_channels_structure(self):
        """Test channel dimensions."""
        config = EcapaTDNNConfig()

        # Should have 5 channel dimensions
        assert len(config.channels) == 5
        # First layer takes input, last outputs for pooling
        assert config.channels[0] == 512
        assert config.channels[4] == 1536

    def test_kernel_sizes(self):
        """Test kernel sizes."""
        config = EcapaTDNNConfig()

        assert len(config.kernel_sizes) == 4
        assert config.kernel_sizes[0] == 5  # Initial TDNN

    def test_dilations(self):
        """Test dilation values."""
        config = EcapaTDNNConfig()

        assert len(config.dilations) == 4
        # First layer has no dilation
        assert config.dilations[0] == 1

    def test_custom_embedding_dim(self):
        """Test custom embedding dimension."""
        config = EcapaTDNNConfig(lin_neurons=256)

        assert config.lin_neurons == 256


class TestDiarizationConfig:
    """Tests for diarization pipeline configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = DiarizationConfig()

        assert config.sample_rate == 16000
        assert config.min_speakers == 1
        assert config.max_speakers == 10
        assert config.segment_duration == 1.5
        assert config.segment_step == 0.75

    def test_mel_parameters(self):
        """Test mel spectrogram parameters."""
        config = DiarizationConfig()

        assert config.n_fft == 400
        assert config.hop_length == 160
        assert config.n_mels == 80
        assert config.fmin == 20
        assert config.fmax == 7600

    def test_embedding_config(self):
        """Test embedded EcapaTDNNConfig."""
        config = DiarizationConfig()

        assert isinstance(config.embedding, EcapaTDNNConfig)
        assert config.embedding.input_dim == 80

    def test_clustering_parameters(self):
        """Test clustering configuration."""
        config = DiarizationConfig()

        assert config.cluster_threshold > 0
        assert config.cluster_threshold < 1

    def test_custom_speaker_range(self):
        """Test custom speaker range."""
        config = DiarizationConfig(min_speakers=2, max_speakers=5)

        assert config.min_speakers == 2
        assert config.max_speakers == 5
