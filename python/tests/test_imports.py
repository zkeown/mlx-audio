"""Basic import tests for mlx-audio."""

import pytest


def test_version():
    """Test version is accessible."""
    from mlx_audio import __version__

    assert __version__ == "0.1.0"


def test_primitives_imports():
    """Test primitives module imports."""
    from mlx_audio.primitives import stft, istft, melspectrogram, mfcc, resample

    assert callable(stft)
    assert callable(istft)
    assert callable(melspectrogram)
    assert callable(mfcc)
    assert callable(resample)


def test_data_imports():
    """Test data module imports."""
    from mlx_audio.data import DataLoader, Dataset, StreamingDataset

    assert DataLoader is not None
    assert Dataset is not None
    assert StreamingDataset is not None


def test_train_imports():
    """Test train module imports."""
    from mlx_audio.train import TrainModule, Trainer, OptimizerConfig

    assert TrainModule is not None
    assert Trainer is not None
    assert OptimizerConfig is not None


def test_models_imports():
    """Test models module imports."""
    from mlx_audio.models.demucs import HTDemucs, HTDemucsConfig

    assert HTDemucs is not None
    assert HTDemucsConfig is not None


def test_types_imports():
    """Test types module imports."""
    from mlx_audio.types import SeparationResult, AudioData

    assert SeparationResult is not None
    assert AudioData is not None


def test_hub_imports():
    """Test hub module imports."""
    from mlx_audio.hub import ModelRegistry, ModelCache

    assert ModelRegistry is not None
    assert ModelCache is not None


def test_top_level_imports():
    """Test top-level convenience imports."""
    import mlx_audio

    # Check high-level API functions exist
    assert callable(mlx_audio.separate)
    assert callable(mlx_audio.transcribe)
    assert callable(mlx_audio.generate)
    assert callable(mlx_audio.embed)

    # Check primitives are re-exported
    assert callable(mlx_audio.stft)
    assert callable(mlx_audio.istft)
    assert callable(mlx_audio.melspectrogram)
    assert callable(mlx_audio.mfcc)

    # Check data is re-exported
    assert mlx_audio.DataLoader is not None
    assert mlx_audio.Dataset is not None

    # Check train is re-exported
    assert mlx_audio.TrainModule is not None
    assert mlx_audio.Trainer is not None


def test_htdemucs_config():
    """Test HTDemucs configuration."""
    from mlx_audio.models.demucs import HTDemucsConfig

    config = HTDemucsConfig()
    assert config.sources == ["drums", "bass", "other", "vocals"]
    assert config.samplerate == 44100
    assert config.num_sources == 4

    config_ft = HTDemucsConfig.htdemucs_ft()
    assert config_ft.channels == 48
    assert config_ft.t_depth == 5


def test_htdemucs_model_creation():
    """Test HTDemucs model can be created."""
    from mlx_audio.models.demucs import HTDemucs, HTDemucsConfig

    config = HTDemucsConfig(
        channels=16,  # Smaller for testing
        depth=2,
        t_depth=2,
        t_heads=4,
    )
    model = HTDemucs(config)

    assert model is not None
    assert model.config.channels == 16


def test_unimplemented_apis_raise():
    """Test that unimplemented APIs raise NotImplementedError."""
    import mlx_audio

    with pytest.raises(NotImplementedError):
        mlx_audio.transcribe("test.wav")

    with pytest.raises(NotImplementedError):
        mlx_audio.generate("test prompt")

    with pytest.raises(NotImplementedError):
        mlx_audio.embed("test.wav")
