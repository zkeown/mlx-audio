"""Model registry for mlx-audio.

Provides a pluggable architecture for model discovery and dispatch.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TaskType(Enum):
    """Types of audio tasks."""

    SEPARATION = "separation"
    TRANSCRIPTION = "transcription"
    GENERATION = "generation"
    EMBEDDING = "embedding"
    VAD = "vad"
    ENHANCEMENT = "enhancement"
    DIARIZATION = "diarization"
    SPEECH = "speech"
    CLASSIFICATION = "classification"
    TAGGING = "tagging"


@dataclass
class ModelSpec:
    """Specification for a registered model.

    Attributes:
        name: Human-readable name
        task: Task this model performs
        model_class: Fully qualified class name (lazy import)
        default_repo: Default HuggingFace repo
        supported_repos: Known compatible repos
        default_params: Default configuration
        capabilities: List of capabilities (e.g., "streaming", "batched")
    """

    name: str
    task: TaskType
    model_class: str
    default_repo: str
    supported_repos: list[str] = field(default_factory=list)
    default_params: dict[str, Any] = field(default_factory=dict)
    capabilities: list[str] = field(default_factory=list)

    def get_model_class(self) -> type:
        """Lazy import of model class."""
        module_path, class_name = self.model_class.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)


class ModelRegistry:
    """Central registry for model discovery and dispatch.

    Design principles:
    - Lazy loading: Models are only imported when first used
    - Extensible: Third parties can register custom models
    - Auto-dispatch: Routes HuggingFace repos to correct implementation
    """

    _instance: ModelRegistry | None = None

    def __init__(self):
        self._specs: dict[str, ModelSpec] = {}
        self._repo_to_spec: dict[str, str] = {}
        self._task_defaults: dict[TaskType, str] = {}
        self._repo_patterns: dict[str, str] = {}

    @classmethod
    def get(cls) -> ModelRegistry:
        """Get singleton registry instance."""
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._register_builtins()
        return cls._instance

    def register(
        self,
        spec: ModelSpec,
        *,
        is_task_default: bool = False,
    ) -> None:
        """Register a model specification.

        Args:
            spec: Model specification
            is_task_default: Set as default for this task type
        """
        self._specs[spec.name] = spec

        # Map repos to spec for auto-dispatch
        for repo in spec.supported_repos + [spec.default_repo]:
            self._repo_to_spec[repo] = spec.name

        if is_task_default:
            self._task_defaults[spec.task] = spec.name

    def get_spec(self, name: str) -> ModelSpec | None:
        """Get model spec by name."""
        return self._specs.get(name)

    def get_spec_for_repo(self, repo: str) -> ModelSpec | None:
        """Find model spec that handles a given HuggingFace repo."""
        # Exact match
        if repo in self._repo_to_spec:
            return self._specs[self._repo_to_spec[repo]]

        # Pattern matching for model families
        for pattern, spec_name in self._repo_patterns.items():
            if pattern in repo.lower():
                return self._specs[spec_name]

        return None

    def get_default_for_task(self, task: TaskType) -> ModelSpec | None:
        """Get default model spec for a task type."""
        spec_name = self._task_defaults.get(task)
        if spec_name:
            return self._specs[spec_name]
        return None

    def resolve(
        self,
        task: TaskType,
        model: str | None = None,
    ) -> tuple[ModelSpec, str]:
        """Resolve task and optional model identifier to spec and repo.

        Args:
            task: Task type
            model: Model name or HuggingFace repo

        Returns:
            Tuple of (ModelSpec, repo_id)

        Raises:
            ValueError: If model cannot be resolved
        """
        if model is None:
            spec = self.get_default_for_task(task)
            if spec is None:
                raise ValueError(f"No default model for task: {task}")
            return spec, spec.default_repo

        # Check if model is a known name
        if model in self._specs:
            spec = self._specs[model]
            return spec, spec.default_repo

        # Check if model is a known repo
        spec = self.get_spec_for_repo(model)
        if spec:
            return spec, model

        # Assume it's a custom repo using task default implementation
        spec = self.get_default_for_task(task)
        if spec is None:
            raise ValueError(f"Cannot resolve model: {model}")
        return spec, model

    def list_models(self, task: TaskType | None = None) -> list[ModelSpec]:
        """List all registered models, optionally filtered by task."""
        specs = list(self._specs.values())
        if task is not None:
            specs = [s for s in specs if s.task == task]
        return specs

    def _register_builtins(self) -> None:
        """Register built-in model implementations."""
        # Separation models
        self.register(
            ModelSpec(
                name="htdemucs",
                task=TaskType.SEPARATION,
                model_class="mlx_audio.models.demucs.HTDemucs",
                default_repo="mlx-community/htdemucs",
                supported_repos=[
                    "mlx-community/htdemucs",
                    "mlx-community/htdemucs-ft",
                    "htdemucs_ft",
                    "htdemucs",
                ],
                default_params={"segment": 6.0, "overlap": 0.25},
                capabilities=["streaming", "batched"],
            ),
            is_task_default=True,
        )

        # Add pattern for demucs models
        self._repo_patterns["demucs"] = "htdemucs"

        # CLAP embedding models
        self.register(
            ModelSpec(
                name="clap-htsat-fused",
                task=TaskType.EMBEDDING,
                model_class="mlx_audio.models.clap.CLAP",
                default_repo="laion/clap-htsat-fused",
                supported_repos=[
                    "laion/clap-htsat-fused",
                    "mlx-community/clap-htsat-fused",
                ],
                default_params={},
                capabilities=["variable_length", "batched", "text_encoding"],
            ),
            is_task_default=True,
        )

        self.register(
            ModelSpec(
                name="clap-htsat-unfused",
                task=TaskType.EMBEDDING,
                model_class="mlx_audio.models.clap.CLAP",
                default_repo="laion/clap-htsat-unfused",
                supported_repos=[
                    "laion/clap-htsat-unfused",
                    "mlx-community/clap-htsat-unfused",
                ],
                capabilities=["batched", "text_encoding"],
            ),
        )

        # Add pattern for CLAP models
        self._repo_patterns["clap"] = "clap-htsat-fused"

        # Whisper transcription models
        self.register(
            ModelSpec(
                name="whisper-large-v3-turbo",
                task=TaskType.TRANSCRIPTION,
                model_class="mlx_audio.models.whisper.Whisper",
                default_repo="mlx-community/whisper-large-v3-turbo",
                supported_repos=[
                    "mlx-community/whisper-large-v3-turbo",
                    "openai/whisper-large-v3-turbo",
                ],
                default_params={"language": None, "task": "transcribe"},
                capabilities=["streaming", "batched", "timestamps", "multilingual"],
            ),
            is_task_default=True,
        )

        self.register(
            ModelSpec(
                name="whisper-large-v3",
                task=TaskType.TRANSCRIPTION,
                model_class="mlx_audio.models.whisper.Whisper",
                default_repo="mlx-community/whisper-large-v3",
                supported_repos=[
                    "mlx-community/whisper-large-v3",
                    "openai/whisper-large-v3",
                ],
                capabilities=["streaming", "batched", "timestamps", "multilingual"],
            ),
        )

        self.register(
            ModelSpec(
                name="whisper-large-v2",
                task=TaskType.TRANSCRIPTION,
                model_class="mlx_audio.models.whisper.Whisper",
                default_repo="mlx-community/whisper-large-v2",
                supported_repos=[
                    "mlx-community/whisper-large-v2",
                    "openai/whisper-large-v2",
                ],
                capabilities=["streaming", "batched", "timestamps", "multilingual"],
            ),
        )

        self.register(
            ModelSpec(
                name="whisper-medium",
                task=TaskType.TRANSCRIPTION,
                model_class="mlx_audio.models.whisper.Whisper",
                default_repo="mlx-community/whisper-medium",
                supported_repos=[
                    "mlx-community/whisper-medium",
                    "openai/whisper-medium",
                ],
                capabilities=["streaming", "batched", "timestamps", "multilingual"],
            ),
        )

        self.register(
            ModelSpec(
                name="whisper-small",
                task=TaskType.TRANSCRIPTION,
                model_class="mlx_audio.models.whisper.Whisper",
                default_repo="mlx-community/whisper-small",
                supported_repos=[
                    "mlx-community/whisper-small",
                    "openai/whisper-small",
                ],
                capabilities=["streaming", "batched", "timestamps", "multilingual"],
            ),
        )

        self.register(
            ModelSpec(
                name="whisper-base",
                task=TaskType.TRANSCRIPTION,
                model_class="mlx_audio.models.whisper.Whisper",
                default_repo="mlx-community/whisper-base",
                supported_repos=[
                    "mlx-community/whisper-base",
                    "openai/whisper-base",
                ],
                capabilities=["streaming", "batched", "timestamps", "multilingual"],
            ),
        )

        self.register(
            ModelSpec(
                name="whisper-tiny",
                task=TaskType.TRANSCRIPTION,
                model_class="mlx_audio.models.whisper.Whisper",
                default_repo="mlx-community/whisper-tiny",
                supported_repos=[
                    "mlx-community/whisper-tiny",
                    "openai/whisper-tiny",
                ],
                capabilities=["streaming", "batched", "timestamps", "multilingual"],
            ),
        )

        # Add pattern for Whisper models
        self._repo_patterns["whisper"] = "whisper-large-v3-turbo"

        # MusicGen generation models
        self.register(
            ModelSpec(
                name="musicgen-small",
                task=TaskType.GENERATION,
                model_class="mlx_audio.models.musicgen.MusicGen",
                default_repo="mlx-community/musicgen-small",
                supported_repos=[
                    "mlx-community/musicgen-small",
                    "facebook/musicgen-small",
                ],
                default_params={"duration": 10.0, "cfg_scale": 3.0},
                capabilities=["variable_length", "cfg", "streaming"],
            ),
        )

        self.register(
            ModelSpec(
                name="musicgen-medium",
                task=TaskType.GENERATION,
                model_class="mlx_audio.models.musicgen.MusicGen",
                default_repo="mlx-community/musicgen-medium",
                supported_repos=[
                    "mlx-community/musicgen-medium",
                    "facebook/musicgen-medium",
                ],
                default_params={"duration": 10.0, "cfg_scale": 3.0},
                capabilities=["variable_length", "cfg", "streaming"],
            ),
            is_task_default=True,
        )

        self.register(
            ModelSpec(
                name="musicgen-large",
                task=TaskType.GENERATION,
                model_class="mlx_audio.models.musicgen.MusicGen",
                default_repo="mlx-community/musicgen-large",
                supported_repos=[
                    "mlx-community/musicgen-large",
                    "facebook/musicgen-large",
                ],
                default_params={"duration": 10.0, "cfg_scale": 3.0},
                capabilities=["variable_length", "cfg", "streaming"],
            ),
        )

        self.register(
            ModelSpec(
                name="musicgen-melody",
                task=TaskType.GENERATION,
                model_class="mlx_audio.models.musicgen.MusicGen",
                default_repo="mlx-community/musicgen-melody",
                supported_repos=[
                    "mlx-community/musicgen-melody",
                    "facebook/musicgen-melody",
                ],
                default_params={"duration": 10.0, "cfg_scale": 3.0},
                capabilities=["variable_length", "cfg", "melody_conditioning"],
            ),
        )

        # Add pattern for MusicGen models
        self._repo_patterns["musicgen"] = "musicgen-medium"

        # EnCodec audio codec models
        self.register(
            ModelSpec(
                name="encodec-32khz",
                task=TaskType.EMBEDDING,  # Using embedding as closest task type
                model_class="mlx_audio.models.encodec.EnCodec",
                default_repo="mlx-community/encodec-32khz",
                supported_repos=[
                    "mlx-community/encodec-32khz",
                    "facebook/encodec_32khz",
                ],
                default_params={},
                capabilities=["encode", "decode", "streaming"],
            ),
        )

        self.register(
            ModelSpec(
                name="encodec-24khz",
                task=TaskType.EMBEDDING,
                model_class="mlx_audio.models.encodec.EnCodec",
                default_repo="mlx-community/encodec-24khz",
                supported_repos=[
                    "mlx-community/encodec-24khz",
                    "facebook/encodec_24khz",
                ],
                default_params={},
                capabilities=["encode", "decode", "streaming"],
            ),
        )

        # Add pattern for EnCodec models
        self._repo_patterns["encodec"] = "encodec-32khz"

        # VAD models
        self.register(
            ModelSpec(
                name="silero-vad",
                task=TaskType.VAD,
                model_class="mlx_audio.models.vad.SileroVAD",
                default_repo="mlx-community/silero-vad",
                supported_repos=[
                    "mlx-community/silero-vad",
                    "snakers4/silero-vad",
                ],
                default_params={"threshold": 0.5},
                capabilities=["streaming", "real_time"],
            ),
            is_task_default=True,
        )

        self.register(
            ModelSpec(
                name="silero-vad-8k",
                task=TaskType.VAD,
                model_class="mlx_audio.models.vad.SileroVAD",
                default_repo="mlx-community/silero-vad-8k",
                supported_repos=[
                    "mlx-community/silero-vad-8k",
                ],
                default_params={"threshold": 0.5, "sample_rate": 8000},
                capabilities=["streaming", "real_time"],
            ),
        )

        # Add pattern for VAD models
        self._repo_patterns["silero"] = "silero-vad"
        self._repo_patterns["vad"] = "silero-vad"

        # Enhancement models
        self.register(
            ModelSpec(
                name="deepfilternet2",
                task=TaskType.ENHANCEMENT,
                model_class="mlx_audio.models.enhance.DeepFilterNet",
                default_repo="mlx-community/deepfilternet2",
                supported_repos=[
                    "mlx-community/deepfilternet2",
                    "deepfilternet/deepfilternet2",
                ],
                default_params={"sample_rate": 48000},
                capabilities=["streaming", "real_time"],
            ),
            is_task_default=True,
        )

        self.register(
            ModelSpec(
                name="deepfilternet2-16k",
                task=TaskType.ENHANCEMENT,
                model_class="mlx_audio.models.enhance.DeepFilterNet",
                default_repo="mlx-community/deepfilternet2-16k",
                supported_repos=[
                    "mlx-community/deepfilternet2-16k",
                ],
                default_params={"sample_rate": 16000},
                capabilities=["streaming", "real_time"],
            ),
        )

        # Add pattern for enhancement models
        self._repo_patterns["deepfilter"] = "deepfilternet2"
        self._repo_patterns["enhance"] = "deepfilternet2"

        # Diarization models
        self.register(
            ModelSpec(
                name="ecapa-tdnn",
                task=TaskType.DIARIZATION,
                model_class="mlx_audio.models.diarization.SpeakerDiarization",
                default_repo="mlx-community/ecapa-tdnn-voxceleb",
                supported_repos=[
                    "mlx-community/ecapa-tdnn-voxceleb",
                    "speechbrain/spkrec-ecapa-voxceleb",
                ],
                default_params={},
                capabilities=["batched", "speaker_embeddings"],
            ),
            is_task_default=True,
        )

        # Add pattern for diarization models
        self._repo_patterns["ecapa"] = "ecapa-tdnn"
        self._repo_patterns["diarize"] = "ecapa-tdnn"

        # Parler-TTS speech models
        self.register(
            ModelSpec(
                name="parler-tts-mini",
                task=TaskType.SPEECH,
                model_class="mlx_audio.models.tts.ParlerTTS",
                default_repo="mlx-community/parler-tts-mini",
                supported_repos=[
                    "mlx-community/parler-tts-mini",
                    "parler-tts/parler-tts-mini-v1",
                ],
                default_params={"temperature": 1.0},
                capabilities=["voice_description", "variable_length"],
            ),
            is_task_default=True,
        )

        self.register(
            ModelSpec(
                name="parler-tts-large",
                task=TaskType.SPEECH,
                model_class="mlx_audio.models.tts.ParlerTTS",
                default_repo="mlx-community/parler-tts-large",
                supported_repos=[
                    "mlx-community/parler-tts-large",
                    "parler-tts/parler-tts-large-v1",
                ],
                capabilities=["voice_description", "variable_length"],
            ),
        )

        # Add pattern for TTS models
        self._repo_patterns["parler"] = "parler-tts-mini"
        self._repo_patterns["tts"] = "parler-tts-mini"


def register_model(
    name: str,
    task: TaskType,
    default_repo: str,
    **kwargs,
) -> Callable[[type], type]:
    """Decorator to register a model class with the registry.

    Example:
        @register_model("my_separator", TaskType.SEPARATION, "user/my-model")
        class MySeparator(BaseModel):
            ...
    """

    def decorator(cls: type) -> type:
        spec = ModelSpec(
            name=name,
            task=task,
            model_class=f"{cls.__module__}.{cls.__name__}",
            default_repo=default_repo,
            supported_repos=kwargs.get("supported_repos", []),
            default_params=kwargs.get("default_params", {}),
            capabilities=kwargs.get("capabilities", []),
        )
        ModelRegistry.get().register(
            spec,
            is_task_default=kwargs.get("is_default", False),
        )
        return cls

    return decorator
