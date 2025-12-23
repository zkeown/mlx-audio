"""Tests for ClassificationResult and TaggingResult."""

import pytest

try:
    import mlx.core as mx
except ImportError:
    mx = None

pytestmark = pytest.mark.skipif(mx is None, reason="MLX not available")


class TestClassificationResult:
    """Tests for ClassificationResult."""

    def test_basic_creation(self, fixed_seed):
        """Test basic ClassificationResult creation."""
        from mlx_audio.types.results import ClassificationResult

        probs = mx.softmax(mx.random.normal((10,)))
        result = ClassificationResult(
            predicted_class=3,
            probabilities=probs,
            class_names=None,
        )

        assert result.predicted_class == 3
        assert result.probabilities.shape == (10,)

    def test_with_class_names(self, fixed_seed, sample_labels):
        """Test ClassificationResult with class names."""
        from mlx_audio.types.results import ClassificationResult

        probs = mx.softmax(mx.random.normal((5,)))
        result = ClassificationResult(
            predicted_class="dog",
            probabilities=probs,
            class_names=sample_labels,
            top_k_classes=["dog", "cat"],
            top_k_probs=[0.7, 0.2],
        )

        assert result.predicted_class == "dog"
        assert result.class_names == sample_labels
        assert result.top_k_classes == ["dog", "cat"]

    def test_confidence_property(self, fixed_seed):
        """Test confidence property returns max probability."""
        from mlx_audio.types.results import ClassificationResult

        # Create known probabilities
        probs = mx.array([0.1, 0.2, 0.5, 0.15, 0.05])
        result = ClassificationResult(
            predicted_class=2,
            probabilities=probs,
        )

        assert abs(result.confidence - 0.5) < 1e-6

    def test_get_probability(self, fixed_seed, sample_labels):
        """Test getting probability for a specific class."""
        from mlx_audio.types.results import ClassificationResult

        probs = mx.array([0.1, 0.2, 0.5, 0.15, 0.05])
        result = ClassificationResult(
            predicted_class=2,
            probabilities=probs,
            class_names=sample_labels,
        )

        # By index
        assert abs(result.get_probability(0) - 0.1) < 1e-6

        # By name
        assert abs(result.get_probability("bird") - 0.5) < 1e-6

    def test_to_dict(self, fixed_seed, sample_labels):
        """Test conversion to dictionary."""
        from mlx_audio.types.results import ClassificationResult

        probs = mx.array([0.1, 0.2, 0.5, 0.15, 0.05])
        result = ClassificationResult(
            predicted_class="bird",
            probabilities=probs,
            class_names=sample_labels,
            top_k_classes=["bird", "cat"],
            top_k_probs=[0.5, 0.2],
        )

        d = result.to_dict()
        assert d["predicted_class"] == "bird"
        assert d["confidence"] == 0.5
        assert d["top_k_classes"] == ["bird", "cat"]

    def test_repr(self, fixed_seed):
        """Test string representation."""
        from mlx_audio.types.results import ClassificationResult

        probs = mx.array([0.3, 0.7])
        result = ClassificationResult(
            predicted_class="cat",
            probabilities=probs,
        )

        s = repr(result)
        assert "ClassificationResult" in s
        assert "cat" in s
        assert "70" in s  # 70% confidence


class TestTaggingResult:
    """Tests for TaggingResult."""

    def test_basic_creation(self, fixed_seed):
        """Test basic TaggingResult creation."""
        from mlx_audio.types.results import TaggingResult

        probs = mx.sigmoid(mx.random.normal((100,)))
        result = TaggingResult(
            tags=[0, 5, 23],
            probabilities=probs,
        )

        assert result.tags == [0, 5, 23]
        assert result.probabilities.shape == (100,)

    def test_with_tag_names(self, fixed_seed, sample_labels):
        """Test TaggingResult with tag names."""
        from mlx_audio.types.results import TaggingResult

        probs = mx.sigmoid(mx.random.normal((5,)))
        result = TaggingResult(
            tags=["dog", "bird"],
            probabilities=probs,
            tag_names=sample_labels,
            threshold=0.5,
        )

        assert result.tags == ["dog", "bird"]
        assert result.threshold == 0.5

    def test_get_probability(self, fixed_seed, sample_labels):
        """Test getting probability for a specific tag."""
        from mlx_audio.types.results import TaggingResult

        probs = mx.array([0.9, 0.3, 0.7, 0.1, 0.8])
        result = TaggingResult(
            tags=["dog", "bird", "train"],
            probabilities=probs,
            tag_names=sample_labels,
        )

        # By index
        assert abs(result.get_probability(0) - 0.9) < 1e-6

        # By name
        assert abs(result.get_probability("bird") - 0.7) < 1e-6

    def test_top_k(self, fixed_seed, sample_labels):
        """Test getting top-k tags."""
        from mlx_audio.types.results import TaggingResult

        probs = mx.array([0.9, 0.3, 0.7, 0.1, 0.8])
        result = TaggingResult(
            tags=["dog", "bird", "train"],
            probabilities=probs,
            tag_names=sample_labels,
        )

        top3 = result.top_k(3)
        assert len(top3) == 3

        # Should be sorted by probability descending
        names = [t[0] for t in top3]
        probs_list = [t[1] for t in top3]
        assert names == ["dog", "train", "bird"]
        assert abs(probs_list[0] - 0.9) < 1e-6
        assert abs(probs_list[1] - 0.8) < 1e-6
        assert abs(probs_list[2] - 0.7) < 1e-6

    def test_above_threshold(self, fixed_seed, sample_labels):
        """Test getting tags above threshold."""
        from mlx_audio.types.results import TaggingResult

        probs = mx.array([0.9, 0.3, 0.7, 0.1, 0.8])
        result = TaggingResult(
            tags=["dog", "bird", "train"],
            probabilities=probs,
            tag_names=sample_labels,
            threshold=0.5,
        )

        # Default threshold
        above = result.above_threshold()
        names = [t[0] for t in above]
        assert set(names) == {"dog", "bird", "train"}

        # Custom threshold
        above = result.above_threshold(0.75)
        names = [t[0] for t in above]
        assert set(names) == {"dog", "train"}

    def test_to_dict(self, fixed_seed, sample_labels):
        """Test conversion to dictionary."""
        from mlx_audio.types.results import TaggingResult

        probs = mx.array([0.9, 0.3, 0.7, 0.1, 0.8])
        result = TaggingResult(
            tags=["dog", "bird", "train"],
            probabilities=probs,
            tag_names=sample_labels,
            threshold=0.5,
        )

        d = result.to_dict()
        assert d["tags"] == ["dog", "bird", "train"]
        assert d["threshold"] == 0.5
        assert len(d["top_5"]) == 5

    def test_repr(self, fixed_seed):
        """Test string representation."""
        from mlx_audio.types.results import TaggingResult

        probs = mx.random.uniform(shape=(10,))
        result = TaggingResult(
            tags=["a", "b", "c"],
            probabilities=probs,
        )

        s = repr(result)
        assert "TaggingResult" in s
        assert "a" in s
