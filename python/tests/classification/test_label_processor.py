"""Tests for LabelProcessor and Ontology."""

import numpy as np
import pytest


class TestLabelProcessor:
    """Tests for LabelProcessor."""

    def test_single_label_encode(self, sample_labels):
        """Test single-label encoding returns integer index."""
        from mlx_audio.data.datasets.classification import LabelProcessor

        processor = LabelProcessor(class_names=sample_labels, is_multilabel=False)

        # Encode single label
        idx = processor.encode(["dog"])
        assert idx == 0

        idx = processor.encode(["train"])
        assert idx == 4

    def test_single_label_decode(self, sample_labels):
        """Test single-label decoding returns label name."""
        from mlx_audio.data.datasets.classification import LabelProcessor

        processor = LabelProcessor(class_names=sample_labels, is_multilabel=False)

        labels = processor.decode(0)
        assert labels == ["dog"]

        labels = processor.decode(3)
        assert labels == ["car"]

    def test_multilabel_encode(self, sample_labels, multilabel_sample):
        """Test multi-label encoding returns binary vector."""
        from mlx_audio.data.datasets.classification import LabelProcessor

        processor = LabelProcessor(class_names=sample_labels, is_multilabel=True)

        # Encode multiple labels
        encoding = processor.encode(multilabel_sample)  # ["dog", "bird"]

        assert isinstance(encoding, np.ndarray)
        assert encoding.shape == (5,)
        assert encoding.dtype == np.float32

        # Check correct positions are 1
        assert encoding[0] == 1.0  # dog
        assert encoding[2] == 1.0  # bird
        assert encoding[1] == 0.0  # cat
        assert encoding[3] == 0.0  # car
        assert encoding[4] == 0.0  # train

    def test_multilabel_decode(self, sample_labels):
        """Test multi-label decoding returns active labels."""
        from mlx_audio.data.datasets.classification import LabelProcessor

        processor = LabelProcessor(class_names=sample_labels, is_multilabel=True)

        # Create binary vector
        encoding = np.array([1.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        labels = processor.decode(encoding)

        assert set(labels) == {"dog", "bird"}

    def test_multilabel_decode_with_threshold(self, sample_labels):
        """Test multi-label decoding with different thresholds."""
        from mlx_audio.data.datasets.classification import LabelProcessor

        processor = LabelProcessor(class_names=sample_labels, is_multilabel=True)

        # Probabilities instead of binary
        probs = np.array([0.9, 0.3, 0.6, 0.1, 0.8], dtype=np.float32)

        # Default threshold 0.5
        labels = processor.decode(probs, threshold=0.5)
        assert set(labels) == {"dog", "bird", "train"}

        # Higher threshold
        labels = processor.decode(probs, threshold=0.7)
        assert set(labels) == {"dog", "train"}

        # Lower threshold
        labels = processor.decode(probs, threshold=0.2)
        assert set(labels) == {"dog", "cat", "bird", "train"}

    def test_unknown_label_single(self, sample_labels):
        """Test encoding unknown label returns -1 for single-label."""
        from mlx_audio.data.datasets.classification import LabelProcessor

        processor = LabelProcessor(class_names=sample_labels, is_multilabel=False)

        # Unknown labels are filtered, returns -1 if no valid labels
        idx = processor.encode(["unknown_class"])
        assert idx == -1

    def test_unknown_label_multilabel(self, sample_labels):
        """Test encoding unknown label returns zeros for multi-label."""
        from mlx_audio.data.datasets.classification import LabelProcessor

        processor = LabelProcessor(class_names=sample_labels, is_multilabel=True)

        # Unknown labels are filtered, returns zeros if no valid labels
        encoding = processor.encode(["unknown_class"])
        assert isinstance(encoding, np.ndarray)
        assert encoding.sum() == 0.0

    def test_num_classes(self, sample_labels):
        """Test num_classes property."""
        from mlx_audio.data.datasets.classification import LabelProcessor

        processor = LabelProcessor(class_names=sample_labels, is_multilabel=False)
        assert processor.num_classes == 5

    def test_get_index(self, sample_labels):
        """Test get_index method."""
        from mlx_audio.data.datasets.classification import LabelProcessor

        processor = LabelProcessor(class_names=sample_labels, is_multilabel=False)

        assert processor.get_index("dog") == 0
        assert processor.get_index("train") == 4

    def test_get_name(self, sample_labels):
        """Test get_name method."""
        from mlx_audio.data.datasets.classification import LabelProcessor

        processor = LabelProcessor(class_names=sample_labels, is_multilabel=False)

        assert processor.get_name(0) == "dog"
        assert processor.get_name(4) == "train"


class TestOntology:
    """Tests for hierarchical Ontology.

    Note: The Ontology class uses class_hierarchy which maps
    child -> list of parents (for direct parent lookup).
    """

    def test_expand_labels(self):
        """Test expanding labels with parents."""
        from mlx_audio.data.datasets.classification import Ontology

        # class_hierarchy: child -> [list of parent classes]
        ontology = Ontology(
            class_hierarchy={
                "dog": ["animal"],
                "cat": ["animal"],
                "animal": ["living_thing"],
            }
        )

        # expand_labels adds all parents
        expanded = ontology.expand_labels(["dog"])
        assert set(expanded) == {"dog", "animal"}

        # Note: the implementation only adds direct parents, not ancestors
        expanded = ontology.expand_labels(["dog", "cat"])
        assert "dog" in expanded
        assert "cat" in expanded
        assert "animal" in expanded

    def test_get_children(self):
        """Test getting children of a class."""
        from mlx_audio.data.datasets.classification import Ontology

        # class_hierarchy: child -> [list of parent classes]
        ontology = Ontology(
            class_hierarchy={
                "dog": ["animal"],
                "cat": ["animal"],
                "bird": ["animal"],
            }
        )

        children = ontology.get_children("animal")
        assert set(children) == {"dog", "cat", "bird"}

        # Leaf has no children
        children = ontology.get_children("dog")
        assert children == []

    def test_get_all_descendants(self):
        """Test getting all descendants (recursive)."""
        from mlx_audio.data.datasets.classification import Ontology

        ontology = Ontology(
            class_hierarchy={
                "dog": ["mammal"],
                "cat": ["mammal"],
                "mammal": ["animal"],
                "bird": ["animal"],
            }
        )

        descendants = ontology.get_all_descendants("animal")
        assert set(descendants) == {"mammal", "dog", "cat", "bird"}

        descendants = ontology.get_all_descendants("mammal")
        assert set(descendants) == {"dog", "cat"}

    def test_ontology_with_label_processor(self, sample_labels):
        """Test LabelProcessor with ontology."""
        from mlx_audio.data.datasets.classification import LabelProcessor, Ontology

        # class_hierarchy: child -> [list of parent classes]
        ontology = Ontology(
            class_hierarchy={
                "dog": ["animal"],
                "cat": ["animal"],
                "bird": ["animal"],
            }
        )

        # Add "animal" to class names
        extended_labels = sample_labels + ["animal"]

        processor = LabelProcessor(
            class_names=extended_labels, is_multilabel=True, ontology=ontology
        )

        # Encoding "dog" should also set "animal" (expanded via ontology)
        encoding = processor.encode(["dog"])

        assert encoding[0] == 1.0  # dog
        assert encoding[5] == 1.0  # animal
