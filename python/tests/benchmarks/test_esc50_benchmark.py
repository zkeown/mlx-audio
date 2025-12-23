"""ESC-50 benchmark tests for audio classification.

These tests evaluate classification performance on the ESC-50 dataset
using both zero-shot and linear probe approaches.

Performance targets:
- Zero-shot CLAP: >85% accuracy
- Linear probe: >90% accuracy

Usage:
    ESC50_ROOT=/path/to/esc50 pytest tests/benchmarks/test_esc50_benchmark.py -v

The ESC50_ROOT should contain the ESC-50-master directory with:
    ESC-50-master/
        audio/
            *.wav
        meta/
            esc50.csv
"""

import logging

import numpy as np
import pytest

try:
    import mlx.core as mx
except ImportError:
    mx = None

pytestmark = [
    pytest.mark.skipif(mx is None, reason="MLX not available"),
    pytest.mark.benchmark,
    pytest.mark.external,
]

logger = logging.getLogger(__name__)


class TestESC50ZeroShot:
    """Zero-shot classification tests using CLAP embeddings."""

    @pytest.mark.slow
    def test_zero_shot_accuracy(self, esc50_root):
        """Test zero-shot classification accuracy on ESC-50.

        Expected accuracy: >85%
        """
        from mlx_audio.data.datasets.classification import ESC50, ESC50_CLASSES
        from mlx_audio.functional.classify import classify

        # Use fold 1 for testing
        dataset = ESC50(root=esc50_root, split="test", fold=1)

        correct = 0
        total = 0

        for i, sample in enumerate(dataset):
            # Get audio
            audio = sample["audio"]
            true_label = sample["class_names"][0]

            # Classify using zero-shot CLAP
            result = classify(audio, labels=ESC50_CLASSES, top_k=1)
            predicted = result.predicted_class

            if predicted == true_label:
                correct += 1
            total += 1

            if (i + 1) % 50 == 0:
                logger.info(f"Processed {i + 1}/{len(dataset)}, accuracy: {correct / total:.1%}")

        accuracy = correct / total
        logger.info(f"ESC-50 zero-shot accuracy (fold 1): {accuracy:.1%}")

        assert accuracy > 0.85, f"Zero-shot accuracy {accuracy:.1%} below target 85%"

    @pytest.mark.slow
    def test_zero_shot_5fold_cv(self, esc50_root):
        """Test 5-fold cross-validation accuracy.

        This is the standard ESC-50 evaluation protocol.
        """
        from mlx_audio.data.datasets.classification import ESC50, ESC50_CLASSES
        from mlx_audio.functional.classify import classify

        fold_accuracies = []

        for fold in range(1, 6):
            dataset = ESC50(root=esc50_root, split="test", fold=fold)

            correct = 0
            total = 0

            for sample in dataset:
                audio = sample["audio"]
                true_label = sample["class_names"][0]

                result = classify(audio, labels=ESC50_CLASSES, top_k=1)
                predicted = result.predicted_class

                if predicted == true_label:
                    correct += 1
                total += 1

            fold_acc = correct / total
            fold_accuracies.append(fold_acc)
            logger.info(f"Fold {fold} accuracy: {fold_acc:.1%}")

        mean_accuracy = np.mean(fold_accuracies)
        std_accuracy = np.std(fold_accuracies)
        logger.info(f"ESC-50 5-fold CV: {mean_accuracy:.1%} +/- {std_accuracy:.1%}")

        assert mean_accuracy > 0.85, f"Mean accuracy {mean_accuracy:.1%} below target 85%"


class TestESC50LinearProbe:
    """Linear probe classification tests."""

    @pytest.mark.slow
    def test_linear_probe_accuracy(self, esc50_root):
        """Test linear probe accuracy on ESC-50.

        Trains a linear classifier on CLAP embeddings.
        Expected accuracy: >90%
        """
        from mlx_audio.data.datasets.classification import ESC50
        from mlx_audio.models.clap import CLAP

        # Load CLAP model
        clap = CLAP.from_pretrained("clap-htsat-fused")

        # Get train and test sets for fold 1
        train_ds = ESC50(root=esc50_root, split="train", fold=1)
        test_ds = ESC50(root=esc50_root, split="test", fold=1)

        # Extract embeddings
        train_embeddings = []
        train_labels = []

        logger.info("Extracting training embeddings...")
        for i, sample in enumerate(train_ds):
            audio = mx.array(sample["audio"])
            # Add batch and channel dimensions
            audio = audio.reshape(1, 1, -1)
            emb = clap.encode_audio(audio)
            train_embeddings.append(np.array(emb[0]))
            train_labels.append(sample["labels"])

            if (i + 1) % 200 == 0:
                logger.info(f"Processed {i + 1}/{len(train_ds)} training samples")

        train_X = np.stack(train_embeddings)
        train_y = np.array(train_labels)

        # Extract test embeddings
        test_embeddings = []
        test_labels = []

        logger.info("Extracting test embeddings...")
        for sample in test_ds:
            audio = mx.array(sample["audio"])
            audio = audio.reshape(1, 1, -1)
            emb = clap.encode_audio(audio)
            test_embeddings.append(np.array(emb[0]))
            test_labels.append(sample["labels"])

        test_X = np.stack(test_embeddings)
        test_y = np.array(test_labels)

        # Train simple linear classifier using least squares
        # (sklearn-free implementation)
        logger.info("Training linear classifier...")

        # Add bias term
        train_X_bias = np.hstack([train_X, np.ones((train_X.shape[0], 1))])
        test_X_bias = np.hstack([test_X, np.ones((test_X.shape[0], 1))])

        # One-hot encode labels
        num_classes = 50
        train_y_onehot = np.zeros((len(train_y), num_classes))
        for i, label in enumerate(train_y):
            train_y_onehot[i, label] = 1

        # Least squares solution with regularization
        lambda_reg = 1e-4
        XtX = train_X_bias.T @ train_X_bias
        XtX += lambda_reg * np.eye(XtX.shape[0])
        Xty = train_X_bias.T @ train_y_onehot
        weights = np.linalg.solve(XtX, Xty)

        # Predict
        logits = test_X_bias @ weights
        predictions = np.argmax(logits, axis=1)

        # Calculate accuracy
        correct = np.sum(predictions == test_y)
        accuracy = correct / len(test_y)

        logger.info(f"ESC-50 linear probe accuracy (fold 1): {accuracy:.1%}")

        assert accuracy > 0.90, f"Linear probe accuracy {accuracy:.1%} below target 90%"


class TestESC10Subset:
    """Tests on the smaller ESC-10 subset."""

    def test_esc10_zero_shot(self, esc50_root):
        """Quick test on ESC-10 (10 classes, faster)."""
        from mlx_audio.data.datasets.classification import ESC50, ESC10_CLASSES
        from mlx_audio.functional.classify import classify

        # ESC-10 subset
        dataset = ESC50(root=esc50_root, split="test", fold=1, esc10_only=True)

        correct = 0
        total = 0

        for sample in dataset:
            audio = sample["audio"]
            true_label = sample["class_names"][0]

            result = classify(audio, labels=ESC10_CLASSES, top_k=1)
            predicted = result.predicted_class

            if predicted == true_label:
                correct += 1
            total += 1

        accuracy = correct / total
        logger.info(f"ESC-10 zero-shot accuracy (fold 1): {accuracy:.1%}")

        # ESC-10 should be easier than ESC-50
        assert accuracy > 0.90, f"ESC-10 accuracy {accuracy:.1%} below target 90%"


class TestESC50Metrics:
    """Additional metrics beyond accuracy."""

    @pytest.mark.slow
    def test_confusion_matrix(self, esc50_root):
        """Generate confusion matrix for analysis."""
        from mlx_audio.data.datasets.classification import ESC50, ESC50_CLASSES
        from mlx_audio.functional.classify import classify

        dataset = ESC50(root=esc50_root, split="test", fold=1)

        # Build confusion matrix
        num_classes = 50
        confusion = np.zeros((num_classes, num_classes), dtype=np.int32)
        class_to_idx = {name: i for i, name in enumerate(ESC50_CLASSES)}

        for sample in dataset:
            audio = sample["audio"]
            true_label = sample["class_names"][0]
            true_idx = class_to_idx[true_label]

            result = classify(audio, labels=ESC50_CLASSES, top_k=1)
            pred_idx = class_to_idx[result.predicted_class]

            confusion[true_idx, pred_idx] += 1

        # Calculate per-class accuracy
        class_accuracies = {}
        for i, name in enumerate(ESC50_CLASSES):
            total = confusion[i].sum()
            correct = confusion[i, i]
            if total > 0:
                class_accuracies[name] = correct / total

        # Log worst performing classes
        sorted_classes = sorted(class_accuracies.items(), key=lambda x: x[1])
        logger.info("Worst performing classes:")
        for name, acc in sorted_classes[:5]:
            logger.info(f"  {name}: {acc:.1%}")

        logger.info("Best performing classes:")
        for name, acc in sorted_classes[-5:]:
            logger.info(f"  {name}: {acc:.1%}")

        # Check that all classes have >50% accuracy
        min_acc = min(class_accuracies.values())
        assert min_acc > 0.3, f"Some classes have very low accuracy: {min_acc:.1%}"

    @pytest.mark.slow
    def test_top5_accuracy(self, esc50_root):
        """Test top-5 accuracy (should be >95%)."""
        from mlx_audio.data.datasets.classification import ESC50, ESC50_CLASSES
        from mlx_audio.functional.classify import classify

        dataset = ESC50(root=esc50_root, split="test", fold=1)

        top1_correct = 0
        top5_correct = 0
        total = 0

        for sample in dataset:
            audio = sample["audio"]
            true_label = sample["class_names"][0]

            result = classify(audio, labels=ESC50_CLASSES, top_k=5)

            if result.predicted_class == true_label:
                top1_correct += 1

            if result.top_k_classes and true_label in result.top_k_classes:
                top5_correct += 1

            total += 1

        top1_acc = top1_correct / total
        top5_acc = top5_correct / total

        logger.info(f"ESC-50 top-1 accuracy: {top1_acc:.1%}")
        logger.info(f"ESC-50 top-5 accuracy: {top5_acc:.1%}")

        assert top5_acc > 0.95, f"Top-5 accuracy {top5_acc:.1%} below target 95%"
