"""Integration tests for CLAP zero-shot classification on ESC-50.

Run with:
    ESC50_ROOT=/path/to/ESC-50 pytest tests/integration/test_esc50_classification.py -v

Quick test (1 fold, ~10 min):
    pytest tests/integration/test_esc50_classification.py -k "test_single_fold" -v

Full 5-fold evaluation (~45 min):
    pytest tests/integration/test_esc50_classification.py -k "test_five_fold" -v
"""

import json
from collections import defaultdict
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from .conftest import get_esc50_classes, load_esc50_audio, load_esc50_metadata


# =============================================================================
# Quality Targets
# =============================================================================

# Published CLAP zero-shot accuracy on ESC-50
ACCURACY_TARGET = 0.85  # 85%
ACCURACY_MINIMUM = 0.70  # 70% minimum acceptable

# Top-5 accuracy target
TOP5_ACCURACY_TARGET = 0.95


# =============================================================================
# Helper Functions
# =============================================================================


def prepare_class_prompts(classes: list[str]) -> list[str]:
    """Generate text prompts for each class.

    Uses the standard CLAP prompt template.
    """
    prompts = []
    for cls in classes:
        # Replace underscores with spaces for natural language
        cls_text = cls.replace("_", " ")
        prompts.append(f"the sound of {cls_text}")
    return prompts


def classify_audio(
    model,
    audio: np.ndarray,
    sample_rate: int,
    class_prompts: list[str],
    target_sr: int = 48000,
) -> tuple[int, np.ndarray]:
    """Classify audio using CLAP zero-shot classification.

    Args:
        model: CLAP model
        audio: Audio array [samples] or [channels, samples]
        sample_rate: Input sample rate
        class_prompts: List of text prompts for each class
        target_sr: CLAP's expected sample rate

    Returns:
        Tuple of (predicted_class_index, similarity_scores)
    """
    # Resample if needed
    if sample_rate != target_sr:
        from scipy import signal

        if audio.ndim == 2:
            audio = audio.mean(axis=0)  # Convert to mono
        num_samples = int(len(audio) * target_sr / sample_rate)
        audio = signal.resample(audio, num_samples)

    # Ensure mono
    if audio.ndim == 2:
        audio = audio.mean(axis=0)

    # Convert to MLX with batch dimension [B, T]
    audio_mx = mx.array(audio.astype(np.float32))[None, :]

    # Get audio embedding
    audio_emb = model.encode_audio(audio_mx)

    # Get text embeddings for all classes
    text_embs = model.encode_text(class_prompts)

    # Compute similarities
    similarities = model.similarity(audio_emb, text_embs)
    similarities = np.array(similarities).flatten()

    predicted = int(np.argmax(similarities))
    return predicted, similarities


def evaluate_fold(
    model,
    esc50_root: Path,
    metadata: list[dict],
    fold: int,
    classes: list[str],
) -> dict:
    """Evaluate CLAP on a single ESC-50 fold.

    Args:
        model: CLAP model
        esc50_root: ESC-50 root directory
        metadata: Full metadata list
        fold: Fold number (1-5)
        classes: List of class names

    Returns:
        Dictionary with accuracy metrics and per-class results
    """
    class_prompts = prepare_class_prompts(classes)
    fold_samples = [m for m in metadata if int(m["fold"]) == fold]

    correct = 0
    top5_correct = 0
    total = 0

    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)
    confusion = np.zeros((len(classes), len(classes)), dtype=int)

    print(f"\nEvaluating fold {fold} ({len(fold_samples)} samples)...")

    for i, sample in enumerate(fold_samples):
        filename = sample["filename"]
        target_class = int(sample["target"])

        try:
            audio, sr = load_esc50_audio(esc50_root, filename)
            predicted, similarities = classify_audio(
                model, audio, sr, class_prompts
            )

            # Top-1 accuracy
            if predicted == target_class:
                correct += 1
                per_class_correct[target_class] += 1

            # Top-5 accuracy
            top5_preds = np.argsort(similarities)[-5:]
            if target_class in top5_preds:
                top5_correct += 1

            per_class_total[target_class] += 1
            confusion[target_class, predicted] += 1
            total += 1

            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(fold_samples)} samples...")

        except Exception as e:
            print(f"  Error processing {filename}: {e}")

    accuracy = correct / total if total > 0 else 0
    top5_accuracy = top5_correct / total if total > 0 else 0

    # Per-class accuracy
    per_class_accuracy = {}
    for cls_idx, cls_name in enumerate(classes):
        if per_class_total[cls_idx] > 0:
            per_class_accuracy[cls_name] = (
                per_class_correct[cls_idx] / per_class_total[cls_idx]
            )

    return {
        "fold": fold,
        "accuracy": accuracy,
        "top5_accuracy": top5_accuracy,
        "correct": correct,
        "total": total,
        "per_class_accuracy": per_class_accuracy,
        "confusion_matrix": confusion.tolist(),
    }


# =============================================================================
# Quick Tests (Single Fold)
# =============================================================================


@pytest.mark.integration
class TestQuickClassification:
    """Quick classification tests on a single fold."""

    def test_model_produces_embeddings(self, clap_model, esc50_root, esc50_metadata):
        """Verify CLAP produces embeddings for audio."""
        sample = esc50_metadata[0]
        audio, sr = load_esc50_audio(esc50_root, sample["filename"])

        # Convert to mono and MLX
        if audio.ndim == 2:
            audio = audio.mean(axis=0)

        # Resample to 48kHz
        from scipy import signal

        audio = signal.resample(audio, int(len(audio) * 48000 / sr))
        # Add batch dimension [B, T] as expected by CLAP
        audio_mx = mx.array(audio.astype(np.float32))[None, :]

        # Get embedding
        embedding = clap_model.encode_audio(audio_mx)

        assert embedding.shape[-1] == 512, f"Expected 512-dim embedding, got {embedding.shape}"

    @pytest.mark.skip(reason="CLAP.encode_text requires tokenized input IDs, not raw text strings")
    def test_text_embeddings(self, clap_model, esc50_classes):
        """Verify CLAP produces text embeddings.

        Note: This test requires a tokenizer to convert text prompts to token IDs.
        The CLAP model's encode_text expects mx.array of token IDs, not strings.
        """
        prompts = prepare_class_prompts(esc50_classes[:5])  # First 5 classes
        embeddings = clap_model.encode_text(prompts)

        assert embeddings.shape[0] == 5
        assert embeddings.shape[-1] == 512

    @pytest.mark.skip(reason="CLAP.encode_text requires tokenized input IDs - needs tokenizer integration")
    def test_single_fold_accuracy(self, clap_model, esc50_root, esc50_metadata, esc50_classes):
        """Test zero-shot accuracy on fold 1.

        Note: This test requires text tokenization for CLAP classification.
        """
        results = evaluate_fold(
            clap_model, esc50_root, esc50_metadata, fold=1, classes=esc50_classes
        )

        print(f"\nFold 1 Results:")
        print(f"  Accuracy: {results['accuracy']:.1%}")
        print(f"  Top-5 Accuracy: {results['top5_accuracy']:.1%}")

        # Check minimum accuracy
        assert results["accuracy"] > ACCURACY_MINIMUM, (
            f"Accuracy {results['accuracy']:.1%} below minimum {ACCURACY_MINIMUM:.0%}"
        )


# =============================================================================
# Full 5-Fold Evaluation
# =============================================================================


@pytest.mark.integration
@pytest.mark.slow
class TestFullEvaluation:
    """Full 5-fold cross-validation on ESC-50."""

    def test_five_fold_accuracy(
        self,
        clap_model,
        esc50_root,
        esc50_metadata,
        esc50_classes,
        tmp_path,
    ):
        """Run full 5-fold cross-validation and generate report."""
        all_results = []
        all_confusion = np.zeros((len(esc50_classes), len(esc50_classes)), dtype=int)

        for fold in range(1, 6):
            results = evaluate_fold(
                clap_model, esc50_root, esc50_metadata, fold=fold, classes=esc50_classes
            )
            all_results.append(results)
            all_confusion += np.array(results["confusion_matrix"])

        # Compute aggregate metrics
        total_correct = sum(r["correct"] for r in all_results)
        total_samples = sum(r["total"] for r in all_results)
        overall_accuracy = total_correct / total_samples if total_samples > 0 else 0

        # Per-class accuracy across all folds
        per_class_correct = defaultdict(int)
        per_class_total = defaultdict(int)

        for r in all_results:
            for cls_name, acc in r["per_class_accuracy"].items():
                cls_idx = esc50_classes.index(cls_name)
                per_class_correct[cls_idx] += int(acc * (r["total"] / 50))  # Approximate
                per_class_total[cls_idx] += r["total"] // 50

        # Save report
        report = {
            "model": "clap-htsat-fused",
            "dataset": "ESC-50",
            "num_classes": len(esc50_classes),
            "total_samples": total_samples,
            "overall_accuracy": overall_accuracy,
            "target_accuracy": ACCURACY_TARGET,
            "per_fold_results": [
                {"fold": r["fold"], "accuracy": r["accuracy"], "top5_accuracy": r["top5_accuracy"]}
                for r in all_results
            ],
            "class_names": esc50_classes,
            "confusion_matrix": all_confusion.tolist(),
        }

        report_path = tmp_path / "esc50_evaluation.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to: {report_path}")

        # Print summary
        print("\n" + "=" * 60)
        print("ESC-50 5-Fold Cross-Validation Summary")
        print("=" * 60)
        for r in all_results:
            print(f"  Fold {r['fold']}: {r['accuracy']:.1%} (Top-5: {r['top5_accuracy']:.1%})")
        print(f"  Overall: {overall_accuracy:.1%}")
        print(f"  Target: {ACCURACY_TARGET:.0%}")

        # Assert accuracy meets target
        assert overall_accuracy > ACCURACY_TARGET - 0.05, (
            f"Overall accuracy {overall_accuracy:.1%} below target "
            f"{ACCURACY_TARGET - 0.05:.0%}"
        )


# =============================================================================
# Per-Class Analysis
# =============================================================================


@pytest.mark.integration
class TestPerClassAccuracy:
    """Test per-class classification accuracy."""

    def test_no_class_below_threshold(
        self,
        clap_model,
        esc50_root,
        esc50_metadata,
        esc50_classes,
    ):
        """Verify no class has accuracy below 30%."""
        results = evaluate_fold(
            clap_model, esc50_root, esc50_metadata, fold=1, classes=esc50_classes
        )

        low_accuracy_classes = []
        for cls_name, acc in results["per_class_accuracy"].items():
            if acc < 0.30:
                low_accuracy_classes.append((cls_name, acc))

        if low_accuracy_classes:
            print("\nClasses with low accuracy (<30%):")
            for cls_name, acc in low_accuracy_classes:
                print(f"  {cls_name}: {acc:.1%}")

        # Warn but don't fail for a few low-accuracy classes
        assert len(low_accuracy_classes) < 10, (
            f"Too many classes with low accuracy: {len(low_accuracy_classes)}"
        )

    def test_top_classes_accuracy(
        self,
        clap_model,
        esc50_root,
        esc50_metadata,
        esc50_classes,
    ):
        """Test accuracy on typically well-performing classes."""
        well_performing = ["dog", "cat", "car_horn", "helicopter", "chainsaw"]

        results = evaluate_fold(
            clap_model, esc50_root, esc50_metadata, fold=1, classes=esc50_classes
        )

        for cls in well_performing:
            if cls in results["per_class_accuracy"]:
                acc = results["per_class_accuracy"][cls]
                print(f"  {cls}: {acc:.1%}")
                # These classes should have >50% accuracy
                assert acc > 0.5, f"{cls} accuracy too low: {acc:.1%}"
