import time
import warnings

warnings.filterwarnings("ignore")

from data.loader import NSLKDDLoader
from patterns.creational import ClassifierRegistry
from patterns.structural import PreprocessingPipelineBuilder
from patterns.behavioral import (
    ClassificationContext,
    BinaryStrategy,
    MulticlassStrategy,
    AnomalyStrategy,
)


def header(title: str) -> None:
    width = 62
    print(f"\n{'═' * width}")
    print(f"  {title}")
    print(f"{'═' * width}")


def section(title: str) -> None:
    print(f"\n  {'─' * 58}")
    print(f"  ▶  {title}")
    print(f"  {'─' * 58}")


def demo_binary_multi_model(X_train, X_test, y_train, y_test):
    header("DEMO 1 — Binary Classification | 3 Models via Factory Method")

    models_to_try = ["random_forest", "gradient_boosting", "neural_net"]
    results = []

    for model_key in models_to_try:
        section(f"Model: {model_key.replace('_', ' ').title()}")

        print("\n  [1/3] CREATIONAL — Factory Method")
        clf = ClassifierRegistry.get(model_key)

        print("\n  [2/3] STRUCTURAL — Decorator Stack")
        clf = (
            PreprocessingPipelineBuilder(clf)
            .with_normalization()
            .with_feature_selection(k=25)
            .build()
        )
        print(f"    Final pipeline name: {clf.name()}")

        print("\n  Training …")
        t0 = time.time()
        clf.train(X_train, y_train)
        elapsed = time.time() - t0
        print(f"    Trained in {elapsed:.1f}s")

        print("\n  [3/3] BEHAVIORAL — Strategy Pattern")
        ctx = ClassificationContext(BinaryStrategy())
        result = ctx.classify(clf, X_test, y_test)

        print(result)
        results.append((model_key, result))

    # Summary table
    header("Binary Classification — Summary")
    print(
        f"  {'Model':<25} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8} {'ROC-AUC':>9}"
    )
    print(f"  {'─' * 25} {'─' * 9} {'─' * 10} {'─' * 8} {'─' * 8} {'─' * 9}")
    for key, r in results:
        roc = f"{r.roc_auc:.4f}" if r.roc_auc else "  N/A  "
        print(
            f"  {key:<25} {r.accuracy:>9.4f} {r.precision:>10.4f} "
            f"{r.recall:>8.4f} {r.f1:>8.4f} {roc:>9}"
        )


def demo_strategy_hotswap(
    X_train, X_test, y_train_mc, y_test_mc, y_train_bin, y_test_bin
):
    header("DEMO 2 — Strategy Hot-Swap (same classifier, 3 strategies)")

    section("Step 1 — Build classifier via Factory Method")
    clf = ClassifierRegistry.get("random_forest")

    section("Step 2 — Wrap with Decorator pipeline")
    clf = (
        PreprocessingPipelineBuilder(clf)
        .with_normalization()
        .with_feature_selection(k=30)
        .build()
    )

    # Train with multiclass labels
    print("\n  Training on multiclass labels …")
    clf.train(X_train, y_train_mc)
    print("  ✓ Training complete")

    section("Step 3 — Strategy Pattern: run all 3 strategies")

    ctx = ClassificationContext(MulticlassStrategy())

    print("\n  ── Running MulticlassStrategy ──")
    r_mc = ctx.classify(clf, X_test, y_test_mc)
    print(r_mc)

    print("\n  ── Swapping to BinaryStrategy (collapse to 0/1) ──")
    ctx.set_strategy(BinaryStrategy())
    y_test_bin_derived = (y_test_mc > 0).astype(int)
    y_pred_bin_derived = (clf.predict(X_test) > 0).astype(int)

    # Re-run binary eval manually for comparison
    from sklearn.metrics import accuracy_score, f1_score

    bin_acc = accuracy_score(y_test_bin_derived, y_pred_bin_derived)
    bin_f1 = f1_score(y_test_bin_derived, y_pred_bin_derived)
    print(f"  Accuracy={bin_acc:.4f}  F1={bin_f1:.4f}")

    print("\n  ── Swapping to AnomalyStrategy (threshold=0.80) ──")
    ctx.set_strategy(AnomalyStrategy(threshold=0.80))
    r_anom = ctx.classify(clf, X_test, y_test_bin_derived)
    print(r_anom)


def demo_full_decorator_stack(X_train, X_test, y_train, y_test):
    header("DEMO 3 — Full Decorator Stack (Normalizer + FeatureSelector + SMOTE)")

    section("Building fully-decorated pipeline")

    clf = ClassifierRegistry.get("gradient_boosting")

    clf = (
        PreprocessingPipelineBuilder(clf)
        .with_normalization(scaler_type="standard")
        .with_feature_selection(k=20)
        .with_smote()
        .build()
    )
    print(f"\n  Pipeline: {clf.name()}")

    print("\n  Training …")
    t0 = time.time()
    clf.train(X_train, y_train)
    print(f"  ✓ Trained in {time.time() - t0:.1f}s")

    ctx = ClassificationContext(BinaryStrategy())
    result = ctx.classify(clf, X_test, y_test)
    print(result)
    print(f"\n  Full classification report:\n{result.report}")


def main():
    print("\n" + "█" * 62)
    print("  NIDS — Network Intrusion Detection System")
    print("  Design Patterns Demo: Factory · Decorator · Strategy")
    print("  Dataset: NSL-KDD")
    print("█" * 62)

    # ── Load binary dataset ────────────────────────────────────────────────
    header("Loading NSL-KDD Dataset")
    loader = NSLKDDLoader(data_dir="./data", sample_frac=1.0)

    print("\n  Loading BINARY labels (normal=0, attack=1) …")
    X_train, X_test, y_train_bin, y_test_bin, features = loader.load("binary")

    print("\n  Loading MULTICLASS labels (0-4) …")
    _, _, y_train_mc, y_test_mc, _ = loader.load("multiclass")

    demo_binary_multi_model(X_train, X_test, y_train_bin, y_test_bin)
    demo_strategy_hotswap(
        X_train, X_test, y_train_mc, y_test_mc, y_train_bin, y_test_bin
    )
    demo_full_decorator_stack(X_train, X_test, y_train_bin, y_test_bin)

    header("All Demos Complete")
    print("""
  PATTERNS DEMONSTRATED
  ─────────────────────
  Creational  →  Factory Method
    ClassifierRegistry.get("model_name") instantiates the
    correct model via a matching concrete factory.
    Client code is decoupled from all concrete classes.

  Structural  →  Decorator
    PreprocessingPipelineBuilder stacks Normalizer,
    FeatureSelector, and SMOTE decorators transparently
    around any classifier — zero changes to core models.

  Behavioral  →  Strategy
    ClassificationContext.set_strategy() hot-swaps between
    Binary, Multiclass, and Anomaly classification logic
    at runtime without modifying the classifier or pipeline.
""")


if __name__ == "__main__":
    main()
