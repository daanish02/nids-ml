"""
behavioral.py — Strategy Pattern
==================================
PATTERN   : Behavioral → Strategy
INTENT    : Define a family of algorithms, encapsulate each one, and make
            them interchangeable. Strategy lets the algorithm vary
            independently from the clients that use it.
ROLE HERE : Three different classification modes are available —
              • BinaryStrategy     : normal vs. attack (2 classes)
              • MulticlassStrategy : DoS / Probe / R2L / U2R / normal (5 classes)
              • AnomalyStrategy    : threshold-based confidence scoring
            The ClassificationContext holds whichever strategy is active and
            can swap it at runtime without touching the classifier or pipeline.

            Usage:
                ctx = ClassificationContext(BinaryStrategy())
                ctx.classify(classifier, X_test, y_test)

                # swap at runtime — no code changes elsewhere
                ctx.set_strategy(MulticlassStrategy())
                ctx.classify(classifier, X_test, y_test)
"""

from __future__ import annotations
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)
from patterns.base import BaseClassifier, ClassificationStrategy


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────


class ClassificationResult:
    """Value object holding all metrics from a single classification run."""

    def __init__(
        self,
        strategy_name: str,
        predictions: np.ndarray,
        y_true: np.ndarray,
        probabilities: np.ndarray | None = None,
    ):
        self.strategy_name = strategy_name
        self.predictions = predictions
        self.y_true = y_true
        self.probabilities = probabilities

        unique_labels = np.unique(np.concatenate([y_true, predictions]))
        avg = "binary" if len(unique_labels) == 2 else "weighted"
        self.accuracy = accuracy_score(y_true, predictions)
        self.precision = precision_score(
            y_true, predictions, average=avg, zero_division=0
        )
        self.recall = recall_score(y_true, predictions, average=avg, zero_division=0)
        self.f1 = f1_score(y_true, predictions, average=avg, zero_division=0)
        self.conf_matrix = confusion_matrix(y_true, predictions)
        self.report = classification_report(y_true, predictions, zero_division=0)

        # ROC-AUC only for binary classification with probabilities
        self.roc_auc = None
        if probabilities is not None and len(np.unique(y_true)) == 2:
            self.roc_auc = roc_auc_score(y_true, probabilities[:, 1])

    def __str__(self) -> str:
        lines = [
            f"\n  Strategy     : {self.strategy_name}",
            f"  Accuracy     : {self.accuracy:.4f}",
            f"  Precision    : {self.precision:.4f}",
            f"  Recall       : {self.recall:.4f}",
            f"  F1-Score     : {self.f1:.4f}",
        ]
        if self.roc_auc is not None:
            lines.append(f"  ROC-AUC      : {self.roc_auc:.4f}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Concrete Strategies
# ─────────────────────────────────────────────────────────────────────────────


class BinaryStrategy(ClassificationStrategy):
    """
    Strategy A — Binary Classification: normal (0) vs. attack (1).
    All multi-class attack labels are collapsed to a single 'attack' label.
    Best for high-speed alerting where attack type is secondary to detection.
    """

    def execute(
        self,
        classifier: BaseClassifier,
        X: np.ndarray,
        y_true: np.ndarray | None = None,
    ) -> ClassificationResult | np.ndarray:

        predictions = classifier.predict(X)
        probabilities = None
        try:
            probabilities = classifier.predict_proba(X)
        except Exception:
            pass

        if y_true is not None:
            return ClassificationResult(
                strategy_name=self.strategy_name(),
                predictions=predictions,
                y_true=y_true,
                probabilities=probabilities,
            )
        return predictions

    def strategy_name(self) -> str:
        return "Binary Strategy (normal vs. attack)"


class MulticlassStrategy(ClassificationStrategy):
    """
    Strategy B — Multiclass Classification: normal + 4 attack families.
    Attack categories in NSL-KDD:
        DoS   — Denial of Service (smurf, neptune, …)
        Probe — Surveillance / port scanning (portsweep, satan, …)
        R2L   — Remote to Local (ftp_write, guess_passwd, …)
        U2R   — User to Root (buffer_overflow, rootkit, …)

    This strategy gives security analysts detailed threat intelligence,
    enabling category-specific countermeasures.
    """

    def execute(
        self,
        classifier: BaseClassifier,
        X: np.ndarray,
        y_true: np.ndarray | None = None,
    ) -> ClassificationResult | np.ndarray:

        predictions = classifier.predict(X)
        probabilities = None
        try:
            probabilities = classifier.predict_proba(X)
        except Exception:
            pass

        if y_true is not None:
            return ClassificationResult(
                strategy_name=self.strategy_name(),
                predictions=predictions,
                y_true=y_true,
                probabilities=probabilities,
            )
        return predictions

    def strategy_name(self) -> str:
        return "Multiclass Strategy (normal / DoS / Probe / R2L / U2R)"


class AnomalyStrategy(ClassificationStrategy):
    """
    Strategy C — Anomaly / Confidence Thresholding.
    Instead of hard labels, uses the classifier's probability output.
    Any sample whose max-class confidence < threshold is flagged as anomalous —
    useful for detecting zero-day attacks not present in training data.
    """

    def __init__(self, threshold: float = 0.75):
        self._threshold = threshold

    def execute(
        self,
        classifier: BaseClassifier,
        X: np.ndarray,
        y_true: np.ndarray | None = None,
    ) -> ClassificationResult | np.ndarray:

        probabilities = classifier.predict_proba(X)
        max_conf = probabilities.max(axis=1)
        hard_preds = classifier.predict(X)

        # Flag low-confidence predictions as anomalies (label = -1)
        anomaly_preds = np.where(max_conf < self._threshold, -1, hard_preds)

        n_anomalies = (anomaly_preds == -1).sum()
        print(
            f"    [Strategy] Anomaly threshold={self._threshold:.2f} → "
            f"{n_anomalies}/{len(X)} samples flagged as anomalous "
            f"({100 * n_anomalies / len(X):.1f}%)"
        )

        if y_true is not None:
            # For metric computation, treat anomaly (-1) as predicted attack
            eval_preds = np.where(anomaly_preds == -1, 1, anomaly_preds)
            return ClassificationResult(
                strategy_name=self.strategy_name(),
                predictions=eval_preds,
                y_true=y_true,
                probabilities=probabilities,
            )
        return anomaly_preds

    def strategy_name(self) -> str:
        return f"Anomaly Strategy (confidence threshold={self._threshold})"


# ─────────────────────────────────────────────────────────────────────────────
# Context  (holds and delegates to the active strategy)
# ─────────────────────────────────────────────────────────────────────────────


class ClassificationContext:
    """
    Context — the object clients interact with.
    It maintains a reference to the active Strategy and delegates
    classification work to it. Strategies are hot-swappable at any time.
    """

    def __init__(self, strategy: ClassificationStrategy):
        self._strategy = strategy
        print(f"  [Context] Initial strategy → {strategy.strategy_name()}")

    def set_strategy(self, strategy: ClassificationStrategy) -> None:
        print(f"  [Context] Strategy swapped → {strategy.strategy_name()}")
        self._strategy = strategy

    def classify(
        self,
        classifier: BaseClassifier,
        X: np.ndarray,
        y_true: np.ndarray | None = None,
    ) -> ClassificationResult | np.ndarray:
        """
        Delegates classification to the current strategy.
        If y_true is provided, returns a full ClassificationResult with metrics.
        Otherwise returns raw predictions.
        """
        return self._strategy.execute(classifier, X, y_true)

    @property
    def active_strategy(self) -> str:
        return self._strategy.strategy_name()
