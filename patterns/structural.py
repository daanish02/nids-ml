"""
structural.py — Decorator Pattern
===================================
PATTERN   : Structural → Decorator
INTENT    : Attach additional responsibilities to an object dynamically.
            Decorators provide a flexible alternative to subclassing.
ROLE HERE : Preprocessing steps (normalization, PCA, feature selection)
            are stacked as decorators around any BaseClassifier.
            This means:
              - The core classifier is NEVER modified.
              - Preprocessing steps are composable and reorderable.
              - Adding a new step (e.g., SMOTE oversampling) = one new class.

            Usage:
                clf = ClassifierRegistry.get("random_forest")
                clf = NormalizerDecorator(clf)
                clf = PCADecorator(clf, n_components=20)
                clf = FeatureSelectorDecorator(clf, k=15)
                clf.train(X_train, y_train)   # all steps fire automatically
"""

from __future__ import annotations
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE

from patterns.base import BaseClassifier


# ─────────────────────────────────────────────────────────────────────────────
# Base Decorator  (wraps any BaseClassifier transparently)
# ─────────────────────────────────────────────────────────────────────────────


class ClassifierDecorator(BaseClassifier):
    """
    Abstract Decorator — holds a reference to a wrapped BaseClassifier
    and delegates all calls to it by default. Concrete decorators override
    only what they need to intercept.
    """

    def __init__(self, classifier: BaseClassifier):
        self._classifier = classifier  # the wrapped component

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self._classifier.train(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._classifier.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._classifier.predict_proba(X)

    def name(self) -> str:
        return self._classifier.name()


# ─────────────────────────────────────────────────────────────────────────────
# Concrete Decorators
# ─────────────────────────────────────────────────────────────────────────────


class NormalizerDecorator(ClassifierDecorator):
    """
    Decorator 1 — Z-score normalization (StandardScaler).
    NSL-KDD features span vastly different numeric ranges
    (e.g., src_bytes can be 0–1e9 while flags are 0/1).
    Normalization prevents high-magnitude features from dominating.
    """

    def __init__(self, classifier: BaseClassifier, scaler_type: str = "standard"):
        super().__init__(classifier)
        self._scaler = StandardScaler() if scaler_type == "standard" else MinMaxScaler()
        self._scaler_type = scaler_type

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        print(
            f"    [Decorator] NormalizerDecorator ({self._scaler_type}) → fitting & transforming X"
        )
        X_scaled = self._scaler.fit_transform(X)
        self._classifier.train(X_scaled, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self._scaler.transform(X)
        return self._classifier.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self._scaler.transform(X)
        return self._classifier.predict_proba(X_scaled)

    def name(self) -> str:
        return f"Normalized({self._classifier.name()})"


class PCADecorator(ClassifierDecorator):
    """
    Decorator 2 — PCA dimensionality reduction.
    NSL-KDD has 41 features, many correlated.
    PCA compresses these into orthogonal components,
    reducing noise and speeding up training.
    """

    def __init__(self, classifier: BaseClassifier, n_components: int = 20):
        super().__init__(classifier)
        self._pca = PCA(n_components=n_components, random_state=42)
        self._n_components = n_components

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        print(
            f"    [Decorator] PCADecorator → reducing to {self._n_components} components "
            f"(from {X.shape[1]} features)"
        )
        X_reduced = self._pca.fit_transform(X)
        explained = self._pca.explained_variance_ratio_.sum() * 100
        print(f"    [Decorator] PCA variance retained: {explained:.1f}%")
        self._classifier.train(X_reduced, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_reduced = self._pca.transform(X)
        return self._classifier.predict(X_reduced)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_reduced = self._pca.transform(X)
        return self._classifier.predict_proba(X_reduced)

    def name(self) -> str:
        return f"PCA-{self._n_components}({self._classifier.name()})"


class FeatureSelectorDecorator(ClassifierDecorator):
    """
    Decorator 3 — Univariate feature selection (SelectKBest / ANOVA F-score).
    Keeps only the top-k features most correlated with the target label,
    removing irrelevant/redundant columns that hurt generalisation.
    """

    def __init__(self, classifier: BaseClassifier, k: int = 20):
        super().__init__(classifier)
        self._selector = SelectKBest(score_func=f_classif, k=k)
        self._k = k

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        print(
            f"    [Decorator] FeatureSelectorDecorator → selecting top {self._k} features"
        )
        X_selected = self._selector.fit_transform(X, y)
        self._classifier.train(X_selected, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_selected = self._selector.transform(X)
        return self._classifier.predict(X_selected)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_selected = self._selector.transform(X)
        return self._classifier.predict_proba(X_selected)

    def name(self) -> str:
        return f"TopK-{self._k}({self._classifier.name()})"


class SMOTEDecorator(ClassifierDecorator):
    """
    Decorator 4 — SMOTE oversampling (train-time only).
    NSL-KDD is class-imbalanced: some attack types are rare.
    SMOTE synthesises minority-class samples during training only —
    test data is never touched, preserving evaluation integrity.
    """

    def __init__(self, classifier: BaseClassifier, random_state: int = 42):
        super().__init__(classifier)
        self._smote = SMOTE(random_state=random_state)

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        print("    [Decorator] SMOTEDecorator → resampling imbalanced classes")
        before = dict(zip(*np.unique(y, return_counts=True)))
        X_res, y_res = self._smote.fit_resample(X, y)
        after = dict(zip(*np.unique(y_res, return_counts=True)))
        print(
            f"    [Decorator] Samples before: {sum(before.values())} → after: {sum(after.values())}"
        )
        self._classifier.train(X_res, y_res)

    def name(self) -> str:
        return f"SMOTE({self._classifier.name()})"


# ─────────────────────────────────────────────────────────────────────────────
# Convenience Builder — constructs a fully-decorated pipeline in one call
# ─────────────────────────────────────────────────────────────────────────────


class PreprocessingPipelineBuilder:
    """
    Fluent builder that stacks decorators in the correct order:
        Raw classifier → Normalizer → Feature Selector → (optional PCA) → (optional SMOTE)
    """

    def __init__(self, classifier: BaseClassifier):
        self._clf = classifier

    def with_normalization(
        self, scaler_type: str = "standard"
    ) -> PreprocessingPipelineBuilder:
        self._clf = NormalizerDecorator(self._clf, scaler_type=scaler_type)
        return self

    def with_feature_selection(self, k: int = 20) -> PreprocessingPipelineBuilder:
        self._clf = FeatureSelectorDecorator(self._clf, k=k)
        return self

    def with_pca(self, n_components: int = 15) -> PreprocessingPipelineBuilder:
        self._clf = PCADecorator(self._clf, n_components=n_components)
        return self

    def with_smote(self) -> PreprocessingPipelineBuilder:
        self._clf = SMOTEDecorator(self._clf)
        return self

    def build(self) -> BaseClassifier:
        return self._clf
