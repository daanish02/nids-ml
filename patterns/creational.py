"""
creational.py — Factory Method Pattern
=======================================
PATTERN   : Creational → Factory Method
INTENT    : Define an interface for creating an object, but let subclasses
            decide which class to instantiate.
ROLE HERE : ClassifierFactory decides which ML model to build based on a
            string key (config-driven), completely decoupling object creation
            from the rest of the pipeline.

            Client code never calls RandomForestClassifier() directly —
            it only calls factory.create_classifier("random_forest").
            Swapping algorithms requires zero changes in the pipeline.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from sklearn.ensemble import (
    RandomForestClassifier as SKLearnRF,
    GradientBoostingClassifier,
)
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from patterns.base import BaseClassifier


# ─────────────────────────────────────────────────────────────────────────────
# Concrete Classifiers  (Products in Factory Method terminology)
# ─────────────────────────────────────────────────────────────────────────────


class RandomForestModel(BaseClassifier):
    """
    Product A: Random Forest — robust to noisy network data,
    naturally handles mixed feature types in NSL-KDD.
    """

    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        self._model = SKLearnRF(
            n_estimators=n_estimators, random_state=random_state, n_jobs=-1
        )

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self._model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)

    def name(self) -> str:
        return "Random Forest"


class SVMModel(BaseClassifier):
    """
    Product B: Support Vector Machine — effective in high-dimensional
    feature spaces, good for detecting boundary-based anomalies.
    """

    def __init__(
        self, kernel: str = "rbf", probability: bool = True, random_state: int = 42
    ):
        self._model = SVC(
            kernel=kernel, probability=probability, random_state=random_state
        )

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self._model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)

    def name(self) -> str:
        return "Support Vector Machine"


class NeuralNetModel(BaseClassifier):
    """
    Product C: Multi-Layer Perceptron — learns non-linear patterns in
    traffic behaviour that rule-based systems miss.
    """

    def __init__(
        self,
        hidden_layer_sizes: tuple = (128, 64),
        max_iter: int = 300,
        random_state: int = 42,
    ):
        self._model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            random_state=random_state,
            early_stopping=True,
            validation_fraction=0.1,
        )

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self._model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)

    def name(self) -> str:
        return "Neural Network (MLP)"


class GradientBoostingModel(BaseClassifier):
    """
    Product D: Gradient Boosting — sequential ensemble that corrects
    previous errors; excellent for imbalanced attack/normal ratios.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        random_state: int = 42,
    ):
        self._model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state,
        )

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self._model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)

    def name(self) -> str:
        return "Gradient Boosting"


# ─────────────────────────────────────────────────────────────────────────────
# Factory (Creator in Factory Method terminology)
# ─────────────────────────────────────────────────────────────────────────────


class ClassifierFactory(ABC):
    """
    Abstract Creator — declares the factory method that subclasses override.
    Also contains core business logic that calls the factory method.
    """

    @abstractmethod
    def create_classifier(self) -> BaseClassifier:
        """Factory method — subclasses decide which product to instantiate."""
        pass

    def get_classifier(self) -> BaseClassifier:
        """
        Business logic that uses the factory method result.
        The creator doesn't know the concrete type — it only uses BaseClassifier.
        """
        classifier = self.create_classifier()
        print(f"  [Factory] Created classifier → {classifier.name()}")
        return classifier


class RandomForestFactory(ClassifierFactory):
    def __init__(self, n_estimators: int = 100):
        self._n_estimators = n_estimators

    def create_classifier(self) -> BaseClassifier:
        return RandomForestModel(n_estimators=self._n_estimators)


class SVMFactory(ClassifierFactory):
    def __init__(self, kernel: str = "rbf"):
        self._kernel = kernel

    def create_classifier(self) -> BaseClassifier:
        return SVMModel(kernel=self._kernel)


class NeuralNetFactory(ClassifierFactory):
    def __init__(self, hidden_layers: tuple = (128, 64)):
        self._hidden_layers = hidden_layers

    def create_classifier(self) -> BaseClassifier:
        return NeuralNetModel(hidden_layer_sizes=self._hidden_layers)


class GradientBoostingFactory(ClassifierFactory):
    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1):
        self._n_estimators = n_estimators
        self._lr = learning_rate

    def create_classifier(self) -> BaseClassifier:
        return GradientBoostingModel(
            n_estimators=self._n_estimators, learning_rate=self._lr
        )


# ─────────────────────────────────────────────────────────────────────────────
# Registry — maps string keys → factory classes (config-driven instantiation)
# ─────────────────────────────────────────────────────────────────────────────


class ClassifierRegistry:
    """
    Central lookup table: string key → factory instance.
    Allows models to be selected via config files or CLI args
    without any if/elif chains in the calling code.
    """

    _registry: dict[str, ClassifierFactory] = {
        "random_forest": RandomForestFactory(),
        "svm": SVMFactory(),
        "neural_net": NeuralNetFactory(),
        "gradient_boosting": GradientBoostingFactory(),
    }

    @classmethod
    def get(cls, model_type: str) -> BaseClassifier:
        factory = cls._registry.get(model_type.lower())
        if factory is None:
            available = list(cls._registry.keys())
            raise ValueError(
                f"Unknown model type: '{model_type}'. Available: {available}"
            )
        return factory.get_classifier()

    @classmethod
    def available_models(cls) -> list[str]:
        return list(cls._registry.keys())
