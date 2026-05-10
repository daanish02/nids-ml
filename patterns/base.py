"""
base.py — Abstract Base Classes for NIDS Design Pattern System
================================================================
Defines the contracts that all classifiers, decorators, and strategies must follow.
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseClassifier(ABC):
    """
    Abstract base for every classifier in the system.
    Both raw classifiers (from Factory) and decorated classifiers implement this.
    """

    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the model to training data."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted labels for input samples."""
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probability estimates."""
        pass

    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the classifier."""
        pass


class ClassificationStrategy(ABC):
    """
    Abstract base for behavioral Strategy pattern.
    Each concrete strategy encapsulates a different classification logic:
    binary (normal vs attack), multiclass (attack category), or anomaly.
    """

    @abstractmethod
    def execute(self, classifier: BaseClassifier, X: np.ndarray) -> np.ndarray:
        """Apply the strategy's classification logic using the given classifier."""
        pass

    @abstractmethod
    def strategy_name(self) -> str:
        pass
