# NIDS — Network Intrusion Detection System
## Design Patterns Assignment — Python Implementation

---

## Assignment Overview

**Topic:** Applying Software Design Patterns to a Scalable AI-Based
Network Intrusion Detection and Traffic Classification System

**Dataset:** NSL-KDD (auto-downloaded on first run)

**Patterns implemented:**
| Type       | Pattern        | Class                          | Role                                        |
| ---------- | -------------- | ------------------------------ | ------------------------------------------- |
| Creational | Factory Method | `ClassifierRegistry`           | Builds the right ML model from a string key |
| Structural | Decorator      | `PreprocessingPipelineBuilder` | Stacks normalizer, feature selector, SMOTE  |
| Behavioral | Strategy       | `ClassificationContext`        | Swaps binary/multiclass/anomaly at runtime  |

---

## Project Structure

```
nids_project/
├── main.py                   ← Entry point — runs all 3 demos
├── requirements.txt
├── data/
│   ├── __init__.py
│   └── loader.py             ← NSL-KDD downloader + preprocessor
└── patterns/
    ├── __init__.py
    ├── base.py               ← Abstract base classes (BaseClassifier, ClassificationStrategy)
    ├── creational.py         ← Factory Method pattern
    ├── structural.py         ← Decorator pattern
    └── behavioral.py         ← Strategy pattern
```

---

## What Each Demo Does

### Demo 1 — Factory Method + Decorator + Binary Strategy
Three models (Random Forest, Gradient Boosting, Neural Net) are each
created through the factory, wrapped with the same decorator pipeline,
and evaluated on binary classification (normal vs. attack).
Shows how the factory decouples model selection from the pipeline.

### Demo 2 — Strategy Hot-Swap
One trained classifier is used with all three strategies in sequence:
Multiclass → Binary → Anomaly. No retraining, no code changes.
Shows how the Strategy pattern makes classification mode interchangeable.

### Demo 3 — Full Decorator Stack
A classifier is wrapped with all four decorators (Normalizer +
FeatureSelector + SMOTE) and evaluated with a final binary strategy.
Shows composable, order-aware preprocessing via chained decorators.

---

## Design Pattern Details

### Creational — Factory Method
```
ClassifierFactory (abstract)
    ├── RandomForestFactory   → RandomForestModel
    ├── SVMFactory            → SVMModel
    ├── NeuralNetFactory      → NeuralNetModel
    └── GradientBoostingFactory → GradientBoostingModel

ClassifierRegistry.get("random_forest")  ← client call
```

### Structural — Decorator
```
BaseClassifier
    └── ClassifierDecorator (abstract)
            ├── NormalizerDecorator     ← Z-score / MinMax scaling
            ├── FeatureSelectorDecorator← ANOVA top-k selection
            ├── PCADecorator            ← dimensionality reduction
            └── SMOTEDecorator          ← minority oversampling

Stacking order:
  clf = NormalizerDecorator(
            FeatureSelectorDecorator(
                SMOTEDecorator(
                    RandomForestModel()
                )
            )
        )
```

### Behavioral — Strategy
```
ClassificationStrategy (abstract)
    ├── BinaryStrategy      ← normal vs. attack (2 classes)
    ├── MulticlassStrategy  ← normal/DoS/Probe/R2L/U2R (5 classes)
    └── AnomalyStrategy     ← confidence threshold flagging

ClassificationContext.set_strategy(new_strategy)  ← hot-swap
```
