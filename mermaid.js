export const architectureDiagram = String.raw`
---
config:
  look: classic
  theme: default
---
flowchart TB
 subgraph DataLayer["Data Layer"]
        Source["NSL-KDD Input Data"]
        Loader["Dataset Loader and Encoding"]
        Prepared["Prepared Training and Test Sets"]
  end
 subgraph ModelCreation["Creational Layer"]
        Registry["Model Registry"]
        Factory["Model Factory"]
        Model["Concrete Classifier"]
  end
 subgraph Preprocessing["Structural Layer"]
        Builder["Preprocessing Builder"]
        Decorators["Composable Preprocessing Steps"]
        Norm["Normalization"]
        Select["Feature Selection"]
        Reduce["Dimensionality Reduction"]
        Resample["Class Balancing"]
  end
 subgraph RuntimeLogic["Behavioral Layer"]
        Context["Classification Context"]
        StrategyA["Binary Strategy"]
        StrategyB["Multiclass Strategy"]
        StrategyC["Anomaly Strategy"]
        Result["Performance Report"]
  end
 subgraph External["External Libraries"]
        Libs["Array Processing, Models, Metrics, Resampling"]
  end
    App["Main Orchestration"] --> Source & Registry & Builder & RuntimeLogic
    Source --> Loader
    Loader --> Prepared & Libs
    Registry --> Factory
    Factory --> Model
    Builder --> Decorators
    Decorators --> Model & Norm & Select & Reduce & Resample & Libs
    Model --> Context & Libs
    Context --> StrategyA & StrategyB & StrategyC & Libs
    StrategyA --> Result
    StrategyB --> Result
    StrategyC --> Result
    Contracts["Core Contracts and Interfaces"] -. guides .-> Model & Decorators & Context
    Prepared --> Builder

     Source:::data
     Loader:::data
     Prepared:::data
     Registry:::model
     Factory:::model
     Model:::model
     Builder:::structural
     Decorators:::structural
     Norm:::structural
     Select:::structural
     Reduce:::structural
     Resample:::structural
     Context:::behavioral
     StrategyA:::behavioral
     StrategyB:::behavioral
     StrategyC:::behavioral
     Result:::behavioral
     Libs:::backend
     App:::entry
     Contracts:::contract
    classDef entry fill:#1f2937,color:#ffffff,stroke:#111827,stroke-width:1px
    classDef data fill:#0f766e,color:#ffffff,stroke:#134e4a,stroke-width:1px
    classDef model fill:#92400e,color:#ffffff,stroke:#78350f,stroke-width:1px
    classDef structural fill:#1d4ed8,color:#ffffff,stroke:#1e3a8a,stroke-width:1px
    classDef behavioral fill:#047857,color:#ffffff,stroke:#064e3b,stroke-width:1px
    classDef contract fill:#334155,color:#ffffff,stroke:#0f172a,stroke-width:1px
    classDef backend fill:#6b7280,color:#ffffff,stroke:#374151,stroke-width:1px
`;

/*
Architecture summary
--------------------
This module presents the project as a layered architecture diagram suitable for
inclusion in a report or paper. The overall system is centered on one
orchestration entry point that coordinates the complete machine learning
pipeline from data acquisition to evaluation.

The first layer handles data acquisition and preparation. It reads the source
dataset, encodes categorical attributes, converts labels into the required
target representation, and produces prepared training and test sets for the
rest of the pipeline. This separation keeps input processing isolated from the
modeling logic.

The shared contracts define the stable structure of the system. One interface
describes the classifier capability, while another defines the strategy used to
interpret predictions. These abstractions preserve interchangeability and keep
the architecture loosely coupled.

The creational layer applies the Factory Method pattern. A registry maps a model
choice to the corresponding factory, and the factory produces the concrete
classifier instance. This centralizes object creation and prevents the rest of
the application from depending on implementation-specific model classes.

The structural layer applies the Decorator pattern. A builder assembles a chain
of preprocessing steps around the classifier, and each step contributes one
optional responsibility such as normalization, feature selection,
dimensionality reduction, or class balancing. The result is a composable
preprocessing pipeline that can be extended without modifying the classifier.

The behavioral layer applies the Strategy pattern. A context object stores the
active classification mode and delegates evaluation to it. The available
strategies support binary, multiclass, and anomaly-oriented interpretation of
the same classifier output, allowing runtime switching without changing the
trained model.

The architecture is therefore layered but flexible. Data flows from the input
source into the loader, then into the factory-produced classifier, passes
through the decorator-based preprocessing chain, and finally reaches the
strategy layer for evaluation. Each stage has a narrow and clearly defined
responsibility: preparing data, selecting models, shaping the pipeline, and
interpreting predictions.

The implementation also relies on external machine learning libraries for array
processing, model training, preprocessing, metrics, and resampling. These
libraries are not part of the project-specific architecture, but they supply the
concrete algorithms wrapped by the pattern-based design.
*/

export default architectureDiagram;