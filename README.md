# ATAM-FET: Experimental Extensions and Reproducibility

This branch contains the **latest experimental extensions and updated implementations** developed after the original project version maintained in the `main` branch.

The primary focus of this branch is the **Adaptive Toxic Attention Module (ATAM)** and the extended **ATAM-FET** experimental workflow.

## About This Branch

The `updates` branch is maintained separately to document and preserve new experimental work without modifying the original project implementation in `main`.

The main additions include:

* Adaptive Toxic Attention Module (ATAM) implementation and evaluation
* Extended ATAM-FET experiments
* Component-level and ablation evaluation
* Additional evaluation and analysis utilities
* Updated reproducibility notebooks
* Supporting federated and edge-oriented implementation components

## Experimental Notebooks

### `notebooks/atamframework.ipynb`

Primary notebook for the **DistilBERT + ATAM** experiment.

It includes:

* Jigsaw Toxic Comment dataset preparation
* Train/validation/test splitting
* DistilBERT tokenization
* ATAM implementation
* Model training and validation
* Multi-label toxicity classification
* Overall and per-label evaluation
* Training curves and confusion matrices

Saved execution outputs are retained for result inspection and reproducibility.

### `notebooks/ATAM FET enhanced.ipynb`

Extended notebook for the **ATAM-FET experimental workflow**.

It includes:

* Baseline and extended model configurations
* Component-level experiments
* Ablation-oriented evaluation
* Additional performance analysis
* Attention analysis and visualization
* Experimental figure and result generation

### `notebooks/atamupdate2.ipynb`

Integrated notebook containing an extended version of the ATAM-FET workflow.

It combines multiple experimental components into a single reproducible workflow and retains execution outputs for result verification.

## Repository Structure

```text
Toxic-Comment-detection/
├── configs/            # Experiment configuration
├── data/               # Dataset and data-related files
├── evaluation/         # Metrics and evaluation utilities
├── federated/          # Federated-learning components
├── models/             # Model architectures and ATAM implementation
├── notebooks/          # Experimental and reproducibility notebooks
├── optimization/       # Model and edge optimization utilities
├── preprocessing/      # Data cleaning and tokenization
├── results/            # Experimental results and generated outputs
├── simulation/         # Prediction and simulation components
├── training/           # Model-training utilities
├── PROJECT_STATUS.md   # Implementation status
├── PROGRESS_TRACKING.md
└── README.md
```

## Reproducibility

The experimental notebooks retain saved outputs so that results, figures, and evaluation steps can be inspected directly.

For the primary ATAM experiment:

```text
notebooks/atamframework.ipynb
```

For the extended ATAM-FET experiments:

```text
notebooks/ATAM FET enhanced.ipynb
notebooks/atamupdate2.ipynb
```

Supporting source files, evaluation utilities, and experiment configurations are included in their respective directories.

## Implementation Status

This branch contains both **executed experimental workflows** and **supporting implementations for extended experimentation**.

The presence of a module does not necessarily indicate that every corresponding experiment has been fully executed. Saved notebook outputs and the project tracking files should be used to verify experimental execution.

For additional implementation details, see:

* `PROJECT_STATUS.md`
* `PROGRESS_TRACKING.md`

