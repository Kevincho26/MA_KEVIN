# Legacy pipeline notebooks

This document describes the current state of the legacy pipeline after the naming cleanup, notebook synchronization, and the ongoing migration of shared logic into `src/`.

## Purpose

The `notebooks/legacy/` directory preserves the original thesis workflow for:

- dataset construction
- exploratory analysis
- supervised modeling
- unsupervised modeling
- association-rule preparation

At the same time, the repository is gradually moving repeated notebook logic into reusable modules under `src/`.

---

## Scope

This document covers the notebooks located in:

```text
notebooks/legacy/
```

The `notebooks/archive/` directory remains outside the active documented workflow and should be treated as archival material.

---

## Notebook naming convention

### Data pipeline / preparation

- `01_build_base_dataset`
- `02_exploratory_data_analysis`
- `03_build_supervised_dataset`
- `04_build_unsupervised_dataset`
- `05_refine_unsupervised_dataset`
- `06_build_association_rules_dataset`

### Supervised models

- `10_logistic_regression`
- `11_decision_tree`
- `12_random_forest`
- `13_xgboost`

### Unsupervised models

- `20_dbscan`
- `21_hierarchical_clustering`
- `22_autoencoder_anomaly_detection`

Each notebook is intended to remain synchronized in both formats:

- `.py`
- `.ipynb`

---

## Canonical processed dataset names

| Previous name | Current name |
|---|---|
| `preprocessed.csv` | `base_dataset.csv` |
| `preprocessed_supervised.csv` | `supervised_modeling_dataset.csv` |
| `preprocessed_unsupervised.csv` | `unsupervised_base_dataset.csv` |
| `processed_unsupervised.csv` | `unsupervised_modeling_dataset.csv` |
| `processed_unsupervised_apriori.csv` | `association_rules_dataset.csv` |

These datasets are generated under:

```text
data/processed/
```

They should be treated as derived pipeline artifacts, not manually maintained source-of-truth files.

---

## Pipeline overview

### Dataset flow

```text
raw data
  ↓
01_build_base_dataset
  ↓
base_dataset.csv
  ├─→ 02_exploratory_data_analysis
  ├─→ 03_build_supervised_dataset
  │     ↓
  │   supervised_modeling_dataset.csv
  │     ├─→ 10_logistic_regression
  │     ├─→ 11_decision_tree
  │     ├─→ 12_random_forest
  │     └─→ 13_xgboost
  │
  └─→ 04_build_unsupervised_dataset
        ↓
      unsupervised_base_dataset.csv
        ├─→ 05_refine_unsupervised_dataset
        │     ↓
        │   unsupervised_modeling_dataset.csv
        │     ├─→ 20_dbscan
        │     ├─→ 21_hierarchical_clustering
        │     └─→ 22_autoencoder_anomaly_detection
        │
        └─→ 06_build_association_rules_dataset
              ↓
            association_rules_dataset.csv
```

---

## Notebook-by-notebook description

### 01_build_base_dataset

**Input**
- raw data from `data/raw/`

**Output**
- `data/processed/base_dataset.csv`

**Purpose**
- consolidate and prepare the base dataset used by the rest of the pipeline

**Note**
- this is the pipeline entry point and still reads raw source files directly

---

### 02_exploratory_data_analysis

**Input**
- `data/processed/base_dataset.csv`

**Output**
- no canonical dataset output

**Purpose**
- explore structure, distributions, relationships, and initial patterns in the base dataset

---

### 03_build_supervised_dataset

**Input**
- `data/processed/base_dataset.csv`

**Output**
- `data/processed/supervised_modeling_dataset.csv`

**Purpose**
- prepare the dataset used by supervised modeling notebooks

**Consumed by**
- `10_logistic_regression`
- `11_decision_tree`
- `12_random_forest`
- `13_xgboost`

---

### 04_build_unsupervised_dataset

**Input**
- `data/processed/base_dataset.csv`

**Output**
- `data/processed/unsupervised_base_dataset.csv`

**Purpose**
- prepare the intermediate dataset used by the unsupervised pipeline

---

### 05_refine_unsupervised_dataset

**Input**
- `data/processed/unsupervised_base_dataset.csv`

**Output**
- `data/processed/unsupervised_modeling_dataset.csv`

**Purpose**
- refine the unsupervised dataset for clustering and anomaly detection

**Consumed by**
- `20_dbscan`
- `21_hierarchical_clustering`
- `22_autoencoder_anomaly_detection`

---

### 06_build_association_rules_dataset

**Input**
- `data/processed/unsupervised_base_dataset.csv`

**Output**
- `data/processed/association_rules_dataset.csv`

**Purpose**
- transform the unsupervised base dataset into a binary association-rules dataset

---

## Recommended execution order

### Data pipeline

1. `01_build_base_dataset`
2. `02_exploratory_data_analysis`
3. `03_build_supervised_dataset`
4. `04_build_unsupervised_dataset`
5. `05_refine_unsupervised_dataset`
6. `06_build_association_rules_dataset`

### Supervised modeling

1. `10_logistic_regression`
2. `11_decision_tree`
3. `12_random_forest`
4. `13_xgboost`

### Unsupervised modeling

1. `20_dbscan`
2. `21_hierarchical_clustering`
3. `22_autoencoder_anomaly_detection`

---

## Reusable project modules under `src/`

As part of the migration away from notebook-local infrastructure, shared logic has been extracted into reusable modules.

### Repository paths

Shared repository path resolution is centralized in:

```text
src/utils/paths.py
```

This module provides reusable path objects such as:

- `REPO_ROOT`
- `DATA_DIR`
- `RAW_DIR`
- `INTERIM_DIR`
- `PROCESSED_DIR`
- `EXTERNAL_DIR`
- `NOTEBOOKS_DIR`
- `SRC_DIR`
- `DOCS_DIR`
- `REPORTS_DIR`

Legacy notebooks should import shared paths from this module instead of redefining local repository-root detection.

### Dataset loaders

Canonical processed datasets are loaded through:

```text
src/data/loaders.py
```

Current loaders include:

- `load_base_dataset()`
- `load_supervised_modeling_dataset()`
- `load_unsupervised_base_dataset()`
- `load_unsupervised_modeling_dataset()`
- `load_association_rules_dataset()`

This removes repeated manual reads of canonical processed CSVs.

### Preprocessing helpers

Reusable preprocessing helpers are currently centralized in:

```text
src/features/preprocessing.py
```

Current helpers include:

- `drop_columns_if_present()`
- `split_features_and_target()`
- `select_numeric_columns()`

These helpers are already used in several pipeline and supervised-model notebooks.

### Feature engineering helpers

Reusable feature-engineering helpers are now centralized in:

```text
src/features/engineering.py
```

Current helpers include:

- `expand_mapped_indicator_columns()`
- `bin_numeric_column_to_indicators()`
- `one_hot_encode_columns()`
- `encode_ordinal_columns()`

These helpers are intended to remove repeated notebook-local logic for:

- indicator expansion from coded categorical variables
- numeric binning into indicator columns
- one-hot encoding of contextual variables
- repeated ordinal encoding across multiple columns

### Feature exports

Shared feature helpers are re-exported through:

```text
src/features/__init__.py
```

This provides a cleaner public surface for the feature-related utilities currently extracted from the legacy notebooks.

### Model helpers

Reusable modeling helpers are now centralized in:

```text
src/models/
```

Current model-related modules include:

- `src/models/evaluation.py`
- `src/models/interpretation.py`
- `src/models/splitting.py`
- `src/models/__init__.py`

Current reusable helpers include:

- `evaluate_binary_classifier()`
- `prepare_feature_importance_df()`
- `plot_feature_importance()`
- `prepare_coefficient_df()`
- `plot_coefficients()`
- `split_supervised_data()`
- `scale_train_test()`

These helpers currently centralize repeated supervised-model notebook logic for:

- binary-classification evaluation outputs
- confusion matrix and ROC plotting
- feature-importance dataframe preparation
- feature-importance plotting and top-feature reporting
- coefficient dataframe preparation for logistic regression
- coefficient plotting and top-coefficient reporting
- supervised train/test splitting
- train/test scaling with shared fitted scaler state

### Model exports

Shared model helpers are re-exported through:

```text
src/models/__init__.py
```

This provides a cleaner public surface for model-related utilities already extracted from the supervised legacy notebooks.

---

## Current migration status

At this stage, the legacy workflow has already been partially migrated away from notebook-local infrastructure.

### Already centralized

- repository path resolution in `src/utils/paths.py`
- canonical processed dataset access in `src/data/loaders.py`
- selected preprocessing helpers in `src/features/preprocessing.py`
- initial feature-engineering helpers in `src/features/engineering.py`
- binary-classification evaluation helpers in `src/models/evaluation.py`
- feature-importance preparation and plotting helpers in `src/models/interpretation.py`
- coefficient preparation and plotting helpers in `src/models/interpretation.py`
- supervised split and scaling helpers in `src/models/splitting.py`

### Already applied in legacy notebooks

- notebooks consuming canonical processed datasets now use shared loaders
- several supervised notebooks use shared preprocessing helpers
- `06_build_association_rules_dataset` uses shared feature-engineering helpers
- `03_build_supervised_dataset` and `04_build_unsupervised_dataset` use shared ordinal-encoding logic
- `10_logistic_regression` reuses shared split, scaling, evaluation, and coefficient-interpretation helpers
- `11_decision_tree`, `12_random_forest`, and `13_xgboost` reuse shared split, evaluation, and feature-importance helpers

### Main exception

- `01_build_base_dataset` still reads raw source files directly because it is the pipeline entry point

---

## Jupytext and `.py` / `.ipynb` synchronization

Legacy notebooks are intended to remain synchronized in both representations:

- `.py`
- `.ipynb`

Synchronization is handled through Jupytext, while notebook cleanup is enforced through pre-commit hooks.

### Practical workflow

1. edit notebook or paired script
2. run the file if needed
3. run `git add`
4. commit
5. if hooks rewrite files, run `git add` again and repeat commit

### Tools involved

- `jupytext --sync`
- `nbstripout`
- `ruff format`
- `ruff check`
- whitespace / EOF hooks

### Useful commands

```bash
jupytext --sync notebooks/legacy/*.ipynb
pre-commit run --all-files
```

When hooks rewrite files, the normal flow is:

```bash
git add .
pre-commit run --all-files
git add .
git commit
```

---

## Running notebooks/scripts from terminal

When legacy notebook scripts are executed directly from the repository root, imports from `src` may require setting `PYTHONPATH` explicitly for the session.

### PowerShell example

```powershell
$env:PYTHONPATH = (Get-Location).Path
python .\notebooks\legacy\03_build_supervised_dataset.py
python .\notebooks\legacy\04_build_unsupervised_dataset.py
python .\notebooks\legacy\06_build_association_rules_dataset.py
```

This is useful when running notebook-paired `.py` files directly without packaging the repository as an installed module.

---

## Known limitations

This phase focused on structure, reuse, and stability of the legacy workflow. Several issues were intentionally left for later.

### Environment dependencies

- `tensorflow` may still be required for `22_autoencoder_anomaly_detection.py`
- `imblearn` may still be required for some supervised notebook variants that use SMOTE

### Technical cleanup still pending

- seaborn warnings in plotting code
- MKL / OpenMP / clustering warnings on Windows
- column names with encoding artifacts in some notebooks
- large notebooks with duplicated experiment sections
- training, tuning, and some model-specific visualization logic still mostly notebook-local

### Architecture still pending

The repository has already started centralizing `src/features` and `src/models`, but there is still room to continue extracting reusable logic into modules such as:

- `src/models/train.py`
- `src/models/tuning.py`

---

## Useful commands to resume work

### PowerShell

```powershell
git status
git log --oneline -10
git grep -n "pd.get_dummies\|pd.cut\|map(" -- "notebooks/legacy/*.py"
Get-ChildItem src
Get-ChildItem src\features
Get-ChildItem src\models
```

---

## Summary

The legacy pipeline is now in a more reusable and portable state in terms of:

- notebook naming
- dataset naming
- path portability
- synchronized notebook/script pairs
- reusable repository paths
- reusable dataset loaders
- reusable preprocessing helpers
- reusable feature-engineering helpers
- reusable model-evaluation helpers
- reusable feature-importance helpers
- reusable coefficient-interpretation helpers
- reusable supervised split and scaling helpers

This README documents that updated baseline for future migration work.
