# Legacy pipeline notebooks

This document describes the current state of the legacy pipeline after the stabilization, naming cleanup, notebook synchronization, and initial migration of shared utilities into `src/`.

## Purpose

The `notebooks/legacy/` directory preserves the original thesis workflow for:

- dataset construction
- exploratory analysis
- supervised modeling
- unsupervised modeling

During the recent cleanup and migration phases, the goals were to:

- professionalize notebook and dataset names
- remove absolute paths tied to a specific local machine
- stabilize `.py` / `.ipynb` synchronization with Jupytext
- centralize shared repository paths
- centralize canonical processed dataset loading
- leave a clearer baseline for future extraction of reusable logic into `src/`

---

## Scope

The main notebooks covered by this document are located in:

```text
notebooks/legacy/
```

The `notebooks/archive/` directory was left out of this phase and is not yet part of the formally documented workflow.

---

## Naming convention

The legacy notebooks follow this convention.

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

Each notebook keeps its synchronized pair in:

- `.py`
- `.ipynb`

---

## Current dataset names

These are the current canonical names of the datasets generated in the legacy pipeline.

| Previous name | Current name |
|---|---|
| `preprocessed.csv` | `base_dataset.csv` |
| `preprocessed_supervised.csv` | `supervised_modeling_dataset.csv` |
| `preprocessed_unsupervised.csv` | `unsupervised_base_dataset.csv` |
| `processed_unsupervised.csv` | `unsupervised_modeling_dataset.csv` |
| `processed_unsupervised_apriori.csv` | `association_rules_dataset.csv` |

The files are generated under:

```text
data/processed/
```

These datasets should be treated as **derived artifacts** of the pipeline, not as manually maintained source-of-truth files.

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
   │      ↓
   │   supervised_modeling_dataset.csv
   │      ├─→ 10_logistic_regression
   │      ├─→ 11_decision_tree
   │      ├─→ 12_random_forest
   │      └─→ 13_xgboost
   │
   └─→ 04_build_unsupervised_dataset
          ↓
       unsupervised_base_dataset.csv
          ├─→ 05_refine_unsupervised_dataset
          │      ↓
          │   unsupervised_modeling_dataset.csv
          │      ├─→ 20_dbscan
          │      ├─→ 21_hierarchical_clustering
          │      └─→ 22_autoencoder_anomaly_detection
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

---

### 02_exploratory_data_analysis

**Input**
- `data/processed/base_dataset.csv`

**Output**
- no canonical dataset output

**Purpose**
- explore structure, distributions, and initial patterns in the base dataset

---

### 03_build_supervised_dataset

**Input**
- `data/processed/base_dataset.csv`

**Output**
- `data/processed/supervised_modeling_dataset.csv`

**Purpose**
- build the final dataset used by the supervised modeling notebooks

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
- build an intermediate dataset oriented toward unsupervised analysis

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

**Note**
- some unsupervised notebooks also use `base_dataset.csv` as additional context for interpreting results

---

### 06_build_association_rules_dataset

**Input**
- `data/processed/unsupervised_base_dataset.csv`

**Output**
- `data/processed/association_rules_dataset.csv`

**Purpose**
- generate the dataset specifically used for association rules

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

## Portable paths

To avoid dependencies on absolute Windows paths, notebooks/scripts were migrated to a portable pattern based on `Path` and repository root detection.

Reference pattern:

```python
from pathlib import Path

def find_repo_root(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "notebooks").exists() and (candidate / "data").exists():
            return candidate
    raise FileNotFoundError("Could not find repository root.")

REPO_ROOT = find_repo_root()
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
RAW_DIR = REPO_ROOT / "data" / "raw"
```

This pattern should be preserved in future notebooks and scripts that still depend on hardcoded paths.

---

## Reusable project modules

As part of the legacy-to-`src/` migration, some shared project utilities were extracted from notebooks into reusable Python modules.

### Repository paths

Shared repository paths are now centralized in:

```text
src/utils/paths.py
```

This module provides reusable path objects such as:

- `REPO_ROOT`
- `RAW_DIR`
- `PROCESSED_DIR`
- `DOCS_DIR`
- `NOTEBOOKS_DIR`

Legacy notebooks should import these paths instead of redefining local path discovery logic.

### Dataset loaders

Canonical processed datasets are now loaded through:

```text
src/data/loaders.py
```

This module currently provides:

- `load_base_dataset()`
- `load_supervised_modeling_dataset()`
- `load_unsupervised_base_dataset()`
- `load_unsupervised_modeling_dataset()`
- `load_association_rules_dataset()`

This reduces repeated `pd.read_csv(...)` calls and centralizes processed dataset access.

### Current migration status

At this stage, legacy notebooks that consume canonical processed datasets have been updated to use shared loaders.

As a result:

- repository path resolution is centralized in `src/utils/paths.py`
- canonical processed dataset loading is centralized in `src/data/loaders.py`

The main exception is `01_build_base_dataset`, which still reads raw source files directly because it is the pipeline entry point that generates `base_dataset.csv`.

---

## Jupytext and `.py` / `.ipynb` synchronization

The legacy notebooks are intended to remain synchronized in both formats:

- `.py`
- `.ipynb`

Synchronization is handled with Jupytext.

### Practical workflow

1. edit file
2. run hooks / `pre-commit`
3. run `git add` again if hooks modified files
4. commit

### Tools involved

- `jupytext --sync`
- `nbstripout`
- `ruff format`
- `ruff check`
- whitespace / EOF hooks

### Useful commands

```bash
jupytext --sync notebooks/legacy/*.py
pre-commit run --all-files
```

When hooks rewrite files, the typical flow is:

```bash
git add .
pre-commit run --all-files
git add .
git commit
```

---

## Current notebook/script pair status

During this phase, missing `.ipynb` pairs were created and all relevant legacy notebooks were aligned with Jupytext.

The goal is for all relevant legacy notebooks to remain consistent with Jupytext.

---

## Known limitations

This phase focused on naming, structure, and stability of the legacy workflow, plus initial extraction of shared utilities into `src/`. Several technical issues were intentionally left for later, since they do not block this documented state.

### Environment dependencies

- `tensorflow` missing in `22_autoencoder_anomaly_detection.py`
- `imblearn` missing in `10_logistic_regression.py`

### Warnings and technical cleanup

- seaborn warnings caused by using `palette` without `hue`
- MKL / OpenMP / `KMeans` warnings on Windows
- column names with encoding artifacts such as `_Î´`
- large notebooks containing duplicated experiment versions

### Architecture

- reusable logic has not yet been seriously migrated into `src/features/` or `src/models/`
- there is still no full centralization of preprocessing, feature engineering, or training/evaluation helpers

### Pending archive status

- `notebooks/archive/` still has no formal analytical role in the active workflow
- its current purpose is archival/documentary only

---

## Recommended next phase

The natural next step after this document is to continue the gradual migration of reusable logic into `src/`.

A reasonable next sequence would be:

- `src/features/preprocessing.py`
- `src/features/engineering.py`
- `src/models/train.py`
- `src/models/evaluate.py`

At the same time, future cleanup may include:

1. deciding the long-term status of `notebooks/archive/`
2. refining environment reproducibility
3. addressing warnings and duplicated experimental code
4. moving stable feature engineering and preprocessing logic out of notebooks

---

## Useful commands to resume work

### PowerShell

```powershell
git status
git log --oneline -10
Get-ChildItem notebooks\legacy
Get-ChildItem data\processed
```

---

## Summary

The legacy pipeline is now in a more stable and professional state in terms of:

- notebook naming
- dataset naming
- path portability
- Jupytext consistency
- reusable repository paths
- reusable dataset loaders
- basic traceability between pipeline stages

This README documents that state as the baseline before further migration of reusable preprocessing, feature engineering, and modeling logic into `src/`.
