from pathlib import Path


def find_repo_root(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()

    for candidate in [start, *start.parents]:
        if (candidate / "notebooks").exists() and (candidate / "data").exists():
            return candidate

    raise FileNotFoundError("Could not find repository root.")


REPO_ROOT = find_repo_root()

DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_DIR = DATA_DIR / "external"

NOTEBOOKS_DIR = REPO_ROOT / "notebooks"
SRC_DIR = REPO_ROOT / "src"
DOCS_DIR = REPO_ROOT / "docs"
REPORTS_DIR = REPO_ROOT / "reports"
