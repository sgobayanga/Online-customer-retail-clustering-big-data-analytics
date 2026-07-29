"""Project settings and file paths for the Online Retail analysis."""

from __future__ import annotations

import os
from pathlib import Path

# Limit numerical libraries to one worker thread. This makes execution more
# predictable on Windows and avoids occasional K-Means/OpenBLAS thread stalls.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHART_DIR = OUTPUT_DIR / "charts"
MODEL_DIR = OUTPUT_DIR / "models"

DATA_FILE_NAME = "Online Retail.xlsx"
RANDOM_STATE = 42
CHOSEN_K = 4

# Set to True when you want every graph to open in a window while the program
# runs. Leave it False when you only want the graphs saved in outputs/charts.
SHOW_PLOTS = False

REQUIRED_COLUMNS = [
    "InvoiceNo",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID",
    "Country",
]


def create_output_folders() -> None:
    """Create all folders needed by the analysis."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def resolve_data_path(command_line_path: str | None = None) -> Path:
    """
    Find the Excel dataset.

    Search order:
    1. A path supplied with --data.
    2. The ONLINE_RETAIL_DATA environment variable.
    3. data/Online Retail.xlsx.
    4. Online Retail.xlsx in the project root.
    """
    candidates: list[Path] = []

    if command_line_path:
        candidates.append(Path(command_line_path).expanduser())

    environment_path = os.getenv("ONLINE_RETAIL_DATA")
    if environment_path:
        candidates.append(Path(environment_path).expanduser())

    candidates.extend(
        [
            DATA_DIR / DATA_FILE_NAME,
            PROJECT_ROOT / DATA_FILE_NAME,
        ]
    )

    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists() and resolved.is_file():
            return resolved

    checked = "\n".join(f"  - {path.resolve()}" for path in candidates)
    raise FileNotFoundError(
        "The dataset could not be found.\n\n"
        f"Copy '{DATA_FILE_NAME}' into the project's data folder:\n"
        f"  {DATA_DIR}\n\n"
        "Alternatively run main.py with:\n"
        '  python main.py --data "C:\\path\\to\\Online Retail.xlsx"\n\n'
        f"Locations checked:\n{checked}"
    )
