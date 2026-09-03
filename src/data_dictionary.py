"""
Dataset discovery, loading and data-dictionary reconciliation.

Covers spec sections 2 (use the supplied dictionary), 44 (automatic data discovery) and
part of 50 (upload validation).

Three real-world problems this module exists to solve, all confirmed against the actual
file shipped in this repository:

1. The CSV is encoded ISO-8859-1, not UTF-8. Reading it as UTF-8 raises UnicodeDecodeError
   on the Spanish place names (Japon, Turquia, Mexico, ...).
2. The CSV column is `shipping date (DateOrders)` with a lowercase "s", while the data
   dictionary - and most write-ups about this dataset - use `Shipping date (DateOrders)`.
   A literal lookup raises KeyError, so all column access goes through resolve().
3. Both CSVs are tracked with Git LFS. Without a successful `git lfs pull` they are either
   ~130-byte pointer files - where pandas would fail with a confusing parser error - or
   absent entirely. `git lfs pull` itself fails for reasons re-running it will not fix
   (bandwidth quota, authentication, objects never pushed), so every failure here reports
   where the loader looked and offers the fallbacks that do not need the git-lfs client.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

import config

ENCODING = "latin-1"
LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"

# GitHub resolves LFS content on the /raw/ path, so a browser download still works when the
# git-lfs client cannot fetch it (a wrong credential, or no git-lfs installed at all).
LFS_BROWSER_BASE = "https://github.com/pr4t3ek/del/raw/main/"


class DatasetError(RuntimeError):
    """Raised when the dataset cannot be loaded, with a user-facing explanation."""


# --------------------------------------------------------------------------------------
# Column-name resolution
# --------------------------------------------------------------------------------------
def _norm(name: str) -> str:
    """Normalise a column name for comparison: casefold, collapse internal whitespace."""
    return re.sub(r"\s+", " ", str(name)).strip().lower()


class ColumnResolver:
    """
    Case- and whitespace-insensitive column lookup.

    Lets the rest of the codebase refer to `Shipping date (DateOrders)` (the dictionary
    spelling) even though the CSV header says `shipping date (DateOrders)`.
    """

    def __init__(self, columns) -> None:
        self._map = {_norm(c): c for c in columns}
        self.columns = list(columns)

    def resolve(self, name: str) -> str | None:
        """Return the actual column name in the frame, or None if absent."""
        return self._map.get(_norm(name))

    def has(self, name: str) -> bool:
        return self.resolve(name) is not None

    def require(self, name: str) -> str:
        actual = self.resolve(name)
        if actual is None:
            raise DatasetError(f"Required column '{name}' is not present in the dataset.")
        return actual

    def resolve_many(self, names) -> list[str]:
        """Resolve a list of names, silently dropping any that are absent."""
        out = []
        for n in names:
            actual = self.resolve(n)
            if actual is not None and actual not in out:
                out.append(actual)
        return out


# --------------------------------------------------------------------------------------
# File discovery and loading
# --------------------------------------------------------------------------------------
def find_file(filename: str, search_path=None) -> Path | None:
    """Locate a data file across the configured search path."""
    for directory in (search_path or config.DATA_SEARCH_PATH):
        candidate = Path(directory) / filename
        if candidate.is_file():
            return candidate
    return None


def is_lfs_pointer(path: Path) -> bool:
    """True when `path` holds a Git LFS pointer rather than the real file contents."""
    try:
        with open(path, "rb") as fh:
            return fh.read(len(LFS_POINTER_PREFIX)) == LFS_POINTER_PREFIX
    except OSError:                                       # pragma: no cover - defensive
        return False


def dataset_available(filename: str | None = None) -> bool:
    """True when the real file - not an unfetched LFS pointer - is on disk somewhere."""
    path = find_file(filename or config.DATASET_FILENAME)
    return path is not None and not is_lfs_pointer(path)


def search_report(filename: str, search_path=None) -> list[tuple[Path, str]]:
    """Describe what each searched directory held, so error messages can show the trail."""
    report = []
    for directory in (search_path or config.DATA_SEARCH_PATH):
        directory = Path(directory)
        candidate = directory / filename
        if not directory.is_dir():
            status = "directory does not exist"
        elif not candidate.is_file():
            status = "no match"
        elif is_lfs_pointer(candidate):
            status = "Git LFS pointer, not fetched"
        else:
            status = "found"
        report.append((directory, status))
    return report


def _recovery_steps(filename: str) -> str:
    """The three ways to obtain a data file, shared by every dataset error message.

    `git lfs pull` is first but cannot be the only answer: it fails for reasons the user
    cannot fix by running it again - LFS bandwidth quota, authentication, or objects that
    were never pushed - so the fallbacks that do not involve the git-lfs client matter.
    """
    return (
        "How to fix, in order of preference:\n"
        "\n"
        "  1. Fetch it with Git LFS, from the repository root:\n"
        "         git lfs install\n"
        "         git lfs pull\n"
        "     This needs a real clone. If `git remote -v` prints nothing, this folder\n"
        "     was never cloned from GitHub and can never fetch the data - re-clone it.\n"
        "\n"
        "  2. If that fails (LFS bandwidth quota, authentication, or the objects were\n"
        "     never pushed), download the file in a browser from\n"
        f"         {LFS_BROWSER_BASE}{filename}\n"
        f"     and save it as  {config.DATA_DIR / filename}\n"
        "     Both CSVs come from the public DataCo Smart Supply Chain dataset, so\n"
        '     searching Kaggle for "DataCo Smart Supply Chain for Big Data Analysis"\n'
        "     is a further fallback.\n"
        "\n"
        "  3. Or start the application and upload it in the browser:\n"
        "         python app.py     ->  http://127.0.0.1:5000/upload"
    )


def _missing_file_message(filename: str) -> str:
    """Explain a file that is absent everywhere, including where the loader looked."""
    lines = [f"{filename} was not found.", "", "Searched, in order:"]
    for directory, status in search_report(filename):
        lines.append(f"  {directory}  ->  {status}")
    lines += [
        "",
        "This repository tracks *.csv with Git LFS, so a plain clone or a \"Download ZIP\"",
        "leaves the file either absent or present only as a ~130-byte pointer.",
        "",
        _recovery_steps(filename),
    ]
    return "\n".join(lines)


def _check_not_lfs_pointer(path: Path) -> None:
    if not is_lfs_pointer(path):
        return
    raise DatasetError(
        f"'{path.name}' is a Git LFS pointer file, not the real data "
        f"({path.stat().st_size} bytes).\n"
        f"Found at: {path}\n"
        "\n"
        + _recovery_steps(path.name)
    )


def load_raw_dataset(path: Path | None = None) -> pd.DataFrame:
    """Load the DataCo dataset with the correct encoding, or raise DatasetError."""
    path = Path(path) if path else find_file(config.DATASET_FILENAME)
    if path is None:
        raise DatasetError(_missing_file_message(config.DATASET_FILENAME))
    _check_not_lfs_pointer(path)
    try:
        return pd.read_csv(path, encoding=ENCODING, low_memory=False)
    except UnicodeDecodeError as exc:                     # pragma: no cover - defensive
        raise DatasetError(
            f"Could not decode '{path.name}'. The DataCo file is ISO-8859-1 encoded; "
            f"reading it as UTF-8 fails on accented place names. ({exc})"
        ) from exc


def load_dictionary(path: Path | None = None) -> pd.DataFrame | None:
    """Load the supplied data dictionary. Returns None if it is unavailable."""
    path = Path(path) if path else find_file(config.DICTIONARY_FILENAME)
    if path is None:
        return None
    try:
        _check_not_lfs_pointer(path)
        dd = pd.read_csv(path, encoding=ENCODING)
    except Exception:                                     # pragma: no cover - defensive
        return None
    dd.columns = [str(c).strip().upper() for c in dd.columns]
    if "FIELDS" not in dd.columns:
        return None
    dd = dd.rename(columns={"FIELDS": "field", "DESCRIPTION": "description"})
    dd["field"] = dd["field"].astype(str).str.strip()
    if "description" in dd.columns:
        # Dictionary descriptions are stored as ":  text"; strip the leading colon.
        dd["description"] = (
            dd["description"].astype(str).str.strip().str.lstrip(":").str.strip()
        )
    else:
        dd["description"] = ""
    return dd[["field", "description"]]


# --------------------------------------------------------------------------------------
# Column classification (spec section 44)
# --------------------------------------------------------------------------------------

# Fields that reveal the delivery outcome, or are the target itself. Screened out of every
# predictive feature set - see src/leakage.py for the reasoning presented to the user.
LEAKAGE_FIELDS = [
    "Late_delivery_risk",
    "Delivery Status",
    "Days for shipping (real)",
    "Shipping date (DateOrders)",
]

# Direct identifiers and personal data. Never used as predictors (spec section 42).
PII_FIELDS = [
    "Customer Email",
    "Customer Password",
    "Customer Fname",
    "Customer Lname",
    "Customer Street",
    "Customer Zipcode",
]

IDENTIFIER_FIELDS = [
    "Order Id",
    "Order Item Id",
    "Customer Id",
    "Order Customer Id",
    "Product Card Id",
    "Order Item Cardprod Id",
    "Category Id",
    "Product Category Id",
    "Department Id",
    "Order Zipcode",
    "Product Image",
]

ECONOMIC_FIELDS = [
    "Order Profit Per Order",
    "Benefit per order",
    "Order Item Profit Ratio",
    "Sales",
    "Order Item Total",
    "Sales per customer",
    "Order Item Discount",
    "Order Item Discount Rate",
]

# Column pairs that are byte-for-byte identical in this dataset. The first element is
# kept, the second dropped. Verified across all 180,519 rows.
DUPLICATE_PAIRS = [
    ("Order Item Total", "Sales per customer"),
    ("Order Profit Per Order", "Benefit per order"),
    ("Category Id", "Product Category Id"),
    ("Customer Id", "Order Customer Id"),
    ("Product Card Id", "Order Item Cardprod Id"),
    ("Product Price", "Order Item Product Price"),
]


def classify_columns(df: pd.DataFrame) -> dict:
    """
    Auto-classify every column into its analytical role.

    Nothing here is hardcoded to a fixed column list: each candidate is resolved against
    the frame that was actually loaded, so a dataset missing some fields degrades
    gracefully instead of raising (spec section 44).
    """
    r = ColumnResolver(df.columns)

    target = r.resolve(config.TARGET)
    decision = r.resolve(config.DECISION_VARIABLE)

    leakage = r.resolve_many(LEAKAGE_FIELDS)
    pii = r.resolve_many(PII_FIELDS)
    identifiers = r.resolve_many(IDENTIFIER_FIELDS)
    economic = r.resolve_many(ECONOMIC_FIELDS)

    # Constant or all-null columns carry no information at all.
    dead = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]

    excluded = set(leakage) | set(pii) | set(identifiers) | set(dead)
    if target:
        excluded.add(target)

    predictors = [c for c in df.columns if c not in excluded]
    numeric = [c for c in predictors if pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in predictors if c not in numeric]

    return {
        "resolver": r,
        "target": target,
        "decision": decision,
        "leakage": leakage,
        "pii": pii,
        "identifiers": identifiers,
        "economic": economic,
        "dead": dead,
        "predictors": predictors,
        "numeric": numeric,
        "categorical": categorical,
    }


def reconcile_with_dictionary(df: pd.DataFrame, dictionary: pd.DataFrame | None) -> dict:
    """
    Compare actual CSV columns against the supplied dictionary (spec section 44).

    On the file shipped here this surfaces two genuine discrepancies:
      * `Order Zipcode` is present in the CSV but undocumented in the dictionary.
      * The CSV spells `shipping date (DateOrders)` in lowercase; the dictionary uses
        `Shipping date (DateOrders)`.
    """
    if dictionary is None:
        return {
            "available": False,
            "undocumented": [],
            "missing_from_csv": [],
            "spelling_mismatches": [],
            "n_documented": 0,
            "n_actual": len(df.columns),
        }

    csv_map = {_norm(c): c for c in df.columns}
    dict_map = {_norm(f): f for f in dictionary["field"]}

    undocumented = [csv_map[k] for k in csv_map if k not in dict_map]
    missing = [dict_map[k] for k in dict_map if k not in csv_map]
    mismatches = [
        {"csv": csv_map[k], "dictionary": dict_map[k]}
        for k in csv_map
        if k in dict_map and csv_map[k] != dict_map[k]
    ]

    return {
        "available": True,
        "undocumented": sorted(undocumented),
        "missing_from_csv": sorted(missing),
        "spelling_mismatches": mismatches,
        "n_documented": len(dict_map),
        "n_actual": len(df.columns),
    }
