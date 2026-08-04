"""Dataset ingestion into data/raw/ (protocol Sec. 5, run-sheet step 2).

Both datasets are public but are *not* fetched implicitly by any downstream
stage. Fetching writes a provenance record (URL, SHA-256, row count, retrieval
date, licence) next to the file so the DVC-tracked raw layer is auditable.

Dataset B is deliberately not auto-downloaded: the REES46 archive is
multi-gigabyte and sits behind Kaggle's authenticated API with a licence that
Sec. 5.3 requires you to confirm by hand before use. `register_dataset_b` takes
a path you already downloaded and records its provenance.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import polars as pl

from ..paths import DATASET_A_RAW, DATASET_B_RAW, RAW
from .schema import (
    DATASET_A_CATEGORICAL,
    DATASET_A_N_ROWS,
    DATASET_A_NUMERIC,
    DATASET_A_TARGET,
    SchemaError,
    validate_event_columns,
)

#: UCI ML Repository id 468 (Sakar et al. 2018). The .../static/public/ path is
#: the repository's stable archive endpoint.
DATASET_A_URL = "https://archive.ics.uci.edu/static/public/468/online+shoppers+purchasing+intention+dataset.zip"
DATASET_A_CITATION = (
    "Sakar, C.O., Polat, S.O., Katircioglu, M., Kastro, Y. (2019). Real-time prediction of "
    "online shoppers' purchasing intention using multilayer perceptron and LSTM recurrent "
    "neural networks. Neural Computing and Applications 31, 6893-6908."
)
DATASET_A_LICENCE = "CC BY 4.0 (UCI Machine Learning Repository)"

DATASET_B_CITATION = (
    "REES46 Marketing Platform. eCommerce behavior data from a multi-category store. "
    "Published via Kaggle."
)


@dataclass
class Provenance:
    dataset: str
    filename: str
    source: str
    sha256: str
    n_rows: int
    n_bytes: int
    retrieved_utc: str
    licence: str
    citation: str

    def write(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        out = directory / f"provenance_{self.dataset}.json"
        out.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return out


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


# --------------------------------------------------------------------------
# Dataset A
# --------------------------------------------------------------------------


def fetch_dataset_a(*, dest: Path = DATASET_A_RAW, overwrite: bool = False) -> Path:
    """Download and unpack the UCI Online Shoppers CSV into data/raw/.

    Network access happens only here, and only when called explicitly.
    """
    import io
    import zipfile

    import requests

    if dest.exists() and not overwrite:
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(DATASET_A_URL, timeout=120)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if len(names) != 1:
            raise SchemaError(f"expected exactly one CSV in the UCI archive, found {names}")
        dest.write_bytes(zf.read(names[0]))

    frame = load_dataset_a(dest)
    Provenance(
        dataset="A",
        filename=dest.name,
        source=DATASET_A_URL,
        sha256=sha256_of(dest),
        n_rows=frame.height,
        n_bytes=dest.stat().st_size,
        retrieved_utc=_now(),
        licence=DATASET_A_LICENCE,
        citation=DATASET_A_CITATION,
    ).write(RAW)
    return dest


def load_dataset_a(path: Path = DATASET_A_RAW) -> pl.DataFrame:
    """Load and schema-check Dataset A (Sec. 5.2, Sec. 7.1 step 1)."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset A not found at {path}. Run `funnel-shap fetch-a` first."
        )

    # The duration columns are integral for the first few thousand rows, so
    # Polars' schema inference types them i64 and then fails on the first
    # fractional value. Pin them to Float64 rather than widening the inference
    # window, which would only move the failure.
    frame = pl.read_csv(
        path, schema_overrides={c: pl.Float64 for c in DATASET_A_NUMERIC}
    )
    expected = set(DATASET_A_NUMERIC) | set(DATASET_A_CATEGORICAL) | {DATASET_A_TARGET}
    missing = expected - set(frame.columns)
    if missing:
        raise SchemaError(f"Dataset A is missing columns {sorted(missing)}")

    if frame.height != DATASET_A_N_ROWS:
        # Not fatal -- a mirror may differ -- but it must be visible, because
        # Sec. 5.2 pins the expected shape.
        import warnings

        warnings.warn(
            f"Dataset A has {frame.height} rows, protocol Sec. 5.2 expects "
            f"{DATASET_A_N_ROWS}. Check the mirror you downloaded from.",
            stacklevel=2,
        )
    return frame


# --------------------------------------------------------------------------
# Dataset B
# --------------------------------------------------------------------------


def register_dataset_b(source: Path, *, dest_dir: Path = DATASET_B_RAW) -> Provenance:
    """Record provenance for an event-log file you downloaded yourself.

    Does not copy multi-GB files; it validates the schema, hashes the file in
    place and writes the provenance record required by Sec. 5.3.
    """
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(source)

    columns = pl.scan_csv(source).collect_schema().names()
    validate_event_columns(columns)
    n_rows = pl.scan_csv(source).select(pl.len()).collect().item()

    prov = Provenance(
        dataset="B",
        filename=str(source),
        source="Kaggle: eCommerce behavior data from a multi-category store (REES46)",
        sha256=sha256_of(source),
        n_rows=int(n_rows),
        n_bytes=source.stat().st_size,
        retrieved_utc=_now(),
        licence="CONFIRM BEFORE PUBLICATION -- Sec. 5.3 requires the Kaggle licence "
        "to be checked and recorded verbatim here.",
        citation=DATASET_B_CITATION,
    )
    prov.write(dest_dir)
    return prov


def scan_dataset_b(pattern: str | Path = None) -> pl.LazyFrame:
    """Lazily scan the Dataset-B event files (Sec. 4.1: Polars lazy API).

    Returns a LazyFrame so sessionisation over millions of events stays
    memory-bound rather than loading the whole log.
    """
    if pattern is None:
        pattern = DATASET_B_RAW / "*.csv"
    lazy = pl.scan_csv(pattern, try_parse_dates=False)
    validate_event_columns(lazy.collect_schema().names())
    return lazy
