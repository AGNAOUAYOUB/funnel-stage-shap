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

#: REES46's own open endpoint. Preferred over the Kaggle mirror named first in
#: Sec. 5.3, which requires API credentials: this is the originating publisher,
#: so the provenance chain is shorter.
DATASET_B_URL_TEMPLATE = "https://data.rees46.com/datasets/marketplace/{month}.csv.gz"
DATASET_B_SOURCE = "REES46 open datasets (https://data.rees46.com/datasets/marketplace/)"
DATASET_B_CITATION = (
    "REES46 Marketing Platform (M. Kechinov). eCommerce behavior data from a "
    "multi-category store. https://rees46.com/en/datasets"
)
#: Sec. 5.3 requires the licence to be confirmed as permitting research publication.
#: It could not be: Kaggle's metadata field reserves rights ("Data files (c) Original
#: Authors") while REES46 publishes the files as free and links academic work built on
#: them. No formal licence text exists. See the licence finding in data/raw/DATASETS.md.
DATASET_B_LICENCE = (
    "UNRESOLVED. Kaggle metadata states 'Data files (c) Original Authors' (a reservation "
    "of rights, not a grant); REES46 publishes them as 'free datasets' with no formal "
    "licence text and links an IEEE paper built on them. Obtain written confirmation from "
    "REES46 that academic use and publication are permitted, and replace this string with "
    "the reply verbatim before submission (Sec. 5.3)."
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

    lazy = pl.scan_parquet(source) if source.suffix == ".parquet" else pl.scan_csv(source)
    validate_event_columns(lazy.collect_schema().names())
    n_rows = lazy.select(pl.len()).collect().item()

    prov = Provenance(
        dataset="B",
        filename=str(source),
        source=DATASET_B_SOURCE,
        sha256=sha256_of(source),
        n_rows=int(n_rows),
        n_bytes=source.stat().st_size,
        retrieved_utc=_now(),
        licence=DATASET_B_LICENCE,
        citation=DATASET_B_CITATION,
    )
    prov.write(dest_dir)
    return prov


def convert_dataset_b_to_parquet(
    source: Path,
    *,
    dest: Path | None = None,
    chunk_rows: int = 2_000_000,
    overwrite: bool = False,
) -> Path:
    """Stream a (possibly gzipped) event CSV into Parquet.

    Polars' lazy CSV scan cannot stream a gzip member, and decompressing the
    month to plain CSV costs several times the disk of the Parquet it would
    become. Converting once up front keeps the Sec. 4.1 requirement (lazy
    Polars for sessionisation) satisfiable via `scan_parquet`, and makes every
    subsequent pass over the log dramatically cheaper.

    The read is chunked so peak memory stays bounded regardless of month size.
    """
    import gzip

    import pyarrow as pa
    import pyarrow.csv as pacsv
    import pyarrow.parquet as pq

    source = Path(source)
    if dest is None:
        name = source.name.removesuffix(".gz").removesuffix(".csv")
        dest = source.with_name(f"{name}.parquet")
    if dest.exists() and not overwrite:
        return dest

    opener = gzip.open if source.suffix == ".gz" else open
    writer: pq.ParquetWriter | None = None
    n_rows = 0

    try:
        with opener(source, "rb") as raw:
            reader = pacsv.open_csv(
                raw,
                read_options=pacsv.ReadOptions(block_size=1 << 24),
                convert_options=pacsv.ConvertOptions(
                    # event_time keeps its trailing " UTC"; sessionize.parse_event_time
                    # handles it, and letting Arrow guess produces nulls instead.
                    column_types={"event_time": pa.string()}
                ),
            )
            validate_event_columns(reader.schema.names)

            batch_buffer: list[pa.RecordBatch] = []
            buffered = 0
            for batch in reader:
                batch_buffer.append(batch)
                buffered += batch.num_rows
                if buffered >= chunk_rows:
                    table = pa.Table.from_batches(batch_buffer)
                    if writer is None:
                        writer = pq.ParquetWriter(dest, table.schema, compression="zstd")
                    writer.write_table(table)
                    n_rows += buffered
                    batch_buffer, buffered = [], 0

            if batch_buffer:
                table = pa.Table.from_batches(batch_buffer)
                if writer is None:
                    writer = pq.ParquetWriter(dest, table.schema, compression="zstd")
                writer.write_table(table)
                n_rows += buffered
    finally:
        if writer is not None:
            writer.close()

    if n_rows == 0:
        dest.unlink(missing_ok=True)
        raise SchemaError(f"{source} produced no rows")

    return dest


def scan_dataset_b(pattern: str | Path | None = None) -> pl.LazyFrame:
    """Lazily scan the Dataset-B event files (Sec. 4.1: Polars lazy API).

    Prefers the Parquet conversion when present, falling back to CSV. Returns a
    LazyFrame so sessionisation over hundreds of millions of events stays
    memory-bound rather than loading the whole log.
    """
    if pattern is None:
        parquet = sorted(DATASET_B_RAW.glob("*.parquet"))
        pattern = DATASET_B_RAW / "*.parquet" if parquet else DATASET_B_RAW / "*.csv"

    text = str(pattern)
    lazy = pl.scan_parquet(pattern) if text.endswith(".parquet") else pl.scan_csv(pattern)
    validate_event_columns(lazy.collect_schema().names())
    return lazy
