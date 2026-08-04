"""Frozen train/validation/test splits (protocol Sec. 7.6, run-sheet step 6).

Sec. 6.2 requires the split to be written to disk once and read identically by
every model and every explanation. This module writes it; nothing else in the
codebase is allowed to partition data.

Two protocols, per Sec. 7.6:

* **temporal** (headline) — train on the earlier period, test on the later one.
  The honest choice for a claim about journeys unfolding over time: a model that
  only works when it has seen the future is not a model of the future.
* **grouped** (robustness) — a visitor never spans train and test, so a user's
  own history cannot leak across the boundary.

A naive random split by session is never produced. It is the third option only
in the sense that Sec. 7.6 names it to forbid it.

**The temporal split is grouped too.** Sec. 7.6 presents these as alternatives,
but a purely temporal cut still lets a returning visitor appear on both sides of
the boundary, which is precisely the identity leakage the grouped protocol
exists to prevent. Rather than choose which leak to accept, the temporal split
assigns each *user* to a period by their first session and cuts on that, so it
is temporal and identity-clean at once. The cost is a small band of users whose
sessions straddle the cut; they are dropped and counted (see
`SplitReport.n_straddling_users`).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl

from ..paths import SPLITS

Protocol = Literal["temporal", "grouped"]
Partition = Literal["train", "val", "test"]

#: Sec. 7.6 ratio.
DEFAULT_RATIOS: tuple[float, float, float] = (0.70, 0.15, 0.15)


class SplitError(ValueError):
    """Raised when a requested split would violate the protocol."""


@dataclass
class SplitReport:
    protocol: str
    seed: int | None
    ratios: tuple[float, float, float]
    n_sessions: int
    n_users: int
    n_train: int
    n_val: int
    n_test: int
    prevalence_train: float
    prevalence_val: float
    prevalence_test: float
    n_straddling_users: int
    boundary_train_val: str | None
    boundary_val_test: str | None

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, default=str), encoding="utf-8")
        return path

    def summary(self) -> str:
        return (
            f"{self.protocol} split: "
            f"train {self.n_train:,} ({self.prevalence_train:.4f}) | "
            f"val {self.n_val:,} ({self.prevalence_val:.4f}) | "
            f"test {self.n_test:,} ({self.prevalence_test:.4f})"
        )


def _validate_ratios(ratios: tuple[float, float, float]) -> None:
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise SplitError(f"ratios must sum to 1.0, got {ratios} summing to {sum(ratios)}")
    if any(r <= 0 for r in ratios):
        raise SplitError(f"every partition must be non-empty, got {ratios}")


def _session_table(
    sessions: pl.DataFrame,
    cutpoints: pl.DataFrame,
    *,
    session_column: str = "session_id",
    time_column: str = "event_time",
    user_column: str = "user_id",
) -> pl.DataFrame:
    """One row per session: user, start time, label."""
    per_session = sessions.group_by(session_column).agg(
        pl.first(user_column).alias("user_id"),
        pl.col(time_column).min().alias("session_start"),
    )
    return per_session.join(
        cutpoints.select([session_column, "label"]), on=session_column, how="inner"
    ).sort("session_start")


def temporal_split(
    sessions: pl.DataFrame,
    cutpoints: pl.DataFrame,
    *,
    ratios: tuple[float, float, float] = DEFAULT_RATIOS,
    **columns,
) -> tuple[pl.DataFrame, SplitReport]:
    """Earlier period trains, later period tests, with users kept whole.

    Users are ordered by the timestamp of their *first* session and cut at the
    ratio boundaries, so every session of a given user lands in one partition.
    Users whose sessions straddle a boundary in time are dropped rather than
    silently reassigned, because keeping them would either leak identity or
    break the temporal ordering.
    """
    _validate_ratios(ratios)
    table = _session_table(sessions, cutpoints, **columns)

    user_first = (
        table.group_by("user_id")
        .agg(
            pl.col("session_start").min().alias("first_seen"),
            pl.col("session_start").max().alias("last_seen"),
        )
        .sort("first_seen")
    )

    n_users = user_first.height
    n_train = int(round(n_users * ratios[0]))
    n_val = int(round(n_users * ratios[1]))

    assignment = user_first.with_columns(
        pl.when(pl.int_range(pl.len()) < n_train)
        .then(pl.lit("train"))
        .when(pl.int_range(pl.len()) < n_train + n_val)
        .then(pl.lit("val"))
        .otherwise(pl.lit("test"))
        .alias("partition")
    )

    # Time boundaries: the first_seen of the first user in each later block.
    boundary_train_val = (
        assignment.filter(pl.col("partition") == "val")["first_seen"].min()
        if n_val
        else None
    )
    boundary_val_test = assignment.filter(pl.col("partition") == "test")["first_seen"].min()

    # A user whose activity continues past the boundary of their own partition
    # would put later sessions into an earlier partition. Drop them.
    straddling = assignment.filter(
        (
            (pl.col("partition") == "train")
            & pl.lit(boundary_train_val is not None)
            & (pl.col("last_seen") >= pl.lit(boundary_train_val))
        )
        | (
            (pl.col("partition") == "val")
            & (pl.col("last_seen") >= pl.lit(boundary_val_test))
        )
    )
    n_straddling = straddling.height

    keep = assignment.join(straddling.select("user_id"), on="user_id", how="anti")
    assigned = table.join(keep.select(["user_id", "partition"]), on="user_id", how="inner")

    return assigned, _report(
        assigned,
        protocol="temporal",
        seed=None,
        ratios=ratios,
        n_users=n_users,
        n_straddling=n_straddling,
        boundary_train_val=str(boundary_train_val) if boundary_train_val else None,
        boundary_val_test=str(boundary_val_test) if boundary_val_test else None,
    )


def grouped_split(
    sessions: pl.DataFrame,
    cutpoints: pl.DataFrame,
    *,
    ratios: tuple[float, float, float] = DEFAULT_RATIOS,
    seed: int = 42,
    **columns,
) -> tuple[pl.DataFrame, SplitReport]:
    """Random by user, so a visitor never spans partitions (Sec. 7.6 robustness arm)."""
    _validate_ratios(ratios)
    table = _session_table(sessions, cutpoints, **columns)

    users = table["user_id"].unique().sort().to_numpy()
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(users)

    n_users = len(shuffled)
    n_train = int(round(n_users * ratios[0]))
    n_val = int(round(n_users * ratios[1]))

    partition = np.empty(n_users, dtype=object)
    partition[:n_train] = "train"
    partition[n_train : n_train + n_val] = "val"
    partition[n_train + n_val :] = "test"

    mapping = pl.DataFrame({"user_id": shuffled, "partition": partition})
    assigned = table.join(mapping, on="user_id", how="inner")

    return assigned, _report(
        assigned,
        protocol="grouped",
        seed=seed,
        ratios=ratios,
        n_users=n_users,
        n_straddling=0,
        boundary_train_val=None,
        boundary_val_test=None,
    )


def _report(
    assigned: pl.DataFrame,
    *,
    protocol: str,
    seed: int | None,
    ratios: tuple[float, float, float],
    n_users: int,
    n_straddling: int,
    boundary_train_val: str | None,
    boundary_val_test: str | None,
) -> SplitReport:
    def part(name: Partition) -> pl.DataFrame:
        return assigned.filter(pl.col("partition") == name)

    train, val, test = part("train"), part("val"), part("test")

    for name, frame in (("train", train), ("val", val), ("test", test)):
        if frame.height == 0:
            raise SplitError(f"{protocol} split produced an empty {name} partition")

    return SplitReport(
        protocol=protocol,
        seed=seed,
        ratios=ratios,
        n_sessions=assigned.height,
        n_users=n_users,
        n_train=train.height,
        n_val=val.height,
        n_test=test.height,
        prevalence_train=float(train["label"].mean()),
        prevalence_val=float(val["label"].mean()),
        prevalence_test=float(test["label"].mean()),
        n_straddling_users=n_straddling,
        boundary_train_val=boundary_train_val,
        boundary_val_test=boundary_val_test,
    )


def freeze_splits(
    sessions: pl.DataFrame,
    cutpoints: pl.DataFrame,
    *,
    suffix: str,
    directory: Path = SPLITS,
    ratios: tuple[float, float, float] = DEFAULT_RATIOS,
    seed: int = 42,
    overwrite: bool = False,
) -> dict[str, SplitReport]:
    """Write both split protocols to disk, once.

    Refuses to overwrite an existing split unless asked explicitly. A silently
    regenerated split would break the Sec. 6.2 guarantee that every model and
    explanation read the identical partition, and the breakage would be
    invisible in the results.
    """
    directory.mkdir(parents=True, exist_ok=True)
    reports: dict[str, SplitReport] = {}

    for protocol in ("temporal", "grouped"):
        target = directory / f"split_{suffix}_{protocol}.parquet"
        if target.exists() and not overwrite:
            raise SplitError(
                f"{target} already exists. The split is frozen (Sec. 6.2); every model and "
                "explanation must read the identical partition. Pass overwrite=True only if "
                "you intend to invalidate every result computed so far."
            )

        if protocol == "temporal":
            assigned, report = temporal_split(sessions, cutpoints, ratios=ratios)
        else:
            assigned, report = grouped_split(sessions, cutpoints, ratios=ratios, seed=seed)

        assigned.select(["session_id", "user_id", "partition", "label"]).write_parquet(target)
        report.write(directory / f"split_{suffix}_{protocol}.json")
        reports[protocol] = report

    return reports


def load_split(suffix: str, protocol: Protocol, *, directory: Path = SPLITS) -> pl.DataFrame:
    """Read a frozen split. The only sanctioned way to learn which rows are test."""
    target = directory / f"split_{suffix}_{protocol}.parquet"
    if not target.exists():
        raise FileNotFoundError(
            f"no frozen {protocol} split at {target}; run `funnel-shap freeze-splits` first"
        )
    return pl.read_parquet(target)
