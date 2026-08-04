"""Tests for the frozen split protocols (protocol Sec. 6.2, 7.6).

A split bug is the second-worst kind after a leakage bug: it produces plausible
numbers that mean nothing. These tests pin the two properties that make the
split trustworthy — no user spans partitions, and the temporal split really is
ordered in time.
"""

from __future__ import annotations

import polars as pl
import pytest

from funnel_shap.data.splits import (
    SplitError,
    freeze_splits,
    grouped_split,
    load_split,
    temporal_split,
)


def _partition_users(assigned: pl.DataFrame) -> dict[str, set]:
    return {
        name: set(assigned.filter(pl.col("partition") == name)["user_id"].to_list())
        for name in ("train", "val", "test")
    }


# ---------------------------------------------------------------------------
# Identity leakage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("protocol", ["temporal", "grouped"])
def test_no_user_spans_partitions(sessionised, cutpoints, protocol) -> None:
    """Sec. 7.6's whole point: a visitor never appears on both sides."""
    fn = temporal_split if protocol == "temporal" else grouped_split
    assigned, _ = fn(sessionised, cutpoints)

    users = _partition_users(assigned)
    assert not users["train"] & users["val"]
    assert not users["train"] & users["test"]
    assert not users["val"] & users["test"]


@pytest.mark.parametrize("protocol", ["temporal", "grouped"])
def test_no_session_appears_twice(sessionised, cutpoints, protocol) -> None:
    fn = temporal_split if protocol == "temporal" else grouped_split
    assigned, _ = fn(sessionised, cutpoints)
    assert assigned["session_id"].n_unique() == assigned.height


# ---------------------------------------------------------------------------
# Temporal ordering
# ---------------------------------------------------------------------------


def test_temporal_split_is_actually_ordered_in_time(sessionised, cutpoints) -> None:
    """Train must end before test begins, or the "temporal" label is a lie."""
    assigned, _ = temporal_split(sessionised, cutpoints)

    train_end = assigned.filter(pl.col("partition") == "train")["session_start"].max()
    test_start = assigned.filter(pl.col("partition") == "test")["session_start"].min()
    assert train_end < test_start


def test_temporal_split_reports_dropped_straddling_users(sessionised, cutpoints) -> None:
    _, report = temporal_split(sessionised, cutpoints)
    assert report.n_straddling_users >= 0
    assert report.boundary_val_test is not None
    # Dropping must be accounted for, not silent.
    assert report.n_sessions <= cutpoints.height


def test_grouped_split_is_not_time_ordered(sessionised, cutpoints) -> None:
    """The robustness arm deliberately ignores time; confirm it differs."""
    grouped, _ = grouped_split(sessionised, cutpoints, seed=42)
    temporal, _ = temporal_split(sessionised, cutpoints)

    g = _partition_users(grouped)["test"]
    t = _partition_users(temporal)["test"]
    assert g != t


# ---------------------------------------------------------------------------
# Determinism and ratios
# ---------------------------------------------------------------------------


def test_grouped_split_is_seeded(sessionised, cutpoints) -> None:
    a, _ = grouped_split(sessionised, cutpoints, seed=42)
    b, _ = grouped_split(sessionised, cutpoints, seed=42)
    c, _ = grouped_split(sessionised, cutpoints, seed=101)

    assert a.sort("session_id").equals(b.sort("session_id"))
    assert not a.sort("session_id").equals(c.sort("session_id"))


def test_temporal_split_is_deterministic(sessionised, cutpoints) -> None:
    a, _ = temporal_split(sessionised, cutpoints)
    b, _ = temporal_split(sessionised, cutpoints)
    assert a.sort("session_id").equals(b.sort("session_id"))


@pytest.mark.parametrize("protocol", ["temporal", "grouped"])
def test_partition_sizes_roughly_match_ratios(sessionised, cutpoints, protocol) -> None:
    fn = temporal_split if protocol == "temporal" else grouped_split
    _, report = fn(sessionised, cutpoints)

    total = report.n_train + report.n_val + report.n_test
    # Sessions per user vary, so the session-level split only approximates the
    # user-level 70/15/15; a wide tolerance is correct here.
    assert 0.55 < report.n_train / total < 0.85
    assert report.n_val > 0 and report.n_test > 0


def test_every_partition_keeps_both_classes(sessionised, cutpoints) -> None:
    """A partition with no positives makes PR-AUC undefined."""
    for fn in (temporal_split, grouped_split):
        assigned, _ = fn(sessionised, cutpoints)
        for name in ("train", "val", "test"):
            part = assigned.filter(pl.col("partition") == name)
            assert part["label"].sum() > 0, f"{fn.__name__}/{name} has no positive sessions"
            assert (~part["label"]).sum() > 0


def test_bad_ratios_are_rejected(sessionised, cutpoints) -> None:
    with pytest.raises(SplitError, match="sum to 1.0"):
        grouped_split(sessionised, cutpoints, ratios=(0.7, 0.2, 0.2))
    with pytest.raises(SplitError, match="non-empty"):
        grouped_split(sessionised, cutpoints, ratios=(1.0, 0.0, 0.0))


# ---------------------------------------------------------------------------
# Freezing
# ---------------------------------------------------------------------------


def test_freeze_writes_both_protocols_and_refuses_to_overwrite(
    sessionised, cutpoints, tmp_path
) -> None:
    reports = freeze_splits(sessionised, cutpoints, suffix="t", directory=tmp_path)
    assert set(reports) == {"temporal", "grouped"}

    for protocol in ("temporal", "grouped"):
        assert (tmp_path / f"split_t_{protocol}.parquet").exists()
        assert (tmp_path / f"split_t_{protocol}.json").exists()

    # Sec. 6.2: a frozen split must not be silently regenerated.
    with pytest.raises(SplitError, match="frozen"):
        freeze_splits(sessionised, cutpoints, suffix="t", directory=tmp_path)

    freeze_splits(sessionised, cutpoints, suffix="t", directory=tmp_path, overwrite=True)


def test_load_split_roundtrips(sessionised, cutpoints, tmp_path) -> None:
    freeze_splits(sessionised, cutpoints, suffix="rt", directory=tmp_path)
    frame = load_split("rt", "temporal", directory=tmp_path)
    assert set(frame.columns) == {"session_id", "user_id", "partition", "label"}
    assert set(frame["partition"].unique()) == {"train", "val", "test"}


def test_load_missing_split_is_an_error(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="freeze-splits"):
        load_split("nope", "temporal", directory=tmp_path)


# ---------------------------------------------------------------------------
# Dataset A (amendment A12)
# ---------------------------------------------------------------------------


def _fake_dataset_a() -> pl.DataFrame:
    """Months in the real file's proportions, including the empty-val trap."""
    sizes = {
        "Feb": 184, "Mar": 1907, "May": 3364, "June": 288, "Jul": 432,
        "Aug": 433, "Sep": 448, "Oct": 549, "Nov": 2998, "Dec": 1727,
    }
    months, revenue = [], []
    for i, (month, n) in enumerate(sizes.items()):
        months += [month] * n
        # Rising prevalence, as in the real data.
        positives = int(n * (0.02 + 0.02 * i))
        revenue += ["TRUE"] * positives + ["FALSE"] * (n - positives)
    return pl.DataFrame({"Month": months, "Revenue": revenue})


def test_dataset_a_split_has_three_non_empty_partitions() -> None:
    """Row-count boundaries put both marks inside November and emptied val."""
    from funnel_shap.data.splits import dataset_a_split

    assigned, report = dataset_a_split(_fake_dataset_a())
    assert report.n_train > 0 and report.n_val > 0 and report.n_test > 0
    assert set(assigned["partition"].unique()) == {"train", "val", "test"}


def test_dataset_a_split_never_straddles_a_month() -> None:
    """A month in two partitions means the held-out period is not held out."""
    from funnel_shap.data.splits import dataset_a_split

    assigned, _ = dataset_a_split(_fake_dataset_a())
    per_month = assigned.group_by("Month").agg(
        pl.col("partition").n_unique().alias("n_partitions")
    )
    assert per_month["n_partitions"].max() == 1


def test_dataset_a_split_is_time_ordered() -> None:
    from funnel_shap.data.splits import DATASET_A_MONTH_ORDER, dataset_a_split

    order = {m: i for i, m in enumerate(DATASET_A_MONTH_ORDER)}
    assigned, _ = dataset_a_split(_fake_dataset_a())

    def months(name: str) -> set[int]:
        sub = assigned.filter(pl.col("partition") == name)
        return {order[m] for m in sub["Month"].unique().to_list()}

    assert max(months("train")) < min(months("val"))
    assert max(months("val")) < min(months("test"))


def test_dataset_a_split_is_deterministic() -> None:
    from funnel_shap.data.splits import dataset_a_split

    frame = _fake_dataset_a()
    a, _ = dataset_a_split(frame)
    b, _ = dataset_a_split(frame)
    assert a.equals(b)


def test_dataset_a_rejects_unknown_months() -> None:
    from funnel_shap.data.splits import dataset_a_split

    frame = pl.DataFrame({"Month": ["Jan"] * 10, "Revenue": ["TRUE"] * 10})
    with pytest.raises(SplitError, match="unexpected Month"):
        dataset_a_split(frame)
