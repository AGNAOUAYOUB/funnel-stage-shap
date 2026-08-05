"""Tests for seeded session subsampling (protocol Sec. 5.3, Appendix C)."""

from __future__ import annotations

import polars as pl
import pytest

from funnel_shap.data.subsample import MIN_SESSIONS, SubsampleError, subsample_users


def _sessions(n_users: int = 40_000, sessions_per_user: int = 3) -> pl.DataFrame:
    rows = []
    for user in range(n_users):
        for s in range(sessions_per_user):
            rows.append({"user_id": user, "session_id": f"{user}_{s}", "event_type": "view"})
            rows.append({"user_id": user, "session_id": f"{user}_{s}", "event_type": "cart"})
    return pl.DataFrame(rows)


@pytest.fixture(scope="module")
def sessions() -> pl.DataFrame:
    return _sessions()


def test_users_are_kept_whole(sessions) -> None:
    """Sampling sessions instead of users would break the Sec. 7.6 split guarantee."""
    sampled, _ = subsample_users(sessions, n_users=20_000, seed=42)

    before = sessions.group_by("user_id").agg(pl.col("session_id").n_unique().alias("n"))
    after = sampled.group_by("user_id").agg(pl.col("session_id").n_unique().alias("n"))
    merged = after.join(before, on="user_id", suffix="_before")

    assert (merged["n"] == merged["n_before"]).all()


def test_sample_is_deterministic(sessions) -> None:
    a, _ = subsample_users(sessions, n_users=20_000, seed=42)
    b, _ = subsample_users(sessions, n_users=20_000, seed=42)
    assert set(a["user_id"].unique()) == set(b["user_id"].unique())


def test_different_seed_gives_a_different_sample(sessions) -> None:
    a, _ = subsample_users(sessions, n_users=20_000, seed=42)
    b, _ = subsample_users(sessions, n_users=20_000, seed=101)
    assert set(a["user_id"].unique()) != set(b["user_id"].unique())


def test_sample_is_independent_of_row_order(sessions) -> None:
    """A sample that depends on iteration order is not reproducible."""
    shuffled = sessions.sample(fraction=1.0, shuffle=True, seed=7)
    a, _ = subsample_users(sessions, n_users=20_000, seed=42)
    b, _ = subsample_users(shuffled, n_users=20_000, seed=42)
    assert set(a["user_id"].unique()) == set(b["user_id"].unique())


def test_larger_sample_is_a_superset_at_the_same_seed(sessions) -> None:
    """Makes a sample-size sensitivity check cheap and interpretable."""
    small, _ = subsample_users(sessions, n_users=20_000, seed=42)
    large, _ = subsample_users(sessions, n_users=30_000, seed=42)
    assert set(small["user_id"].unique()).issubset(set(large["user_id"].unique()))


def test_requesting_more_users_than_exist_keeps_everything(sessions) -> None:
    sampled, report = subsample_users(sessions, n_users=10_000_000, seed=42)
    assert sampled.height == sessions.height
    assert report.n_users_after == report.n_users_before


def test_sampling_below_the_protocol_floor_is_refused(sessions) -> None:
    with pytest.raises(SubsampleError, match=f"{MIN_SESSIONS:,}"):
        subsample_users(sessions, n_users=100, seed=42)


def test_report_records_what_happened(sessions) -> None:
    _, report = subsample_users(sessions, n_users=20_000, seed=42)

    assert report.seed == 42
    assert report.target_users == 20_000
    assert report.n_users_after == 20_000
    assert report.n_sessions_after < report.n_sessions_before
    assert report.n_events_after < report.n_events_before
    assert "20,000" in report.summary()
