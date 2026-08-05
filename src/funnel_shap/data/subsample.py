"""Seeded session subsampling for tractability (protocol Sec. 5.3, Appendix C).

Sec. 5.3 sets the requirement at ">=52k sessions" and explicitly permits
subsampling "for tractability, documented and seeded". October 2019 alone yields
5,366,181 sessions after filtering -- roughly 100x the floor -- so the binding
constraint is compute, not evidence. Appendix C budgets TimeSHAP at ~1-5k
sequences, which the full month would swamp by three orders of magnitude.

**Sampling is by user, never by session.** Both split protocols (Sec. 7.6) keep
a visitor whole; sampling sessions independently would tear users apart before
the splitter ever sees them, silently breaking the identity-leakage guarantee it
is there to provide. Sampling users keeps every sampled visitor's full journey
history intact.

The sample is a deterministic function of (seed, user id) via a hash, not of the
iteration order, so it is stable across reruns, Polars versions and row
orderings -- and a larger sample is a strict superset of a smaller one at the
same seed, which makes a sensitivity check cheap.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

#: Sec. 5.3's floor. Sampling below this is a protocol violation, not a
#: tractability decision.
MIN_SESSIONS = 52_000


class SubsampleError(ValueError):
    """Raised when a subsample would violate the protocol."""


@dataclass(frozen=True)
class SubsampleReport:
    seed: int
    n_users_before: int
    n_users_after: int
    n_sessions_before: int
    n_sessions_after: int
    n_events_before: int
    n_events_after: int
    target_users: int

    def summary(self) -> str:
        return (
            f"subsample (seed={self.seed}): "
            f"users {self.n_users_before:,} -> {self.n_users_after:,}  |  "
            f"sessions {self.n_sessions_before:,} -> {self.n_sessions_after:,}  |  "
            f"events {self.n_events_before:,} -> {self.n_events_after:,}"
        )


def subsample_users(
    sessions: pl.DataFrame,
    *,
    n_users: int,
    seed: int = 42,
    user_column: str = "user_id",
    session_column: str = "session_id",
) -> tuple[pl.DataFrame, SubsampleReport]:
    """Keep every session of a seeded random sample of ``n_users`` users."""
    users = sessions.select(user_column).unique()
    n_before = users.height

    if n_users >= n_before:
        keep = users
    else:
        # Deterministic in (seed, user_id): stable across reruns and orderings,
        # and nested across sample sizes at a fixed seed.
        keep = (
            users.with_columns(
                (pl.col(user_column).cast(pl.Utf8) + pl.lit(f"|{seed}"))
                .hash(seed=seed)
                .alias("_h")
            )
            .sort("_h")
            .head(n_users)
            .select(user_column)
        )

    sampled = sessions.join(keep, on=user_column, how="inner")

    n_sessions_after = sampled[session_column].n_unique()
    if n_sessions_after < MIN_SESSIONS:
        raise SubsampleError(
            f"subsample yields {n_sessions_after:,} sessions, below the Sec. 5.3 floor of "
            f"{MIN_SESSIONS:,}. Increase n_users."
        )

    report = SubsampleReport(
        seed=seed,
        n_users_before=n_before,
        n_users_after=keep.height,
        n_sessions_before=sessions[session_column].n_unique(),
        n_sessions_after=n_sessions_after,
        n_events_before=sessions.height,
        n_events_after=sampled.height,
        target_users=n_users,
    )
    return sampled, report
