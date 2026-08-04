"""Synthetic REES46-shaped event log for tests and smoke runs.

Dataset B is licence-gated and multi-gigabyte, so the pipeline must be
exercisable without it. This generator produces an event stream with the same
schema and the same structural quirks that break naive implementations:
one-second timestamp granularity with colliding events, sessions that never
reach S2 or S3, repeat views, category switches, cart removals, and a purchase
rate in the same ballpark as the real log.

It is a *test fixture*, not a simulation. No result in the paper may come from
it; it exists so that leakage invariants and feature definitions can be pinned
before the real data arrives.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl


def make_event_log(
    n_users: int = 200,
    *,
    seed: int = 42,
    start: dt.datetime | None = None,
    purchase_rate: float = 0.12,
) -> pl.DataFrame:
    """Generate a synthetic event log with REES46 columns."""
    rng = np.random.default_rng(seed)
    start = start or dt.datetime(2019, 10, 1, 0, 0, 0)

    rows: list[dict] = []
    for user in range(1, n_users + 1):
        n_sessions = int(rng.integers(1, 4))
        clock = start + dt.timedelta(minutes=float(rng.integers(0, 60 * 24 * 25)))

        for session in range(n_sessions):
            # Gap large enough to force a session cut at any of 15/30/60 min.
            clock += dt.timedelta(minutes=float(rng.integers(90, 400)))
            n_views = int(rng.integers(1, 12))
            categories = rng.integers(1, 6, size=n_views)
            products = rng.integers(100, 140, size=n_views)

            session_rows: list[dict] = []
            for i in range(n_views):
                # Many events share a second, as in the real log.
                clock += dt.timedelta(seconds=int(rng.integers(0, 90)))
                session_rows.append(
                    _row(clock, "view", int(products[i]), int(categories[i]), user, session, rng)
                )

            reaches_cart = rng.random() < 0.30 and n_views >= 2
            if reaches_cart:
                clock += dt.timedelta(seconds=int(rng.integers(5, 120)))
                session_rows.append(
                    _row(clock, "cart", int(products[-1]), int(categories[-1]), user, session, rng)
                )
                if rng.random() < 0.25:
                    clock += dt.timedelta(seconds=int(rng.integers(5, 90)))
                    session_rows.append(
                        _row(
                            clock, "remove_from_cart", int(products[-1]),
                            int(categories[-1]), user, session, rng,
                        )
                    )
                if rng.random() < purchase_rate / 0.30:
                    clock += dt.timedelta(seconds=int(rng.integers(10, 300)))
                    session_rows.append(
                        _row(
                            clock, "purchase", int(products[-1]),
                            int(categories[-1]), user, session, rng,
                        )
                    )

            rows.extend(session_rows)

    frame = pl.DataFrame(rows)
    return frame.with_columns(
        pl.col("event_time").dt.strftime("%Y-%m-%d %H:%M:%S") + pl.lit(" UTC")
    )


def _row(
    clock: dt.datetime,
    event_type: str,
    product: int,
    category: int,
    user: int,
    session: int,
    rng: np.random.Generator,
) -> dict:
    return {
        "event_time": clock,
        "event_type": event_type,
        "product_id": product,
        "category_id": category,
        "category_code": f"cat.{category}",
        "brand": f"brand{product % 7}",
        "price": float(np.round(rng.uniform(5, 500), 2)),
        "user_id": user,
        "user_session": f"{user}-{session}",
    }
