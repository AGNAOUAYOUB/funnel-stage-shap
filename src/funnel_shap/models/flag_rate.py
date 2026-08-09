"""Incremental precision as a function of the contact budget.

The decision analysis in the paper originally rested on one operating point per
stage, chosen by maximising $F_1$ on validation. That rule is
prevalence-sensitive: as prevalence approaches one half it is increasingly
satisfied by flagging everything, and the intent stage's prevalence is 0.52. The
threshold it selected there flags 99% of sessions, so the finding that the stage
adds nothing over a blanket policy was close to circular.

This module reports the quantity a manager actually allocates against. A *flag
rate* is a budget: the share of sessions the organisation can afford to contact.
At each budget the incremental precision is the precision among the
highest-scoring sessions minus the stage prevalence --- how much better than
contacting everyone the model does --- and its reciprocal is the number of
contacts needed for one conversion beyond what a blanket rule already delivers.

Both quantities are measured against the blanket rule rather than against not
contacting, because not contacting is not the alternative available to an
organisation that already has a mailing list.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl

DEFAULT_RATES: tuple[float, ...] = (0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 1.00)


def incremental_precision(
    y_true: np.ndarray, scores: np.ndarray, rate: float
) -> float:
    """Precision among the top ``rate`` share of scores, minus the prevalence.

    At ``rate == 1`` this is exactly zero: flagging everything *is* the blanket
    rule, so the incremental value of the model must vanish there. The identity
    is a useful check on the whole table.
    """
    if not 0 < rate <= 1:
        raise ValueError(f"flag rate must lie in (0, 1], got {rate}")

    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    if y_true.shape != scores.shape:
        raise ValueError("labels and scores must have the same shape")

    k = max(1, int(round(rate * len(y_true))))
    flagged = y_true[np.argsort(-scores)[:k]]
    return float(flagged.mean() - y_true.mean())


def sweep_table(
    by_stage: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    *,
    rates: Sequence[float] = DEFAULT_RATES,
    stage_order: Sequence[str] | None = None,
) -> pl.DataFrame:
    """Mean and dispersion of incremental precision per stage and budget.

    ``by_stage`` maps a stage to one ``(y_true, scores)`` pair per seed. The
    contact requirement is the reciprocal of the mean, and is left null when the
    mean is not positive: the reciprocal of a near-zero quantity is unstable, and
    reporting a very large point estimate would imply a precision the data do not
    support.
    """
    stages = list(stage_order) if stage_order is not None else list(by_stage)

    rows = []
    for rate in rates:
        for stage in stages:
            arms = by_stage.get(stage) or []
            if not arms:
                continue
            values = np.array(
                [incremental_precision(y, p, rate) for y, p in arms], dtype=float
            )
            mean = float(values.mean())
            rows.append(
                {
                    "flag_rate": float(rate),
                    "stage": stage,
                    "incremental_precision_mean": round(mean, 4),
                    "incremental_precision_sd": (
                        round(float(values.std(ddof=1)), 4) if values.size > 1 else None
                    ),
                    "contacts_per_incremental": (
                        round(1.0 / mean, 1) if mean > 0 else None
                    ),
                    "n_seeds": int(values.size),
                }
            )
    return pl.DataFrame(rows)
