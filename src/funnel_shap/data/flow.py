"""Data-flow accounting for the raw -> filter -> final N diagram (protocol Sec. 7.1 step 3).

Every filter in the cleaning and sessionisation pipeline registers what it
removed. The resulting table is the CONSORT-style data-flow figure the protocol
requires, and it doubles as the reproducibility check: if a rerun drops a
different number of events, the diff is visible immediately.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl


@dataclass(frozen=True)
class FlowStep:
    step: str
    rule: str
    n_events: int
    n_sessions: int | None
    events_removed: int
    sessions_removed: int | None


@dataclass
class DataFlow:
    """Ordered record of how many rows survived each pipeline stage."""

    label: str
    steps: list[FlowStep] = field(default_factory=list)

    def record(
        self,
        step: str,
        rule: str,
        n_events: int,
        n_sessions: int | None = None,
    ) -> None:
        prev = self.steps[-1] if self.steps else None
        events_removed = 0 if prev is None else prev.n_events - n_events
        if prev is None or prev.n_sessions is None or n_sessions is None:
            sessions_removed = None
        else:
            sessions_removed = prev.n_sessions - n_sessions

        self.steps.append(
            FlowStep(
                step=step,
                rule=rule,
                n_events=n_events,
                n_sessions=n_sessions,
                events_removed=events_removed,
                sessions_removed=sessions_removed,
            )
        )

    def to_frame(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "step": [s.step for s in self.steps],
                "rule": [s.rule for s in self.steps],
                "n_events": [s.n_events for s in self.steps],
                "n_sessions": [s.n_sessions for s in self.steps],
                "events_removed": [s.events_removed for s in self.steps],
                "sessions_removed": [s.sessions_removed for s in self.steps],
            }
        )

    def write(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_frame().write_csv(path)
        path.with_suffix(".json").write_text(
            json.dumps(
                {"label": self.label, "steps": [s.__dict__ for s in self.steps]}, indent=2
            ),
            encoding="utf-8",
        )
        return path

    def summary(self) -> str:
        lines = [f"Data flow: {self.label}", "-" * 72]
        for s in self.steps:
            sessions = "" if s.n_sessions is None else f"  sessions={s.n_sessions:,}"
            removed = f"  (-{s.events_removed:,} events)" if s.events_removed else ""
            lines.append(f"{s.step:<32} events={s.n_events:>12,}{sessions}{removed}")
            lines.append(f"{'':<32} rule: {s.rule}")
        return "\n".join(lines)
