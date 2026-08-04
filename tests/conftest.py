"""Shared fixtures."""

from __future__ import annotations

import polars as pl
import pytest

from funnel_shap.data.journey import stage_cutpoints
from funnel_shap.data.sessionize import SessionizeConfig, sessionize
from funnel_shap.data.synthetic import make_event_log


@pytest.fixture(scope="session")
def raw_events() -> pl.DataFrame:
    return make_event_log(n_users=250, seed=42)


@pytest.fixture(scope="session")
def sessionised(raw_events: pl.DataFrame) -> pl.DataFrame:
    return sessionize(raw_events.lazy(), SessionizeConfig()).collect()


@pytest.fixture(scope="session")
def cutpoints(sessionised: pl.DataFrame) -> pl.DataFrame:
    return stage_cutpoints(sessionised.lazy())
