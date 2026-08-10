"""Is the attribution trajectory signal, or structured noise?

The migration finding --- navigation entropy carrying 3.5% of attribution mass
at S1, 22.4% at S2 and 7.4% at S3 --- rests on models whose ranking skill is
modest, and at S2 it is the weakest of the three. SHAP measures fidelity to the
*model*, not to the world, so a model with almost no discriminative skill can be
explained perfectly faithfully while its attributions describe nothing but the
covariance structure of the feature matrix and the idiosyncrasies of one fit.
Faithfulness, stability and reproducibility checks are all silent on this: an
attribution can be faithful, stable, reproducible, and empty.

The test is a permutation null. Labels are shuffled within the training
partition, the stage model is refitted on the shuffled labels, and the same
grouped attribution shares are computed against the same reference grouping.
Under permutation there is by construction no relationship between features and
outcome, so any share the null still produces is what the pipeline manufactures
from feature geometry alone --- a group of correlated, high-variance features
will absorb attribution mass whether or not it predicts anything.

Reading the result:

* A group whose observed share sits far outside its null distribution carries
  information about the outcome.
* A group whose observed share sits inside the null is explained by feature
  structure, and any narrative built on it is unsupported however faithful the
  attribution is.

The decisive case for this paper is the S2 navigation peak. If it survives, the
mid-funnel finding is real; if it does not, it must be withdrawn. Both outcomes
are reportable, which is why the test is worth running rather than assuming.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from ..data.journey import MODELLING_STAGES, StageName
from ..seeds import set_global_seed
from .run_stage_shap import explain_stages, reference_groups
from .stage_shap import grouped_attribution


@dataclass
class NullDraw:
    """One permutation round's grouped shares, per stage."""

    round_index: int
    seed: int
    shares: dict[tuple[StageName, str], float] = field(default_factory=dict)


def _shares(explanations, groups: dict[str, list[str]]) -> dict[tuple[str, str], float]:
    out: dict[tuple[str, str], float] = {}
    for stage, explanation in explanations.items():
        table = grouped_attribution(explanation.attribution, groups)
        for row in table.to_dicts():
            out[(stage, row["group"])] = float(row["share"])
    return out


def permute_labels(
    features_by_stage: dict[StageName, pl.DataFrame], rng: np.random.Generator
) -> dict[StageName, pl.DataFrame]:
    """Shuffle the label column within each stage.

    Permuting per stage rather than once across the union is deliberate: each
    stage's label vector keeps its own prevalence, so the null model faces the
    same class balance as the real one and the comparison isolates the
    feature--label relationship rather than confounding it with a prevalence
    change.
    """
    out = {}
    for stage, frame in features_by_stage.items():
        labels = frame["label"].to_numpy().copy()
        rng.shuffle(labels)
        out[stage] = frame.with_columns(pl.Series("label", labels))
    return out


def run_permutation_null(
    features_by_stage: dict[StageName, pl.DataFrame],
    *,
    suffix: str,
    protocol: str = "temporal",
    n_rounds: int = 20,
    seed: int = 42,
    max_explain: int = 1500,
    background_size: int = 500,
    progress: bool = True,
) -> tuple[dict[tuple[str, str], float], list[NullDraw]]:
    """Observed shares and a permutation null distribution for each.

    The observed run uses the same reduced explanation budget as the null
    rounds. Comparing a 5,000-row observed attribution against 1,500-row null
    draws would confound the effect with the sample size of the explanation set.
    """
    set_global_seed(seed)
    observed_explanations = explain_stages(
        features_by_stage,
        suffix=suffix,
        protocol=protocol,
        seed=seed,
        max_explain=max_explain,
        background_size=background_size,
    )
    if not observed_explanations:
        raise ValueError("no stage could be explained; the null has nothing to test")

    # The grouping is fixed from the observed run and reused for every null
    # round. Letting each round derive its own grouping would compare shares
    # computed over different partitions of the feature space.
    groups = reference_groups(observed_explanations)
    observed = _shares(observed_explanations, groups)

    draws: list[NullDraw] = []
    rng = np.random.default_rng(seed)
    for index in range(n_rounds):
        round_seed = int(rng.integers(0, 2**31 - 1))
        permuted = permute_labels(features_by_stage, np.random.default_rng(round_seed))
        set_global_seed(seed)
        null_explanations = explain_stages(
            permuted,
            suffix=suffix,
            protocol=protocol,
            seed=seed,
            max_explain=max_explain,
            background_size=background_size,
        )
        draws.append(
            NullDraw(
                round_index=index,
                seed=round_seed,
                shares=_shares(null_explanations, groups),
            )
        )
        if progress:
            print("permutation round %d/%d done" % (index + 1, n_rounds), flush=True)

    return observed, draws


def null_table(
    observed: dict[tuple[str, str], float], draws: list[NullDraw]
) -> pl.DataFrame:
    """Observed share against its permutation null, with a one-sided p-value.

    The p-value is the proportion of null rounds whose share reaches the
    observed one, with the customary add-one correction so that a share never
    matched by the null reports ``1/(R+1)`` rather than zero --- a permutation
    test cannot certify a probability smaller than its own resolution.
    """
    if not draws:
        raise ValueError("no null draws; the comparison has no reference")

    rows = []
    for (stage, group), value in observed.items():
        null = np.array([d.shares.get((stage, group), 0.0) for d in draws])
        at_least = int((null >= value).sum())
        rows.append(
            {
                "stage": stage,
                "group": group,
                "observed_share": round(value, 4),
                "null_mean": round(float(null.mean()), 4),
                "null_sd": round(float(null.std(ddof=1)), 4) if null.size > 1 else None,
                "null_max": round(float(null.max()), 4),
                "p_value": round((at_least + 1) / (len(draws) + 1), 4),
                "n_rounds": len(draws),
            }
        )

    order = {stage: i for i, stage in enumerate(MODELLING_STAGES)}
    return (
        pl.DataFrame(rows)
        .with_columns(pl.col("stage").replace_strict(order, default=99).alias("_o"))
        .sort(["_o", "observed_share"], descending=[False, True])
        .drop("_o")
    )
