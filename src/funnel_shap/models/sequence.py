"""GRU sequence model over the event stream (protocol Sec. 9.3).

The cross-paradigm arm. Its purpose is not to beat the tabular stage models on
PR-AUC — it is to provide a *second, independent* attribution source so that H4
can ask whether two different explanation paradigms agree on which drivers
matter (Sec. 11.2, RQ3).

**It is trained on the same prefixes, not on whole sessions.** A recurrent model
fed the full event sequence would see the purchase event and the anti-leakage
protocol (Sec. 7.4) would be broken for this arm alone, which would make any
comparison to the tree models meaningless — the two would be explaining
different information sets, not different paradigms.

Sequences are right-padded and a mask is carried, rather than left-padded: the
prefix's *final* events are the ones nearest the decision point, and keeping
them at fixed positions from the start makes the per-timestep attributions that
TimeSHAP produces directly comparable across sessions of different length.

**Amendment A30 (2026-08-07): convergence training.**  The original 8-epoch
fixed-duration training was replaced with early stopping on validation PR-AUC
(patience=10), a cosine-annealing LR schedule, gradient clipping, and support
for stacked GRU layers with inter-layer dropout. This ensures a fair
cross-paradigm comparison (revision A.1) by training the GRU to convergence
rather than for an arbitrary fixed duration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

#: Event vocabulary, fixed so encoding is stable across runs and datasets.
EVENT_TYPES: tuple[str, ...] = ("view", "cart", "remove_from_cart")

#: Per-timestep features. Deliberately small: the point is a second view of the
#: same information, not a better model.
SEQUENCE_FEATURES: tuple[str, ...] = (
    "event_type_view",
    "event_type_cart",
    "event_type_remove_from_cart",
    "log_gap_s",
    "log_price",
    "is_new_product",
    "is_new_category",
)


@dataclass
class SequenceBatch:
    """Padded sequences plus the mask and labels."""

    X: np.ndarray  # (n_sessions, max_len, n_features)
    mask: np.ndarray  # (n_sessions, max_len) bool
    y: np.ndarray  # (n_sessions,)
    session_ids: list[str]
    feature_names: list[str]

    def __len__(self) -> int:
        return len(self.y)


def build_sequences(
    prefix_events: pl.DataFrame,
    *,
    max_len: int = 32,
    session_column: str = "session_id",
) -> SequenceBatch:
    """Turn a stage prefix event frame into padded per-timestep features.

    ``prefix_events`` must come from :func:`funnel_shap.data.journey.prefix_events`,
    so the anti-leakage cut has already been applied and no purchase event can
    be present.
    """
    etype = pl.col("event_type").cast(pl.Utf8)
    frame = prefix_events.sort([session_column, "event_idx"]).with_columns(
        *[(etype == t).cast(pl.Float32).alias(f"event_type_{t}") for t in EVENT_TYPES],
        (
            pl.col("event_time").diff().over(session_column).dt.total_seconds().fill_null(0)
        ).alias("_gap_s"),
        (~pl.col("product_id").is_first_distinct().over(session_column))
        .not_()
        .cast(pl.Float32)
        .alias("is_new_product"),
    )

    if "category_id" in frame.columns:
        frame = frame.with_columns(
            (~pl.col("category_id").is_first_distinct().over(session_column))
            .not_()
            .cast(pl.Float32)
            .alias("is_new_category")
        )
    else:
        frame = frame.with_columns(pl.lit(0.0, dtype=pl.Float32).alias("is_new_category"))

    price = pl.col("price") if "price" in frame.columns else pl.lit(0.0)
    frame = frame.with_columns(
        # log1p on gaps and prices: both are heavy-tailed by orders of
        # magnitude, and an unscaled tail dominates the recurrent update.
        pl.col("_gap_s").clip(lower_bound=0).log1p().cast(pl.Float32).alias("log_gap_s"),
        price.clip(lower_bound=0).log1p().cast(pl.Float32).alias("log_price"),
    )

    grouped = frame.group_by(session_column, maintain_order=True).agg(
        [pl.col(c).alias(c) for c in SEQUENCE_FEATURES] + [pl.first("label").alias("label")]
    )

    n = grouped.height
    X = np.zeros((n, max_len, len(SEQUENCE_FEATURES)), dtype=np.float32)
    mask = np.zeros((n, max_len), dtype=bool)

    for j, column in enumerate(SEQUENCE_FEATURES):
        for i, values in enumerate(grouped[column].to_list()):
            # Keep the LAST max_len events: the tail is nearest the decision.
            take = values[-max_len:]
            X[i, : len(take), j] = np.asarray(take, dtype=np.float32)
            if j == 0:
                mask[i, : len(take)] = True

    return SequenceBatch(
        X=X,
        mask=mask,
        y=grouped["label"].cast(pl.Int8).to_numpy(),
        session_ids=grouped[session_column].to_list(),
        feature_names=list(SEQUENCE_FEATURES),
    )


def build_gru(
    n_features: int,
    *,
    hidden: int = 128,
    num_layers: int = 2,
    dropout: float = 0.2,
    seed: int = 42,
):
    """A GRU classifier with the forward signature TimeSHAP expects.

    Supports stacked layers with inter-layer dropout (amendment A30).
    TimeSHAP calls the model with a numpy array of shape
    ``(batch, timesteps, features)`` and expects ``(batch, 1)`` scores back, so
    the wrapper is part of the contract rather than a convenience.
    """
    import torch
    from torch import nn

    torch.manual_seed(seed)

    class GRUClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gru = nn.GRU(
                n_features,
                hidden,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden, 1),
                nn.Sigmoid(),
            )

        def forward(self, x):
            if not torch.is_tensor(x):
                x = torch.as_tensor(np.asarray(x, dtype=np.float32))
            out, _ = self.gru(x)
            return self.head(out[:, -1, :])

    return GRUClassifier()


def train_gru(
    batch: SequenceBatch,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    *,
    seed: int = 42,
    epochs: int = 80,
    patience: int = 10,
    batch_size: int = 512,
    learning_rate: float = 1e-3,
    hidden: int = 128,
    num_layers: int = 2,
    dropout: float = 0.2,
    grad_clip: float = 1.0,
):
    """Train with class-weighted BCE, early stopping on validation PR-AUC.

    Amendment A30: convergence training replaces the original fixed 8-epoch run.

    - **Early stopping** with configurable patience on validation PR-AUC
      ensures the model trains until improvement plateaus rather than for an
      arbitrary duration. The best checkpoint is restored at the end.
    - **Cosine annealing** LR schedule prevents the learning rate from
      being too aggressive in later epochs as the model approaches convergence.
    - **Gradient clipping** stabilises training on long sequences where
      gradients through many timesteps can accumulate.
    - **Stacked GRU layers with dropout** give the model enough capacity to
      match the tree baseline while regularising against overfitting.

    Class weighting rather than resampling: Sec. 9.4 permits either, and
    reweighting the loss avoids synthesising event sequences, which SMOTE would
    have to do and which has no defensible interpretation here.
    """
    import torch
    from sklearn.metrics import average_precision_score
    from torch import nn

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = build_gru(
        batch.X.shape[2],
        hidden=hidden,
        num_layers=num_layers,
        dropout=dropout,
        seed=seed,
    )
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=epochs, eta_min=learning_rate * 0.01
    )

    y_train = batch.y[train_idx].astype(np.float32)
    pos_weight = float((len(y_train) - y_train.sum()) / max(y_train.sum(), 1.0))
    loss_fn = nn.BCELoss(reduction="none")

    X_train = torch.as_tensor(batch.X[train_idx])
    y_train_t = torch.as_tensor(y_train).unsqueeze(1)
    X_val = torch.as_tensor(batch.X[val_idx])
    y_val = batch.y[val_idx]

    best_state, best_score = None, -np.inf
    epochs_without_improvement = 0

    for _epoch in range(epochs):
        model.train()
        order = np.random.permutation(len(train_idx))
        for start in range(0, len(order), batch_size):
            chunk = order[start : start + batch_size]
            optimiser.zero_grad()
            predicted = model(X_train[chunk])
            target = y_train_t[chunk]
            weights = torch.where(target > 0, pos_weight, 1.0)
            loss = (loss_fn(predicted, target) * weights).mean()
            loss.backward()
            # Gradient clipping prevents exploding gradients on long sequences.
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimiser.step()

        scheduler.step()

        model.eval()
        with torch.no_grad():
            scores = model(X_val).squeeze(1).numpy()
        score = float(average_precision_score(y_val, scores))

        if score > best_score:
            best_score = score
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, best_score


def predict_sequences(model, X: np.ndarray, *, batch_size: int = 1024) -> np.ndarray:
    """Scores for a padded sequence array."""
    import torch

    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            chunk = torch.as_tensor(X[start : start + batch_size])
            out.append(model(chunk).squeeze(1).numpy())
    return np.concatenate(out) if out else np.empty(0)
