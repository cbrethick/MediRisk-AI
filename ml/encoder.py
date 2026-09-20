"""
A small, explicit feature encoder.

Why not sklearn's ColumnTransformer? Because the finished model has to run
*inside the browser*, and every transformation therefore has to be replayable
in TypeScript from a JSON spec. A hand-rolled encoder whose entire state is a
few dictionaries is trivially serialisable; a pickled ColumnTransformer is not.

The rules are deliberately boring:
  * numeric / ordinal -> impute with the TRAINING median, and add a companion
    `<name>__isna` indicator so the model can learn that "BMI not recorded"
    is itself informative (30% of adults have no BMI in MEPS).
  * binary            -> impute with the TRAINING mode, plus an indicator.
  * categorical       -> one-hot over a frozen category list. An unseen or
    missing category becomes an all-zero row, which is the honest encoding of
    "we don't know".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    CATEGORICAL_FEATURES, CONDITION_COLS, ENGINEERED_BINARY, NUMERIC_FEATURES,
    ORDINAL_FEATURES,
)

CONTINUOUS = NUMERIC_FEATURES + ORDINAL_FEATURES
BINARY = ENGINEERED_BINARY + CONDITION_COLS


class MepsEncoder:
    """Fit on the training split only, then frozen."""

    def __init__(self) -> None:
        self.medians: dict[str, float] = {}
        self.modes: dict[str, float] = {}
        self.categories: dict[str, list[float]] = {}
        self.feature_names_: list[str] = []

    # ------------------------------------------------------------------ fit --
    def fit(self, df: pd.DataFrame) -> "MepsEncoder":
        for c in CONTINUOUS:
            self.medians[c] = float(np.nanmedian(df[c].astype(float)))
        for c in BINARY:
            v = df[c].dropna()
            self.modes[c] = float(v.mode().iloc[0]) if len(v) else 0.0
        for c in CATEGORICAL_FEATURES:
            self.categories[c] = sorted(float(x) for x in df[c].dropna().unique())

        names: list[str] = []
        for c in CONTINUOUS:
            names += [c, f"{c}__isna"]
        for c in BINARY:
            names += [c, f"{c}__isna"]
        for c in CATEGORICAL_FEATURES:
            names += [f"{c}={int(v)}" for v in self.categories[c]]
        self.feature_names_ = names
        return self

    # -------------------------------------------------------------- transform --
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        blocks: list[np.ndarray] = []
        for c in CONTINUOUS:
            col = df[c].astype(float).to_numpy()
            isna = np.isnan(col).astype(float)
            blocks.append(np.where(np.isnan(col), self.medians[c], col))
            blocks.append(isna)
        for c in BINARY:
            col = df[c].astype(float).to_numpy()
            isna = np.isnan(col).astype(float)
            blocks.append(np.where(np.isnan(col), self.modes[c], col))
            blocks.append(isna)
        for c in CATEGORICAL_FEATURES:
            col = df[c].astype(float).to_numpy()
            for v in self.categories[c]:
                blocks.append((col == v).astype(float))
        return np.column_stack(blocks)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    # ----------------------------------------------------------------- export --
    def to_spec(self) -> dict:
        """Everything the TypeScript re-implementation needs."""
        return {
            "continuous": CONTINUOUS,
            "binary": BINARY,
            "categorical": CATEGORICAL_FEATURES,
            "medians": self.medians,
            "modes": self.modes,
            "categories": {k: [int(x) for x in v] for k, v in self.categories.items()},
            "featureNames": self.feature_names_,
        }
