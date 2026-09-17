"""Pass selection (behaviour policy): which visible option would a typical professional choose?

Why this exists (M2 finding): with EV alone, the "best" option is usually a longer, more
forward, riskier pass than players choose (median P(complete) 0.77 vs 0.96). Part of that
may be genuine conservatism. But part is limitation B: options nobody plays are avoided
for reasons a static frame can't show (marking intensity, body shape, defender movement).
The completion model is calibrated on *attempted* passes, but for never-attempted
options it may be optimistic, and no data can check that.

Off-policy evaluation offers the standard remedy: only trust value estimates for actions
the behaviour policy actually takes with non-trivial probability. So recommendations are
restricted to **plausible** options: those this model says a professional would pick at
least `threshold` of the time in this situation.

The policy uses only geometric/pressure features (PASS_FEATURES), not our value estimates,
so "plausible" is defined independently of what we then rank.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from phds.features.pass_features import PASS_FEATURES


def softmax_by_group(scores: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Softmax of raw scores within each group (pass). `groups` must be contiguous."""
    s = pd.Series(scores)
    shifted = s - s.groupby(groups).transform("max")
    e = np.exp(shifted.to_numpy())
    return e / pd.Series(e).groupby(groups).transform("sum").to_numpy()


class SelectionModel:
    def __init__(self, **params):
        self.params = {
            "n_estimators": 400, "learning_rate": 0.05, "num_leaves": 63,
            "min_child_samples": 200, "subsample": 0.8, "subsample_freq": 1,
            "colsample_bytree": 0.8, "reg_lambda": 5.0, "verbose": -1,
        } | params  # fmt: skip

    def fit(self, options: pd.DataFrame, chosen: np.ndarray):
        self.model = lgb.LGBMClassifier(**self.params).fit(options[PASS_FEATURES], chosen)
        return self

    def predict(self, options: pd.DataFrame) -> np.ndarray:
        """P(option is chosen), normalised over the options of each pass (`event_id`)."""
        raw = self.model.predict(options[PASS_FEATURES], raw_score=True)
        return softmax_by_group(raw, options["event_id"].to_numpy())
