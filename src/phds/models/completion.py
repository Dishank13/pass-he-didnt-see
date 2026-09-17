"""Pass completion: P(a pass to teammate j at p_j is completed | freeze frame).

Model ladder:
  distance  logistic on length / forward progress / lateral change. "Short passes succeed."
  physics   logistic on the interception time margins (+ length). A hand-built prior that
            knows why lanes fail, even ones nobody attempts.
  lgbm      gradient boosting on every feature in PASS_FEATURES.

All models train with `weight` from `pass_targets.linkage_weights`, so predicted
probabilities are calibrated for passes to *visible* teammates, not for the subset
of passes we managed to link.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler
from tqdm import tqdm

from phds.data.frame_store import FrameStore
from phds.features.pass_features import PASS_FEATURES, option_features

DISTANCE_FEATURES = ["length", "dx", "abs_dy"]
PHYSICS_FEATURES = ["physics_margin", "lane_margin", "target_margin", "length"]


def real_pass_features(passes: pd.DataFrame, store: FrameStore, frames: pd.DataFrame) -> pd.DataFrame:
    """Features for each linked pass to its intended target (row-aligned with `passes`)."""
    vis = frames.set_index("event_id")["visible_frac"]
    rows = np.zeros((len(passes), len(PASS_FEATURES)), np.float32)
    it = passes[["event_id", "x", "y", "target_x", "target_y", "under_pressure"]].itertuples(index=False)
    for i, (eid, x, y, tx, ty, up) in enumerate(tqdm(it, total=len(passes), desc="pass features")):
        rows[i] = option_features(store[eid], np.array([x, y]), np.array([[tx, ty]]),
                                  bool(up), vis.get(eid, np.nan))[0]  # fmt: skip
    return pd.DataFrame(rows, columns=PASS_FEATURES, index=passes.index)


class LogitCompletion:
    """Logistic regression on a few features, each expanded with splines (monotone-ish shapes)."""

    def __init__(self, features: list[str]):
        self.features = features

    def fit(self, X: pd.DataFrame, y, w):
        self.model = make_pipeline(
            SplineTransformer(n_knots=6, degree=3), StandardScaler(),
            LogisticRegression(C=1.0, max_iter=3000),
        )  # fmt: skip
        self.model.fit(X[self.features], y, logisticregression__sample_weight=w)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X[self.features])[:, 1]


class CalibratedCompletion:
    """Wrap a completion model with cross-fitted isotonic calibration.

    1. Split *training* matches into k groups. For each fold, fit on k-1 groups and predict
       the held-out group. This gives out-of-fold predictions for every training pass.
    2. Fit a weighted isotonic regression from out-of-fold predictions to outcomes.
    3. Refit the base model on all training data and apply the isotonic map at predict time.

    Evaluation data never touches the calibrator. Isotonic regression is monotone, so ranking
    (AUC) is essentially unchanged. Only the probability scale moves.
    """

    def __init__(self, make_model, n_folds: int = 5, seed: int = 0):
        self.make_model, self.n_folds, self.seed = make_model, n_folds, seed

    def fit(self, X: pd.DataFrame, y, w, groups):
        from sklearn.isotonic import IsotonicRegression

        y, w, groups = np.asarray(y), np.asarray(w), np.asarray(groups)
        rng = np.random.default_rng(self.seed)
        uniq = rng.permutation(np.unique(groups))
        fold_of = {g: i % self.n_folds for i, g in enumerate(uniq)}
        fold = np.array([fold_of[g] for g in groups])
        oof = np.zeros(len(y))
        for k in range(self.n_folds):
            tr, te = fold != k, fold == k
            oof[te] = self.make_model().fit(X[tr], y[tr], w[tr]).predict(X[te])
        self.calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self.calibrator.fit(oof, y, sample_weight=w)
        self.base = self.make_model().fit(X, y, w)
        return self

    def predict_raw(self, X: pd.DataFrame) -> np.ndarray:
        return self.base.predict(X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.calibrator.predict(self.base.predict(X))


class LGBMCompletion:
    def __init__(self, **params):
        self.params = {
            "n_estimators": 600, "learning_rate": 0.05, "num_leaves": 63,
            "min_child_samples": 100, "subsample": 0.8, "subsample_freq": 1,
            "colsample_bytree": 0.8, "reg_lambda": 5.0, "verbose": -1,
        } | params  # fmt: skip

    def fit(self, X: pd.DataFrame, y, w):
        self.model = lgb.LGBMClassifier(**self.params).fit(X[PASS_FEATURES], y, sample_weight=w)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X[PASS_FEATURES])[:, 1]

    def importance(self) -> pd.Series:
        gain = self.model.booster_.feature_importance("gain")
        return pd.Series(gain / gain.sum(), index=PASS_FEATURES).sort_values(ascending=False)
