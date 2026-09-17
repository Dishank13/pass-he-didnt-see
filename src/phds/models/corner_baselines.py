"""Baselines for corner receiver and shot prediction.

A GNN result means nothing without a ladder of simpler models under it:

Receiver (who touches the ball first?)
  1. uniform:  equal probability for every visible non-taker. The chance level.
  2. hotspot:  a Gaussian around where first touches usually happen (fit on train).
               Answers: is "stand in the right spot" all there is to it?
  3. LightGBM: per-player gradient boosting on handcrafted node + context features,
               with probabilities normalised within each corner (softmax of raw scores).

Shot (does the corner lead to a shot within 20 s?)
  1. base rate: the training shot rate for every corner.
  2. logistic regression on graph-level summary features.
  3. LightGBM on graph features + per-team aggregates of node features.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from phds.features.corner_graph import NODE_FEATURES, CornerGraphs

X_IDX, Y_IDX, ATT_IDX = (NODE_FEATURES.index(k) for k in ("x", "y", "attacker"))


def masked_softmax(scores: np.ndarray, cand: np.ndarray) -> np.ndarray:
    s = np.where(cand, scores, -np.inf)
    s = s - s.max(axis=1, keepdims=True)
    p = np.where(cand, np.exp(s), 0.0)
    return p / p.sum(axis=1, keepdims=True)


# --- receiver ------------------------------------------------------------------------
def uniform_receiver(g: CornerGraphs) -> np.ndarray:
    return g.cand / g.cand.sum(axis=1, keepdims=True)


class HotspotReceiver:
    """Score each candidate by a 2D Gaussian density centred on typical first-touch spots.

    Fit separately for attackers and defenders: defenders' first touches (clearances,
    keeper claims) cluster closer to goal than attackers'.
    """

    def fit(self, g: CornerGraphs, target_dist: np.ndarray):
        pos = np.stack([g.x[..., X_IDX] * 120, g.x[..., Y_IDX] * 80], -1)  # [N, M, 2]
        self.params = {}
        for team in (0.0, 1.0):
            w = target_dist * (g.x[..., ATT_IDX] == team)
            w_flat, p_flat = w.reshape(-1), pos.reshape(-1, 2)
            mu = (w_flat[:, None] * p_flat).sum(0) / w_flat.sum()
            var = (w_flat[:, None] * (p_flat - mu) ** 2).sum(0) / w_flat.sum()
            self.params[team] = (mu, var, w_flat.sum() / target_dist.sum())
        return self

    def predict(self, g: CornerGraphs) -> np.ndarray:
        pos = np.stack([g.x[..., X_IDX] * 120, g.x[..., Y_IDX] * 80], -1)
        score = np.full(g.cand.shape, -np.inf)
        for team, (mu, var, prior) in self.params.items():
            sel = g.x[..., ATT_IDX] == team
            ll = -0.5 * (((pos - mu) ** 2) / var).sum(-1) + np.log(prior)
            score = np.where(sel, ll, score)
        return masked_softmax(score, g.cand)


def receiver_node_table(g: CornerGraphs) -> np.ndarray:
    """Node features + graph features + within-corner context ranks, one row per node."""
    m = g.x.shape[1]
    node = g.x
    graph = np.repeat(g.g[:, None, :], m, axis=1)
    # Rank of closeness to goal within the player's own team (1 = closest).
    dist_goal = np.where(g.mask, node[..., NODE_FEATURES.index("dist_goal")], np.inf)
    same_team = node[..., ATT_IDX][:, :, None] == node[..., ATT_IDX][:, None, :]
    closer = (dist_goal[:, None, :] < dist_goal[:, :, None]) & same_team & g.mask[:, None, :]
    rank = closer.sum(-1, keepdims=True).astype(np.float32) / 11
    return np.concatenate([node, graph, rank], axis=-1)


class LGBMReceiver:
    def __init__(self, **params):
        self.params = {
            "n_estimators": 400, "learning_rate": 0.03, "num_leaves": 15,
            "min_child_samples": 20, "subsample": 0.8, "subsample_freq": 1,
            "colsample_bytree": 0.8, "reg_lambda": 1.0, "verbose": -1,
        } | params  # fmt: skip

    def fit(self, g: CornerGraphs, target_dist: np.ndarray, has_label: np.ndarray):
        """target_dist: [N, M] soft or one-hot targets. LightGBM's cross_entropy objective
        accepts labels in [0, 1], so soft labels plug in directly."""
        table = receiver_node_table(g)
        rows = has_label[:, None] & g.cand
        self.model = lgb.LGBMRegressor(objective="cross_entropy", **self.params)
        self.model.fit(table[rows], target_dist[rows])
        return self

    def predict(self, g: CornerGraphs) -> np.ndarray:
        table = receiver_node_table(g)
        n, m, f = table.shape
        raw = self.model.predict(table.reshape(-1, f), raw_score=True).reshape(n, m)
        return masked_softmax(raw, g.cand)


# --- shot ----------------------------------------------------------------------------
def shot_table(g: CornerGraphs) -> np.ndarray:
    """Graph features + mean/min/max of node features over attackers and over defenders."""
    feats = [g.g]
    att = g.x[..., ATT_IDX] == 1
    for team_mask in (att & g.mask, ~att & g.mask):
        cnt = team_mask.sum(1, keepdims=True).clip(min=1)
        x = g.x
        feats.append((x * team_mask[..., None]).sum(1) / cnt)
        feats.append(np.where(team_mask[..., None], x, np.inf).min(1))
        feats.append(np.where(team_mask[..., None], x, -np.inf).max(1))
    t = np.concatenate(feats, axis=1)
    return np.where(np.isfinite(t), t, 0.0).astype(np.float32)


class BaseRateShot:
    def fit(self, g: CornerGraphs):
        self.rate = g.shot.mean()
        return self

    def predict(self, g: CornerGraphs) -> np.ndarray:
        return np.full(len(g.shot), self.rate)


class LogisticShot:
    def fit(self, g: CornerGraphs):
        self.model = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))
        self.model.fit(g.g, g.shot)
        return self

    def predict(self, g: CornerGraphs) -> np.ndarray:
        return self.model.predict_proba(g.g)[:, 1]


class LGBMShot:
    def fit(self, g: CornerGraphs):
        self.model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.02, num_leaves=7,
                                        min_child_samples=30, subsample=0.8, subsample_freq=1,
                                        colsample_bytree=0.7, reg_lambda=2.0, verbose=-1)  # fmt: skip
        self.model.fit(shot_table(g), g.shot)
        return self

    def predict(self, g: CornerGraphs) -> np.ndarray:
        return self.model.predict_proba(shot_table(g))[:, 1]
