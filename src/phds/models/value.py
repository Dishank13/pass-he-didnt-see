"""Possession value: how good is it for team T to have the ball at location L?

Label (VAEP-style, Decroos et al.): for each on-ball action by team T,
    scores   = T scores within the next `horizon` actions (same period)
    concedes = T concedes within the next `horizon` actions
    V(state) = P(scores) - P(concedes)

Why a probability of scoring *soon* rather than "did this possession end in a goal"?
It's a smoother, denser signal (1-2% positives instead of <1%), it values a state by
what typically happens next regardless of who keeps the ball, and it's the standard
choice in the literature, so results are comparable.

Two feature sets:
    loc: x, y, distance and angle to goal. An xT-like baseline: where you are.
    ctx: loc + what the freeze frame shows around L (defensive pressure, goal-side
         defenders, support ahead). Computed identically for the real ball carrier
         (training) and for a hypothetical receiver (counterfactual scoring in M2c).

Caveat for M2c: a counterfactual receiver's context is measured from the frame at the
moment of the *pass*. By the time the ball arrives, defenders will have moved. Frame
context is therefore an optimistic snapshot, and the loc/ctx comparison shows how
much it matters.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from tqdm import tqdm

from phds.data.frame_store import Frame, FrameStore
from phds.geometry.coords import GOAL_CENTER, angle_to_goal, distance_to_goal

ACTION_TYPES = {
    "Pass", "Carry", "Dribble", "Shot", "Clearance", "Interception", "Ball Recovery",
    "Miscontrol", "Dispossessed", "Goal Keeper", "Block", "Foul Committed", "Foul Won",
}  # fmt: skip
LOC_FEATURES = ["x", "y", "dist_goal", "angle_goal"]
CTX_FEATURES = [
    "opp_nearest", "opp_within_5", "opp_within_10", "opp_goal_side_cone", "mates_ahead",
    "mates_within_10", "n_visible", "visible_frac",
]  # fmt: skip


def action_labels(events: pd.DataFrame, horizon: int = 10) -> pd.DataFrame:
    """On-ball actions with `scores` / `concedes` labels over the next `horizon` actions."""
    a = events[events["type"].isin(ACTION_TYPES) & events["x"].notna()]
    a = a.sort_values(["match_id", "index"]).reset_index(drop=True)
    goal_for_team = np.where(
        (a["type"] == "Shot") & (a["shot_outcome"] == "Goal"), a["team_id"], -1
    )  # own goals appear as separate event types and are rare; ignored here
    scores = np.zeros(len(a), bool)
    concedes = np.zeros(len(a), bool)
    match, period, team = a["match_id"].to_numpy(), a["period"].to_numpy(), a["team_id"].to_numpy()
    goal_idx = np.flatnonzero(goal_for_team != -1)
    for g in goal_idx:
        lo = max(0, g - horizon + 1)
        window = np.arange(lo, g + 1)
        window = window[(match[window] == match[g]) & (period[window] == period[g])]
        scores[window[team[window] == goal_for_team[g]]] = True
        concedes[window[team[window] != goal_for_team[g]]] = True
    out = a[["match_id", "event_id", "index", "period", "team_id", "type", "x", "y"]].copy()
    out["scores"], out["concedes"] = scores, concedes
    return out


def possession_labels(events: pd.DataFrame) -> pd.DataFrame:
    """Possession-level value labels for actions by the team *in possession*.

        xg_rest          xG of shots by the team in the rest of this possession (from this action on)
        xg_against_next  xG of shots by the opponent in their next possession
        poss_value       xg_rest - xg_against_next

    Why this label (after the 10-action goal label proved myopic in M2): a failed pass
    ends our possession, so it forfeits everything the possession would still have
    produced, and it hands the opponent a possession that starts nearby. Summing xG
    rather than counting goals keeps the target dense (13% of possessions have a shot,
    vs ~1% of actions preceding a goal).
    """
    ev = events.sort_values(["match_id", "index"])
    shot_xg = np.where(ev["type"].eq("Shot"), ev["shot_xg"].fillna(0.0), 0.0)
    ev = ev.assign(_xg=np.where(ev["team_id"].eq(ev["possession_team_id"]), shot_xg, 0.0))
    # Rest-of-possession xG: reverse cumulative sum within (match, possession).
    grp = ev.groupby(["match_id", "possession"], sort=False)["_xg"]
    ev["xg_rest"] = grp.transform(lambda s: s[::-1].cumsum()[::-1])
    poss = ev.groupby(["match_id", "possession"], sort=True).agg(
        team=("possession_team_id", "first"), xg=("_xg", "sum")).reset_index()  # fmt: skip
    # Next possession by the *other* team: scan forward within each match.
    nxt = np.full(len(poss), 0.0)
    for _, g in poss.groupby("match_id", sort=False):
        teams, xgs, idx = g["team"].to_numpy(), g["xg"].to_numpy(), g.index.to_numpy()
        for j in range(len(g)):
            k = j + 1
            while k < len(g) and teams[k] == teams[j]:
                k += 1
            nxt[idx[j]] = xgs[k] if k < len(g) else 0.0
    poss["xg_against_next"] = nxt
    ev = ev.merge(poss[["match_id", "possession", "xg_against_next"]], on=["match_id", "possession"])
    a = ev[ev["type"].isin(ACTION_TYPES) & ev["x"].notna() & ev["team_id"].eq(ev["possession_team_id"])]
    out = a[["event_id", "xg_rest", "xg_against_next"]].copy()
    out["poss_value"] = out["xg_rest"] - out["xg_against_next"]
    return out


def context_features(frame: Frame | None, xy: np.ndarray, exclude_xy: list[np.ndarray] | None = None,
                     visible_frac: float = np.nan) -> dict:  # fmt: skip
    """Frame context around location xy, from the attacking team's perspective.

    `exclude_xy`: locations of players to leave out of teammate counts (the player at xy,
    and for counterfactual passes also the passer), matched within 0.5 yd.
    """
    if frame is None or len(frame) == 0:
        return dict.fromkeys(CTX_FEATURES, np.nan)
    opp = frame.xy[~frame.teammate].astype(float)
    mates = frame.xy[frame.teammate].astype(float)
    for ex in exclude_xy or []:
        if len(mates):
            d = np.linalg.norm(mates - ex, axis=1)
            if d.min() <= 0.5:  # drop only the single closest match
                mates = np.delete(mates, np.argmin(d), axis=0)
    d_opp = np.linalg.norm(opp - xy, axis=1) if len(opp) else np.array([])
    d_mate = np.linalg.norm(mates - xy, axis=1) if len(mates) else np.array([])
    # Goal-side defenders inside a 30-degree cone from xy towards the goal centre.
    to_goal = GOAL_CENTER - xy
    cone = 0
    if len(opp):
        v = opp - xy
        cos = (v @ to_goal) / (np.linalg.norm(v, axis=1) * np.linalg.norm(to_goal) + 1e-9)
        closer = np.linalg.norm(opp - GOAL_CENTER, axis=1) < np.linalg.norm(to_goal)
        cone = int(((cos > np.cos(np.radians(30))) & closer).sum())
    return {
        "opp_nearest": float(min(d_opp.min(), 30.0)) if len(d_opp) else 30.0,
        "opp_within_5": int((d_opp < 5).sum()),
        "opp_within_10": int((d_opp < 10).sum()),
        "opp_goal_side_cone": cone,
        "mates_ahead": int((mates[:, 0] > xy[0]).sum()) if len(mates) else 0,
        "mates_within_10": int((d_mate < 10).sum()),
        "n_visible": len(frame),
        "visible_frac": visible_frac,
    }


def loc_features(xy: np.ndarray) -> dict:
    xy = np.asarray(xy, float)
    return {"x": xy[..., 0], "y": xy[..., 1], "dist_goal": distance_to_goal(xy),
            "angle_goal": angle_to_goal(xy)}  # fmt: skip


def build_value_table(actions: pd.DataFrame, store: FrameStore, frames: pd.DataFrame) -> pd.DataFrame:
    """Actions (with a frame) + loc and ctx features."""
    vis = frames.set_index("event_id")["visible_frac"]
    a = actions[actions["event_id"].isin(store.event_ids)].reset_index(drop=True)
    xy = a[["x", "y"]].to_numpy(float)
    loc = pd.DataFrame(loc_features(xy))
    ctx = [
        context_features(store[e], xy[i], exclude_xy=[xy[i]], visible_frac=vis.get(e, np.nan))
        for i, e in enumerate(tqdm(a["event_id"], desc="value ctx"))
    ]
    return pd.concat([a, loc.drop(columns=["x", "y"]), pd.DataFrame(ctx)], axis=1)


class ValueModel:
    """Two calibrated-ish LightGBM classifiers: P(score soon) and P(concede soon)."""

    def __init__(self, features: list[str], **params):
        self.features = features
        self.params = {
            "n_estimators": 500, "learning_rate": 0.03, "num_leaves": 31,
            "min_child_samples": 200, "subsample": 0.8, "subsample_freq": 1,
            "colsample_bytree": 0.8, "reg_lambda": 5.0, "verbose": -1,
        } | params  # fmt: skip

    def fit(self, table: pd.DataFrame):
        X = table[self.features]
        self.score_model = lgb.LGBMClassifier(**self.params).fit(X, table["scores"])
        self.concede_model = lgb.LGBMClassifier(**self.params).fit(X, table["concedes"])
        return self

    def predict(self, table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        X = table[self.features]
        return self.score_model.predict_proba(X)[:, 1], self.concede_model.predict_proba(X)[:, 1]

    def value(self, table: pd.DataFrame) -> np.ndarray:
        s, c = self.predict(table)
        return s - c


class PossessionValueModel:
    """V(state) = E[xG rest of our possession] - E[xG of their next possession], one regressor."""

    def __init__(self, features: list[str], **params):
        self.features = features
        self.params = {
            "n_estimators": 500, "learning_rate": 0.03, "num_leaves": 31,
            "min_child_samples": 200, "subsample": 0.8, "subsample_freq": 1,
            "colsample_bytree": 0.8, "reg_lambda": 5.0, "verbose": -1,
        } | params  # fmt: skip

    def fit(self, table: pd.DataFrame):
        t = table[table["poss_value"].notna()]
        self.model = lgb.LGBMRegressor(**self.params).fit(t[self.features], t["poss_value"])
        return self

    def value(self, table: pd.DataFrame) -> np.ndarray:
        return self.model.predict(table[self.features])
