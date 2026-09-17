"""Expected value of every passing option in a freeze frame.

For a pass by team T with frame F, each visible teammate j (at p_j) is an option:

    P_j        = P(pass to j completes)                        completion model (calibrated)
    V_succ(j)  = P(T scores soon) - P(T concedes soon)          value model, ball at p_j, frame context
    V_fail(j)  = -[P(opp scores soon) - P(opp concedes soon)]   value model, opponent's ball near p_j
    EV_j       = P_j * V_succ(j) + (1 - P_j) * V_fail(j)

"The pass he didn't see" is the gap between the best visible option and the pass actually
played:  delta_EV = max_j EV_j - EV_actual.

Decisions this module makes, and why:
  * Failure location = the target's position, mirrored into the opponent's frame. A failed pass
    is usually lost somewhere along the lane, and the target end is the conservative choice
    (a turnover near our own goal is costly). This is a known simplification.
  * The opponent's value uses the location-only model: their context would need the frame
    re-expressed from their side at a moment we don't observe.
  * Options are the *visible* teammates only (limitation D). "Best visible option" is the claim.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from phds.data.frame_store import Frame
from phds.features.pass_features import PASS_FEATURES, option_features
from phds.geometry.coords import to_opponent_perspective
from phds.models.value import CTX_FEATURES, LOC_FEATURES, context_features, loc_features


@dataclass
class OptionScores:
    player_idx: np.ndarray  # [k] frame player_idx of each option
    xy: np.ndarray  # [k, 2]
    p_complete: np.ndarray  # [k]
    v_success: np.ndarray  # [k]
    v_fail: np.ndarray  # [k]
    ev: np.ndarray  # [k]
    features: pd.DataFrame  # [k, PASS_FEATURES] completion features (for support checks)

    @property
    def best(self) -> int:
        return int(np.argmax(self.ev))

    def index_of(self, player_idx: int) -> int | None:
        hit = np.flatnonzero(self.player_idx == player_idx)
        return int(hit[0]) if len(hit) else None


def option_tables(frame: Frame, passer_xy, under_pressure: bool = False, visible_frac: float = np.nan):
    """Model inputs for every visible-teammate option in one frame, or None if there are none.

    Returns (player_idx [k], xy [k, 2], completion features, success-value features,
    failure-value features). Shared by single-pass and batch scoring, so both paths are
    identical by construction.
    """
    passer_xy = np.asarray(passer_xy, float)
    opts = frame.teammate & ~frame.actor
    if not opts.any():
        return None
    xy = frame.xy[opts].astype(float)
    comp = pd.DataFrame(option_features(frame, passer_xy, xy, under_pressure, visible_frac),
                        columns=PASS_FEATURES)  # fmt: skip
    ctx = pd.DataFrame([context_features(frame, pt, exclude_xy=[pt], visible_frac=visible_frac)
                        for pt in xy])  # fmt: skip
    succ = pd.concat([pd.DataFrame(loc_features(xy)), ctx], axis=1)[LOC_FEATURES + CTX_FEATURES]
    fail = pd.DataFrame(loc_features(to_opponent_perspective(xy)))[LOC_FEATURES]
    return frame.player_idx[opts], xy, comp, succ, fail


class PassOptionScorer:
    def __init__(self, completion_model, value_ctx=None, value_loc=None, value_sets: dict | None = None):
        """completion_model.predict(DataFrame[PASS_FEATURES]) -> P(complete).

        Value models expose .value(DataFrame) -> V. Pass either one (value_ctx, value_loc)
        pair, or `value_sets={"name": (ctx_model, loc_model), ...}` to compute EV under several
        value definitions at once. The first set is the primary one used by `score()`.
        """
        self.completion = completion_model
        self.value_sets = value_sets or {"ev": (value_ctx, value_loc)}

    def _predict(self, comp, succ, fail):
        p = np.clip(self.completion.predict(comp), 0.0, 1.0)
        out = {}
        for name, (ctx_model, loc_model) in self.value_sets.items():
            v_succ = ctx_model.value(succ)
            v_fail = -loc_model.value(fail)
            out[name] = (v_succ, v_fail, p * v_succ + (1 - p) * v_fail)
        return p, out

    def score(self, frame: Frame, passer_xy, under_pressure: bool = False,
              visible_frac: float = np.nan) -> OptionScores | None:  # fmt: skip
        tables = option_tables(frame, passer_xy, under_pressure, visible_frac)
        if tables is None:
            return None
        pidx, xy, comp, succ, fail = tables
        p, out = self._predict(comp, succ, fail)
        v_succ, v_fail, ev = next(iter(out.values()))
        return OptionScores(pidx, xy, p, v_succ, v_fail, ev, comp)

    def score_batch(self, items) -> pd.DataFrame:
        """items: iterable of (event_id, frame, passer_xy, under_pressure, visible_frac).

        One row per (event_id, option): p_complete, the completion features, and for each value
        set `name`: v_success_<name>, v_fail_<name>, ev_<name> (with a single default set the
        columns are plain v_success, v_fail, ev). Predictions run once over all options.
        """
        comps, succs, fails, keys = [], [], [], []
        for event_id, frame, passer_xy, up, vf in items:
            tables = option_tables(frame, passer_xy, up, vf)
            if tables is None:
                continue
            pidx, xy, comp, succ, fail = tables
            comps.append(comp), succs.append(succ), fails.append(fail)
            keys.append(pd.DataFrame({"event_id": event_id, "player_idx": pidx,
                                      "opt_x": xy[:, 0], "opt_y": xy[:, 1]}))  # fmt: skip
        if not keys:
            return pd.DataFrame()
        comp, succ, fail = (pd.concat(t, ignore_index=True) for t in (comps, succs, fails))
        p, preds = self._predict(comp, succ, fail)
        out = pd.concat([pd.concat(keys, ignore_index=True), comp], axis=1)
        out["p_complete"] = p
        single = list(preds) == ["ev"]
        for name, (v_succ, v_fail, ev) in preds.items():
            sfx = "" if single else f"_{name}"
            out[f"v_success{sfx}"], out[f"v_fail{sfx}"], out[f"ev{sfx}"] = v_succ, v_fail, ev
        return out


def decision_summary(scores: OptionScores, actual_player_idx: int) -> dict:
    """Per-pass decision metrics given which option was actually played."""
    k = scores.index_of(actual_player_idx)
    if k is None:
        return {}
    ev, best = scores.ev, scores.best
    n = len(ev)
    return {
        "n_options": n,
        "ev_actual": float(ev[k]),
        "ev_best": float(ev[best]),
        "delta_ev": float(ev[best] - ev[k]),
        "p_actual": float(scores.p_complete[k]),
        "p_best": float(scores.p_complete[best]),
        "actual_is_best": bool(k == best),
        # 1.0 = chose the best option, 0.0 = chose the worst (ties share the average rank).
        "choice_percentile": float(((ev < ev[k]).sum() + 0.5 * ((ev == ev[k]).sum() - 1)) / max(n - 1, 1)),
        "best_player_idx": int(scores.player_idx[best]),
        "best_x": float(scores.xy[best, 0]),
        "best_y": float(scores.xy[best, 1]),
    }


def summarise_decisions(opts: pd.DataFrame, target_idx: pd.Series, plausible: float,
                        value_defs=("goal10", "poss")) -> pd.DataFrame:
    """Per-pass decision summary from the long options table.

    opts: one row per (event_id, option) with player_idx, opt_x/opt_y, p_complete, policy_p,
          length, dx, physics_margin, is_actual and ev_<v> for each value definition.
    target_idx: player_idx of the option actually played, indexed by event_id.
    plausible: selection-probability threshold. The played option always counts as plausible.

    For each value definition v and scope in {all, plausible}: the best option's EV / P /
    policy P / geometry, delta_ev (best - actual), and whether the played option was the best.
    choice_pct_<v> ranks the played option among *all* options (0 worst, 1 best, ties averaged).
    """
    o = opts.copy()
    g = o.groupby("event_id", sort=False)
    o["n_options"] = g["ev_poss"].transform("size")
    o["plausible"] = (o["policy_p"] >= plausible) | o["is_actual"]
    for v in value_defs:  # choice percentile: 0 = worst option by EV, 1 = best (ties averaged)
        o[f"pct_{v}"] = (g[f"ev_{v}"].rank(method="average") - 1) / (o["n_options"] - 1).clip(lower=1)
    rows = pd.DataFrame({"event_id": o.loc[o["is_actual"], "event_id"].to_numpy()})
    act = o[o["is_actual"]].set_index("event_id")
    rows = rows.merge(act[["n_options", "policy_p", "p_complete", "length", "dx", "physics_margin"]]
                      .add_prefix("actual_").rename(columns={"actual_n_options": "n_options"}),
                      left_on="event_id", right_index=True)  # fmt: skip
    rows["n_plausible"] = rows["event_id"].map(o.groupby("event_id")["plausible"].sum())
    for v in value_defs:
        ev = f"ev_{v}"
        rows[f"ev_actual_{v}"] = rows["event_id"].map(act[ev])
        rows[f"choice_pct_{v}"] = rows["event_id"].map(act[f"pct_{v}"])
        for scope, sub in [("all", o), ("plausible", o[o["plausible"]])]:
            best = sub.loc[sub.groupby("event_id", sort=False)[ev].idxmax()].set_index("event_id")
            pre = f"best_{scope}_{v}_"
            for col in ["player_idx", "opt_x", "opt_y", ev, "p_complete", "policy_p", "length", "dx"]:
                rows[pre + col.replace(ev, "ev")] = rows["event_id"].map(best[col])
            rows[f"delta_ev_{scope}_{v}"] = rows[pre + "ev"] - rows[f"ev_actual_{v}"]
            rows[f"actual_is_best_{scope}_{v}"] = rows[pre + "player_idx"].eq(
                rows["event_id"].map(target_idx))  # fmt: skip
    return rows
