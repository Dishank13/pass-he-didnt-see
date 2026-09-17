"""Features for "a pass from the passer to a teammate standing at p_j".

The same function describes a real pass (training) and a hypothetical pass to any
other visible teammate (counterfactual scoring), so the model can't tell them
apart by construction.

Physics prior: interception time margin
---------------------------------------
Limitation B: the completion model only sees passes that players chose to attempt.
Lanes nobody tries (e.g. through two defenders) are under-represented, so a purely
learned model can be naively optimistic about them. We add a crude physical model
that knows *why* such passes fail:

    ball time to lane point s:     t_ball = s / v_ball
    defender time to that point:   t_def  = reaction + perpendicular_distance / v_player
    margin = min over defenders (t_def - t_ball)    (< 0: some defender can get there first)

computed along the lane and at the target (a defender can also arrive before the
receiver controls it). It's a cheap stand-in for a pitch-control model: no velocities,
constant speeds. M3 quantifies what's lost without velocities.
"""

from __future__ import annotations

import numpy as np

from phds.data.frame_store import Frame
from phds.geometry.coords import distance_to_goal

V_BALL = 17.0  # yd/s, a firm ground pass (~15.5 m/s)
V_PLAYER = 7.0  # yd/s, sprinting defender
REACTION = 0.5  # s

PASS_FEATURES = [
    "length", "dx", "abs_dy", "passer_x", "passer_y", "target_x", "target_y", "target_dist_goal",
    "lane_min_perp", "lane_opp_within_2", "lane_opp_within_5", "lane_margin", "target_margin",
    "physics_margin", "passer_opp_nearest", "passer_opp_within_3", "target_opp_nearest",
    "target_opp_within_3", "target_opp_within_5", "under_pressure", "n_visible", "visible_frac",
]  # fmt: skip


def option_features(frame: Frame, passer_xy: np.ndarray, targets_xy: np.ndarray,
                    under_pressure: bool = False, visible_frac: float = np.nan) -> np.ndarray:  # fmt: skip
    """[n_targets, len(PASS_FEATURES)] feature matrix for passes from passer_xy to each target."""
    targets_xy = np.asarray(targets_xy, float).reshape(-1, 2)
    passer_xy = np.asarray(passer_xy, float)
    opp = frame.xy[~frame.teammate].astype(float)  # [m, 2]
    n = len(targets_xy)

    vec = targets_xy - passer_xy  # [n, 2]
    length = np.linalg.norm(vec, axis=1)
    safe_len = np.maximum(length, 1e-6)

    if len(opp):
        rel = opp[None, :, :] - passer_xy  # [1, m, 2]
        t = (rel * vec[:, None, :]).sum(-1) / safe_len[:, None] ** 2  # projection fraction [n, m]
        closest = passer_xy + np.clip(t, 0, 1)[..., None] * vec[:, None, :]
        perp = np.linalg.norm(opp[None] - closest, axis=-1)  # [n, m]
        on_lane = (t > 0) & (t < 1)
        perp_lane = np.where(on_lane, perp, np.inf)
        t_ball = np.clip(t, 0, 1) * length[:, None] / V_BALL
        lane_margin = np.where(on_lane, REACTION + perp / V_PLAYER - t_ball, np.inf).min(1)
        d_target = np.linalg.norm(opp[None] - targets_xy[:, None], axis=-1)  # [n, m]
        target_margin = (REACTION + d_target / V_PLAYER - (length / V_BALL)[:, None]).min(1)
        d_passer = np.linalg.norm(opp - passer_xy, axis=1)
        lane_min_perp = perp_lane.min(1)
        feats_opp = [
            np.minimum(lane_min_perp, 30.0), (perp_lane < 2).sum(1), (perp_lane < 5).sum(1),
            np.minimum(lane_margin, 5.0), np.minimum(target_margin, 5.0),
            np.minimum(np.minimum(lane_margin, target_margin), 5.0),
            np.full(n, min(d_passer.min(), 30.0)), np.full(n, (d_passer < 3).sum()),
            np.minimum(d_target.min(1), 30.0), (d_target < 3).sum(1), (d_target < 5).sum(1),
        ]  # fmt: skip
    else:  # no visible opponents: everything maximally open
        feats_opp = [np.full(n, 30.0), np.zeros(n), np.zeros(n), np.full(n, 5.0), np.full(n, 5.0),
                     np.full(n, 5.0), np.full(n, 30.0), np.zeros(n), np.full(n, 30.0), np.zeros(n),
                     np.zeros(n)]  # fmt: skip

    cols = [
        length, vec[:, 0], np.abs(vec[:, 1]), np.full(n, passer_xy[0]), np.full(n, passer_xy[1]),
        targets_xy[:, 0], targets_xy[:, 1], distance_to_goal(targets_xy), *feats_opp,
        np.full(n, float(under_pressure)), np.full(n, len(frame)), np.full(n, visible_frac),
    ]  # fmt: skip
    return np.column_stack(cols).astype(np.float32)
