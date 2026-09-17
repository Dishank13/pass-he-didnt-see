"""Pass features from broadcast tracking under four information conditions (M3).

For every pass (passer -> targeted teammate, outcome known) we describe the moment of the
pass four ways:

    condition   players used                      velocities   what it mimics
    full_vel    all 22 (detected + extrapolated)  yes          full tracking
    full_pos    all 22                            no           a perfect snapshot
    vis_vel     detected (on screen) only         yes          broadcast tracking, visible players
    vis_pos     detected only                     no           StatsBomb 360 freeze frame

Every condition gets the M2 static features (`option_features`: geometry, lane blocking,
constant-speed interception margins). Velocity conditions add *dynamic* margins, which
project each defender along their current velocity during the reaction time before they
sprint (the time-to-intercept idea behind the Spearman / Metrica pitch-control models):

    t_def = reaction + |p + v * reaction - point| / v_max

plus receiver movement (speed along and across the pass line) and defenders' closing speed.

Comparing conditions on the *same passes* isolates what each missing ingredient costs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from phds.data.frame_store import Frame
from phds.data.skillcorner import match_meta, to_statsbomb
from phds.features.pass_features import PASS_FEATURES, REACTION, V_BALL, V_PLAYER, option_features

VELOCITY_LAG_FRAMES = 5  # 0.5 s at 10 fps: backward difference, uses only the past
VELOCITY_FEATURES = [
    "dyn_lane_margin", "dyn_target_margin", "dyn_physics_margin", "receiver_speed",
    "receiver_v_along", "receiver_v_across", "max_defender_closing_speed", "passer_speed",
]  # fmt: skip
CONDITIONS = ("full_vel", "full_pos", "vis_vel", "vis_pos")


def velocity_features(opp_xy, opp_v, passer_xy, passer_v, target_xy, target_v) -> dict:
    """Velocity-aware interception margins and movement features (yards, yards/s)."""
    passer_xy, target_xy = np.asarray(passer_xy, float), np.asarray(target_xy, float)
    vec = target_xy - passer_xy
    length = max(np.linalg.norm(vec), 1e-6)
    unit = vec / length
    out = {
        "receiver_speed": float(np.linalg.norm(target_v)),
        "receiver_v_along": float(target_v @ unit),
        "receiver_v_across": float(abs(target_v[0] * unit[1] - target_v[1] * unit[0])),
        "passer_speed": float(np.linalg.norm(passer_v)),
    }
    if len(opp_xy) == 0:
        return out | {"dyn_lane_margin": 5.0, "dyn_target_margin": 5.0, "dyn_physics_margin": 5.0,
                      "max_defender_closing_speed": 0.0}  # fmt: skip
    drift = opp_xy + opp_v * REACTION  # where each defender is when they start to react
    t = np.clip(((opp_xy - passer_xy) @ vec) / length**2, 0, 1)
    lane_pt = passer_xy + t[:, None] * vec
    on_lane = (t > 0) & (t < 1)
    t_ball = t * length / V_BALL
    lane_margin = np.where(on_lane, REACTION + np.linalg.norm(drift - lane_pt, axis=1) / V_PLAYER - t_ball, np.inf)
    target_margin = REACTION + np.linalg.norm(drift - target_xy, axis=1) / V_PLAYER - length / V_BALL
    to_target = target_xy - opp_xy
    closing = (opp_v * to_target).sum(1) / np.maximum(np.linalg.norm(to_target, axis=1), 1e-6)
    lm, tm = float(min(lane_margin.min(), 5.0)), float(min(target_margin.min(), 5.0))
    return out | {"dyn_lane_margin": lm, "dyn_target_margin": tm, "dyn_physics_margin": min(lm, tm),
                  "max_defender_closing_speed": float(closing.max())}  # fmt: skip


def build_match_passes(paths: dict) -> pd.DataFrame:
    meta = match_meta(paths["match"])
    L, W = meta["length"], meta["width"]
    roles = meta["players"].set_index("player_id")
    ev = pd.read_csv(paths["events"], low_memory=False)
    pl = pd.read_parquet(paths["players"]).sort_values(["frame", "player_id"])
    f_arr = pl["frame"].to_numpy()

    def at(frame):
        lo, hi = np.searchsorted(f_arr, frame, "left"), np.searchsorted(f_arr, frame, "right")
        return pl.iloc[lo:hi]

    passes = ev[ev["event_type"].eq("player_possession") & ev["end_type"].eq("pass")
                & ev["pass_outcome"].isin(["successful", "unsuccessful"]) & ev["player_targeted_id"].notna()]  # fmt: skip
    rows = []
    for p in passes.itertuples(index=False):
        f = int(p.frame_end)
        now, before = at(f), at(f - VELOCITY_LAG_FRAMES)
        if now.empty or before.empty:
            continue
        ltr = p.attacking_side == "left_to_right"
        now = now.merge(before[["player_id", "x", "y"]], on="player_id", how="left", suffixes=("", "_prev"))
        x, y = to_statsbomb(now["x"], now["y"], ltr, L, W)
        xp, yp = to_statsbomb(now["x_prev"].fillna(now["x"]), now["y_prev"].fillna(now["y"]), ltr, L, W)
        xy = np.column_stack([x, y])
        vel = (xy - np.column_stack([xp, yp])) / (VELOCITY_LAG_FRAMES / 10.0)
        pid = now["player_id"].to_numpy()
        team = roles["team_id"].reindex(pid).to_numpy()
        passer_i = np.flatnonzero(pid == p.player_id)
        target_i = np.flatnonzero(pid == int(p.player_targeted_id))
        if len(passer_i) != 1 or len(target_i) != 1:
            continue
        pi, ti = int(passer_i[0]), int(target_i[0])
        mate = team == p.team_id
        detected = now["is_detected"].to_numpy(bool)
        keeper = roles["role"].reindex(pid).eq("GK").to_numpy()

        row = {
            "match_id": p.match_id, "frame": f, "period": p.period, "minute": p.minute_start,
            "block": f"{p.match_id}_{p.period}_{int(p.minute_start) // 10}", "team_id": p.team_id,
            "passer_id": p.player_id, "target_id": int(p.player_targeted_id),
            "completed": int(p.pass_outcome == "successful"), "sk_xpass": p.player_targeted_xpass_completion,
            "high_pass": p.high_pass, "passer_detected": bool(detected[pi]),
            "target_detected": bool(detected[ti]), "n_detected": int(detected.sum()),
            "n_players": len(pid), "velocity_missing": int(now["x_prev"].isna().sum()),
        }  # fmt: skip
        for cond in CONDITIONS:
            keep = np.ones(len(pid), bool) if cond.startswith("full") else detected.copy()
            keep[[pi, ti]] = True  # passer and target always present (sample is restricted later)
            frame = Frame(xy=xy[keep], teammate=mate[keep], actor=(np.arange(len(pid)) == pi)[keep],
                          keeper=keeper[keep], player_idx=pid[keep])  # fmt: skip
            static = option_features(frame, xy[pi], xy[[ti]], False, np.nan)[0]
            feats = dict(zip(PASS_FEATURES, static))
            if cond.endswith("vel"):
                opp = keep & ~mate
                feats |= velocity_features(xy[opp], vel[opp], xy[pi], vel[pi], xy[ti], vel[ti])
            row |= {f"{cond}__{k}": v for k, v in feats.items()}
        rows.append(row)
    return pd.DataFrame(rows)


def condition_columns(df: pd.DataFrame, cond: str) -> list[str]:
    return [c for c in df.columns if c.startswith(f"{cond}__")]
