"""Infer labels that 360 freeze frames don't contain directly.

The core problem (limitation A in the plan)
-------------------------------------------
A freeze frame is a set of anonymous dots: location + teammate/actor/keeper flags.
It has no player IDs. So "who touched the corner first" has to be *inferred*.

Step 1: find the **first touch** after the corner, i.e. the first on-ball event by
either team within a few seconds.

Step 2: link that toucher back to a dot in the corner's freeze frame. Two methods:

* **nearest**: the player (of the toucher's team) in the corner frame nearest the
  touch location. Simple, but players move 5-10 yd while the ball is in the air,
  and defenders stand in tight clusters, so it is often ambiguous.

* **assign**: the touch event usually has *its own* freeze frame, in
  which the toucher is flagged as `actor`. We match every player of that team
  between the two frames by minimum total displacement (Hungarian algorithm).
  The actor's partner in the corner frame is the label. This uses the whole team's
  movement as a constraint. Confidence is the *regret*: how much the total
  displacement grows if that particular pairing is forbidden. Low regret means
  another pairing explains the movement almost as well, so the label is ambiguous.

Goalkeeper touches are resolved exactly, via the `keeper` flag present in every frame.

The final label combines both methods: take a confident assignment, else a
confident nearest match that the assignment doesn't contradict. If both are
confident but disagree, drop the corner. When both are confident, their agreement
rate is an empirical estimate of label noise, reported in the M0 audit (~94%).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from phds.geometry.coords import to_opponent_perspective

# Events that mean a player actually touched the ball. Duels/pressures/fouls aren't touches.
TOUCH_TYPES = {
    "Ball Receipt*", "Clearance", "Interception", "Goal Keeper", "Shot", "Miscontrol",
    "Block", "Pass", "Ball Recovery", "Dribble", "Carry",
}  # fmt: skip

# The penalty area in StatsBomb coordinates (attacking towards x = 120).
BOX_X_MIN, BOX_Y_MIN, BOX_Y_MAX = 102.0, 18.0, 62.0


@dataclass(frozen=True)
class LinkResult:
    idx: int | None  # position in the candidate array, or None
    score: float  # nearest: distance d1; assign: displacement of the linked pair
    confidence: float  # nearest: d2 - d1; assign: regret (yd of extra total displacement)
    status: str  # "ok" | "ambiguous" | "too_far" | "no_candidates"


def nearest_player(
    candidates_xy: np.ndarray, target_xy: np.ndarray, max_dist: float = 5.0, margin: float = 1.5
) -> LinkResult:
    """Match a target location to the nearest candidate player, refusing ambiguous cases."""
    candidates_xy = np.asarray(candidates_xy, dtype=float).reshape(-1, 2)
    if len(candidates_xy) == 0:
        return LinkResult(None, np.inf, 0.0, "no_candidates")
    d = np.linalg.norm(candidates_xy - np.asarray(target_xy, dtype=float), axis=1)
    order = np.argsort(d)
    d1 = float(d[order[0]])
    d2 = float(d[order[1]]) if len(d) > 1 else np.inf
    idx = int(order[0])
    if d1 > max_dist:
        return LinkResult(idx, d1, d2 - d1, "too_far")
    if d2 - d1 < margin:
        return LinkResult(idx, d1, d2 - d1, "ambiguous")
    return LinkResult(idx, d1, d2 - d1, "ok")


def _assignment_cost(cost: np.ndarray) -> float:
    r, c = linear_sum_assignment(cost)
    return float(cost[r, c].sum())


def link_by_assignment(
    before_xy: np.ndarray,
    after_xy: np.ndarray,
    after_actor: int,
    max_move: float,
    min_regret: float = 1.5,
) -> LinkResult:
    """Link `after_xy[after_actor]` to a player in `before_xy` via optimal assignment.

    Both arrays hold one team's players, in the same coordinate frame. Either frame
    may miss players (off camera). So a pairing costs `min(distance, max_move)`,
    and a pairing at the cap counts as "no partner". The cap also stops a single
    far-off player from dominating the solution.
    """
    before_xy = np.asarray(before_xy, dtype=float).reshape(-1, 2)
    after_xy = np.asarray(after_xy, dtype=float).reshape(-1, 2)
    if len(before_xy) == 0 or len(after_xy) == 0:
        return LinkResult(None, np.inf, 0.0, "no_candidates")

    dist = np.linalg.norm(after_xy[:, None, :] - before_xy[None, :, :], axis=-1)
    cost = np.minimum(dist, max_move)
    rows, cols = linear_sum_assignment(cost)
    match = dict(zip(rows, cols))
    if after_actor not in match or dist[after_actor, match[after_actor]] >= max_move:
        return LinkResult(None, np.inf, 0.0, "too_far")

    j = int(match[after_actor])
    best = float(cost[rows, cols].sum())
    forbidden = cost.copy()
    forbidden[after_actor, j] = 1e6  # re-solve without this pairing
    regret = _assignment_cost(forbidden) - best
    status = "ok" if regret >= min_regret else "ambiguous"
    return LinkResult(j, float(dist[after_actor, j]), float(regret), status)


def assignment_excess_costs(
    before_xy: np.ndarray, after_xy: np.ndarray, after_actor: int, max_move: float
) -> np.ndarray:
    """For each candidate j in `before_xy`: extra total displacement (yd) if the actor
    were player j, relative to the best candidate. Basis for soft labels.

    Forcing actor -> j is solved as cost[actor, j] + an optimal assignment of everyone
    else with row `actor` and column j removed. The best candidate has excess 0. The
    regret used in `link_by_assignment` is the second-smallest excess.
    """
    before_xy = np.asarray(before_xy, dtype=float).reshape(-1, 2)
    after_xy = np.asarray(after_xy, dtype=float).reshape(-1, 2)
    cost = np.minimum(np.linalg.norm(after_xy[:, None] - before_xy[None], axis=-1), max_move)
    other_rows = [i for i in range(len(after_xy)) if i != after_actor]
    totals = np.empty(len(before_xy))
    for j in range(len(before_xy)):
        other_cols = [k for k in range(len(before_xy)) if k != j]
        rest = cost[np.ix_(other_rows, other_cols)]
        totals[j] = cost[after_actor, j] + (_assignment_cost(rest) if rest.size else 0.0)
    return totals - totals.min()


def soft_label(excess: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Turn excess costs (yd) into a probability distribution over candidates.

    p_j ∝ exp(-excess_j / T). T is a hyperparameter: T -> 0 recovers the hard label,
    large T spreads the target over plausible alternatives.
    """
    z = -np.asarray(excess, dtype=float) / max(temperature, 1e-6)
    z -= z.max()
    p = np.exp(z)
    return p / p.sum()


def combine_links(near: LinkResult, near_idx, assign: LinkResult | None, assign_idx):
    """Combine the two linking methods into one label (thresholds chosen in the M0 audit).

    * a confident assignment wins;
    * otherwise a confident nearest match counts if the assignment doesn't point elsewhere;
    * if both are confident but disagree, the example is dropped ("conflict").

    `near_idx` / `assign_idx` are the linked ids (e.g. player_idx) for the two results.
    Returns (status, linked id or None, method).
    """
    n_ok = near_idx if near.status == "ok" else None
    a_ok = assign_idx if assign is not None and assign.status == "ok" else None
    if a_ok is not None and n_ok is not None and a_ok != n_ok:
        return "conflict", None, "both"
    if a_ok is not None:
        return "ok", a_ok, "assign"
    if n_ok is not None and (assign is None or assign.idx is None or assign_idx == n_ok):
        return "ok", n_ok, "nearest"
    status = assign.status if assign is not None else near.status
    return status, None, "assign" if assign is not None else "nearest"


def timestamp_seconds(ts: pd.Series) -> pd.Series:
    """'HH:MM:SS.mmm' (clock within a period) -> float seconds."""
    parts = ts.str.split(":", expand=True).astype(float)
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


def first_touch_after(match_events: pd.DataFrame, corner: pd.Series, max_seconds: float = 6.0):
    """The first on-ball event after `corner` in the same period, or None.

    `match_events` must be one match's events, sorted by `index`, with a `t` column
    (seconds, from `timestamp_seconds`). Failed receipts ("Ball Receipt*" with an
    outcome) are not touches: the ball never arrived. The lower time bound matters:
    StatsBomb has occasional glitched timestamps (e.g. "00:00:01" in minute 47).
    """
    dt = match_events["t"] - corner["t"]
    after = match_events[
        (match_events["index"] > corner["index"])
        & (match_events["period"] == corner["period"])
        & dt.between(0, max_seconds)
        & match_events["type"].isin(TOUCH_TYPES)
        & match_events["x"].notna()
    ]
    failed_receipt = (after["type"] == "Ball Receipt*") & after["ball_receipt_outcome"].notna()
    after = after[~failed_receipt & (after["event_id"] != corner["event_id"])]
    return None if after.empty else after.iloc[0]


def max_move_yards(dt: float) -> float:
    """Generous bound on how far a player can move in dt seconds (sprint ~9 yd/s)."""
    return 3.0 + 9.0 * max(dt, 0.0)


def build_corner_dataset(
    events: pd.DataFrame,
    players: pd.DataFrame,
    frames: pd.DataFrame,
    max_dist: float = 5.0,
    margin: float = 1.5,
    min_regret: float = 1.0,
    shot_window_s: float = 20.0,
) -> pd.DataFrame:
    """One row per corner, with the inferred first-touch label.

    Columns of note:
      label_team      "attack" | "defence" | None (no touch in the window)
      receiver_idx    `player_idx` (players.parquet, corner's frame) of the first toucher
      label_status    "ok" | "conflict" | "ambiguous" | "too_far" | "no_candidates" |
                      "no_touch" | "no_frame"
      label_method    which method produced the label ("keeper" | "assign" | "nearest" | "both")
      nearest_idx / nearest_status      the nearest-player method, always computed
      assign_idx / assign_status        the assignment method, when a touch frame exists
      shot_within     the corner team shot within `shot_window_s` s in the same possession
      soft_player_idx / soft_excess     candidates of the toucher's team and their excess
                      cost in yd (0 = best). Feed to `soft_label` for a training target.
    """
    ev = events.sort_values(["match_id", "index"]).copy()
    ev["t"] = timestamp_seconds(ev["timestamp"])
    corners = ev[(ev["type"] == "Pass") & (ev["pass_type"] == "Corner")]
    corners = corners.merge(
        frames[["event_id", "visible_frac", "n_players", "n_teammates", "n_opponents"]],
        on="event_id",
        how="left",
    )
    events_by_match = {k: g for k, g in ev.groupby("match_id", sort=False)}

    # First pass: find each corner's first touch (no player data needed). Then keep
    # only the freeze frames we'll use. The full players table has ~25M rows.
    touches = {
        c["event_id"]: first_touch_after(events_by_match[c["match_id"]], c)
        for _, c in corners.iterrows()
    }
    needed = set(corners["event_id"]) | {t["event_id"] for t in touches.values() if t is not None}
    needed_players = players[players["event_id"].isin(needed)].copy()
    needed_players["event_id"] = needed_players["event_id"].astype(str)
    players_by_event = {k: g for k, g in needed_players.groupby("event_id", sort=False)}

    rows = []
    for _, c in corners.iterrows():
        mev = events_by_match[c["match_id"]]
        row = {
            "match_id": c["match_id"], "event_id": c["event_id"], "team_id": c["team_id"],
            "team": c["team"], "taker": c["player"], "minute": c["minute"],
            "x": c["x"], "y": c["y"], "end_x": c["pass_end_x"], "end_y": c["pass_end_y"],
            "pass_outcome": c["pass_outcome"], "pass_height": c["pass_height"],
            "pass_technique": c["pass_technique"], "visible_frac": c["visible_frac"],
            "n_players": c["n_players"], "n_teammates": c["n_teammates"],
            "n_opponents": c["n_opponents"],
            "into_box": bool(
                c["pass_end_x"] >= BOX_X_MIN and BOX_Y_MIN <= c["pass_end_y"] <= BOX_Y_MAX
            ),
        }  # fmt: skip

        shots = mev[
            (mev["type"] == "Shot")
            & (mev["team_id"] == c["team_id"])
            & (mev["possession"] == c["possession"])
            & (mev["period"] == c["period"])
            & (mev["t"] - c["t"]).between(0, shot_window_s)
        ]
        row["shot_within"] = not shots.empty
        row["xg_within"] = float(shots["shot_xg"].fillna(0).sum())

        frame_players = players_by_event.get(c["event_id"])
        touch = touches[c["event_id"]]
        if frame_players is None:
            row.update(label_team=None, receiver_idx=None, label_status="no_frame")
            rows.append(row)
            continue
        if touch is None:
            row.update(label_team=None, receiver_idx=None, label_status="no_touch")
            rows.append(row)
            continue

        attack = touch["team_id"] == c["team_id"]
        dt = float(touch["t"] - c["t"])
        target = np.array([touch["x"], touch["y"]])
        if attack:
            cand = frame_players[frame_players["teammate"] & ~frame_players["actor"]]
        else:
            # The defender's event is in the defending team's frame. Rotate it into ours.
            target = to_opponent_perspective(target)
            cand = frame_players[~frame_players["teammate"]]
        cand_xy = cand[["x", "y"]].to_numpy()
        row.update(
            label_team="attack" if attack else "defence",
            touch_type=touch["type"], touch_player=touch["player"], touch_dt=dt,
            touch_x=float(target[0]), touch_y=float(target[1]),
        )  # fmt: skip

        near = nearest_player(cand_xy, target, max_dist, margin)
        row.update(
            nearest_idx=int(cand["player_idx"].iloc[near.idx]) if near.idx is not None else None,
            nearest_status=near.status, nearest_d=near.score, nearest_margin=near.confidence,
        )  # fmt: skip

        touch_players = players_by_event.get(touch["event_id"])
        assign = None
        if touch_players is not None and touch_players["actor"].any():
            tp = touch_players
            tp_xy = tp[["x", "y"]].to_numpy()
            if not attack:
                tp_xy = to_opponent_perspective(tp_xy)
            # In the touch frame, "teammate" means the toucher's team.
            same_team = tp["teammate"].to_numpy()
            after_xy = tp_xy[same_team]
            after_actor = int(np.flatnonzero(tp["actor"].to_numpy()[same_team])[0])
            assign = link_by_assignment(
                cand_xy, after_xy, after_actor, max_move_yards(dt), min_regret
            )
            row.update(
                assign_idx=(
                    int(cand["player_idx"].iloc[assign.idx]) if assign.idx is not None else None
                ),
                assign_status=assign.status, assign_move=assign.score,
                assign_regret=assign.confidence,
            )  # fmt: skip

        # Soft-label ingredients: excess cost per candidate (assignment if available,
        # else distance to the touch location), in yards.
        if assign is not None and assign.status != "no_candidates":
            excess = assignment_excess_costs(cand_xy, after_xy, after_actor, max_move_yards(dt))
        elif len(cand_xy):
            d = np.linalg.norm(cand_xy - target, axis=1)
            excess = d - d.min()
        else:
            excess = np.array([])
        row.update(
            soft_player_idx=cand["player_idx"].astype(int).tolist(),
            soft_excess=[float(e) for e in excess],
        )

        # Goalkeepers are flagged in every frame. If the keeper touched it, the label is exact.
        keepers = cand[cand["keeper"]] if "keeper" in cand else cand.iloc[:0]
        if touch.get("position") == "Goalkeeper" and len(keepers) == 1:
            k = int(keepers["player_idx"].iloc[0])
            row.update(label_status="ok", receiver_idx=k, label_method="keeper",
                       soft_player_idx=[k], soft_excess=[0.0])  # fmt: skip
            rows.append(row)
            continue

        status, receiver, method = combine_links(
            near, row["nearest_idx"], assign, row.get("assign_idx") if assign is not None else None
        )
        row.update(label_status=status, receiver_idx=receiver, label_method=method)
        rows.append(row)
    return pd.DataFrame(rows)
