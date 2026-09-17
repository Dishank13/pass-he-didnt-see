"""Open-play passes with an identified *intended* target in the freeze frame.

For the pass-value model, the unit is "a pass from the passer to teammate j, who
stands at p_j in the frame". To train it we need, for real passes, which dot the
passer was aiming at. That's the same anonymous-dot problem as corners, solved the
same way:

    pass event --related_events--> Ball Receipt* of the intended recipient
      (exists for ~100% of completed and ~73% of incomplete passes; for failed passes
       its location is where the recipient was, not where the ball was intercepted)
    receipt location  -> nearest teammate in the pass frame           (method "nearest")
    receipt frame     -> whole-team assignment back to the pass frame (method "assign")
    receipt by a keeper -> the flagged keeper                          (method "keeper")

Why not use the pass end location as the target? For intercepted passes it lies on
the interceptor, so "a defender at the destination" would leak the outcome. And a
counterfactual pass to teammate j has no end location, only p_j.

Passes whose target can't be identified are dropped, and that isn't random: failed
passes lose their target far more often (no receipt for most "Out" passes).
`linkage_weights` gives models the inverse-probability weights to correct for this.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from shapely.geometry import Point
from tqdm import tqdm

from phds.data.frame_store import FrameStore
from phds.data.freeze_frames import visible_area_polygon
from phds.data.labels import (
    combine_links,
    link_by_assignment,
    max_move_yards,
    nearest_player,
    timestamp_seconds,
)

OUTCOMES = {"Complete": 1, "Incomplete": 0, "Out": 0}  # offside/unknown/injury excluded
LENGTH_BINS = [0, 10, 20, 30, 45, 200]  # yd, strata for the linkage-rate correction


def build_pass_dataset(
    events: pd.DataFrame,
    store: FrameStore,
    max_dist: float = 5.0,
    margin: float = 1.5,
    min_regret: float = 1.0,
) -> pd.DataFrame:
    """One row per open-play pass with a freeze frame (target columns are NaN when unlinked)."""
    ev = events.copy()
    ev["t"] = timestamp_seconds(ev["timestamp"])
    receipts = ev[ev["type"] == "Ball Receipt*"].set_index("event_id")
    receipt_ids = set(receipts.index)
    passes = ev[
        (ev["type"] == "Pass")
        & ev["pass_type"].isna()
        & ev["pass_outcome"].isin(list(OUTCOMES))
        & ev["event_id"].isin(store.event_ids)
    ]

    rows = []
    for p in tqdm(passes.itertuples(index=False), total=len(passes), desc="passes"):
        f = store[p.event_id]
        row = {
            "match_id": p.match_id, "event_id": p.event_id, "period": p.period, "minute": p.minute,
            "team_id": p.team_id, "team": p.team, "player_id": p.player_id, "player": p.player,
            "position": p.position, "recipient": p.pass_recipient, "x": p.x, "y": p.y,
            "end_x": p.pass_end_x, "end_y": p.pass_end_y, "length": p.pass_length,
            "outcome": p.pass_outcome, "completed": OUTCOMES[p.pass_outcome],
            "under_pressure": p.under_pressure, "pass_height": p.pass_height,
            "n_teammates": int((f.teammate & ~f.actor).sum()), "n_opponents": int((~f.teammate).sum()),
            "has_actor": bool(f.actor.any()),
        }  # fmt: skip
        rows.append(row)

        rid = next((r for r in p.related_events if r in receipt_ids), None)
        if rid is None:
            row["link_status"] = "no_receipt"
            continue
        rec = receipts.loc[rid]
        cand = f.teammate & ~f.actor
        cand_xy, cand_pid = f.xy[cand].astype(float), f.player_idx[cand]
        target = np.array([rec["x"], rec["y"]], dtype=float)
        dt = float(rec["t"] - p.t) if rec["period"] == p.period else 0.0
        row.update(receipt_x=target[0], receipt_y=target[1], receipt_dt=dt)
        if len(cand_xy) == 0:
            row["link_status"] = "no_candidates"
            continue

        near = nearest_player(cand_xy, target, max_dist, margin)
        near_idx = int(cand_pid[near.idx]) if near.idx is not None else None

        assign, assign_idx = None, None
        rf = store.get(rid)
        if rf is not None and rf.actor.any():
            same = rf.teammate  # receipt is by the passing team: same perspective as the pass
            after_xy = rf.xy[same].astype(float)
            after_actor = int(np.flatnonzero(rf.actor[same])[0])
            assign = link_by_assignment(cand_xy, after_xy, after_actor, max_move_yards(dt), min_regret)
            assign_idx = int(cand_pid[assign.idx]) if assign.idx is not None else None

        keepers = np.flatnonzero(f.keeper[cand])
        if rec["position"] == "Goalkeeper" and len(keepers) == 1:
            status, target_idx, method = "ok", int(cand_pid[keepers[0]]), "keeper"
        else:
            status, target_idx, method = combine_links(near, near_idx, assign, assign_idx)
        row.update(link_status=status, link_method=method, target_idx=target_idx,
                   nearest_d=near.score, assign_regret=assign.confidence if assign else np.nan)  # fmt: skip
        if target_idx is not None:
            k = int(np.flatnonzero(f.player_idx == target_idx)[0])
            row.update(target_x=float(f.xy[k, 0]), target_y=float(f.xy[k, 1]))
    return pd.DataFrame(rows)


def receipt_in_view(passes: pd.DataFrame, frames: pd.DataFrame, buffer: float = 1.0) -> pd.Series:
    """Was the intended recipient's receipt location inside the pass frame's camera view?

    NaN when the pass has no receipt event (recipient unknown).
    """
    polys = frames.set_index("event_id")["visible_area"]
    out = np.full(len(passes), np.nan)
    has = passes["receipt_x"].notna().to_numpy()
    for i in tqdm(np.flatnonzero(has), desc="receipt in view"):
        r = passes.iloc[i]
        poly = visible_area_polygon(polys.get(r["event_id"]))
        if poly is not None:
            out[i] = float(poly.buffer(buffer).contains(Point(r["receipt_x"], r["receipt_y"])))
    return pd.Series(out, index=passes.index, name="receipt_in_view")


def linkage_weights(passes: pd.DataFrame) -> pd.Series:
    """Weights that make linked passes represent *all passes to a visible recipient*.

    The completion model scores passes to teammates visible in the frame, so that's the
    population its probabilities must be calibrated for. Linkage fails far more often for
    failed passes, partly because their recipient is off camera (30% vs 14%) and partly
    because failed receipts are located less precisely. Within each (completed, length bin)
    stratum:

        target count  = all passes in stratum x P(recipient in view | stratum, has receipt)
        weight        = target count / linked passes in stratum

    Assumption: within a stratum, linked passes are representative of in-view passes.
    """
    linked = passes["link_status"].eq("ok")
    strata = [passes["completed"], pd.cut(passes["length"], LENGTH_BINS, include_lowest=True)]
    n_total = linked.groupby(strata, observed=True).transform("size")
    in_view_share = passes["receipt_in_view"].groupby(strata, observed=True).transform("mean")
    n_linked = linked.groupby(strata, observed=True).transform("sum")
    return (n_total * in_view_share / n_linked).where(linked, 0.0)
