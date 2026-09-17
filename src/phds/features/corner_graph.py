"""Turn corners + freeze frames into padded graph tensors.

Each corner is a complete graph over the visible players (<= 22 nodes). We pad
to MAX_NODES and carry a mask, so a whole dataset is one dense array. At this
size, dense masked attention is exactly equivalent to message passing on a
complete graph, with no graph library needed.

Canonicalisation (as in TacticAI): corners from the far flag (y = 80) are
reflected to the near flag (y = 0), so the model never has to learn left/right
wing symmetry from ~1k examples. The test suite checks that a corner and its
mirror image give identical features.

Only information available *at the moment of the kick* is used. Pass end
location, height, technique and outcome would leak the target.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from phds.geometry.coords import PITCH_WIDTH, angle_to_goal, distance_to_goal, flip_y

MAX_NODES = 22
NODE_FEATURES = [
    "x", "y", "attacker", "taker", "keeper", "dist_goal", "angle_goal", "dist_taker",
    "in_box", "in_six", "nearest_opp_dist", "opp_within_2", "mates_within_2",
    "dist_near_post", "dist_far_post",
]  # fmt: skip
GRAPH_FEATURES = [
    "visible_frac", "n_attackers", "n_defenders", "attackers_in_box", "defenders_in_box", "women",
]  # fmt: skip
NEAR_POST = np.array([120.0, 36.0])  # after canonicalisation the corner is taken from y = 0
FAR_POST = np.array([120.0, 44.0])


@dataclass
class CornerGraphs:
    x: np.ndarray  # [N, MAX_NODES, F] node features
    mask: np.ndarray  # [N, MAX_NODES] valid node
    cand: np.ndarray  # [N, MAX_NODES] can be the first toucher (valid and not the taker)
    g: np.ndarray  # [N, G] graph-level features
    target: np.ndarray  # [N] node index of the confident first toucher, -1 if none
    excess: np.ndarray  # [N, MAX_NODES] soft-label excess cost (yd), inf = not a candidate
    shot: np.ndarray  # [N] shot within 20 s
    meta: pd.DataFrame  # one row per corner: ids, split, label info

    def subset(self, idx) -> CornerGraphs:
        return CornerGraphs(self.x[idx], self.mask[idx], self.cand[idx], self.g[idx],
                            self.target[idx], self.excess[idx], self.shot[idx],
                            self.meta.iloc[idx].reset_index(drop=True))  # fmt: skip

    def save(self, path):
        np.savez_compressed(path, x=self.x, mask=self.mask, cand=self.cand, g=self.g,
                            target=self.target, excess=self.excess, shot=self.shot)  # fmt: skip
        self.meta.to_parquet(str(path).replace(".npz", "_meta.parquet"), index=False)

    @classmethod
    def load(cls, path) -> CornerGraphs:
        a = np.load(path)
        meta = pd.read_parquet(str(path).replace(".npz", "_meta.parquet"))
        return cls(a["x"], a["mask"], a["cand"], a["g"], a["target"], a["excess"], a["shot"], meta)


def node_features(xy: np.ndarray, attacker: np.ndarray, taker: np.ndarray, keeper: np.ndarray):
    """Per-player features for one canonicalised frame. xy: [n, 2]."""
    n = len(xy)
    d = np.linalg.norm(xy[:, None] - xy[None], axis=-1)
    np.fill_diagonal(d, np.inf)
    same = attacker[:, None] == attacker[None]
    opp_d = np.where(~same, d, np.inf)
    mate_d = np.where(same, d, np.inf)
    taker_xy = xy[taker][0] if taker.any() else np.array([120.0, 0.0])
    in_box = (xy[:, 0] >= 102) & (xy[:, 1] >= 18) & (xy[:, 1] <= 62)
    in_six = (xy[:, 0] >= 114) & (xy[:, 1] >= 30) & (xy[:, 1] <= 50)
    nearest_opp = opp_d.min(axis=1) if n > 1 else np.full(n, np.inf)
    f = np.column_stack([
        xy[:, 0] / 120.0,
        xy[:, 1] / 80.0,
        attacker, taker, keeper,
        distance_to_goal(xy) / 40.0,
        angle_to_goal(xy) / np.pi,
        np.linalg.norm(xy - taker_xy, axis=1) / 60.0,
        in_box, in_six,
        np.minimum(nearest_opp, 20.0) / 10.0,  # off-camera opponents -> capped
        (opp_d < 2.0).sum(axis=1) / 3.0,
        (mate_d < 2.0).sum(axis=1) / 3.0,
        np.linalg.norm(xy - NEAR_POST, axis=1) / 40.0,
        np.linalg.norm(xy - FAR_POST, axis=1) / 40.0,
    ])  # fmt: skip
    return f.astype(np.float32)


def build_corner_graphs(
    corners: pd.DataFrame, players: pd.DataFrame, matches: pd.DataFrame
) -> CornerGraphs:
    """Corners with a freeze frame -> padded arrays. `players` may be the full table."""
    c = corners[corners["label_status"] != "no_frame"].reset_index(drop=True)
    pl = players[players["event_id"].isin(set(c["event_id"]))].copy()
    pl["event_id"] = pl["event_id"].astype(str)
    by_event = {k: g.sort_values("player_idx") for k, g in pl.groupby("event_id", sort=False)}
    gender = matches.set_index("match_id")["competition_gender"]

    n = len(c)
    X = np.zeros((n, MAX_NODES, len(NODE_FEATURES)), np.float32)
    mask = np.zeros((n, MAX_NODES), bool)
    cand = np.zeros((n, MAX_NODES), bool)
    G = np.zeros((n, len(GRAPH_FEATURES)), np.float32)
    target = np.full(n, -1, np.int64)
    excess = np.full((n, MAX_NODES), np.inf, np.float32)
    keep = np.ones(n, bool)

    for i, row in c.iterrows():
        fp = by_event.get(row["event_id"])
        if fp is None or len(fp) == 0:
            keep[i] = False
            continue
        fp = fp.iloc[:MAX_NODES]
        xy = fp[["x", "y"]].to_numpy(float)
        flipped = row["y"] > PITCH_WIDTH / 2
        if flipped:
            xy = flip_y(xy)
        att = fp["teammate"].to_numpy(bool)
        tak = fp["actor"].to_numpy(bool)
        kee = fp["keeper"].to_numpy(bool)
        m = len(fp)
        X[i, :m] = node_features(xy, att, tak, kee)
        mask[i, :m] = True
        cand[i, :m] = ~tak

        in_box = (xy[:, 0] >= 102) & (xy[:, 1] >= 18) & (xy[:, 1] <= 62)
        G[i] = [row["visible_frac"], att.sum() / 11, (~att).sum() / 11, (att & in_box).sum() / 11,
                (~att & in_box).sum() / 11,
                float(gender.get(row["match_id"]) == "female")]  # fmt: skip

        pos = {int(p): k for k, p in enumerate(fp["player_idx"])}
        if row["label_status"] == "ok" and pd.notna(row["receiver_idx"]):
            target[i] = pos.get(int(row["receiver_idx"]), -1)
        soft_idx = row.get("soft_player_idx")
        if soft_idx is not None and len(soft_idx):
            for p, e in zip(soft_idx, row["soft_excess"]):
                if int(p) in pos:
                    excess[i, pos[int(p)]] = e

    meta = c[["match_id", "event_id", "team", "split", "label_status", "label_team",
              "label_method", "shot_within", "xg_within", "y"]].copy()  # fmt: skip
    meta["flipped"] = meta["y"] > PITCH_WIDTH / 2
    graphs = CornerGraphs(X, mask, cand, G, target, excess,
                          c["shot_within"].to_numpy(bool), meta)  # fmt: skip
    return graphs.subset(np.flatnonzero(keep))
