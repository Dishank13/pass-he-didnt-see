"""Suggest small defender adjustments that reduce the attackers' chance of winning the first touch.

TacticAI generated defensive setups with a conditional generative model. We start
with something simpler and fully inspectable: a constrained greedy local search
using the trained receiver model as the objective.

    objective  P(attacking team wins the first touch) = sum of receiver probs over attackers
    moves      each defender (keeper excluded) may shift up to `max_move` yd from where they stood,
               staying on the pitch and >= `min_gap` yd from every other player
    search     coordinate descent: repeatedly try every candidate move for every defender,
               apply the single best improving move, stop when nothing improves enough

Why not shot probability? In M1 the shot models barely beat the base rate, so optimising
them would optimise noise. The receiver model has real signal.

Honesty caveat: the model is correlational. It learned which setups tend to precede
attacker first touches, not what would happen if a defender moved. Small moves keep the
suggested setup close to configurations seen in training, but the output is a hypothesis
for a coach to evaluate, not a causal claim.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from phds.features.corner_graph import NODE_FEATURES, node_features

X_IDX, Y_IDX, ATT_IDX, TAKER_IDX, KEEPER_IDX = (
    NODE_FEATURES.index(k) for k in ("x", "y", "attacker", "taker", "keeper")
)
DIRECTIONS = np.array([[np.cos(a), np.sin(a)] for a in np.linspace(0, 2 * np.pi, 8, endpoint=False)])

# predictor(x [B, M, F], mask [B, M], cand [B, M], g [B, G]) -> receiver probs [B, M]
Predictor = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]


@dataclass
class Suggestion:
    start_xy: np.ndarray  # [n, 2] original positions (canonical frame)
    final_xy: np.ndarray  # [n, 2] suggested positions
    p_attack_before: float
    p_attack_after: float
    moves: list[tuple[int, float]]  # (node index, cumulative P(attack) after the move)


def p_attack(probs: np.ndarray, x: np.ndarray) -> np.ndarray:
    return (probs * (x[..., ATT_IDX] == 1)).sum(-1)


def _graph_row(g_row: np.ndarray, xy: np.ndarray, att: np.ndarray) -> np.ndarray:
    """Recompute the position-dependent graph features (in-box counts) for moved players."""
    out = g_row.copy()
    in_box = (xy[:, 0] >= 102) & (xy[:, 1] >= 18) & (xy[:, 1] <= 62)
    out[3], out[4] = (att & in_box).sum() / 11, (~att & in_box).sum() / 11
    return out


def suggest_defence(
    predictor: Predictor,
    x: np.ndarray,
    mask: np.ndarray,
    cand: np.ndarray,
    g: np.ndarray,
    max_move: float = 3.0,
    step_sizes: tuple[float, ...] = (1.0, 2.0, 3.0),
    min_gap: float = 1.0,
    max_rounds: int = 6,
    min_gain: float = 0.005,
) -> Suggestion:
    """Greedy constrained search for one corner. x/mask/cand: [M, ...], g: [G]."""
    n = int(mask.sum())
    xy0 = np.column_stack([x[:n, X_IDX] * 120, x[:n, Y_IDX] * 80]).astype(float)
    att = x[:n, ATT_IDX] == 1
    tak = x[:n, TAKER_IDX] == 1
    kee = x[:n, KEEPER_IDX] == 1
    movable = np.flatnonzero(~att & ~kee)

    def batch_eval(configs: list[np.ndarray]) -> np.ndarray:
        X = np.repeat(x[None], len(configs), 0)
        G = np.repeat(g[None], len(configs), 0)
        for b, xy in enumerate(configs):
            X[b, :n] = node_features(xy, att, tak, kee)
            G[b] = _graph_row(g, xy, att)
        M_, C_ = np.repeat(mask[None], len(configs), 0), np.repeat(cand[None], len(configs), 0)
        return p_attack(predictor(X, M_, C_, G), X)

    xy = xy0.copy()
    current = float(batch_eval([xy])[0])
    before, moves = current, []
    for _ in range(max_rounds):
        configs, who = [], []
        for i in movable:
            for step in step_sizes:
                for d in DIRECTIONS:
                    new = xy[i] + step * d
                    if np.linalg.norm(new - xy0[i]) > max_move + 1e-9:
                        continue
                    if not (0 <= new[0] <= 120 and 0 <= new[1] <= 80):
                        continue
                    others = np.delete(xy, i, axis=0)
                    if np.linalg.norm(others - new, axis=1).min() < min_gap:
                        continue
                    trial = xy.copy()
                    trial[i] = new
                    configs.append(trial)
                    who.append(int(i))
        if not configs:
            break
        scores = batch_eval(configs)
        best = int(np.argmin(scores))
        if current - scores[best] < min_gain:
            break
        xy, current = configs[best], float(scores[best])
        moves.append((who[best], current))
    return Suggestion(xy0, xy, before, current, moves)
