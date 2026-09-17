import numpy as np

from phds.features.corner_graph import GRAPH_FEATURES, MAX_NODES, NODE_FEATURES, node_features
from phds.models.defence_search import X_IDX, Y_IDX, p_attack, suggest_defence

ATT = NODE_FEATURES.index("attacker")


def _setup():
    # Taker, one attacker at the penalty spot, two defenders, a keeper.
    xy = np.array([[120.0, 0.1], [108.0, 40.0], [112.0, 30.0], [104.0, 55.0], [119.0, 40.0]])
    att = np.array([True, True, False, False, False])
    tak = np.array([True, False, False, False, False])
    kee = np.array([False, False, False, False, True])
    x = np.zeros((MAX_NODES, len(NODE_FEATURES)), np.float32)
    x[:5] = node_features(xy, att, tak, kee)
    mask = np.zeros(MAX_NODES, bool)
    mask[:5] = True
    cand = mask.copy()
    cand[0] = False
    return x, mask, cand, np.zeros(len(GRAPH_FEATURES), np.float32)


def toy_predictor(X, mask, cand, G):
    """Attacker's share rises the further the nearest defender is from them."""
    pos = np.stack([X[..., X_IDX] * 120, X[..., Y_IDX] * 80], -1)
    att = X[..., ATT] == 1
    probs = np.zeros(mask.shape)
    for b in range(len(X)):
        a = np.flatnonzero(att[b] & cand[b])[0]
        defs = np.flatnonzero(~att[b] & mask[b])
        gap = np.linalg.norm(pos[b, defs] - pos[b, a], axis=1).min()
        p_att = 1 / (1 + np.exp(-(gap - 4)))
        probs[b, a] = p_att
        probs[b, defs] = (1 - p_att) / len(defs)
    return probs


def test_search_moves_defender_towards_attacker_within_constraints():
    x, mask, cand, g = _setup()
    s = suggest_defence(toy_predictor, x, mask, cand, g, max_move=3.0)
    assert s.p_attack_after < s.p_attack_before
    moved = np.linalg.norm(s.final_xy - s.start_xy, axis=1)
    assert moved.max() <= 3.0 + 1e-6
    assert moved[[0, 1, 4]].max() == 0  # taker, attacker and keeper never move
    # The defender nearest the attacker closed the gap.
    attacker = s.start_xy[1]
    assert np.linalg.norm(s.final_xy[2] - attacker) < np.linalg.norm(s.start_xy[2] - attacker)
    others = np.delete(s.final_xy, 2, axis=0)
    assert np.linalg.norm(others - s.final_xy[2], axis=1).min() >= 1.0


def test_p_attack_sums_attacker_probabilities():
    x, *_ = _setup()
    probs = np.zeros((1, MAX_NODES))
    probs[0, 1], probs[0, 2] = 0.7, 0.3
    assert np.isclose(p_attack(probs, x[None])[0], 0.7)
