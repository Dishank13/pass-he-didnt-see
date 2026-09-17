import numpy as np
import pandas as pd

from phds.features.corner_graph import MAX_NODES, build_corner_graphs
from phds.geometry.coords import flip_y

rng = np.random.default_rng(1)
XY = np.column_stack([rng.uniform(95, 120, 18), rng.uniform(15, 65, 18)])


def _corner(event_id, corner_y, xy, status="ok", receiver_idx=3):
    players = pd.DataFrame({
        "event_id": event_id, "player_idx": np.arange(len(xy)), "x": xy[:, 0], "y": xy[:, 1],
        "teammate": np.arange(len(xy)) < 9, "actor": np.arange(len(xy)) == 0,
        "keeper": np.arange(len(xy)) == 17,
    })  # fmt: skip
    players.loc[0, ["x", "y"]] = [120.0, corner_y]
    corner = dict(
        match_id=1, event_id=event_id, team="A", split="train", label_status=status,
        label_team="attack", label_method="assign", shot_within=True, xg_within=0.1,
        y=corner_y, visible_frac=0.2, receiver_idx=receiver_idx,
        soft_player_idx=[3, 4], soft_excess=[0.0, 2.0],
    )  # fmt: skip
    return corner, players


def _build(corner_rows, player_frames):
    matches = pd.DataFrame({"match_id": [1], "competition_gender": ["male"]})
    return build_corner_graphs(pd.DataFrame(corner_rows), pd.concat(player_frames), matches)


def test_mirrored_corners_have_identical_features():
    c1, p1 = _corner("a", 0.1, XY)
    c2, p2 = _corner("b", 79.9, flip_y(XY))
    g = _build([c1, c2], [p1, p2])
    np.testing.assert_allclose(g.x[0], g.x[1], atol=1e-5)
    assert list(g.meta["flipped"]) == [False, True]


def test_padding_masks_targets_and_soft_labels():
    c, p = _corner("a", 0.1, XY)
    g = _build([c], [p])
    assert g.x.shape == (1, MAX_NODES, g.x.shape[2])
    assert g.mask[0].sum() == 18 and not g.mask[0, 18:].any()
    assert not g.cand[0, 0] and g.cand[0, 1:18].all()  # taker can't be the first toucher
    assert g.target[0] == 3
    assert g.excess[0, 3] == 0 and g.excess[0, 4] == 2 and np.isinf(g.excess[0, 5])


def test_non_ok_corner_has_no_hard_target():
    c, p = _corner("a", 0.1, XY, status="ambiguous")
    assert _build([c], [p]).target[0] == -1


def test_gnn_masks_padding_and_taker():
    import torch

    from phds.models.corner_gnn import CornerGNN

    c, p = _corner("a", 0.1, XY)
    g = _build([c], [p])
    model = CornerGNN(g.x.shape[2], g.g.shape[1], hidden=16, layers=2, heads=2).eval()
    t = lambda a: torch.as_tensor(a)
    recv, shot = model(t(g.x), t(g.mask), t(g.cand), t(g.g))
    probs = torch.softmax(recv, -1)[0]
    assert torch.isfinite(shot).all()
    assert probs[0] == 0 and (probs[18:] == 0).all()  # taker and padding get no probability
    assert torch.isclose(probs.sum(), torch.tensor(1.0))
    # Padding content must not affect predictions.
    x2 = g.x.copy()
    x2[0, 18:] = 99.0
    recv2, shot2 = model(t(x2), t(g.mask), t(g.cand), t(g.g))
    assert torch.allclose(recv[0, :18], recv2[0, :18], atol=1e-5)
    assert torch.allclose(shot, shot2, atol=1e-5)
