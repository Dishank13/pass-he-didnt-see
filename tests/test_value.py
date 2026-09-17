import numpy as np
import pandas as pd

from phds.data.frame_store import Frame
from phds.models.value import action_labels, context_features, possession_labels


def _actions(rows):
    base = dict(match_id=1, period=1, x=60.0, y=40.0, shot_outcome=None)
    return pd.DataFrame([base | dict(index=i, event_id=f"e{i}") | r for i, r in enumerate(rows)])


def test_action_labels_score_and_concede_windows():
    rows = [dict(type="Pass", team_id=1)] * 3 + [dict(type="Pass", team_id=2),
            dict(type="Shot", team_id=1, shot_outcome="Goal")]  # fmt: skip
    lab = action_labels(_actions(rows), horizon=3)
    # Window = last 3 actions up to and including the goal: indices 2, 3, 4.
    np.testing.assert_array_equal(lab["scores"], [False, False, True, False, True])
    np.testing.assert_array_equal(lab["concedes"], [False, False, False, True, False])


def test_action_labels_do_not_cross_periods():
    rows = [dict(type="Pass", team_id=1), dict(type="Shot", team_id=1, shot_outcome="Goal", period=2)]
    lab = action_labels(_actions(rows), horizon=5)
    np.testing.assert_array_equal(lab["scores"], [False, True])


def test_context_features_counts():
    frame = Frame(
        xy=np.array([[100, 40], [105, 40], [90, 40], [104, 55], [60, 40]], dtype=np.float32),
        teammate=np.array([True, False, False, False, True]),
        actor=np.array([True, False, False, False, False]),
        keeper=np.zeros(5, bool), player_idx=np.arange(5),
    )  # fmt: skip
    f = context_features(frame, np.array([100.0, 40.0]), exclude_xy=[np.array([100.0, 40.0])])
    assert f["opp_nearest"] == 5.0
    assert f["opp_within_5"] == 0 and f["opp_within_10"] == 1  # strict: 5 yd and 10 yd are excluded
    assert f["opp_goal_side_cone"] == 1  # only (105, 40) sits between xy and goal
    assert f["mates_ahead"] == 0 and f["mates_within_10"] == 0  # self excluded; (60,40) far behind
    assert f["n_visible"] == 5


def test_context_features_without_frame_is_nan():
    f = context_features(None, np.array([50.0, 40.0]))
    assert all(np.isnan(v) for v in f.values())


def test_possession_labels():
    base = dict(match_id=1, period=1, x=60.0, y=40.0, shot_outcome=None, shot_xg=np.nan)
    rows = [
        dict(possession=1, possession_team_id=1, team_id=1, type="Pass"),
        dict(possession=1, possession_team_id=1, team_id=2, type="Pressure"),  # defender, not an action row
        dict(possession=1, possession_team_id=1, team_id=1, type="Shot", shot_xg=0.10),
        dict(possession=1, possession_team_id=1, team_id=1, type="Pass"),       # after the shot
        dict(possession=2, possession_team_id=2, team_id=2, type="Pass"),
        dict(possession=2, possession_team_id=2, team_id=2, type="Shot", shot_xg=0.30),
        dict(possession=3, possession_team_id=2, team_id=2, type="Pass"),       # same team again
        dict(possession=4, possession_team_id=1, team_id=1, type="Carry"),
    ]
    ev = pd.DataFrame([base | dict(index=i, event_id=f"e{i}") | r for i, r in enumerate(rows)])
    lab = possession_labels(ev).set_index("event_id")
    assert "e1" not in lab.index  # the defender's pressure isn't a state of "team 1 has the ball"
    np.testing.assert_allclose(lab.loc["e0", ["xg_rest", "xg_against_next", "poss_value"]], [0.10, 0.30, -0.20])
    np.testing.assert_allclose(lab.loc["e3", "xg_rest"], 0.0)  # shot already happened
    np.testing.assert_allclose(lab.loc["e4", ["xg_rest", "xg_against_next"]], [0.30, 0.0])  # next opp poss = 4
    np.testing.assert_allclose(lab.loc["e6", "xg_against_next"], 0.0)
