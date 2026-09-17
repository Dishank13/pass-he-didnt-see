import numpy as np

from phds.data.skillcorner import to_statsbomb
from phds.features.tracking_passes import velocity_features


def test_to_statsbomb_orientation():
    # Centre spot -> (60, 40); right touchline end (x=+52, y=0) attacked left-to-right -> x=120.
    x, y = to_statsbomb([0.0, 52.0, 0.0], [0.0, 0.0, 34.0], [True, True, True], 104, 68)
    np.testing.assert_allclose(x, [60, 120, 60])
    np.testing.assert_allclose(y, [40, 40, 0])  # y up in SkillCorner = top of the StatsBomb pitch
    # Attacking right-to-left: the same physical goal is now at x = 0 in the attacking frame.
    x2, y2 = to_statsbomb([52.0, 0.0], [0.0, 34.0], [False, False], 104, 68)
    np.testing.assert_allclose(x2, [0, 60])
    np.testing.assert_allclose(y2, [40, 80])


def test_defender_running_into_lane_reduces_dynamic_margin():
    passer, target = np.array([50.0, 40.0]), np.array([70.0, 40.0])
    opp = np.array([[60.0, 46.0]])  # 6 yd from the lane midpoint
    still = velocity_features(opp, np.zeros((1, 2)), passer, np.zeros(2), target, np.zeros(2))
    towards = velocity_features(opp, np.array([[0.0, -6.0]]), passer, np.zeros(2), target, np.zeros(2))
    away = velocity_features(opp, np.array([[0.0, 6.0]]), passer, np.zeros(2), target, np.zeros(2))
    assert towards["dyn_lane_margin"] < still["dyn_lane_margin"] < away["dyn_lane_margin"]
    assert towards["max_defender_closing_speed"] > 0 > away["max_defender_closing_speed"]


def test_receiver_movement_components():
    f = velocity_features(np.empty((0, 2)), np.empty((0, 2)), np.array([50.0, 40.0]), np.zeros(2),
                          np.array([70.0, 40.0]), np.array([3.0, 4.0]))  # fmt: skip
    assert np.isclose(f["receiver_speed"], 5) and np.isclose(f["receiver_v_along"], 3)
    assert np.isclose(f["receiver_v_across"], 4) and f["dyn_physics_margin"] == 5.0
