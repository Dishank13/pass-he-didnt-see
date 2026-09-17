import numpy as np

from phds.geometry.coords import (
    angle_to_goal,
    distance_to_goal,
    flip_y,
    to_opponent_perspective,
)

rng = np.random.default_rng(0)
PTS = rng.uniform([0, 0], [120, 80], size=(50, 2))


def test_flip_y_is_involution():
    np.testing.assert_allclose(flip_y(flip_y(PTS)), PTS)


def test_flip_y_does_not_mutate_input():
    before = PTS.copy()
    flip_y(PTS)
    np.testing.assert_array_equal(PTS, before)


def test_flip_y_preserves_pairwise_distances():
    d = np.linalg.norm(PTS[:, None] - PTS[None], axis=-1)
    f = flip_y(PTS)
    np.testing.assert_allclose(np.linalg.norm(f[:, None] - f[None], axis=-1), d)


def test_goal_features_invariant_under_flip_y():
    # Distance/angle to goal must not depend on which wing you're on.
    np.testing.assert_allclose(distance_to_goal(flip_y(PTS)), distance_to_goal(PTS))
    np.testing.assert_allclose(angle_to_goal(flip_y(PTS)), angle_to_goal(PTS), atol=1e-7)


def test_opponent_perspective_round_trip():
    np.testing.assert_allclose(to_opponent_perspective(to_opponent_perspective(PTS)), PTS)
    np.testing.assert_allclose(to_opponent_perspective(np.array([0.0, 0.0])), [120.0, 80.0])


def test_angle_to_goal_on_goal_line_is_pi():
    np.testing.assert_allclose(angle_to_goal(np.array([120.0, 40.0])), np.pi, atol=1e-4)  # arccos is ill-conditioned near -1
