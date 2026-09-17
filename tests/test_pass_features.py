import numpy as np

from phds.data.frame_store import Frame
from phds.features.pass_features import PASS_FEATURES, option_features

F = {name: i for i, name in enumerate(PASS_FEATURES)}


def _frame(opponents):
    xy = np.array([[50, 40], [70, 40], [70, 60]] + opponents, dtype=np.float32)
    n_mates = 3
    return Frame(xy=xy, teammate=np.arange(len(xy)) < n_mates, actor=np.arange(len(xy)) == 0,
                 keeper=np.zeros(len(xy), bool), player_idx=np.arange(len(xy)))  # fmt: skip


def test_defender_in_lane_blocks_only_that_lane():
    frame = _frame([[60, 40.5]])  # sits on the straight lane to (70, 40)
    f = option_features(frame, np.array([50, 40]), np.array([[70, 40], [70, 60]]))
    assert f[0, F["lane_min_perp"]] < 1 and f[0, F["lane_opp_within_2"]] == 1
    assert f[0, F["lane_margin"]] < 0  # defender reaches the lane before the ball
    assert f[1, F["lane_min_perp"]] > 5 and f[1, F["lane_margin"]] > 0


def test_geometry_features():
    f = option_features(_frame([]), np.array([50, 40]), np.array([[70, 60]]))
    assert np.isclose(f[0, F["length"]], np.hypot(20, 20))
    assert f[0, F["dx"]] == 20 and f[0, F["abs_dy"]] == 20
    assert f[0, F["physics_margin"]] == 5.0  # no opponents -> capped open


def test_defender_behind_passer_is_not_in_lane():
    frame = _frame([[45, 40]])
    f = option_features(frame, np.array([50, 40]), np.array([[70, 40]]))
    assert f[0, F["lane_opp_within_2"]] == 0 and f[0, F["passer_opp_nearest"]] == 5


def test_calibrated_completion_fixes_a_biased_model():
    import pandas as pd

    from phds.models.completion import CalibratedCompletion

    class Overconfident:  # predicts a probability that's systematically 0.15 too high
        def fit(self, X, y, w):
            return self

        def predict(self, X):
            return np.clip(X["p_true"].to_numpy() + 0.15, 0, 1)

    rng = np.random.default_rng(0)
    p_true = rng.uniform(0.2, 0.8, 20000)
    X = pd.DataFrame({"p_true": p_true})
    y = (rng.uniform(size=len(p_true)) < p_true).astype(int)
    groups = rng.integers(0, 100, len(p_true))
    model = CalibratedCompletion(Overconfident).fit(X, y, np.ones(len(y)), groups)
    raw_gap = model.predict_raw(X).mean() - y.mean()
    cal_gap = model.predict(X).mean() - y.mean()
    assert raw_gap > 0.1 and abs(cal_gap) < 0.01
