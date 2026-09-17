import numpy as np
import pandas as pd

from phds.data.frame_store import Frame
from phds.features.pass_features import PASS_FEATURES
from phds.models.pass_value import PassOptionScorer, decision_summary


class ToyCompletion:
    """Completion falls with the physics margin closing: blocked lanes fail."""

    def predict(self, X: pd.DataFrame):
        return 1 / (1 + np.exp(-3 * X["physics_margin"].to_numpy()))


class ToyValue:
    """Value rises linearly towards the opponent goal (x = 120), always positive,
    so handing the opponent the ball always costs us something."""

    def value(self, X: pd.DataFrame):
        return X["x"].to_numpy() / 1200


def _frame():
    xy = np.array([
        [50, 40],   # passer (actor)
        [55, 30],   # safe square ball
        [90, 40],   # forward option, lane blocked by the defender below
        [85, 70],   # forward option, open
        [70, 40.3],  # defender in the lane to (90, 40)
    ], dtype=np.float32)  # fmt: skip
    return Frame(xy=xy, teammate=np.array([True, True, True, True, False]),
                 actor=np.array([True, False, False, False, False]), keeper=np.zeros(5, bool),
                 player_idx=np.arange(5))  # fmt: skip


def test_scorer_prefers_open_forward_option():
    s = PassOptionScorer(ToyCompletion(), ToyValue(), ToyValue()).score(_frame(), [50, 40])
    np.testing.assert_array_equal(s.player_idx, [1, 2, 3])
    assert s.p_complete[1] < 0.5 < s.p_complete[2]  # blocked vs open
    assert s.player_idx[s.best] == 3
    assert list(s.features.columns) == PASS_FEATURES
    # EV identity
    np.testing.assert_allclose(s.ev, s.p_complete * s.v_success + (1 - s.p_complete) * s.v_fail)


def test_failure_value_is_opponent_value_mirrored():
    s = PassOptionScorer(ToyCompletion(), ToyValue(), ToyValue()).score(_frame(), [50, 40])
    # Losing the ball at x=90 hands the opponent the ball at their x=30: our value -30/1200.
    np.testing.assert_allclose(s.v_fail[1], -30 / 1200, atol=1e-6)


def test_decision_summary():
    s = PassOptionScorer(ToyCompletion(), ToyValue(), ToyValue()).score(_frame(), [50, 40])
    d = decision_summary(s, actual_player_idx=1)
    assert d["n_options"] == 3 and not d["actual_is_best"]
    assert d["delta_ev"] > 0 and d["best_player_idx"] == 3
    assert decision_summary(s, actual_player_idx=3)["choice_percentile"] == 1.0
    assert decision_summary(s, actual_player_idx=99) == {}


def test_batch_scoring_matches_single_pass_scoring():
    scorer = PassOptionScorer(ToyCompletion(), ToyValue(), ToyValue())
    single = scorer.score(_frame(), [50, 40])
    batch = scorer.score_batch([("e1", _frame(), [50, 40], False, np.nan),
                                ("e2", _frame(), [50, 40], False, np.nan)])  # fmt: skip
    assert len(batch) == 6
    e1 = batch[batch["event_id"] == "e1"]
    np.testing.assert_allclose(e1["ev"].to_numpy(), single.ev)
    np.testing.assert_array_equal(e1["player_idx"].to_numpy(), single.player_idx)


def test_multiple_value_sets_in_one_batch():
    class Flat:
        def value(self, X):
            return np.full(len(X), 0.01)

    scorer = PassOptionScorer(ToyCompletion(), value_sets={"toy": (ToyValue(), ToyValue()),
                                                           "flat": (Flat(), Flat())})  # fmt: skip
    out = scorer.score_batch([("e1", _frame(), [50, 40], False, np.nan)])
    for c in ["ev_toy", "ev_flat", "v_success_flat", "v_fail_toy"]:
        assert c in out.columns
    # Flat value: EV = p*0.01 - (1-p)*0.01 = 0.01*(2p - 1), so it ranks options purely by P(complete).
    np.testing.assert_allclose(out["ev_flat"], 0.01 * (2 * out["p_complete"] - 1))
    single = scorer.score(_frame(), [50, 40])
    np.testing.assert_allclose(single.ev, out["ev_toy"])


def test_summarise_decisions():
    from phds.models.pass_value import summarise_decisions

    # One pass, three options. Option 7 was played; option 8 has the highest EV but is implausible;
    # option 9 is the best plausible option.
    opts = pd.DataFrame({
        "event_id": ["p"] * 3, "player_idx": [7, 8, 9], "opt_x": [50.0, 90.0, 70.0], "opt_y": [40.0] * 3,
        "p_complete": [0.95, 0.4, 0.85], "policy_p": [0.6, 0.02, 0.38], "length": [10.0, 40.0, 20.0],
        "dx": [0.0, 30.0, 10.0], "physics_margin": [2.0, -0.2, 1.0], "is_actual": [True, False, False],
        "ev_goal10": [0.01, 0.05, 0.03], "ev_poss": [0.02, 0.01, 0.03],
    })  # fmt: skip
    d = summarise_decisions(opts, pd.Series({"p": 7}), plausible=0.10).iloc[0]
    assert d["n_options"] == 3 and d["n_plausible"] == 2
    assert d["best_all_goal10_player_idx"] == 8 and d["best_plausible_goal10_player_idx"] == 9
    assert np.isclose(d["delta_ev_all_goal10"], 0.04) and np.isclose(d["delta_ev_plausible_goal10"], 0.02)
    assert d["choice_pct_goal10"] == 0.0 and d["choice_pct_poss"] == 0.5
    assert not d["actual_is_best_all_poss"] and d["best_all_poss_player_idx"] == 9
