import numpy as np
import pandas as pd

from phds.data.labels import build_corner_dataset, link_by_assignment, nearest_player


def test_nearest_player_ok():
    m = nearest_player(np.array([[110, 40], [100, 30]]), np.array([111, 41]))
    assert (m.idx, m.status) == (0, "ok")


def test_nearest_player_ambiguous_when_two_close():
    m = nearest_player(np.array([[110, 40], [110.5, 41]]), np.array([110.2, 40.5]))
    assert m.status == "ambiguous"


def test_nearest_player_too_far():
    m = nearest_player(np.array([[60, 40]]), np.array([110, 40]))
    assert m.status == "too_far"


def test_nearest_player_no_candidates():
    assert nearest_player(np.empty((0, 2)), np.array([1, 1])).status == "no_candidates"


def _event(i, t, type_, team_id, x, y, **kw):
    base = dict(
        match_id=1, event_id=f"e{i}", index=i, period=1, minute=0, timestamp=f"00:00:{t:06.3f}",
        type=type_, possession=5, team_id=team_id, team=f"T{team_id}", player=f"P{i}",
        x=x, y=y, pass_type=None, pass_end_x=np.nan, pass_end_y=np.nan, pass_outcome=None,
        pass_height=None, pass_technique=None, shot_xg=np.nan, ball_receipt_outcome=None,
        position=None,
    )  # fmt: skip
    base.update(kw)
    return base


def _frame(event_id, rows=None):
    # Corner team (teammate=True) attacks x=120. Actor at the corner flag.
    rows = rows or [
        (0, True, True, 120.0, 0.1),
        (1, True, False, 110.0, 40.0),  # attacker near the penalty spot
        (2, True, False, 100.0, 60.0),
        (3, False, False, 112.0, 30.0),  # defender near the post
        (4, False, False, 104.0, 50.0),
    ]
    players = pd.DataFrame(rows, columns=["player_idx", "teammate", "actor", "x", "y"])
    players["event_id"], players["match_id"], players["keeper"] = event_id, 1, False
    frames = pd.DataFrame(
        [dict(event_id=event_id, visible_frac=0.3, n_players=len(rows), n_teammates=3,
              n_opponents=2)]
    )
    return players, frames


def test_corner_attacking_receiver_and_shot():
    events = pd.DataFrame(
        [
            _event(0, 1.0, "Pass", 10, 120.0, 0.1, pass_type="Corner", pass_end_x=110.0,
                   pass_end_y=41.0, pass_outcome="Complete"),
            _event(1, 2.0, "Ball Receipt*", 10, 110.5, 40.5),
            _event(2, 2.5, "Shot", 10, 110.5, 40.5, shot_xg=0.2),
        ]
    )  # fmt: skip
    players, frames = _frame("e0")
    out = build_corner_dataset(events, players, frames).iloc[0]
    assert out["label_team"] == "attack" and out["label_status"] == "ok"
    assert out["receiver_idx"] == 1 and out["label_method"] == "nearest"
    assert out["into_box"] and out["shot_within"] and abs(out["xg_within"] - 0.2) < 1e-9


def test_corner_defensive_first_touch_uses_opponent_perspective():
    # The defender's clearance is recorded from the defending team's view:
    # (8, 50) there == (112, 30) in the corner team's frame -> player_idx 3.
    events = pd.DataFrame(
        [
            _event(0, 1.0, "Pass", 10, 120.0, 0.1, pass_type="Corner", pass_end_x=112.0,
                   pass_end_y=30.0, pass_outcome="Incomplete"),
            _event(1, 2.0, "Ball Receipt*", 10, 112.0, 30.0, ball_receipt_outcome="Incomplete"),
            _event(2, 2.1, "Clearance", 20, 8.0, 50.0),
        ]
    )  # fmt: skip
    players, frames = _frame("e0")
    out = build_corner_dataset(events, players, frames).iloc[0]
    assert out["label_team"] == "defence" and out["receiver_idx"] == 3
    assert out["touch_type"] == "Clearance" and not out["shot_within"]


def test_corner_without_touch_in_window():
    events = pd.DataFrame(
        [
            _event(0, 1.0, "Pass", 10, 120.0, 0.1, pass_type="Corner", pass_end_x=60.0,
                   pass_end_y=90.0, pass_outcome="Out"),
            _event(1, 30.0, "Pass", 20, 5.0, 40.0),
        ]
    )  # fmt: skip
    players, frames = _frame("e0")
    out = build_corner_dataset(events, players, frames).iloc[0]
    assert out["label_status"] == "no_touch" and not out["into_box"]


def test_assignment_links_actor_through_team_movement():
    before = np.array([[110, 40], [104, 42]])
    after = np.array([[104.5, 40], [100, 50]])  # actor (row 0) ran in from 110
    m = link_by_assignment(before, after, after_actor=0, max_move=16.5)
    assert (m.idx, m.status) == (0, "ok") and m.confidence > 1.5


def test_assignment_handles_player_missing_from_one_frame():
    before = np.array([[110, 40]])
    after = np.array([[111, 41], [60, 10]])  # second player not in the corner frame
    assert link_by_assignment(before, after, after_actor=0, max_move=10).idx == 0
    assert link_by_assignment(before, after, after_actor=1, max_move=10).status == "too_far"


def test_corner_prefers_assignment_when_touch_frame_exists():
    corner_rows = [(0, True, True, 120.0, 0.1), (1, True, False, 110.0, 40.0),
                   (2, True, False, 104.0, 42.0), (3, False, False, 90.0, 40.0),
                   (4, False, False, 80.0, 40.0)]
    touch_rows = [(0, True, False, 104.5, 40.0), (1, True, False, 100.0, 50.0),
                  (2, False, False, 90.0, 40.0)]
    players_c, frames = _frame("e0", corner_rows)
    players_t, _ = _frame("e1", touch_rows)
    players_t.loc[0, "actor"] = True
    events = pd.DataFrame(
        [
            _event(0, 1.0, "Pass", 10, 120.0, 0.1, pass_type="Corner", pass_end_x=104.5,
                   pass_end_y=40.0, pass_outcome="Complete"),
            _event(1, 2.5, "Ball Receipt*", 10, 104.5, 40.0),
        ]
    )  # fmt: skip
    out = build_corner_dataset(events, pd.concat([players_c, players_t]), frames).iloc[0]
    assert out["nearest_idx"] == 2  # fooled: player 2 stood closest to where the ball landed
    assert out["assign_idx"] == 1  # team movement says the runner from the penalty spot
    # Both methods are confident but disagree, so the corner is dropped rather than guessed.
    assert out["label_status"] == "conflict" and out["receiver_idx"] is None


def test_corner_assignment_rescues_label_nearest_rejects():
    corner_rows = [(0, True, True, 120.0, 0.1), (1, True, False, 112.0, 40.0),
                   (2, True, False, 95.0, 55.0), (3, False, False, 90.0, 40.0),
                   (4, False, False, 80.0, 40.0)]
    # Runner covered 8 yd before the touch: too far for nearest (max 5), fine for assignment.
    touch_rows = [(0, True, False, 104.0, 40.0), (1, True, False, 94.0, 56.0),
                  (2, False, False, 90.0, 40.0)]
    players_c, frames = _frame("e0", corner_rows)
    players_t, _ = _frame("e1", touch_rows)
    players_t.loc[0, "actor"] = True
    events = pd.DataFrame(
        [
            _event(0, 1.0, "Pass", 10, 120.0, 0.1, pass_type="Corner", pass_end_x=104.0,
                   pass_end_y=40.0, pass_outcome="Complete"),
            _event(1, 2.5, "Ball Receipt*", 10, 104.0, 40.0),
        ]
    )  # fmt: skip
    out = build_corner_dataset(events, pd.concat([players_c, players_t]), frames).iloc[0]
    assert out["nearest_status"] == "too_far"
    assert out["label_status"] == "ok" and out["label_method"] == "assign"
    assert out["receiver_idx"] == 1


def test_goalkeeper_touch_uses_keeper_flag():
    players, frames = _frame("e0")
    players.loc[players["player_idx"] == 4, "keeper"] = True  # defender at (104, 50) is the keeper
    events = pd.DataFrame(
        [
            _event(0, 1.0, "Pass", 10, 120.0, 0.1, pass_type="Corner", pass_end_x=112.0,
                   pass_end_y=30.0, pass_outcome="Incomplete"),
            # Keeper claims at (112, 30) in our frame = (8, 50) in theirs. It's nearest to
            # defender 3, but the flag says the keeper is player 4.
            _event(1, 2.0, "Goal Keeper", 20, 8.0, 50.0, position="Goalkeeper"),
        ]
    )  # fmt: skip
    out = build_corner_dataset(events, players, frames).iloc[0]
    assert out["nearest_idx"] == 3
    assert (out["label_method"], out["receiver_idx"], out["label_status"]) == ("keeper", 4, "ok")
