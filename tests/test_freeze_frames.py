from phds.data.freeze_frames import parse_event, parse_three_sixty

CORNER = {
    "id": "e1", "index": 231, "period": 1, "timestamp": "00:05:55.492", "minute": 5, "second": 55,
    "type": {"id": 30, "name": "Pass"}, "possession": 10,
    "possession_team": {"id": 780, "name": "Portugal"},
    "play_pattern": {"id": 2, "name": "From Corner"}, "team": {"id": 780, "name": "Portugal"},
    "player": {"id": 1, "name": "A"}, "location": [120.0, 0.1],
    "pass": {"recipient": {"id": 2, "name": "B"}, "end_location": [102.7, 46.0],
             "outcome": {"id": 9, "name": "Incomplete"}, "type": {"id": 61, "name": "Corner"}},
}  # fmt: skip


def test_parse_event_pass_fields():
    row = parse_event(CORNER, match_id=99)
    assert row["pass_type"] == "Corner"
    assert row["pass_outcome"] == "Incomplete"
    assert row["pass_recipient_id"] == 2  # intended recipient is kept even on incomplete passes
    assert (row["x"], row["y"], row["pass_end_x"], row["pass_end_y"]) == (120.0, 0.1, 102.7, 46.0)


def test_completed_pass_outcome_made_explicit():
    e = {**CORNER, "pass": {k: v for k, v in CORNER["pass"].items() if k != "outcome"}}
    assert parse_event(e, 99)["pass_outcome"] == "Complete"


def test_non_pass_has_no_pass_fields():
    e = {k: v for k, v in CORNER.items() if k != "pass"}
    e["type"] = {"id": 43, "name": "Carry"}
    row = parse_event(e, 99)
    assert row["pass_outcome"] is None and row["pass_recipient_id"] is None


def test_parse_three_sixty_counts_and_area():
    rec = {
        "event_uuid": "e1",
        "visible_area": [0, 0, 60, 0, 60, 80, 0, 80, 0, 0],  # left half = 50% of the pitch
        "freeze_frame": [
            {"teammate": True, "actor": True, "keeper": False, "location": [10, 10]},
            {"teammate": False, "actor": False, "keeper": True, "location": [5, 40]},
        ],
    }
    players, frames = parse_three_sixty([rec], match_id=99)
    assert len(players) == 2 and players[0]["actor"] and players[1]["keeper"]
    f = frames[0]
    assert (f["n_teammates"], f["n_opponents"], f["has_actor"]) == (1, 1, True)
    assert abs(f["visible_frac"] - 0.5) < 1e-9


def test_missing_visible_area_gives_nan_area():
    rec = {"event_uuid": "e2", "visible_area": [], "freeze_frame": []}
    _, frames = parse_three_sixty([rec], match_id=99)
    assert frames[0]["n_players"] == 0 and frames[0]["visible_frac"] != frames[0]["visible_frac"]


def test_visible_area_polygon_accepts_numpy_arrays():
    import numpy as np

    from phds.data.freeze_frames import visible_area_polygon

    poly = visible_area_polygon(np.array([0, 0, 60, 0, 60, 80, 0, 80, 0, 0], dtype=np.float32))
    assert abs(poly.area - 4800) < 1e-6
    assert visible_area_polygon(np.array([], dtype=np.float32)) is None
