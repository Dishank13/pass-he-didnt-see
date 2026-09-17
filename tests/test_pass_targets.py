import numpy as np
import pandas as pd

from phds.data.frame_store import FrameStore
from phds.data.pass_targets import build_pass_dataset, linkage_weights


def _ev(eid, type_, t, x, y, **kw):
    base = dict(
        match_id=1, event_id=eid, period=1, minute=0, timestamp=f"00:00:{t:06.3f}", type=type_,
        team_id=10, team="A", player_id=1, player="P", position="Center Midfield", x=x, y=y,
        pass_type=None, pass_outcome=None, pass_recipient=None, pass_end_x=np.nan,
        pass_end_y=np.nan, pass_length=np.nan, pass_height=None, under_pressure=False,
        related_events=[],
    )  # fmt: skip
    base.update(kw)
    return base


def _players(eid, rows):
    df = pd.DataFrame(rows, columns=["player_idx", "teammate", "actor", "keeper", "x", "y"])
    df["event_id"] = eid
    return df


def _pass(outcome, receipt_xy, receipt_frame=None, position="Center Midfield"):
    events = pd.DataFrame([
        _ev("p1", "Pass", 1.0, 50.0, 40.0, pass_outcome=outcome, pass_length=15.0,
            pass_end_x=receipt_xy[0], pass_end_y=receipt_xy[1], related_events=["r1"]),
        _ev("r1", "Ball Receipt*", 2.0, *receipt_xy, position=position),
    ])  # fmt: skip
    frame = _players("p1", [
        (0, True, True, False, 50.0, 40.0),   # passer
        (1, True, False, False, 64.0, 40.0),  # target
        (2, True, False, False, 60.0, 20.0),
        (3, True, False, True, 5.0, 40.0),    # own keeper
        (4, False, False, False, 58.0, 41.0),
    ])  # fmt: skip
    players = [frame] + ([receipt_frame] if receipt_frame is not None else [])
    return events, FrameStore.from_players(pd.concat(players))


def test_completed_pass_links_to_target_dot():
    events, store = _pass("Complete", (65.0, 40.5))
    row = build_pass_dataset(events, store).iloc[0]
    assert row["link_status"] == "ok" and row["target_idx"] == 1
    assert (row["target_x"], row["target_y"]) == (64.0, 40.0)  # frame position, not receipt position
    assert row["completed"] == 1 and row["n_teammates"] == 3


def test_incomplete_pass_uses_receipt_frame_assignment():
    # Receipt 2 s later: the target ran 7 yd. Too far for "nearest", fine for assignment.
    receipt = _players("r1", [
        (0, True, True, False, 71.0, 40.0),  # receiver (target who ran)
        (1, True, False, False, 61.0, 21.0),
        (2, True, False, True, 5.0, 40.0),
    ])  # fmt: skip
    events, store = _pass("Incomplete", (71.0, 40.0), receipt)
    row = build_pass_dataset(events, store).iloc[0]
    assert row["link_method"] == "assign" and row["target_idx"] == 1 and row["completed"] == 0


def test_backpass_to_keeper_uses_keeper_flag():
    events, store = _pass("Complete", (12.0, 45.0), position="Goalkeeper")
    row = build_pass_dataset(events, store).iloc[0]
    assert (row["link_method"], row["target_idx"]) == ("keeper", 3)


def test_linkage_weights_restore_in_view_completion_rate():
    # 100 completed passes (all in view, 80 linked); 40 failed passes, half off camera,
    # of the 20 in view only 10 linked. True in-view completion = 100 / 120.
    df = pd.DataFrame({
        "completed": [1] * 100 + [0] * 40,
        "length": [15.0] * 140,
        "receipt_in_view": [1.0] * 100 + [1.0] * 20 + [0.0] * 20,
        "link_status": ["ok"] * 80 + ["too_far"] * 20 + ["ok"] * 10 + ["too_far"] * 30,
    })  # fmt: skip
    w = linkage_weights(df)
    linked = df["link_status"] == "ok"
    naive = df.loc[linked, "completed"].mean()
    weighted = np.average(df.loc[linked, "completed"], weights=w[linked])
    assert abs(naive - 80 / 90) < 1e-9 and abs(weighted - 100 / 120) < 1e-9
