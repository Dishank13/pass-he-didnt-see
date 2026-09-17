"""Parse cached StatsBomb JSON into tidy Parquet tables.

    matches.parquet  one row per match with 360 data
    events.parquet   one row per event (only the fields we use)
    players.parquet  one row per player per 360 freeze frame (long format)
    frames.parquet   one row per freeze frame: visible-area polygon + summary counts

Every event and its freeze frame use the perspective of the team *performing*
the event (attacking towards x = 120). When you join an event from team A to a
frame from team B, convert with `phds.geometry.coords.to_opponent_perspective`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from shapely.geometry import Polygon
from tqdm import tqdm

from phds.data.load_statsbomb import cached_json, matches_with_360
from phds.geometry.coords import PITCH_LENGTH, PITCH_WIDTH

PROCESSED_DIR = Path(__file__).resolve().parents[3] / "data" / "processed"


def _get(d, *path, default=None):
    for key in path:
        if not isinstance(d, dict) or key not in d:
            return default
        d = d[key]
    return d


def _xy(loc, i):
    return float(loc[i]) if loc is not None and len(loc) > i else np.nan


def parse_event(e: dict, match_id: int) -> dict:
    loc = e.get("location")
    p = e.get("pass")
    pass_end = _get(p, "end_location")
    carry_end = _get(e, "carry", "end_location")
    return {
        "match_id": match_id,
        "event_id": e["id"],
        "index": e["index"],
        "period": e["period"],
        "timestamp": e["timestamp"],
        "minute": e["minute"],
        "second": e["second"],
        "type": e["type"]["name"],
        "possession": e.get("possession"),
        "possession_team_id": _get(e, "possession_team", "id"),
        "play_pattern": _get(e, "play_pattern", "name"),
        "team_id": _get(e, "team", "id"),
        "team": _get(e, "team", "name"),
        "player_id": _get(e, "player", "id"),
        "player": _get(e, "player", "name"),
        "position": _get(e, "position", "name"),
        "x": _xy(loc, 0),
        "y": _xy(loc, 1),
        "duration": e.get("duration"),
        "under_pressure": bool(e.get("under_pressure", False)),
        "related_events": e.get("related_events", []),
        # pass
        "pass_recipient_id": _get(p, "recipient", "id"),
        "pass_recipient": _get(p, "recipient", "name"),
        "pass_end_x": _xy(pass_end, 0),
        "pass_end_y": _xy(pass_end, 1),
        "pass_length": _get(p, "length"),
        "pass_angle": _get(p, "angle"),
        "pass_height": _get(p, "height", "name"),
        # StatsBomb omits `outcome` for completed passes. We make it explicit.
        "pass_outcome": _get(p, "outcome", "name", default="Complete") if p is not None else None,
        "pass_type": _get(p, "type", "name"),
        "pass_body_part": _get(p, "body_part", "name"),
        "pass_technique": _get(p, "technique", "name"),
        "pass_cross": bool(_get(p, "cross", default=False)),
        "pass_switch": bool(_get(p, "switch", default=False)),
        "pass_cut_back": bool(_get(p, "cut_back", default=False)),
        "pass_through_ball": bool(_get(p, "through_ball", default=False)),
        "pass_shot_assist": bool(_get(p, "shot_assist", default=False)),
        "pass_goal_assist": bool(_get(p, "goal_assist", default=False)),
        # carry / shot / receipt
        "carry_end_x": _xy(carry_end, 0),
        "carry_end_y": _xy(carry_end, 1),
        "shot_xg": _get(e, "shot", "statsbomb_xg"),
        "shot_outcome": _get(e, "shot", "outcome", "name"),
        "ball_receipt_outcome": _get(e, "ball_receipt", "outcome", "name"),
    }


def visible_area_polygon(flat: list[float]) -> Polygon | None:
    """StatsBomb stores the polygon as a flat [x1, y1, x2, y2, ...] list."""
    if not flat or len(flat) < 6:
        return None
    ring = np.asarray(flat, dtype=float).reshape(-1, 2)
    return Polygon(ring).buffer(0)  # buffer(0) repairs occasional self-intersections


def parse_three_sixty(records: list[dict], match_id: int) -> tuple[list[dict], list[dict]]:
    players, frames = [], []
    for r in records:
        uuid = r["event_uuid"]
        va = r.get("visible_area") or []
        poly = visible_area_polygon(va)
        area = poly.area if poly is not None else np.nan
        ff = r.get("freeze_frame") or []
        frames.append(
            {
                "match_id": match_id,
                "event_id": uuid,
                "visible_area": [float(v) for v in va],
                "visible_area_sqyd": area,
                "visible_frac": area / (PITCH_LENGTH * PITCH_WIDTH),
                "n_players": len(ff),
                "n_teammates": sum(bool(p["teammate"]) for p in ff),
                "n_opponents": sum(not p["teammate"] for p in ff),
                "has_actor": any(p.get("actor", False) for p in ff),
            }
        )
        for i, p in enumerate(ff):
            players.append(
                {
                    "match_id": match_id,
                    "event_id": uuid,
                    "player_idx": i,
                    "teammate": bool(p["teammate"]),
                    "actor": bool(p.get("actor", False)),
                    "keeper": bool(p.get("keeper", False)),
                    "x": float(p["location"][0]),
                    "y": float(p["location"][1]),
                }
            )
    return players, frames


# Explicit schemas: a match where e.g. no shot happened would otherwise infer a
# different column type and break the streaming writer. `event_id` is
# dictionary-encoded, which matters for players.parquet (~25M rows, one per dot).
_EVENT_ID = pa.dictionary(pa.int32(), pa.string())

PLAYERS_SCHEMA = pa.schema([
    ("match_id", pa.int32()), ("event_id", _EVENT_ID), ("player_idx", pa.int8()),
    ("teammate", pa.bool_()), ("actor", pa.bool_()), ("keeper", pa.bool_()),
    ("x", pa.float32()), ("y", pa.float32()),
])  # fmt: skip

FRAMES_SCHEMA = pa.schema([
    ("match_id", pa.int32()), ("event_id", pa.string()),
    ("visible_area", pa.list_(pa.float32())), ("visible_area_sqyd", pa.float32()),
    ("visible_frac", pa.float32()), ("n_players", pa.int8()), ("n_teammates", pa.int8()),
    ("n_opponents", pa.int8()), ("has_actor", pa.bool_()),
])  # fmt: skip

_STR = pa.string()
EVENTS_SCHEMA = pa.schema([
    ("match_id", pa.int32()), ("event_id", _STR), ("index", pa.int32()), ("period", pa.int8()),
    ("timestamp", _STR), ("minute", pa.int16()), ("second", pa.int8()), ("type", _STR),
    ("possession", pa.int16()), ("possession_team_id", pa.int32()), ("play_pattern", _STR),
    ("team_id", pa.int32()), ("team", _STR), ("player_id", pa.int64()), ("player", _STR),
    ("position", _STR), ("x", pa.float32()), ("y", pa.float32()), ("duration", pa.float32()),
    ("under_pressure", pa.bool_()), ("related_events", pa.list_(_STR)),
    ("pass_recipient_id", pa.int64()), ("pass_recipient", _STR),
    ("pass_end_x", pa.float32()), ("pass_end_y", pa.float32()), ("pass_length", pa.float32()),
    ("pass_angle", pa.float32()), ("pass_height", _STR), ("pass_outcome", _STR),
    ("pass_type", _STR), ("pass_body_part", _STR), ("pass_technique", _STR),
    ("pass_cross", pa.bool_()), ("pass_switch", pa.bool_()), ("pass_cut_back", pa.bool_()),
    ("pass_through_ball", pa.bool_()), ("pass_shot_assist", pa.bool_()),
    ("pass_goal_assist", pa.bool_()), ("carry_end_x", pa.float32()),
    ("carry_end_y", pa.float32()), ("shot_xg", pa.float32()), ("shot_outcome", _STR),
    ("ball_receipt_outcome", _STR),
])  # fmt: skip


def _to_table(rows: list[dict], schema: pa.Schema) -> pa.Table:
    return pa.Table.from_pylist(rows, schema=schema)


def build_tables(out_dir: Path = PROCESSED_DIR) -> dict[str, int]:
    """Stream every match into Parquet, one row group per match. Returns row counts.

    Memory stays roughly constant however many matches there are. Only one
    match's rows are ever held in Python objects.
    """
    matches = matches_with_360()
    out_dir.mkdir(parents=True, exist_ok=True)
    matches.to_parquet(out_dir / "matches.parquet", index=False)

    writers = {
        name: pq.ParquetWriter(out_dir / f"{name}.parquet", schema, compression="zstd")
        for name, schema in [("events", EVENTS_SCHEMA), ("players", PLAYERS_SCHEMA),
                             ("frames", FRAMES_SCHEMA)]
    }  # fmt: skip
    counts = {"matches": len(matches), "events": 0, "players": 0, "frames": 0}
    issues = []
    try:
        for mid in tqdm(matches["match_id"], desc="parse"):
            ev = [parse_event(e, mid) for e in cached_json(f"events/{mid}.json")]
            try:
                pl, fr = parse_three_sixty(cached_json(f"three-sixty/{mid}.json"), mid)
            except (ValueError, OSError) as e:
                # Upstream files are occasionally corrupt (e.g. three-sixty/3845506.json has
                # a run of spaces overwriting part of the JSON). Keep the events, skip the frames.
                issues.append({"match_id": int(mid), "file": "three-sixty", "error": repr(e)})
                pl, fr = [], []
            for name, rows, schema in [("events", ev, EVENTS_SCHEMA),
                                       ("players", pl, PLAYERS_SCHEMA),
                                       ("frames", fr, FRAMES_SCHEMA)]:  # fmt: skip
                if rows:
                    writers[name].write_table(_to_table(rows, schema))
                    counts[name] += len(rows)
    finally:
        for w in writers.values():
            w.close()
    (out_dir / "data_issues.json").write_text(json.dumps(issues, indent=2), encoding="utf-8")
    if issues:
        print(f"{len(issues)} matches with unreadable files, see data_issues.json")
    return counts


def load_table(name: str, in_dir: Path = PROCESSED_DIR, columns=None, filters=None) -> pd.DataFrame:
    """Read a processed table. `filters` uses pyarrow syntax, e.g.
    [("event_id", "in", ids)], which avoids loading all 25M player rows."""
    return pd.read_parquet(in_dir / f"{name}.parquet", columns=columns, filters=filters)
