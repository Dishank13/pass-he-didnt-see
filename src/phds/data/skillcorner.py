"""SkillCorner open data (broadcast tracking, A-League 2024/25, MIT licence) for the velocity study.

https://github.com/SkillCorner/opendata. Credit: SkillCorner. Per match:
    {id}_match.json                    lineups, pitch size, team sides
    {id}_tracking_extrapolated.jsonl   10 fps; every player each frame with `is_detected`
                                       (on screen) or extrapolated; camera footprint polygon
    {id}_dynamic_events.csv            player possessions, incl. passes with the *targeted*
                                       player and outcome (successful / unsuccessful)

Why this dataset for M3: it has the two things StatsBomb 360 lacks, velocities (from
consecutive frames) and every player's position (detected or estimated), plus a *real*
per-player visibility flag from the same broadcast camera. So "360-like" conditions can be
built from the actual camera view instead of being simulated.

Caveat: extrapolated players are SkillCorner's estimates, not ground truth. The
"full information" condition is therefore the best available reconstruction, not reality.

Tracking JSONL (~90 MB per match) is streamed into compact Parquet, then deleted.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests

RAW = "https://raw.githubusercontent.com/SkillCorner/opendata/master/data"
MEDIA = "https://media.githubusercontent.com/media/SkillCorner/opendata/master/data"  # Git LFS files
SK_DIR = Path(__file__).resolve().parents[3] / "data" / "raw" / "skillcorner"


def list_matches() -> list[int]:
    """Match ids with data folders. (matches.json lists more matches than are published.)"""
    url = "https://api.github.com/repos/SkillCorner/opendata/contents/data/matches"
    return sorted(int(x["name"]) for x in requests.get(url, timeout=60).json() if x["type"] == "dir")


def _download(url: str, path: Path):
    if path.exists():
        return
    tmp = path.with_suffix(path.suffix + ".tmp")
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(tmp, "wb") as fh:
            fh.writelines(r.iter_content(1 << 20))
    tmp.replace(path)


def parse_tracking(jsonl_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """-> (players: frame, period, player_id, x, y, is_detected;
           frames: frame, period, ball_x, ball_y, camera corner coordinates, possession group)."""
    pl = {k: [] for k in ("frame", "player_id", "x", "y", "is_detected")}
    fr = []
    with open(jsonl_path, encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            if not d.get("player_data") or d.get("period") is None:
                continue
            f = d["frame"]
            ball = d.get("ball_data") or {}
            cam = d.get("image_corners_projection") or {}
            fr.append({
                "frame": f, "period": d["period"], "ball_x": ball.get("x"), "ball_y": ball.get("y"),
                **{k: cam.get(k) for k in ("x_top_left", "y_top_left", "x_bottom_left", "y_bottom_left",
                                           "x_bottom_right", "y_bottom_right", "x_top_right", "y_top_right")},
                "possession_group": (d.get("possession") or {}).get("group"),
            })  # fmt: skip
            for p in d["player_data"]:
                pl["frame"].append(f)
                pl["player_id"].append(p["player_id"])
                pl["x"].append(p["x"])
                pl["y"].append(p["y"])
                pl["is_detected"].append(bool(p.get("is_detected", False)))
    players = pd.DataFrame(pl).astype({"frame": "int32", "player_id": "int32", "x": "float32",
                                       "y": "float32"})  # fmt: skip
    frames = pd.DataFrame(fr)
    return players, frames


def fetch_match(match_id: int, out_dir: Path = SK_DIR) -> dict[str, Path]:
    """Download one match, convert tracking to Parquet, delete the raw JSONL. Idempotent."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "match": out_dir / f"{match_id}_match.json",
        "events": out_dir / f"{match_id}_dynamic_events.csv",
        "players": out_dir / f"{match_id}_players.parquet",
        "frames": out_dir / f"{match_id}_frames.parquet",
    }
    _download(f"{RAW}/matches/{match_id}/{match_id}_match.json", paths["match"])
    _download(f"{RAW}/matches/{match_id}/{match_id}_dynamic_events.csv", paths["events"])
    if not paths["players"].exists():
        jsonl = out_dir / f"{match_id}_tracking.jsonl"
        _download(f"{MEDIA}/matches/{match_id}/{match_id}_tracking_extrapolated.jsonl", jsonl)
        players, frames = parse_tracking(jsonl)
        players.to_parquet(paths["players"], index=False)
        frames.to_parquet(paths["frames"], index=False)
        jsonl.unlink()  # ~90 MB; the Parquet holds everything we use
    return paths


# --- coordinates -----------------------------------------------------------------------
def to_statsbomb(x, y, attacking_left_to_right, length: float, width: float):
    """SkillCorner metres (origin at centre, y up) -> StatsBomb-style yards (120 x 80, origin top-left,
    attacking towards x = 120).

    Axis scaling is anisotropic (104 m -> 120, 68 m -> 80), matching StatsBomb's own convention
    of a nominal pitch. Speeds computed afterwards are in these "yards".
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    sign = np.where(np.asarray(attacking_left_to_right, bool), 1.0, -1.0)
    x, y = x * sign, y * sign  # 180-degree rotation when attacking right-to-left
    return (x + length / 2) * 120.0 / length, (width / 2 - y) * 80.0 / width


def match_meta(match_json: Path) -> dict:
    m = json.loads(Path(match_json).read_text(encoding="utf-8"))
    players = pd.DataFrame([
        {"player_id": p["id"], "team_id": p["team_id"], "role": p["player_role"]["acronym"],
         "name": p["short_name"]} for p in m["players"]
    ])  # fmt: skip
    return {"length": float(m["pitch_length"]), "width": float(m["pitch_width"]), "players": players,
            "home_team_id": m["home_team"]["id"], "away_team_id": m["away_team"]["id"]}  # fmt: skip
