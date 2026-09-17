"""Pitch plots of a freeze frame: players, visible area, passes."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon as MplPolygon
from mplsoccer import Pitch

ATTACK_COLOR = "#d7301f"
DEFENCE_COLOR = "#2b8cbe"


def plot_freeze_frame(
    frame_players: pd.DataFrame,
    visible_area: list[float] | None = None,
    passes: list[dict] | None = None,
    highlight_idx: int | None = None,
    title: str | None = None,
    half: bool = False,
    ax=None,
):
    """Draw one 360 freeze frame in StatsBomb coordinates.

    frame_players: rows from players.parquet for one event_id.
    passes: dicts like {"start": (x, y), "end": (x, y), "color": "k", "label": "actual"}.
    highlight_idx: a `player_idx` to ring (e.g. the inferred receiver).
    """
    pitch = Pitch(pitch_type="statsbomb", half=half, pitch_color="white", line_color="#888")
    if ax is None:
        _, ax = pitch.draw(figsize=(9, 6) if not half else (6, 6))
    else:
        pitch.draw(ax=ax)

    if visible_area is not None and len(visible_area) >= 6:
        ring = np.asarray(visible_area, dtype=float).reshape(-1, 2)
        ax.add_patch(MplPolygon(ring, closed=True, facecolor="#999", alpha=0.15, zorder=1))

    fp = frame_players
    for is_mate, color in [(True, ATTACK_COLOR), (False, DEFENCE_COLOR)]:
        sub = fp[fp["teammate"] == is_mate]
        pitch.scatter(sub["x"], sub["y"], s=120, c=color, edgecolors="k", zorder=3, ax=ax)
        keepers = sub[sub["keeper"]]
        pitch.scatter(keepers["x"], keepers["y"], s=120, c=color, marker="s", edgecolors="k",
                      zorder=4, ax=ax)  # fmt: skip
    actor = fp[fp["actor"]]
    pitch.scatter(actor["x"], actor["y"], s=260, facecolors="none", edgecolors="gold", lw=2.5,
                  zorder=5, ax=ax)  # fmt: skip
    if highlight_idx is not None:
        h = fp[fp["player_idx"] == highlight_idx]
        pitch.scatter(h["x"], h["y"], s=380, facecolors="none", edgecolors="k", lw=2, ls="--",
                      zorder=5, ax=ax)  # fmt: skip

    for p in passes or []:
        pitch.arrows(*p["start"], *p["end"], width=2, headwidth=5, color=p.get("color", "k"),
                     alpha=0.8, zorder=6, ax=ax, label=p.get("label"))  # fmt: skip
    if any(p.get("label") for p in passes or []):
        ax.legend(loc="upper left", fontsize=8)
    if title:
        ax.set_title(title, fontsize=11)
    return ax


def save(fig_or_ax, path):
    fig = fig_or_ax.figure if hasattr(fig_or_ax, "figure") else fig_or_ax
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
