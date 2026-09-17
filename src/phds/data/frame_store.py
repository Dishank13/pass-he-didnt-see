"""Constant-time access to any freeze frame, without pandas groupbys over 25M rows.

players.parquet is sorted once by event, into flat numpy arrays plus an offset
index. A frame is then just a slice:

    store = FrameStore.from_players(load_table("players"))
    f = store["87a790b7-..."]      # Frame(xy [n, 2], teammate [n], actor [n], keeper [n])

Memory: ~25M rows x (2 float32 + 3 bool + int8 index) ≈ 280 MB.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Frame:
    xy: np.ndarray  # [n, 2] float
    teammate: np.ndarray  # [n] bool: same team as the event's actor
    actor: np.ndarray  # [n] bool
    keeper: np.ndarray  # [n] bool
    player_idx: np.ndarray  # [n] position in the original freeze_frame list

    def __len__(self) -> int:
        return len(self.xy)


class FrameStore:
    def __init__(self, event_ids, starts, stops, xy, teammate, actor, keeper, player_idx):
        self._index = {e: i for i, e in enumerate(event_ids)}
        self._starts, self._stops = starts, stops
        self._xy, self._teammate, self._actor = xy, teammate, actor
        self._keeper, self._player_idx = keeper, player_idx

    @classmethod
    def from_players(cls, players: pd.DataFrame) -> FrameStore:
        codes, uniques = pd.factorize(players["event_id"], sort=False)
        order = np.argsort(codes, kind="stable")  # stable: keeps player_idx order within a frame
        codes = codes[order]
        bounds = np.flatnonzero(np.diff(codes)) + 1
        starts = np.r_[0, bounds]
        stops = np.r_[bounds, len(codes)]
        event_ids = np.asarray(uniques.astype(str))[codes[starts]]
        take = lambda col, dtype: players[col].to_numpy(dtype)[order]
        xy = np.column_stack([take("x", np.float32), take("y", np.float32)])
        return cls(event_ids, starts, stops, xy, take("teammate", bool), take("actor", bool),
                   take("keeper", bool), take("player_idx", np.int16))  # fmt: skip

    @property
    def event_ids(self):
        return self._index.keys()

    def __contains__(self, event_id) -> bool:
        return event_id in self._index

    def __len__(self) -> int:
        return len(self._index)

    def get(self, event_id) -> Frame | None:
        i = self._index.get(event_id)
        if i is None:
            return None
        s = slice(self._starts[i], self._stops[i])
        return Frame(self._xy[s], self._teammate[s], self._actor[s], self._keeper[s],
                     self._player_idx[s])  # fmt: skip

    def __getitem__(self, event_id) -> Frame:
        f = self.get(event_id)
        if f is None:
            raise KeyError(event_id)
        return f
