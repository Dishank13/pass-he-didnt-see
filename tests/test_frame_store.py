import numpy as np
import pandas as pd

from phds.data.frame_store import FrameStore


def test_frame_store_slices_match_pandas_rows():
    rng = np.random.default_rng(0)
    rows = []
    for e in ["b", "a", "c"]:
        for i in range(int(rng.integers(2, 6))):
            rows.append(dict(event_id=e, player_idx=i, x=rng.uniform(0, 120), y=rng.uniform(0, 80),
                             teammate=bool(i % 2), actor=i == 0, keeper=False))  # fmt: skip
    # Interleave events (as a real table might be), keeping player order within each event.
    players = pd.DataFrame(rows).sample(frac=1, random_state=1).sort_values("player_idx", kind="stable")
    store = FrameStore.from_players(players.astype({"event_id": "category"}))
    assert len(store) == 3 and "a" in store and "z" not in store
    for e, g in players.groupby("event_id", observed=True):
        f = store[e]
        np.testing.assert_allclose(f.xy, g[["x", "y"]].to_numpy(), rtol=1e-6)
        np.testing.assert_array_equal(f.player_idx, g["player_idx"].to_numpy())
        np.testing.assert_array_equal(f.actor, g["actor"].to_numpy())
    assert store.get("z") is None
