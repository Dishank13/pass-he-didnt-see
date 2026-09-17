"""Build padded corner graph arrays from corners.parquet + players.parquet.

    python scripts/build_corner_graphs.py   ->  data/processed/corner_graphs.npz (+ _meta.parquet)
"""

from phds.data.freeze_frames import PROCESSED_DIR, load_table
from phds.features.corner_graph import build_corner_graphs

if __name__ == "__main__":
    corners = load_table("corners")
    ids = corners.loc[corners["label_status"] != "no_frame", "event_id"].tolist()
    players = load_table("players", filters=[("event_id", "in", ids)])
    g = build_corner_graphs(corners, players, load_table("matches"))
    g.save(PROCESSED_DIR / "corner_graphs.npz")
    print(f"{len(g.meta)} corner graphs; x {g.x.shape}")
    print("hard targets per split:", g.meta.assign(t=g.target >= 0).groupby("split")["t"].sum().to_dict())
    print("with soft label per split:",
          g.meta.assign(s=(g.excess < 1e9).any(1)).groupby("split")["s"].sum().to_dict())
    print("shot rate per split:", g.meta.groupby("split")["shot_within"].mean().round(3).to_dict())
