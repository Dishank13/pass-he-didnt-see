"""Build processed Parquet tables and the labelled corner dataset.

    python scripts/build_dataset.py                 # everything
    python scripts/build_dataset.py --corners-only  # reuse tables, rebuild corners.parquet

Outputs (data/processed/): matches, events, players, frames, corners (.parquet)
"""

import sys

from phds.data.freeze_frames import PROCESSED_DIR, build_tables, load_table
from phds.data.labels import build_corner_dataset
from phds.data.splits import assign_splits, check_no_leakage

if __name__ == "__main__":
    if "--corners-only" not in sys.argv:
        for name, n in build_tables().items():
            print(f"{name:8s} {n:>12,d} rows")

    events, frames = load_table("events"), load_table("frames")
    corners = build_corner_dataset(events, load_table("players"), frames)
    split = assign_splits(load_table("matches"))
    corners["split"] = corners["match_id"].map(split)
    check_no_leakage(corners)
    corners.to_parquet(PROCESSED_DIR / "corners.parquet", index=False)

    print(f"\ncorners  {len(corners):>10,d} rows")
    print(corners["label_status"].value_counts().to_string())
    print(corners.groupby("split")["label_status"].apply(lambda s: (s == "ok").sum()).to_string())
