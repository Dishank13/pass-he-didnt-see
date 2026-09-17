"""Build data/processed/passes.parquet: open-play passes with linked intended targets.

    python scripts/build_pass_dataset.py
"""

import gc

from phds.data.frame_store import FrameStore
from phds.data.freeze_frames import PROCESSED_DIR, load_table
from phds.data.pass_targets import build_pass_dataset, linkage_weights, receipt_in_view
from phds.data.splits import assign_splits, check_no_leakage

if __name__ == "__main__":
    players = load_table("players", columns=["event_id", "player_idx", "teammate", "actor",
                                             "keeper", "x", "y"])  # fmt: skip
    store = FrameStore.from_players(players)
    del players
    gc.collect()

    events = load_table("events")
    passes = build_pass_dataset(events, store)
    passes["split"] = passes["match_id"].map(assign_splits(load_table("matches")))
    passes["receipt_in_view"] = receipt_in_view(passes, load_table("frames", columns=["event_id", "visible_area"]))
    passes["weight"] = linkage_weights(passes)
    check_no_leakage(passes)
    passes.to_parquet(PROCESSED_DIR / "passes.parquet", index=False)

    print(f"{len(passes):,} open-play passes with a frame")
    print(passes["link_status"].value_counts().to_string())
    ok = passes[passes["link_status"] == "ok"]
    print("\nlinked by method:", ok["link_method"].value_counts().to_dict())
    print("link rate by outcome:", passes.groupby("outcome")["link_status"].apply(
        lambda s: round((s == "ok").mean(), 3)).to_dict())
    print(f"completion: all {passes['completed'].mean():.3f} | linked naive {ok['completed'].mean():.3f}"
          f" | linked weighted (in-view population) {(ok['completed'] * ok['weight']).sum() / ok['weight'].sum():.3f}")
    print("linked per split:", ok["split"].value_counts().to_dict())
