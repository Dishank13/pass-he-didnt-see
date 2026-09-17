"""M2c: out-of-fold expected value and plausibility for every option of every linked open-play pass.

    python scripts/score_passes.py [--folds 4] [--plausible 0.10]

For each of K match-grouped folds, fit on the *other* folds:
  * completion model (LightGBM + cross-fitted isotonic calibration)
  * two value definitions, each with a frame-context model (success) and a location model (failure):
        goal10  P(score within 10 actions) - P(concede within 10 actions)     [VAEP-style]
        poss    xG rest of possession - xG of opponent's next possession       [possession value]
then score this fold's passes. Next, fit a pass-selection model out-of-fold on those options.
No pass is judged by a model that saw its match.

Outputs (data/processed/):
    pass_options.parquet     one row per (pass, visible teammate option)
    pass_decisions.parquet   one row per pass: actual vs best (all options and plausible options)
"""

from __future__ import annotations

import argparse
import gc

import numpy as np
import pandas as pd
from tqdm import tqdm

from phds.data.frame_store import FrameStore
from phds.data.freeze_frames import PROCESSED_DIR, load_table
from phds.eval.metrics import topk_accuracy
from phds.features.pass_features import PASS_FEATURES
from phds.models.completion import CalibratedCompletion, LGBMCompletion
from phds.models.pass_value import PassOptionScorer, summarise_decisions
from phds.models.selection import SelectionModel
from phds.models.value import CTX_FEATURES, LOC_FEATURES, PossessionValueModel, ValueModel

VALUE_DEFS = ("goal10", "poss")


def fold_scores(passes, train_table, value_table, store, vis, fold_of, k):
    tr = passes["fold"] != k
    print(f"fold {k}: fitting on {tr.sum():,} passes")
    completion = CalibratedCompletion(LGBMCompletion).fit(
        train_table[tr], train_table.loc[tr, "completed"], train_table.loc[tr, "weight"],
        train_table.loc[tr, "match_id"],
    )  # fmt: skip
    vt = value_table[~value_table["match_id"].map(fold_of).eq(k)]
    value_sets = {
        "goal10": (ValueModel(LOC_FEATURES + CTX_FEATURES).fit(vt), ValueModel(LOC_FEATURES).fit(vt)),
        "poss": (PossessionValueModel(LOC_FEATURES + CTX_FEATURES).fit(vt),
                 PossessionValueModel(LOC_FEATURES).fit(vt)),
    }  # fmt: skip
    scorer = PassOptionScorer(completion, value_sets=value_sets)
    fold_passes = passes[~tr]
    items = (
        (r.event_id, store[r.event_id], (r.x, r.y), bool(r.under_pressure), vis.get(r.event_id, np.nan))
        for r in fold_passes.itertuples(index=False)
    )
    opts = scorer.score_batch(tqdm(items, total=len(fold_passes), desc=f"score fold {k}", mininterval=30))
    opts["fold"] = k
    return opts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--plausible", type=float, default=0.10)
    ap.add_argument("--decisions-only", action="store_true", help="recompute decisions from saved options")
    args = ap.parse_args()

    passes = load_table("passes")
    passes = passes[passes["link_status"] == "ok"].reset_index(drop=True)
    if args.decisions_only:
        opts = pd.read_parquet(PROCESSED_DIR / "pass_options.parquet")
        passes["fold"] = passes["event_id"].map(opts.groupby("event_id")["fold"].first())
        write_decisions(opts, passes, args.plausible)
        return
    feats = pd.read_parquet(PROCESSED_DIR / "pass_features.parquet")
    assert (feats["event_id"].to_numpy() == passes["event_id"].to_numpy()).all()
    train_table = pd.concat([passes[["match_id", "completed", "weight"]], feats.drop(columns="event_id")], axis=1)
    value_table = pd.read_parquet(PROCESSED_DIR / "value_table.parquet")
    vis = load_table("frames", columns=["event_id", "visible_frac"]).set_index("event_id")["visible_frac"]

    players = load_table("players", columns=["event_id", "player_idx", "teammate", "actor", "keeper", "x", "y"])
    store = FrameStore.from_players(players)
    del players
    gc.collect()

    rng = np.random.default_rng(0)
    matches = rng.permutation(np.sort(passes["match_id"].unique()))
    fold_of = {m: i % args.folds for i, m in enumerate(matches)}
    passes["fold"] = passes["match_id"].map(fold_of)

    opts = pd.concat([fold_scores(passes, train_table, value_table, store, vis, fold_of, k)
                      for k in range(args.folds)], ignore_index=True)  # fmt: skip
    opts["is_actual"] = opts["player_idx"].to_numpy() == opts["event_id"].map(
        passes.set_index("event_id")["target_idx"]).to_numpy()  # fmt: skip

    # Behaviour policy, out-of-fold over the same match folds.
    opts["policy_p"] = np.nan
    for k in range(args.folds):
        tr, te = opts["fold"] != k, opts["fold"] == k
        model = SelectionModel().fit(opts[tr], opts.loc[tr, "is_actual"].to_numpy())
        opts.loc[te, "policy_p"] = model.predict(opts[te].reset_index(drop=True))
    act = opts[opts["is_actual"]]
    print(f"selection model: mean P(actual option) {act['policy_p'].mean():.3f}")

    # Top-1 / top-3 accuracy of predicting the actual choice (tie-aware), per pass.
    wide = opts.assign(r=opts.groupby("event_id").cumcount())
    pivot = wide.pivot(index="event_id", columns="r", values="policy_p").fillna(0)
    P = pivot.to_numpy()
    T = wide[wide["is_actual"]].set_index("event_id")["r"].reindex(pivot.index).to_numpy().astype(int)
    print(f"selection top-1 {topk_accuracy(P, T, 1).mean():.3f}, top-3 {topk_accuracy(P, T, 3).mean():.3f}")

    keep = ["event_id", "player_idx", "opt_x", "opt_y", "fold", "is_actual", "p_complete", "policy_p",
            *[f"{c}_{v}" for v in VALUE_DEFS for c in ("v_success", "v_fail", "ev")], *PASS_FEATURES]  # fmt: skip
    opts[keep].to_parquet(PROCESSED_DIR / "pass_options.parquet", index=False)

    write_decisions(opts, passes, args.plausible)


def write_decisions(opts, passes, plausible):
    dec = summarise_decisions(opts, passes.set_index("event_id")["target_idx"], plausible, VALUE_DEFS)
    dec = passes[["event_id", "match_id", "split", "fold", "period", "minute", "team_id", "team", "player_id",
                  "player", "position", "recipient", "x", "y", "target_idx", "target_x", "target_y",
                  "completed", "outcome", "under_pressure", "weight"]].merge(dec, on="event_id")  # fmt: skip
    dec.to_parquet(PROCESSED_DIR / "pass_decisions.parquet", index=False)

    print(f"{len(opts):,} options for {len(dec):,} passes; median options {dec['n_options'].median():.0f}, "
          f"median plausible {dec['n_plausible'].median():.0f}")  # fmt: skip
    for v in VALUE_DEFS:
        print(f"[{v}] actual = best (all) {dec[f'actual_is_best_all_{v}'].mean():.1%} | actual = best (plausible) "
              f"{dec[f'actual_is_best_plausible_{v}'].mean():.1%} | mean choice pct {dec[f'choice_pct_{v}'].mean():.3f} | "
              f"median P(best all) {dec[f'best_all_{v}_p_complete'].median():.3f} vs actual {dec['actual_p_complete'].median():.3f}")  # fmt: skip


if __name__ == "__main__":
    main()
