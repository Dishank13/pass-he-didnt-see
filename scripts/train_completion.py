"""M2a: pass completion models.

    python scripts/train_completion.py                # develop: train on train, report on val
    python scripts/train_completion.py --split test   # final: train on train+val, report on Euro 2024

Metrics are weighted by the selection-correction weights (population = passes to visible
teammates) with 95% match-level cluster-bootstrap CIs.
"""

from __future__ import annotations

import argparse
import gc
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from phds.data.frame_store import FrameStore
from phds.data.freeze_frames import PROCESSED_DIR, load_table
from phds.eval import metrics as M
from phds.models.completion import (
    DISTANCE_FEATURES,
    PHYSICS_FEATURES,
    CalibratedCompletion,
    LGBMCompletion,
    LogitCompletion,
    real_pass_features,
)

REPORTS = Path(__file__).resolve().parents[1] / "reports" / "m2"
FEATURES_PATH = PROCESSED_DIR / "pass_features.parquet"


def load_linked_passes() -> pd.DataFrame:
    passes = load_table("passes")
    passes = passes[passes["link_status"] == "ok"].reset_index(drop=True)
    if not FEATURES_PATH.exists():
        players = load_table("players", columns=["event_id", "player_idx", "teammate", "actor",
                                                 "keeper", "x", "y"])  # fmt: skip
        store = FrameStore.from_players(players)
        del players
        gc.collect()
        feats = real_pass_features(passes, store, load_table("frames", columns=["event_id", "visible_frac"]))
        feats.insert(0, "event_id", passes["event_id"].to_numpy())
        feats.to_parquet(FEATURES_PATH, index=False)
    feats = pd.read_parquet(FEATURES_PATH)
    assert (feats["event_id"].to_numpy() == passes["event_id"].to_numpy()).all()
    feats = feats.drop(columns="event_id")
    # Feature versions win: e.g. `length` is the distance to the target's *frame* position,
    # whereas passes.length is StatsBomb's pass length (to where the ball went).
    passes = passes.rename(columns={c: f"sb_{c}" for c in feats.columns if c in passes.columns})
    return pd.concat([passes, feats], axis=1)


def report(name, p, df, n_boot):
    y, w, g = df["completed"].to_numpy(float), df["weight"].to_numpy(float), df["match_id"].to_numpy()
    ll, br = M.log_loss_each(p, y), M.brier_each(p, y)
    return {
        "model": name,
        "log_loss": M.weighted_mean_ci(ll, w, g, n_boot=n_boot),
        "brier": M.weighted_mean_ci(br, w, g, n_boot=n_boot),
        "auc": M.cluster_bootstrap(lambda i: M.auc(p[i], y[i], w[i]), g, n_boot=n_boot),
        "ece": M.ece(p, y, n_bins=15, w=w),
        "mean_pred": float(np.average(p, weights=w)),
        "observed": float(np.average(y, weights=w)),
        "reliability": M.reliability_bins(p, y, n_bins=15, w=w),
        "_ll": ll,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--n-boot", type=int, default=500)
    args = ap.parse_args()

    df = load_linked_passes()
    if args.split == "val":
        train, ev = df[df["split"] == "train"], df[df["split"] == "val"]
    else:
        train, ev = df[df["split"] != "test"], df[df["split"] == "test"]
    print(f"train {len(train):,} passes | eval ({args.split}) {len(ev):,} passes")
    y, w = train["completed"].to_numpy(), train["weight"].to_numpy()

    models = {
        "distance": LogitCompletion(DISTANCE_FEATURES).fit(train, y, w),
        "physics": LogitCompletion(PHYSICS_FEATURES).fit(train, y, w),
        # Final model: LightGBM + cross-fitted isotonic calibration on training matches.
        "lgbm": CalibratedCompletion(LGBMCompletion).fit(train, y, w, train["match_id"].to_numpy()),
    }
    preds = {name: m.predict(ev) for name, m in models.items()}
    preds["lgbm_raw"] = models["lgbm"].predict_raw(ev)  # same trees, before calibration

    rows = []
    for name in ["distance", "physics", "lgbm_raw", "lgbm"]:
        rows.append(report(name, preds[name], ev, args.n_boot))
        r = rows[-1]
        print(f"  {name:9s} logloss {M.fmt_ci(r['log_loss'])}  brier {M.fmt_ci(r['brier'])}  "
              f"auc {M.fmt_ci(r['auc'])}  ece {r['ece']:.4f}  mean pred {r['mean_pred']:.3f} "
              f"vs observed {r['observed']:.3f}")  # fmt: skip

    g = ev["match_id"].to_numpy()
    wv = ev["weight"].to_numpy(float)
    by = {r["model"]: r["_ll"] for r in rows}
    comparisons = {}
    for a, b in [("physics", "distance"), ("lgbm", "physics"), ("lgbm", "lgbm_raw")]:
        diff = by[a] - by[b]
        comparisons[f"log loss: {a} - {b}"] = M.weighted_mean_ci(diff, wv, g, n_boot=args.n_boot)
        print(f"  log loss {a} - {b}: {M.fmt_ci(comparisons[f'log loss: {a} - {b}'], digits=4)}")

    importance = models["lgbm"].base.importance()
    print("top features (gain share):", importance.head(8).round(3).to_dict())

    REPORTS.mkdir(parents=True, exist_ok=True)
    out = {"split": args.split, "n_train": len(train), "n_eval": len(ev),
           "models": [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows],
           "comparisons": comparisons, "lgbm_importance": importance.round(4).to_dict()}  # fmt: skip
    (REPORTS / f"completion_{args.split}.json").write_text(json.dumps(out, indent=2, default=float))
    np.savez_compressed(REPORTS / f"completion_predictions_{args.split}.npz",
                        event_id=ev["event_id"].to_numpy(), **preds)  # fmt: skip
    (PROCESSED_DIR / "models").mkdir(exist_ok=True)
    with open(PROCESSED_DIR / "models" / f"completion_{args.split}.pkl", "wb") as fh:
        pickle.dump(models, fh)
    print(f"-> {REPORTS}")


if __name__ == "__main__":
    main()
