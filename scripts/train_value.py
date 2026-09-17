"""M2b: possession value models (location-only vs frame context).

    python scripts/train_value.py                 # develop: train on train, report on val
    python scripts/train_value.py --split test    # final: train on train+val, report on Euro 2024
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
from phds.data.splits import assign_splits
from phds.eval import metrics as M
from phds.models.value import (
    CTX_FEATURES,
    LOC_FEATURES,
    PossessionValueModel,
    ValueModel,
    action_labels,
    build_value_table,
    possession_labels,
)

REPORTS = Path(__file__).resolve().parents[1] / "reports" / "m2"
TABLE_PATH = PROCESSED_DIR / "value_table.parquet"


def load_value_table() -> pd.DataFrame:
    if not TABLE_PATH.exists():
        actions = action_labels(load_table("events"))
        players = load_table("players", columns=["event_id", "player_idx", "teammate", "actor",
                                                 "keeper", "x", "y"])  # fmt: skip
        store = FrameStore.from_players(players)
        del players
        gc.collect()
        table = build_value_table(actions, store, load_table("frames", columns=["event_id", "visible_frac"]))
        table["split"] = table["match_id"].map(assign_splits(load_table("matches")))
        table.to_parquet(TABLE_PATH, index=False)
    table = pd.read_parquet(TABLE_PATH)
    if "poss_value" not in table.columns:
        labels = possession_labels(load_table("events"))
        table = table.merge(labels, on="event_id", how="left")  # NaN: action by the non-possessing team
        table.to_parquet(TABLE_PATH, index=False)
    return table


def report(name, target, p, df, n_boot):
    y, g = df[target].to_numpy(float), df["match_id"].to_numpy()
    ll = M.log_loss_each(p, y)
    return {
        "model": name, "target": target,
        "log_loss": M.mean_ci(ll, g, n_boot=n_boot),
        "auc": M.cluster_bootstrap(lambda i: M.auc(p[i], y[i]), g, n_boot=n_boot),
        "ece": M.ece(p, y, n_bins=20), "mean_pred": float(p.mean()), "observed": float(y.mean()),
        "reliability": M.reliability_bins(p, y, n_bins=20), "_ll": ll,
    }  # fmt: skip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--n-boot", type=int, default=300)
    args = ap.parse_args()

    df = load_value_table()
    if args.split == "val":
        train, ev = df[df["split"] == "train"], df[df["split"] == "val"]
    else:
        train, ev = df[df["split"] != "test"], df[df["split"] == "test"]
    print(f"train {len(train):,} actions | eval ({args.split}) {len(ev):,} | "
          f"score rate {train['scores'].mean():.4f}, concede rate {train['concedes'].mean():.4f}")  # fmt: skip

    models = {"loc": ValueModel(LOC_FEATURES), "ctx": ValueModel(LOC_FEATURES + CTX_FEATURES)}
    rows, preds = [], {}
    for name, model in models.items():
        model.fit(train)
        s, c = model.predict(ev)
        preds[f"{name}_scores"], preds[f"{name}_concedes"] = s, c
        for target, p in [("scores", s), ("concedes", c)]:
            rows.append(report(name, target, p, ev, args.n_boot))
            r = rows[-1]
            print(f"  {name:4s} {target:8s} logloss {M.fmt_ci(r['log_loss'], digits=4)}  "
                  f"auc {M.fmt_ci(r['auc'])}  ece {r['ece']:.4f}")  # fmt: skip

    # Possession value (regression): xG rest of possession - xG of opponent's next possession.
    ptrain, pev = train[train["poss_value"].notna()], ev[ev["poss_value"].notna()]
    yv, gv = pev["poss_value"].to_numpy(), pev["match_id"].to_numpy()
    base = float(ptrain["poss_value"].mean())
    se_base = (yv - base) ** 2
    poss_rows, se = [], {}
    for name, feats in [("poss_loc", LOC_FEATURES), ("poss_ctx", LOC_FEATURES + CTX_FEATURES)]:
        model = PossessionValueModel(feats).fit(ptrain)
        models[name] = model
        pred = model.value(pev)
        preds[name] = model.value(ev)
        se[name] = (yv - pred) ** 2
        r2 = M.cluster_bootstrap(lambda i, e=se[name]: 1 - e[i].mean() / se_base[i].mean(), gv, n_boot=args.n_boot)
        order = np.argsort(pred)
        deciles = [(float(pred[b].mean()), float(yv[b].mean())) for b in np.array_split(order, 10)]
        poss_rows.append({"model": name, "r2_vs_mean": r2, "rmse": float(np.sqrt(se[name].mean())),
                          "decile_calibration": deciles})  # fmt: skip
        print(f"  {name:8s} R2 vs train mean {M.fmt_ci(r2, digits=4)}  rmse {np.sqrt(se[name].mean()):.4f}  "
              f"deciles pred->obs {[(round(a, 3), round(b, 3)) for a, b in (deciles[0], deciles[4], deciles[9])]}")  # fmt: skip

    g = ev["match_id"].to_numpy()
    comparisons = {}
    comparisons["poss_value MSE: ctx - loc"] = M.paired_diff_ci(se["poss_ctx"], se["poss_loc"], gv, n_boot=args.n_boot)
    print(f"  poss_value MSE ctx - loc: {M.fmt_ci(comparisons['poss_value MSE: ctx - loc'], digits=6)}")
    for target in ("scores", "concedes"):
        a = next(r["_ll"] for r in rows if r["model"] == "ctx" and r["target"] == target)
        b = next(r["_ll"] for r in rows if r["model"] == "loc" and r["target"] == target)
        comparisons[f"{target} log loss: ctx - loc"] = M.paired_diff_ci(a, b, g, n_boot=args.n_boot)
        print(f"  {target} log loss ctx - loc: {M.fmt_ci(comparisons[f'{target} log loss: ctx - loc'], digits=5)}")

    REPORTS.mkdir(parents=True, exist_ok=True)
    out = {"split": args.split, "n_train": len(train), "n_eval": len(ev),
           "models": [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows],
           "possession_value": poss_rows, "comparisons": comparisons}  # fmt: skip
    (REPORTS / f"value_{args.split}.json").write_text(json.dumps(out, indent=2, default=float))
    np.savez_compressed(REPORTS / f"value_predictions_{args.split}.npz",
                        event_id=ev["event_id"].to_numpy(), **preds)  # fmt: skip
    (PROCESSED_DIR / "models").mkdir(exist_ok=True)
    with open(PROCESSED_DIR / "models" / f"value_{args.split}.pkl", "wb") as fh:
        pickle.dump(models, fh)
    print(f"-> {REPORTS}")


if __name__ == "__main__":
    main()
