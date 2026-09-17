"""M3: what do snapshots without velocities (and with a broadcast camera view) lose?

    python scripts/velocity_study.py

Data: SkillCorner open broadcast tracking (20 A-League 2024/25 matches). Task: predict whether a pass
to the targeted teammate is completed, from the moment of the pass, under four information
conditions (see phds.features.tracking_passes). All conditions are evaluated on the *same*
passes, those where passer and target are on screen, so a 360-style frame would contain the pass.

Protocol: leave-one-match-out out-of-fold predictions. Paired differences between conditions,
with bootstrap CIs over 10-minute match blocks (20 matches is too few clusters for a
match-level bootstrap. Blocks are a pragmatic middle ground and are stated as such).
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from phds.data.freeze_frames import PROCESSED_DIR
from phds.data.skillcorner import fetch_match, list_matches
from phds.eval import metrics as M
from phds.features.pass_features import PASS_FEATURES
from phds.features.tracking_passes import CONDITIONS, VELOCITY_FEATURES, build_match_passes

REPORTS = Path(__file__).resolve().parents[1] / "reports" / "m3"
CACHE = PROCESSED_DIR / "tracking_passes.parquet"
STATIC_PHYSICS = ["physics_margin", "lane_margin", "target_margin", "length"]
DYNAMIC_PHYSICS = ["dyn_physics_margin", "dyn_lane_margin", "dyn_target_margin", "length"]
CONTRASTS = [
    ("velocity (all players)", "full_vel", "full_pos"),
    ("velocity (on-screen players)", "vis_vel", "vis_pos"),
    ("visibility (no velocity)", "full_pos", "vis_pos"),
    ("visibility (with velocity)", "full_vel", "vis_vel"),
    ("total: full tracking vs 360-like", "full_vel", "vis_pos"),
]


def load_passes() -> pd.DataFrame:
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    frames = []
    for m in list_matches():
        df = build_match_passes(fetch_match(m))
        print(f"match {m}: {len(df)} passes")
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(CACHE, index=False)
    return out


def physics_model():
    return make_pipeline(SplineTransformer(n_knots=5, degree=3), StandardScaler(),
                         LogisticRegression(C=1.0, max_iter=3000))  # fmt: skip


def lgbm_model():
    return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=40,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=2.0,
                              verbose=-1)  # fmt: skip


def oof(df: pd.DataFrame, cols: list[str], make) -> np.ndarray:
    """Leave-one-match-out predictions."""
    pred = np.zeros(len(df))
    y = df["completed"].to_numpy()
    for m in df["match_id"].unique():
        te = (df["match_id"] == m).to_numpy()
        model = make().fit(df.loc[~te, cols], y[~te])
        pred[te] = model.predict_proba(df.loc[te, cols])[:, 1]
    return pred


def main():
    df = load_passes()
    print(f"{len(df):,} passes from {df['match_id'].nunique()} matches; completion {df['completed'].mean():.3f}")
    print(f"target on screen {df['target_detected'].mean():.3f}, passer on screen {df['passer_detected'].mean():.3f}, "
          f"median players on screen {df['n_detected'].median():.0f}")  # fmt: skip
    s = df[df["passer_detected"] & df["target_detected"]].reset_index(drop=True)
    print(f"analysis sample (passer and target on screen): {len(s):,}, completion {s['completed'].mean():.3f}")

    preds = {}
    for cond in CONDITIONS:
        has_vel = cond.endswith("vel")
        phys = [f"{cond}__{c}" for c in (DYNAMIC_PHYSICS if has_vel else STATIC_PHYSICS)]
        allf = [f"{cond}__{c}" for c in PASS_FEATURES + (VELOCITY_FEATURES if has_vel else [])]
        allf = [c for c in allf if s[c].notna().any()]
        preds[f"physics|{cond}"] = oof(s, phys, physics_model)
        preds[f"lgbm|{cond}"] = oof(s, allf, lgbm_model)
        if has_vel:  # nested: do velocity margins add information *on top of* the static ones?
            both = [f"{cond}__{c}" for c in dict.fromkeys(STATIC_PHYSICS + DYNAMIC_PHYSICS)]
            preds[f"physics_nested|{cond}"] = oof(s, both, physics_model)

    # External validation: the M2 completion model, trained on StatsBomb 360 (train+val), applied as-is.
    with open(PROCESSED_DIR / "models" / "completion_test.pkl", "rb") as fh:
        m2 = pickle.load(fh)["lgbm"]
    for cond in ("full_pos", "vis_pos"):
        X = s[[f"{cond}__{c}" for c in PASS_FEATURES]].copy()
        X.columns = PASS_FEATURES
        preds[f"m2_statsbomb_transfer|{cond}"] = m2.predict(X)

    y, groups = s["completed"].to_numpy(float), s["block"].to_numpy()
    rows, ll = [], {}
    for name, p in preds.items():
        p = np.clip(p, 1e-6, 1 - 1e-6)
        ll[name] = M.log_loss_each(p, y)
        rows.append({
            "model": name.split("|")[0], "condition": name.split("|")[1],
            "log_loss": M.mean_ci(ll[name], groups, n_boot=500), "brier": float(M.brier_each(p, y).mean()),
            "auc": M.cluster_bootstrap(lambda i, p=p: M.auc(p[i], y[i]), groups, n_boot=500),
            "ece": M.ece(p, y), "mean_pred": float(p.mean()), "reliability": M.reliability_bins(p, y),
        })  # fmt: skip
        r = rows[-1]
        print(f"  {name:32s} logloss {M.fmt_ci(r['log_loss'])}  auc {M.fmt_ci(r['auc'])}  ece {r['ece']:.3f}")
    sk = s["sk_xpass"].notna().to_numpy()
    sk_p, sk_y = s["sk_xpass"].to_numpy()[sk], y[sk]
    sk_auc = M.cluster_bootstrap(lambda i: M.auc(sk_p[i], sk_y[i]), groups[sk], n_boot=500)
    print(f"  SkillCorner xpass_completion (their model, reference; n={sk.sum()}): auc {M.fmt_ci(sk_auc)}")

    contrasts = {}
    nested = [("physics_nested", "physics", "velocity added to static margins (all players)", "full_vel", "full_pos"),
              ("physics_nested", "physics", "velocity added to static margins (on-screen)", "vis_vel", "vis_pos")]
    plan = [(m, m, label, a, b) for m in ("physics", "lgbm") for label, a, b in CONTRASTS] + nested
    for model_a, model_b, label, a, b in plan:
        pa, pb = preds[f"{model_a}|{a}"], preds[f"{model_b}|{b}"]
        d_ll = M.paired_diff_ci(ll[f"{model_a}|{a}"], ll[f"{model_b}|{b}"], groups, n_boot=1000)
        d_auc = M.cluster_bootstrap(lambda i, pa=pa, pb=pb: M.auc(pa[i], y[i]) - M.auc(pb[i], y[i]),
                                    groups, n_boot=1000)  # fmt: skip
        contrasts[f"{model_a}: {label}"] = {"log_loss_diff": d_ll, "auc_diff": d_auc}
        print(f"  {model_a:14s} {label:48s} dlogloss {M.fmt_ci(d_ll, digits=4)}  dAUC {M.fmt_ci(d_auc)}")

    REPORTS.mkdir(parents=True, exist_ok=True)
    out = {"n_passes": len(df), "n_sample": len(s), "completion": float(s["completed"].mean()),
           "target_on_screen": float(df["target_detected"].mean()),
           "median_on_screen": float(df["n_detected"].median()), "rows": rows, "contrasts": contrasts,
           "skillcorner_xpass_auc": sk_auc}  # fmt: skip
    (REPORTS / "velocity_study.json").write_text(json.dumps(out, indent=2, default=float))
    np.savez_compressed(REPORTS / "velocity_study_predictions.npz", y=y, block=groups,
                        **{k.replace("|", "__"): v for k, v in preds.items()})  # fmt: skip
    print(f"-> {REPORTS}")


if __name__ == "__main__":
    main()
