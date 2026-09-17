"""Analyse blind-test answer files, following docs/human_eval_protocol.md.

    python scripts/analyse_human_eval.py responses/*.json

Each file is the JSON a rater downloads from the app's blind test. The answer key comes from
reports/m4/eval_key.json (written by scripts/export_demo.py).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
KEY = ROOT / "reports" / "m4" / "eval_key.json"


def load_judgments(paths: list[Path], key: dict) -> pd.DataFrame:
    rows = []
    for rater_i, path in enumerate(paths):
        s = json.loads(Path(path).read_text(encoding="utf-8"))
        if s.get("seed") != key["seed"]:
            raise ValueError(f"{path}: answers are for eval set seed {s.get('seed')}, key is {key['seed']}")
        rater = s.get("rater") or {}
        for item_id, a in s["answers"].items():
            k = key["key"].get(item_id)
            if k is None:
                continue
            rows.append({
                "rater": f"r{rater_i}", "background": rater.get("experience"), "item": item_id,
                "choice": a["choice"], "ms": a["ms"], "third": k["third"], "delta_ev": k["delta_ev"],
                "prefers_model": np.nan if a["choice"] == "unsure" else float(a["choice"] == k["model_option"]),
            })  # fmt: skip
    return pd.DataFrame(rows)


def two_way_bootstrap(df: pd.DataFrame, n_boot: int = 10_000, seed: int = 0) -> tuple[float, float, float]:
    """Proportion preferring the model, resampling raters and moments independently (crossed design)."""
    d = df.dropna(subset=["prefers_model"])
    raters, items = d["rater"].unique(), d["item"].unique()
    mat = d.pivot_table(index="rater", columns="item", values="prefers_model", aggfunc="mean")
    vals = mat.to_numpy()
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        ri = rng.integers(0, len(raters), len(raters))
        ci = rng.integers(0, len(items), len(items))
        stats[b] = np.nanmean(vals[np.ix_(ri, ci)])
    return float(np.nanmean(vals)), float(np.nanquantile(stats, 0.025)), float(np.nanquantile(stats, 0.975))


def fleiss_kappa(df: pd.DataFrame) -> float:
    d = df[df["choice"].isin(["A", "B"])]
    counts = d.pivot_table(index="item", columns="choice", values="rater", aggfunc="count").fillna(0)
    n = counts.sum(axis=1)
    counts = counts[n >= 2]
    n = counts.sum(axis=1)
    p_j = counts.sum() / counts.to_numpy().sum()
    P_i = ((counts**2).sum(axis=1) - n) / (n * (n - 1))
    P_bar, P_e = P_i.mean(), (p_j**2).sum()
    return float((P_bar - P_e) / (1 - P_e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", type=Path)
    args = ap.parse_args()
    key = json.loads(KEY.read_text(encoding="utf-8"))
    df = load_judgments(args.files, key)
    print(f"{df['rater'].nunique()} raters, {df['item'].nunique()} moments, {len(df)} judgments; "
          f"'can't choose' {df['choice'].eq('unsure').mean():.1%}; median response {df['ms'].median() / 1000:.1f}s")  # fmt: skip

    est, lo, hi = two_way_bootstrap(df)
    verdict = ("humans prefer the model's option" if lo > 0.5 else
               "humans prefer the professional's choice" if hi < 0.5 else
               "not distinguishable with this sample")  # fmt: skip
    print(f"\nPRIMARY: prefers model option {est:.3f} (95% CI {lo:.3f}–{hi:.3f}) -> {verdict}")

    print("\nEXPLORATORY")
    for col in ["third", "background"]:
        g = df.dropna(subset=["prefers_model"]).groupby(col)["prefers_model"].agg(["mean", "size"])
        print(f"  by {col}:\n{g.round(3).to_string()}")
    per_item = df.dropna(subset=["prefers_model"]).groupby("item").agg(share=("prefers_model", "mean"),
                                                                       delta_ev=("delta_ev", "first"))  # fmt: skip
    if len(per_item) >= 5:
        rho = spearmanr(per_item["delta_ev"], per_item["share"]).statistic
        print(f"  Spearman(EV gap, share preferring model) across moments: {rho:.3f}")
    if df["rater"].nunique() >= 2:
        print(f"  Fleiss' kappa (A/B choices): {fleiss_kappa(df):.3f}")


if __name__ == "__main__":
    main()
