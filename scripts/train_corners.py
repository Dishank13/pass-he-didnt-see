"""M1 experiments: corner receiver + shot prediction, baselines vs GNN.

    python scripts/train_corners.py                 # develop: report on validation
    python scripts/train_corners.py --split test    # final: train on train+val, report on Euro 2024
    python scripts/train_corners.py --rescore       # recompute metrics from saved predictions

Protocol
  * Model and hyperparameter choices are made on validation only.
  * No model sees the evaluation split during training. In validation mode the GNN
    early-stops on an inner 1/7 of *training* matches (the baselines have no early stopping).
  * The test split (Euro 2024) is scored once, with choices frozen. For the final run
    the models are retrained on train+val (GNN epochs fixed from the validation runs).
  * Receiver metrics use only confident labels. Soft labels are a *training* device.
  * All metrics carry 95% cluster-bootstrap CIs (resampling matches).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from phds.data.freeze_frames import PROCESSED_DIR
from phds.data.labels import soft_label
from phds.eval import metrics as M
from phds.features.corner_graph import CornerGraphs
from phds.models import corner_baselines as B
from phds.models.corner_gnn import CornerGNN, receiver_loss

REPORTS = Path(__file__).resolve().parents[1] / "reports" / "m1"
MODELS = PROCESSED_DIR / "models"  # gitignored; default CornerGNN hyperparameters


# --- targets -------------------------------------------------------------------------
def receiver_targets(g: CornerGraphs, regime: str, temperature: float = 1.0):
    """Return (target_dist [N, M], has_label [N]).

    hard:   one-hot on confident ('ok') corners only.
    hybrid: one-hot on confident corners + soft distributions for ambiguous/conflict corners.
    """
    n, m = g.cand.shape
    dist = np.zeros((n, m), np.float32)
    has = g.target >= 0
    dist[has, g.target[has]] = 1.0
    if regime == "hybrid":
        soft_rows = (~has) & g.meta["label_status"].isin(["ambiguous", "conflict"]).to_numpy()
        for i in np.flatnonzero(soft_rows):
            finite = np.isfinite(g.excess[i]) & g.cand[i]
            if finite.sum() == 0:
                soft_rows[i] = False
                continue
            dist[i, finite] = soft_label(g.excess[i, finite], temperature)
        has = has | soft_rows
    return dist, has


# --- GNN training --------------------------------------------------------------------
def _t(a):
    return torch.as_tensor(a)


def train_gnn(train: CornerGraphs, dist, has, val: CornerGraphs | None, seed: int,
              epochs: int = 150, patience: int = 25, shot_weight: float = 1.0,
              hidden: int = 64, layers: int = 3, lr: float = 1e-3, batch: int = 128):  # fmt: skip
    """Train one GNN. With `val`, early-stop on val loss and return the best epoch count."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = CornerGNN(train.x.shape[2], train.g.shape[1], hidden=hidden, layers=layers)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    X, mask, cand, G = _t(train.x), _t(train.mask), _t(train.cand), _t(train.g)
    D, H, S = _t(dist), _t(has), _t(train.shot.astype(np.float32))
    bce = torch.nn.BCEWithLogitsLoss()

    if val is not None:
        vdist, vhas = receiver_targets(val, "hard")
        V = [_t(a) for a in (val.x, val.mask, val.cand, val.g)]
        VD, VH, VS = _t(vdist), _t(vhas), _t(val.shot.astype(np.float32))

    best, best_state, best_epoch, bad = np.inf, None, epochs, 0
    for epoch in range(1, epochs + 1):
        model.train()
        for idx in np.array_split(rng.permutation(len(X)), max(1, len(X) // batch)):
            idx = _t(idx)
            recv, shot = model(X[idx], mask[idx], cand[idx], G[idx])
            loss = receiver_loss(recv, D[idx], H[idx]) + shot_weight * bce(shot, S[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        if val is None:
            continue
        model.eval()
        with torch.no_grad():
            recv, shot = model(*V)
            vloss = (receiver_loss(recv, VD, VH) + shot_weight * bce(shot, VS)).item()
        if vloss < best - 1e-4:
            best, best_epoch, bad = vloss, epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_epoch


def gnn_predict(models, g: CornerGraphs):
    """Ensemble: average probabilities (not logits) across seeds."""
    recv_p, shot_p = [], []
    with torch.no_grad():
        for model in models:
            model.eval()
            recv, shot = model(_t(g.x), _t(g.mask), _t(g.cand), _t(g.g))
            recv_p.append(torch.softmax(recv, -1).numpy())
            shot_p.append(torch.sigmoid(shot).numpy())
    return np.mean(recv_p, 0), np.mean(shot_p, 0)


# --- evaluation ----------------------------------------------------------------------
def receiver_report(name, probs, g: CornerGraphs, n_boot):
    ev = g.target >= 0
    p, t, groups = probs[ev], g.target[ev], g.meta["match_id"].to_numpy()[ev]
    team = g.meta["label_team"].to_numpy()[ev]
    top1, top3 = M.topk_accuracy(p, t, 1), M.topk_accuracy(p, t, 3)
    nll = M.receiver_nll(p, t)
    return {
        "model": name,
        "top1": M.mean_ci(top1, groups, n_boot=n_boot),
        "top3": M.mean_ci(top3, groups, n_boot=n_boot),
        "nll": M.mean_ci(nll, groups, n_boot=n_boot),
        "top3_attack": float(top3[team == "attack"].mean()),
        "top3_defence": float(top3[team == "defence"].mean()),
        "_per_corner": {"top1": top1, "top3": top3, "nll": nll, "groups": groups},
    }


def shot_report(name, p, g: CornerGraphs, n_boot):
    y, groups = g.shot.astype(float), g.meta["match_id"].to_numpy()
    ll, br = M.log_loss_each(p, y), M.brier_each(p, y)
    return {
        "model": name,
        "log_loss": M.mean_ci(ll, groups, n_boot=n_boot),
        "brier": M.mean_ci(br, groups, n_boot=n_boot),
        "auc": M.cluster_bootstrap(lambda i: M.auc(p[i], y[i]), groups, n_boot=n_boot),
        "ece": M.ece(p, y),
        "reliability": M.reliability_bins(p, y),
        "_per_corner": {"log_loss": ll, "brier": br, "groups": groups},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--quick", action="store_true", help="1 seed, fewer epochs (debugging)")
    ap.add_argument("--rescore", action="store_true", help="recompute metrics from saved predictions")
    args = ap.parse_args()
    torch.set_num_threads(max(1, torch.get_num_threads()))

    g = CornerGraphs.load(PROCESSED_DIR / "corner_graphs.npz")
    split = g.meta["split"].to_numpy()
    if args.split == "val":
        train = g.subset(np.flatnonzero(split == "train"))
        evalset = g.subset(np.flatnonzero(split == "val"))
        # The GNN early-stops on an inner slice of *training* matches, never on the
        # evaluation set. Otherwise it would get a selection advantage the baselines don't.
        rng = np.random.default_rng(123)
        train_matches = train.meta["match_id"].unique()
        inner = set(rng.choice(train_matches, size=len(train_matches) // 7, replace=False))
        is_inner = train.meta["match_id"].isin(inner).to_numpy()
        gnn_train, val_for_gnn = train.subset(np.flatnonzero(~is_inner)), train.subset(np.flatnonzero(is_inner))
    else:
        train = gnn_train = g.subset(np.flatnonzero(split != "test"))
        evalset, val_for_gnn = g.subset(np.flatnonzero(split == "test")), None
    print(f"train {len(train.meta)} graphs | eval ({args.split}) {len(evalset.meta)} graphs, "
          f"{(evalset.target >= 0).sum()} confident receiver labels")  # fmt: skip

    if args.rescore:
        # Recompute every metric from saved predictions (e.g. after a metric fix).
        saved = np.load(REPORTS / f"predictions_{args.split}.npz", allow_pickle=True)
        assert (saved["event_id"] == evalset.meta["event_id"].to_numpy()).all()
        preds = {k: saved[k] for k in saved.files if k != "event_id"}
        log = {k: v for k, v in json.loads((REPORTS / f"results_{args.split}.json").read_text()).items()
               if k == "gnn_epochs"}  # fmt: skip
    else:
        preds, log = fit_and_predict(args, train, gnn_train, val_for_gnn, evalset)
    write_report(args, preds, log, evalset)


def fit_and_predict(args, train, gnn_train, val_for_gnn, evalset):
    """Train every model and return ({"recv_<model>" | "shot_<model>": predictions}, log)."""
    seeds = 1 if args.quick else args.seeds
    epochs = 40 if args.quick else 150
    T = args.temperature
    preds, log = {}, {}
    t0 = time.time()
    preds["recv_uniform"] = B.uniform_receiver(evalset)
    for regime in ("hard", "hybrid"):
        dist, has = receiver_targets(train, regime, T)
        print(f"[{regime}] training corners with a receiver target: {has.sum()}")
        preds[f"recv_hotspot_{regime}"] = B.HotspotReceiver().fit(train, dist).predict(evalset)
        preds[f"recv_lgbm_{regime}"] = B.LGBMReceiver().fit(train, dist, has).predict(evalset)

        models, epochs_used = [], []
        for seed in range(seeds):
            if args.split == "test":
                # Frozen epoch count from the validation run (saved in the val results).
                val_log = json.loads((REPORTS / "results_val.json").read_text())["gnn_epochs"]
                # Scale up slightly: the final model sees ~35% more data than the inner-train fit.
                ep = int(np.median(val_log[regime]) * 1.2)
                model, _ = train_gnn(train, dist, has, None, seed, epochs=ep)
            else:
                gdist, ghas = receiver_targets(gnn_train, regime, T)
                model, ep = train_gnn(gnn_train, gdist, ghas, val_for_gnn, seed, epochs=epochs)
            models.append(model)
            epochs_used.append(ep)
            print(f"    gnn[{regime}] seed {seed}: {ep} epochs ({time.time() - t0:.0f}s)")
        log.setdefault("gnn_epochs", {})[regime] = epochs_used
        MODELS.mkdir(parents=True, exist_ok=True)
        for seed, model in enumerate(models):
            torch.save(model.state_dict(), MODELS / f"corner_gnn_{regime}_{args.split}_s{seed}.pt")
        preds[f"recv_gnn_{regime}"], preds[f"shot_gnn_{regime}"] = gnn_predict(models, evalset)

    preds["shot_base_rate"] = B.BaseRateShot().fit(train).predict(evalset)
    preds["shot_logistic"] = B.LogisticShot().fit(train).predict(evalset)
    preds["shot_lgbm"] = B.LGBMShot().fit(train).predict(evalset)
    print(f"trained in {time.time() - t0:.0f}s")
    return preds, log


RECV_ORDER = ["uniform", "hotspot_hard", "lgbm_hard", "gnn_hard", "hotspot_hybrid", "lgbm_hybrid",
              "gnn_hybrid"]  # fmt: skip
SHOT_ORDER = ["base_rate", "logistic", "lgbm", "gnn_hard", "gnn_hybrid"]


def write_report(args, preds, log, evalset):
    recv_rows = [receiver_report(m, preds[f"recv_{m}"], evalset, args.n_boot) for m in RECV_ORDER]
    shot_rows = [shot_report(m, preds[f"shot_{m}"], evalset, args.n_boot) for m in SHOT_ORDER]
    for r in recv_rows:
        print(f"  receiver {r['model']:16s} top1 {M.fmt_ci(r['top1'], True)}  "
              f"top3 {M.fmt_ci(r['top3'], True)}  nll {r['nll'][0]:.3f}  "
              f"(top3 att {r['top3_attack']:.2f} / def {r['top3_defence']:.2f})")  # fmt: skip
    for r in shot_rows:
        print(f"  shot     {r['model']:16s} logloss {M.fmt_ci(r['log_loss'])}  "
              f"auc {M.fmt_ci(r['auc'])}  ece {r['ece']:.3f}")  # fmt: skip

    # Pre-specified paired comparisons.
    comparisons = {}
    by = {r["model"]: r["_per_corner"] for r in recv_rows}
    for a, b in [("gnn_hybrid", "lgbm_hybrid"), ("gnn_hard", "lgbm_hard"),
                 ("gnn_hybrid", "gnn_hard"), ("lgbm_hybrid", "lgbm_hard"),
                 ("lgbm_hybrid", "hotspot_hybrid")]:  # fmt: skip
        comparisons[f"receiver top3: {a} - {b}"] = M.paired_diff_ci(
            by[a]["top3"], by[b]["top3"], by[a]["groups"], n_boot=args.n_boot)
    sby = {r["model"]: r["_per_corner"] for r in shot_rows}
    for a, b in [("gnn_hybrid", "lgbm"), ("lgbm", "base_rate"), ("gnn_hybrid", "base_rate")]:
        comparisons[f"shot log loss: {a} - {b}"] = M.paired_diff_ci(
            sby[a]["log_loss"], sby[b]["log_loss"], sby[a]["groups"], n_boot=args.n_boot)
    print("paired differences (95% CI):")
    for k, v in comparisons.items():
        print(f"  {k:45s} {M.fmt_ci(v)}")

    REPORTS.mkdir(parents=True, exist_ok=True)
    strip = lambda rows: [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    out = {"split": args.split, "n_eval_graphs": len(evalset.meta),
           "n_eval_receiver": int((evalset.target >= 0).sum()), "temperature": args.temperature,
           "receiver": strip(recv_rows), "shot": strip(shot_rows),
           "comparisons": comparisons, **log}  # fmt: skip
    (REPORTS / f"results_{args.split}.json").write_text(json.dumps(out, indent=2, default=float))
    np.savez_compressed(REPORTS / f"predictions_{args.split}.npz", **preds,
                        event_id=evalset.meta["event_id"].to_numpy())  # fmt: skip
    print(f"-> {REPORTS}")


if __name__ == "__main__":
    main()
