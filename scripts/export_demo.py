"""Export model outputs as small JSON files for the static demo app.

    python scripts/export_demo.py   ->  app/public/data/{moments,eval_set,stats}.json

moments.json    passes from major-tournament finals: goal assists, the largest gaps to the best
                plausible option ("the pass he didn't see"), and strong decisions
eval_set.json   40 blind A/B moments for the human evaluation (see docs/human_eval_protocol.md),
                drawn from non-final matches so they can't overlap the explorer. The answer key is
                written separately to reports/m4/eval_key.json and is never served by the app.
stats.json      headline numbers from M0-M3 reports

All EVs use the possession value (units: expected-goal difference) and out-of-fold scoring.
Data: StatsBomb Open Data (non-commercial, attribution required).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from phds.data.freeze_frames import PROCESSED_DIR, load_table
from phds.data.load_statsbomb import cached_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "app" / "public" / "data"
REPORTS = ROOT / "reports"
FINALS = ["FIFA World Cup 2022", "UEFA Euro 2024", "UEFA Euro 2020", "UEFA Women's Euro 2022",
          "UEFA Women's Euro 2025", "Women's World Cup 2023"]  # fmt: skip
EVAL_SIZE, EVAL_SEED = 40, 2026
R = lambda v, d=4: None if v is None or (isinstance(v, float) and np.isnan(v)) else round(float(v), d)


def zone(x: float, y: float) -> str:
    """Plain-language location, from the attacking team's perspective (attacking towards x = 120)."""
    side = "left" if y < 80 / 3 else "right" if y > 160 / 3 else "central"
    if x >= 102 and 18 <= y <= 62:
        return "in the box" if side == "central" else f"in the box, {side} side"
    if x >= 90:
        return "at the edge of the box" if side == "central" else f"on the {side} wing near the box"
    if x >= 80:
        return f"in the final third, {side}" if side != "central" else "in the final third, central"
    if x >= 40:
        return f"in midfield, {side}" if side != "central" else "in central midfield"
    return f"in the defensive third, {side}" if side != "central" else "in the defensive third, central"


def load_all():
    dec = pd.read_parquet(PROCESSED_DIR / "pass_decisions.parquet")
    opts = pd.read_parquet(PROCESSED_DIR / "pass_options.parquet", columns=[
        "event_id", "player_idx", "opt_x", "opt_y", "is_actual", "p_complete", "policy_p",
        "v_success_poss", "v_fail_poss", "ev_poss"])  # fmt: skip
    matches = load_table("matches")
    matches["comp"] = matches["competition_competition_name"] + " " + matches["season_season_name"]
    events = load_table("events", columns=["event_id", "match_id", "pass_goal_assist", "pass_shot_assist"])
    dec = dec.merge(matches[["match_id", "comp", "competition_stage_name", "home_team_home_team_name",
                             "away_team_away_team_name", "home_score", "away_score", "match_date"]], on="match_id")
    dec = dec.merge(events[["event_id", "pass_goal_assist", "pass_shot_assist"]], on="event_id", how="left")
    return dec, opts


def nickname_map(match_ids) -> dict[str, str]:
    """StatsBomb event names are full legal names ("Ángel Fabián Di María Hernández"); lineups carry
    the familiar ones ("Ángel Di María"). Returns full name -> nickname for the given matches."""
    out = {}
    for mid in match_ids:
        for team in cached_json(f"lineups/{int(mid)}.json"):
            for pl in team["lineup"]:
                if pl.get("player_nickname"):
                    out[pl["player_name"]] = pl["player_nickname"]
    return out


def frame_payload(event_ids):
    players = load_table("players", filters=[("event_id", "in", list(event_ids))])
    players["event_id"] = players["event_id"].astype(str)
    frames = load_table("frames", columns=["event_id", "visible_area"],
                        filters=[("event_id", "in", list(event_ids))]).set_index("event_id")  # fmt: skip
    out = {}
    for eid, g in players.groupby("event_id"):
        va = frames.loc[eid, "visible_area"] if eid in frames.index else []
        out[eid] = {
            "players": [{"idx": int(r.player_idx), "x": R(r.x, 2), "y": R(r.y, 2), "teammate": bool(r.teammate),
                         "actor": bool(r.actor), "keeper": bool(r.keeper)} for r in g.itertuples()],
            "visible_area": [[R(va[i], 2), R(va[i + 1], 2)] for i in range(0, len(va) - 1, 2)],
        }
    return out


def moment_record(r, o: pd.DataFrame, frame: dict, kind: str, plausible: float = 0.10) -> dict:
    o = o.sort_values("ev_poss", ascending=False)
    best_all = o.iloc[0]
    pl = o[(o["policy_p"] >= plausible) | o["is_actual"]]
    best_pl = pl.iloc[0]
    return {
        "id": r.event_id, "kind": kind, "competition": r.comp, "stage": r.competition_stage_name,
        "match": f"{r.home_team_home_team_name} {r.home_score}–{r.away_score} {r.away_team_away_team_name}",
        "date": str(r.match_date), "minute": int(r.minute), "team": r.team, "passer": r.player,
        "recipient": r.recipient, "outcome": r.outcome, "goal_assist": bool(r.pass_goal_assist),
        "passer_xy": [R(r.x, 2), R(r.y, 2)],
        "actual_idx": int(r.target_idx), "best_plausible_idx": int(best_pl.player_idx),
        "best_all_idx": int(best_all.player_idx),
        "ev_actual": R(r.ev_actual_poss), "ev_best_plausible": R(best_pl.ev_poss),
        "delta_ev_plausible": R(best_pl.ev_poss - r.ev_actual_poss),
        "best_plausible_zone": zone(best_pl.opt_x, best_pl.opt_y),
        "actual_zone": zone(r.target_x, r.target_y),
        "options": [{"idx": int(q.player_idx), "x": R(q.opt_x, 2), "y": R(q.opt_y, 2), "p": R(q.p_complete, 3),
                     "v_success": R(q.v_success_poss), "v_fail": R(q.v_fail_poss), "ev": R(q.ev_poss),
                     "policy_p": R(q.policy_p, 3), "plausible": bool(q.policy_p >= plausible or q.is_actual),
                     "actual": bool(q.is_actual)} for q in o.itertuples()],
        **frame,
    }


def select_finals(dec: pd.DataFrame) -> pd.DataFrame:
    f = dec[dec["comp"].isin(FINALS) & dec["competition_stage_name"].eq("Final")]
    picks = []
    for _, g in f.groupby("match_id"):
        assists = g[g["pass_goal_assist"].fillna(False)].assign(kind="goal assist")
        missed = (g[(g["completed"] == 1) & (g["n_plausible"] >= 2) & ~g["event_id"].isin(assists["event_id"])]
                  .nlargest(3, "delta_ev_plausible_poss").assign(kind="the pass he didn't see"))  # fmt: skip
        good = (g[g["actual_is_best_plausible_poss"] & (g["n_plausible"] >= 2) & (g["actual_p_complete"] < 0.85)
                  & ~g["event_id"].isin(pd.concat([assists, missed])["event_id"])]
                .nlargest(2, "ev_actual_poss").assign(kind="best option, taken under risk"))  # fmt: skip
        picks.append(pd.concat([assists, missed, good]))
    return pd.concat(picks).sort_values(["comp", "minute"])


def select_eval(dec: pd.DataFrame, exclude_matches: set) -> pd.DataFrame:
    """Blind A/B candidates: the best plausible option differs clearly from the played pass."""
    c = dec[~dec["match_id"].isin(exclude_matches) & ~dec["actual_is_best_plausible_poss"]
            & (dec["n_plausible"] >= 2)].copy()  # fmt: skip
    c["sep"] = np.hypot(c["best_plausible_poss_opt_x"] - c["target_x"], c["best_plausible_poss_opt_y"] - c["target_y"])
    c = c[(c["sep"] >= 10) & (c["delta_ev_plausible_poss"] >= c["delta_ev_plausible_poss"].quantile(0.5))]
    c["third"] = np.where(c["x"] >= 80, "final third", "build-up")
    rng = np.random.default_rng(EVAL_SEED)
    # One pass per match at most, half from build-up and half from the final third.
    c = c.sample(frac=1, random_state=EVAL_SEED).drop_duplicates("match_id")
    picked = pd.concat([g.head(EVAL_SIZE // 2) for _, g in c.groupby("third")])
    picked["model_is_a"] = rng.integers(0, 2, len(picked)).astype(bool)  # fixed A/B order, stored
    return picked


def stats_payload() -> dict:
    def j(p):
        path = REPORTS / p
        return json.loads(path.read_text()) if path.exists() else None

    m1, comp, val, m3 = j("m1/results_test.json"), j("m2/completion_test.json"), j("m2/value_test.json"), j("m3/velocity_study.json")
    pick = lambda rows, key, name: next(r for r in rows if r[key] == name)
    return {
        "m0": {"matches": 426, "freeze_frames": 1357627, "corners_labelled": 1364, "passes_linked": 220029},
        "m1": {"receiver_top3_gnn": pick(m1["receiver"], "model", "gnn_hard")["top3"],
               "receiver_top3_lgbm": pick(m1["receiver"], "model", "lgbm_hard")["top3"],
               "receiver_top3_uniform": pick(m1["receiver"], "model", "uniform")["top3"],
               "team_auc_gnn": 0.647, "team_auc_lgbm": 0.565},
        "m2": {"completion_auc": pick(comp["models"], "model", "lgbm")["auc"],
               "completion_ece": pick(comp["models"], "model", "lgbm")["ece"],
               "physics_auc": pick(comp["models"], "model", "physics")["auc"],
               "value_r2": pick(val["possession_value"], "model", "poss_ctx")["r2_vs_mean"],
               "selection_top3": 0.904, "split_half_choice": 0.193, "split_half_regret": 0.489},
        "m3": {"lgbm_full": next(r for r in m3["rows"] if r["model"] == "lgbm" and r["condition"] == "full_vel")["auc"],
               "lgbm_360like": next(r for r in m3["rows"] if r["model"] == "lgbm" and r["condition"] == "vis_pos")["auc"],
               "total_gap": m3["contrasts"]["lgbm: total: full tracking vs 360-like"]["auc_diff"],
               "transfer_auc": next(r for r in m3["rows"] if r["model"] == "m2_statsbomb_transfer" and r["condition"] == "vis_pos")["auc"]},
    }  # fmt: skip


def main():
    dec, opts = load_all()
    finals = select_finals(dec)
    evalset = select_eval(dec, exclude_matches=set(finals["match_id"]))
    ids = set(finals["event_id"]) | set(evalset["event_id"])
    frames = frame_payload(ids)
    opts = opts[opts["event_id"].isin(ids)]
    by_event = dict(tuple(opts.groupby("event_id")))

    nick = nickname_map(finals["match_id"].unique())
    finals = finals.assign(player=finals["player"].map(lambda n: nick.get(n, n)),
                           recipient=finals["recipient"].map(lambda n: nick.get(n, n) if isinstance(n, str) else n))  # fmt: skip
    moments = [moment_record(r, by_event[r.event_id], frames[r.event_id], r.kind)
               for r in finals.itertuples() if r.event_id in frames]  # fmt: skip
    eval_items, eval_key = [], {}
    for r in evalset.itertuples():
        if r.event_id not in frames:
            continue
        rec = moment_record(r, by_event[r.event_id], frames[r.event_id], "eval")
        a, b = (rec["best_plausible_idx"], rec["actual_idx"]) if r.model_is_a else (rec["actual_idx"], rec["best_plausible_idx"])
        # Blind: strip everything that reveals which option was played or what the model thinks.
        eval_items.append({
            "id": rec["id"], "players": rec["players"], "visible_area": rec["visible_area"],
            "passer_xy": rec["passer_xy"],
            "option_a": next([o["x"], o["y"]] for o in rec["options"] if o["idx"] == a),
            "option_b": next([o["x"], o["y"]] for o in rec["options"] if o["idx"] == b),
        })  # fmt: skip
        # Answer key stays out of the served app (reports/m4/eval_key.json) to keep raters blind.
        eval_key[rec["id"]] = {"model_option": "A" if r.model_is_a else "B", "ev_model": rec["ev_best_plausible"],
                               "ev_actual": rec["ev_actual"], "delta_ev": rec["delta_ev_plausible"],
                               "third": r.third, "competition": rec["competition"], "outcome": rec["outcome"]}

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "moments.json").write_text(json.dumps({"moments": moments}, separators=(",", ":")), encoding="utf-8")
    (OUT / "eval_set.json").write_text(json.dumps({"seed": EVAL_SEED, "items": eval_items}, separators=(",", ":")), encoding="utf-8")
    (OUT / "stats.json").write_text(json.dumps(stats_payload(), indent=1), encoding="utf-8")
    key_dir = REPORTS / "m4"
    key_dir.mkdir(parents=True, exist_ok=True)
    (key_dir / "eval_key.json").write_text(json.dumps({"seed": EVAL_SEED, "key": eval_key}, indent=1), encoding="utf-8")
    kinds = pd.Series([m["kind"] for m in moments]).value_counts().to_dict()
    print(f"{len(moments)} moments {kinds}; {len(eval_items)} eval items "
          f"(model option = A in {sum(k['model_option'] == 'A' for k in eval_key.values())})")  # fmt: skip
    for f in OUT.iterdir():
        print(f"  {f.name}: {f.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
