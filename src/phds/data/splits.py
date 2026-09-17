"""Train/val/test splits that can't leak.

Why not split events at random?
Events from the same match share everything a model can latch onto: the teams,
their set-piece routines, the camera angle and calibration, the weather, the score
state. A random event-level split puts near-duplicate corners (same routine, same
match) in both train and test, which inflates accuracy. So:

* **Test = a whole held-out tournament** (default: Euro 2024). This measures
  generalisation to new matches *and* a new time period / squads.
* **Validation = whole matches** sampled from the remaining competitions,
  stratified by competition so every league is represented.

Caveat worth stating in the write-up: teams recur across tournaments (e.g. Spain
appears in Euro 2020, WC 2022 and Euro 2024), so a team's set-piece habits can
still carry over from train to test. A stricter team-held-out split is an
optional robustness check.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_TEST = ((55, 282),)  # (competition_id, season_id): UEFA Euro 2024


def assign_splits(
    matches: pd.DataFrame,
    test_competitions: tuple[tuple[int, int], ...] = DEFAULT_TEST,
    val_frac: float = 0.15,
    seed: int = 0,
) -> pd.Series:
    """Return a Series mapping match_id -> "train" | "val" | "test"."""
    comp_key = list(zip(matches["competition_competition_id"], matches["season_season_id"]))
    is_test = pd.Series([k in set(test_competitions) for k in comp_key], index=matches.index)
    split = pd.Series("train", index=matches.index)
    split[is_test] = "test"

    rng = np.random.default_rng(seed)
    rest = matches[~is_test]
    for _, g in rest.groupby(["competition_competition_id", "season_season_id"]):
        n_val = round(len(g) * val_frac)
        val_idx = rng.choice(g.index.to_numpy(), size=n_val, replace=False)
        split[val_idx] = "val"

    return pd.Series(split.to_numpy(), index=matches["match_id"].to_numpy(), name="split")


def check_no_leakage(df: pd.DataFrame, split_col: str = "split", group_col: str = "match_id"):
    """Raise if any match appears in more than one split."""
    n_splits = df.groupby(group_col)[split_col].nunique()
    leaked = n_splits[n_splits > 1]
    if not leaked.empty:
        raise AssertionError(f"{len(leaked)} {group_col}s appear in multiple splits")
