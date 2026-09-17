import pandas as pd
import pytest

from phds.data.splits import assign_splits, check_no_leakage


def _matches():
    rows = []
    mid = 0
    for comp, season, n in [(55, 282, 10), (43, 106, 40), (9, 281, 20)]:
        for _ in range(n):
            rows.append(dict(match_id=mid, competition_competition_id=comp, season_season_id=season))
            mid += 1
    return pd.DataFrame(rows)


def test_test_split_is_exactly_the_held_out_tournament():
    m = _matches()
    s = assign_splits(m)
    test_ids = set(m.loc[m.competition_competition_id == 55, "match_id"])
    assert set(s[s == "test"].index) == test_ids


def test_val_is_stratified_and_deterministic():
    m = _matches()
    s1, s2 = assign_splits(m, seed=3), assign_splits(m, seed=3)
    pd.testing.assert_series_equal(s1, s2)
    per_comp = m.assign(split=s1.to_numpy()).groupby("competition_competition_id")["split"]
    assert (per_comp.apply(lambda x: (x == "val").sum()).loc[[43, 9]] == [6, 3]).all()


def test_every_match_gets_exactly_one_split():
    m = _matches()
    s = assign_splits(m)
    assert s.index.is_unique and len(s) == len(m)


def test_check_no_leakage_detects_overlap():
    ok = pd.DataFrame(dict(match_id=[1, 1, 2], split=["train", "train", "test"]))
    check_no_leakage(ok)
    bad = pd.DataFrame(dict(match_id=[1, 1], split=["train", "test"]))
    with pytest.raises(AssertionError):
        check_no_leakage(bad)
