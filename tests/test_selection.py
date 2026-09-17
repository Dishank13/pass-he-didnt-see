import numpy as np
import pandas as pd

from phds.features.pass_features import PASS_FEATURES
from phds.models.selection import SelectionModel, softmax_by_group


def test_softmax_by_group_normalises_each_pass():
    p = softmax_by_group(np.array([0.0, 0.0, 1.0, 5.0, 5.0]), np.array(["a", "a", "b", "b", "b"]))
    np.testing.assert_allclose(p[:2], [0.5, 0.5])
    np.testing.assert_allclose(p[2:].sum(), 1.0)
    assert p[3] == p[4] > p[2]


def test_selection_model_learns_preference_for_short_passes():
    rng = np.random.default_rng(0)
    rows, chosen = [], []
    for e in range(600):
        lengths = rng.uniform(5, 50, 6)
        pick = int(np.argmin(lengths + rng.normal(0, 3, 6)))  # players prefer short passes
        for j, length in enumerate(lengths):
            r = dict.fromkeys(PASS_FEATURES, 0.0) | {"length": length, "event_id": f"e{e}"}
            rows.append(r)
            chosen.append(j == pick)
    opts = pd.DataFrame(rows)
    train = opts["event_id"].str[1:].astype(int) < 500
    model = SelectionModel(n_estimators=100, min_child_samples=20).fit(opts[train], np.array(chosen)[train])
    test = opts[~train].reset_index(drop=True)
    p = model.predict(test)
    np.testing.assert_allclose(pd.Series(p).groupby(test["event_id"]).sum(), 1.0)
    short = test.groupby("event_id")["length"].transform("min") == test["length"]
    assert p[short.to_numpy()].mean() > 0.4  # well above 1/6
