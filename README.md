# The Pass He Didn't See

A mini TacticAI: given where every visible player was (StatsBomb 360 freeze frames),
what was the best passing option, and how much expected value did the actual pass
gain or lose?

**Status:** M0 (data) and M1 (corners) complete. Next: M2 open-play pass value.

### M0 in numbers
- 426 matches, 1.36M StatsBomb 360 freeze frames, ~25M player positions
- 4,020 corners → 1,364 with a reliable first-touch label (inferred by linking anonymous
  freeze-frame players across frames with Hungarian matching; ~94% cross-method agreement)
- Leakage-safe split: Euro 2024 held out as the test tournament

See `notebooks/00_data_audit.ipynb` and `docs/learning/M0_walkthrough.md`.

### M1 in numbers (Euro 2024 held-out test, 95% match-level bootstrap CIs)
- **First-touch player:** GNN top-3 accuracy 55% [48, 63] vs 16% chance. LightGBM 51% [44, 58];
  the difference is not significant.
- **Which team wins the first touch:** GNN AUC 0.65 vs LightGBM 0.57, a paired difference of
  **+0.08 [0.02, 0.14]**. Relational structure is where the graph model earns its keep.
- **Shot within 20 s:** AUC ~0.62 on test, ~0.55 on validation. The setup alone barely predicts it.
- **Defensive suggestions** (≤3 yd moves): P(attackers win first touch) 0.78 → 0.59. Random moves
  of the same size: no change. ~1/3 of the effect survives re-scoring by a different model family.
  Tactical content: protect the near post and six-yard box.

See `notebooks/01_corners.ipynb` and `docs/learning/M1_walkthrough.md`.

## Quickstart
```bash
python -m venv .venv
.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python -m pip install -e ".[dev]"
.venv/Scripts/python scripts/download_data.py    # ~850 files, cached gzip in data/raw
.venv/Scripts/python scripts/build_dataset.py    # Parquet tables + labelled corners
.venv/Scripts/python scripts/build_corner_graphs.py
.venv/Scripts/python scripts/train_corners.py    # validation; add --split test for the final run
.venv/Scripts/python -m pytest
```

## Layout
- `src/phds/data`: download, parsing, label inference, leakage-safe splits
- `src/phds/geometry`: coordinate conventions and symmetry augmentation
- `src/phds/viz`: pitch plots
- `notebooks/`: numbered, narrative analyses
- `docs/learning/`: per-milestone walkthroughs

Data: StatsBomb Open Data. See [ATTRIBUTION.md](ATTRIBUTION.md).
