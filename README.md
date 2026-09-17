# The Pass He Didn't See

A mini TacticAI: given where every visible player was (StatsBomb 360 freeze frames),
what was the best passing option, and how much expected value did the actual pass
gain or lose?

**Status:** M0 (data foundation and audit) complete. Next: M1 corners.

### M0 in numbers
- 426 matches, 1.36M StatsBomb 360 freeze frames, ~25M player positions
- 4,020 corners → 1,364 with a reliable first-touch label (inferred by linking anonymous
  freeze-frame players across frames with Hungarian matching; ~94% cross-method agreement)
- Leakage-safe split: Euro 2024 held out as the test tournament

See `notebooks/00_data_audit.ipynb` and `docs/learning/M0_walkthrough.md`.

## Quickstart
```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"
.venv/Scripts/python scripts/download_data.py    # ~850 files, cached gzip in data/raw
.venv/Scripts/python scripts/build_dataset.py    # Parquet tables + labelled corners
.venv/Scripts/python -m pytest
```

## Layout
- `src/phds/data`: download, parsing, label inference, leakage-safe splits
- `src/phds/geometry`: coordinate conventions and symmetry augmentation
- `src/phds/viz`: pitch plots
- `notebooks/`: numbered, narrative analyses
- `docs/learning/`: per-milestone walkthroughs

Data: StatsBomb Open Data. See [ATTRIBUTION.md](ATTRIBUTION.md).
