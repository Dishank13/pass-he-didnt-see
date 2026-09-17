# M0 walkthrough: data foundation and audit

**Goal of M0:** turn StatsBomb Open Data into clean, leakage-safe tables. Then answer three
questions *before* modelling: how much data is there, what does a freeze frame actually
show, and can we trust the corner labels?

Read the code in this order: `geometry/coords.py` → `data/load_statsbomb.py` →
`data/freeze_frames.py` → `data/labels.py` → `data/splits.py` → `notebooks/00_data_audit.ipynb`.

---

## 1. The data model

StatsBomb gives two files per match:

| File | Unit | What's in it |
|---|---|---|
| `events/{match}.json` | one on-ball action | type, player, team, location, pass/shot/carry details, `possession` sequence number |
| `three-sixty/{match}.json` | one frame per event | `freeze_frame`: every *visible* player's location + `teammate`/`actor`/`keeper` flags; `visible_area` polygon |

Things that aren't obvious from the docs, and that shape everything later:

1. **No player identities in frames.** A freeze frame is anonymous dots. You know which dot
   is the passer (`actor`) and which team each dot is on, and nothing more. Any per-player
   label (receiver, marker) has to be *inferred*. See section 3.
2. **Perspective flips per event.** Each event and its frame are expressed from the acting
   team's point of view, attacking towards x = 120. A defender's clearance after your corner
   is in the *other* team's coordinates. `to_opponent_perspective` (a 180° rotation) converts.
   Getting this wrong silently mislabels every defensive touch. There's a unit test for it.
3. **Intended recipients on failed passes.** Incomplete passes still carry `pass.recipient`.
   That gives M2's completion model a positive *and* a negative label for the same target type.
4. **Not every event has a frame.** Broadcast cuts (replays, close-ups) mean some events have
   no frame. Corners are hit harder than open play (see audit), plausibly because broadcasts
   cut away during set-piece setup. This is a *selection effect*: frames aren't missing at random.
5. **Upstream data can be corrupt.** One 360 file (match 3845506) contains a run of spaces
   overwriting part of the JSON. The pipeline logs it to `data/processed/data_issues.json`
   and keeps that match's events, rather than crashing or silently dropping it.

### Storage decisions
- Raw JSON is cached gzip-compressed (516 MB for 426 matches). Processed Parquet is ~400 MB.
- `players.parquet` is long format (~25M rows: one row per visible player per frame). It's
  written **one match at a time** with explicit schemas, float32 coordinates and
  dictionary-encoded `event_id`. That's ~22 bytes/row in memory. The first naive version
  (a list of Python dicts) would have needed >10 GB. The downloader had the same bug:
  thread-pool futures held every parsed file, which reached 9 GB before it stalled.
  Lesson: at this scale, hold data in columnar form and stream it.

## 2. Symmetry and coordinates

The pitch is 120 × 80 yd, origin top-left, y pointing down. Football is approximately
invariant to **reflecting across the long axis** (y → 80 − y: swap wings). That's our
augmentation. Reflecting across halfway (x → 120 − x) is **not** valid: it swaps the goal
being attacked. TacticAI used a D2 group (both flips) only because corners can be
canonicalised to one side. `tests/test_coords.py` checks that goal distance and angle are
invariant under the flip. Later, any model's predictions should be too (M1 test).

## 3. Inferring the corner's first touch (limitation A)

**Label definition (following TacticAI):** the first player of *either* team to touch the ball.

**Step 1: first touch.** The first event within 6 s after the corner whose type is a real
touch (receipt, clearance, keeper action, shot…). Two details matter:
- A `Ball Receipt*` *with* an outcome is a failed receipt. The ball never arrived, so it's not a touch.
- The window needs a **lower** bound too. StatsBomb has rare timestamp glitches (an event
  stamped `00:00:01` in minute 47). Without the bound, those became "first touches" with
  a 20-minute negative delay. The smoke test found it.

**Step 2: link the toucher to a dot in the corner frame.** The hard part.

*Method A, nearest:* the dot closest to where the touch happened. It fails for two reasons:
players move 5–10 yd during a 1–2 s delivery (attackers make runs), and defenders stand in
tight zonal clusters (the nearest and second nearest are within a yard).

*Method B, assignment:* the touch event has **its own freeze frame**, where the toucher is
the `actor`. So we match the whole team between the two frames by solving a linear assignment
problem (Hungarian algorithm, `scipy.optimize.linear_sum_assignment`) on the displacement
matrix. The actor's partner is the label. Two design details:
- **Capped costs** `min(distance, max_move)`: players can be off camera in either frame, and
  an uncapped far pair would distort everyone else's matching. `max_move = 3 + 9·dt` yd (sprint speed).
- **Confidence = regret:** re-solve with the chosen pairing forbidden. The increase in total
  displacement tells you how much better this explanation is than the next best. It's a
  *global* margin, analogous to the "2nd nearest − nearest" margin but using the whole team.

*Combining:* take a confident assignment. Otherwise take a confident nearest match that the
assignment doesn't contradict. If both are confident but disagree, **drop** the corner.
Dropping biases the sample (toward clean, well-filmed corners), but mislabels are worse.
With only ~1.4k labelled corners, a 10% label-noise rate caps achievable accuracy and blurs model comparisons.

*Goalkeepers:* every frame flags the keeper. If the first touch is by a goalkeeper, the label
is exact. That one rule rescued 212 labels (+18%). It came from noticing that "Goal Keeper"
touches had the lowest yield (27%) of any touch type.

**Label-noise estimate:** where both methods are independently confident, they agree
**94.3%** of the time (96.3% for attacking touches, 91.6% for defensive ones; n = 460). A small
60-match smoke test had suggested 98.7%. It's a good reminder to re-measure on the full data.
The methods use different evidence, so agreement is a reasonable proxy for accuracy. It's an
*upper bound*: both can be fooled by the same crossing runs.

## 4. Splits (leakage)

- **Test = Euro 2024, all 51 matches.** It's held out completely: new matches, a later time
  period, different squads.
- **Validation = 15% of matches** from each other competition (stratified).
- Never split at the event level. Corners from one match share routines, camera setup and game
  state, so a random split leaks. `check_no_leakage` runs inside `build_dataset.py`.
- **Known residual leakage:** national teams recur (Spain: Euro 2020, WC 2022, Euro 2024), so
  team-specific routines can carry over. M1 reports a team-held-out robustness check.

## 5. Results (from `notebooks/00_data_audit.ipynb`, figures in `reports/m0/`)

**Coverage.** 426 matches across 11 competitions (AFCON has 1 match and effectively no frames;
MLS 2023 has 6 matches at 44% frame coverage). 1.59M events, **1.36M freeze frames (86%)**,
~25M player positions.

**What a frame shows** (medians):

| | players visible | teammates | opponents | pitch visible |
|---|---|---|---|---|
| corner | 19 | 8 | 11 | 16% |
| open-play pass | 16 | 8 | 9 | 29% |

Corners are the best case for player coverage (almost everyone is in the box, in shot), even
though the camera shows only 16% of the pitch. Open-play frames miss ~6 players on average.
That's why "best *visible* option" is the honest framing for M2.

**Passes (M2 preview).** 322k open-play passes with frames, 84.1% completed. 51k incomplete
passes, 74% with an intended recipient. The completion model has plenty of negatives.

**Corners.**

| | count |
|---|---|
| corners | 4,020 |
| with freeze frame | 2,468 (61%, vs 86% for events overall) |
| first touch found | 2,351 |
| **usable label** | **1,364** (58% of touched): 634 attack / 730 defence |
| dropped: ambiguous / too far / conflict | 832 / 132 / 22 |
| split (train / val / test) | 1,007 / 192 / 165 |

- Label methods among usable labels: assignment 859, keeper flag 291, nearest 214.
- **Selection check:** the shot-within-20s rate is 37.3% for corners *without* a frame and 37.1%
  *with* one. So missing frames don't appear to be selected on corner danger.
- **Threshold sensitivity** (share of touched corners labelled / share dropped as conflicts):
  loosening to `min_regret=0.5` gains ~7 points of yield but doubles-to-triples conflicts
  (2.6%, vs 1.0% at the chosen `min_regret=1.0, margin=1.5`). Tightening to `2.0 / 3.0` costs
  14 points. The defaults sit at a reasonable knee.

### What this means for M1 (the key risk, surfaced early)
**~1,000 training corners is small**, about 7x fewer than TacticAI's 7,176. Consequences:
1. Report **top-3 accuracy with bootstrap CIs**. On 165 test corners, a 95% CI on top-1
   accuracy is roughly ±7 points, so small differences between models are not real.
2. Consider **soft labels** instead of dropping ambiguous corners. With 832 ambiguous corners,
   a training target spread over the 2–3 plausible players (weighted by assignment cost)
   uses the data instead of discarding it. Evaluate only on the confident set.
3. Keep **shot prediction** (graph-level, all 2,468 corners with frames, no identity label
   needed) as an equally important M1 target. It doesn't depend on limitation A at all.

## 6. Try this

1. **Tighten the label.** In `notebooks/00_data_audit.ipynb`, set `min_regret=2.0, margin=3.0`.
   How many labels survive? Keep that stricter corner set around. In M1, check whether the
   GNN-vs-baseline ranking changes when trained on it.
2. **Break the perspective.** Comment out the `to_opponent_perspective` call in `labels.py`
   and run `pytest`. Which test fails, and what would the defensive labels have looked like?
   (Plot a few with `plot_freeze_frame`.)
3. **Selection bias in frames.** Compare the `shot_within` rate for corners *with* vs *without*
   a freeze frame (the `no_frame` status). If they differ, the 360 subset isn't representative.
   Say so in the write-up.
4. **Upper bound on the noise estimate.** Find corners where the methods agree but the toucher's
   displacement (`assign_move`) is >8 yd. Plot 5 and judge by eye whether the label looks right.
5. **Visible area vs label yield.** Bin corners by `visible_frac`. Does the ok-label rate drop for
   tight camera shots? Is that a reason to add `visible_frac` as a model input?
