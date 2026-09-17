# M2 walkthrough: the pass he didn't see

**Question:** given every visible player at the moment of a pass, which option had the highest
expected value, and how does the pass actually played compare?

Read in this order: `data/frame_store.py` → `data/pass_targets.py` → `features/pass_features.py` →
`models/completion.py` → `models/value.py` → `models/pass_value.py` → `scripts/score_passes.py` →
`notebooks/02_pass_value.ipynb`. Figures and tables are in `reports/m2/`.

---

## 1. Decomposing "best pass" into things with ground truth

There's no label for "the best pass". There are labels for its ingredients:

| ingredient | question | label | model |
|---|---|---|---|
| completion | will a pass to *j* arrive? | real passes: completed or not | LightGBM + isotonic calibration |
| success value | how good is it for us to have the ball at *p_j*? | did we score within 10 actions? | LightGBM, location + frame context |
| failure cost | how bad is it if they win it near *p_j*? | did *they* score within 10 actions? | same value model, mirrored location |

```
EV_j = P_j · V_us(p_j) + (1 − P_j) · V_fail(p_j),    V_fail(p_j) = −V_them(mirror(p_j))
```

**Why multiply calibrated probabilities?** EV is a sum of products of probabilities. If P_j is
systematically 2 points too high, every risky option gets inflated, and the "best option" drifts
toward risky passes. This is why calibration matters more here than AUC: the probabilities feed
further calculations rather than just ranking.

## 2. Possession value (VAEP-style)

For every on-ball action by team T: did T score (or concede) within the next 10 actions?
V = P(score) − P(concede). This is the VAEP formulation (Decroos et al., 2019). Compared with
**xT** (expected threat: a grid of scoring chances learned from move/shot transitions), it's a
supervised classifier that can take any features. That lets us add freeze-frame context
(defenders goal-side, nearest opponent, support ahead), which xT can't use.

Base rates are tiny: ~1.1% of actions precede a goal within 10 actions, 0.2% precede conceding.
With rare events, **log loss differences look small in absolute terms** (0.001) and still be
highly significant. Read them next to AUC and the paired CIs.

## 3. Pitch control and the physics prior

**Pitch control** (Taki & Hasegawa; Spearman's 2018 model) assigns each pitch location to the
team that would reach it first, given positions, velocities, reaction times and ball travel time.
We have no velocities (limitation 1). `features/pass_features.py` implements the core idea in its
simplest form: an **interception time margin**.

```
t_ball(s) = s / v_ball          t_def = reaction + perpendicular distance / v_player
margin    = min over defenders (t_def − t_ball)       (negative ⇒ someone gets there first)
```

This matters for **limitation B (selection bias)**. A learned model only sees passes players chose,
and nobody tries passes straight through two defenders, so the model has little evidence about
them. The physics margin encodes *why* such passes fail, so the completion model penalises blocked
lanes even where attempts are rare. On validation, the 4-feature physics logistic alone reaches AUC
0.86 with ECE 0.007, and it's the second-most important LightGBM feature.

M3 quantifies what the missing velocities cost.

## 4. The intended target: anonymous dots again

Training "a pass to *j* at *p_j*" needs *which dot* the passer aimed at. The M0 machinery was reused:
the intended recipient's Ball Receipt event (present even for failed passes), linked back to the
pass frame by nearest player, whole-team assignment, or the keeper flag.

**Why not the pass end location?** For an intercepted pass it lies *on the interceptor*, so "a defender
at the destination" would leak the outcome and give a model that looks brilliant and fails on
counterfactuals. The target is always the teammate's frame position, for real and hypothetical passes alike.

**Selection and linkage.** 68% of passes link. Failures link far less often (31% vs 76%). Diagnosis:
30% of failed passes' recipients are **off camera** (vs 14% of completed ones), and failed receipts are
located less precisely. Two consequences:
1. The population the model should be calibrated for is *passes to visible teammates*. Off-camera
   targets are never options anyway.
2. Linked passes over-represent completions (93.5% vs ~87%), so we **reweight** within
   (outcome × length) strata to the in-view population (`linkage_weights`). Training and every
   metric use these weights.

## 5. Calibration, done without touching evaluation data

Weighted LightGBM over-predicted completion by 1.7 points on validation. The fix is **cross-fitted isotonic
regression**: out-of-fold predictions on *training* matches (5 match-grouped folds) → a monotone map
→ applied to the final model. Validation ECE went 0.017 → 0.006 with AUC unchanged (isotonic is
monotone, so rankings are preserved).

## 6. Scoring every pass honestly: out-of-fold EV

Per-player aggregates are only meaningful if no pass is scored by a model that trained on it.
`score_passes.py` uses 4 match-grouped folds, fits all three models on 3 folds, and scores every option
of every pass in the 4th. Euro 2024 test metrics for the components are reported separately (trained on
train+val, scored once).

## 7. The diagnostic that reshaped M2

The first full scoring run used the goal10 value, and the numbers looked wrong:

| | played pass | "best" option |
|---|---|---|
| median P(complete) | 0.96 | 0.77 |
| median length / forward progress | 15 yd / +2 yd | 26 yd / +14 yd |
| played = best | | 14.7% (chance 15.8%) |

Players choose the 78th-percentile option by completion probability but the 39th by success value.
EV's within-pass ranking correlated 0.60 with success value and 0.07 with completion probability.
**Before blaming the players, blame the model.** Two causes:

1. **Myopic value.** With a 10-action horizon, a midfield turnover rarely costs a goal *within
   10 actions*, so failure looked nearly free (median failure value −0.004). A possession-level value
   fixes the incentive: a turnover forfeits the rest of our possession and starts theirs.
2. **Selection bias (limitation B).** Out-of-fold completion probabilities are calibrated even on risky
   *attempted* forward passes (0.755 predicted vs 0.762 observed). But options nobody played may have been
   avoided for reasons the frame can't show. No data can check the model there. The standard off-policy
   remedy is to **only trust value estimates for actions the behaviour policy actually takes**, so a
   **pass-selection model** defines "plausible" options (P(chosen) ≥ 10%, fixed in advance).

These changes were motivated by a diagnostic, not chosen blind. The notebook shows both value definitions
side by side so the effect is visible rather than hidden.

## 8. Results

### Components (Euro 2024 test set, scored once)

| model | metric |
|---|---|
| completion, distance-only logistic | log loss 0.349 |
| completion, **physics prior** (4 features) | log loss 0.241, AUC 0.891 |
| completion, **LightGBM + isotonic** | **log loss 0.220, AUC 0.912, ECE 0.011, mean pred 0.880 = observed 0.880** |
| value goal10, location only / + frame context | score AUC 0.78 / **0.79**; concede AUC 0.79 / **0.83** |
| value poss, location only / + frame context | R² 0.039 / **0.055**, decile-calibrated (top decile 0.045 predicted vs 0.041 observed) |

Physics prior vs distance: log loss −0.108 [−0.118, −0.098]. LightGBM vs physics: −0.022 [−0.025, −0.019].
Frame context improves both value models (paired CIs exclude zero).

### Decision layer (220k passes, 1.5M options, out-of-fold)

| | played | best (all), goal10 | best (all), poss | best (plausible), poss |
|---|---|---|---|---|
| median P(complete) | 0.962 | 0.765 | 0.849 | 0.960 |
| median forward progress | +1.9 yd | +14.1 yd | +10.9 yd | +2.6 yd |
| played = best | | 14.7% | 20.5% | 50.1% (median 2 plausible options) |
| mean choice percentile | | 0.517 | 0.593 | |

- **The selection model** is strong: the played option is its top pick 58% of the time, top-3 90%
  (mean P(played) 0.46 vs 0.16 uniform).
- **Predictive validity:** EV of the chosen pass predicts what happens next *beyond* the passer's location.
  Goal within 10 actions: AUC 0.734 → **0.764**. Realised possession value: Spearman 0.157 → **0.169**.
- **Missed EV and outcomes:** larger gaps to the best plausible option go with worse realised value beyond
  location, ρ = −0.063 [−0.068, −0.058]. Partly mechanical: large-gap passes are riskier and fail more
  (89% vs 96% completion).

### Is decision quality a player trait?

Split-half reliability (Spearman across random halves of a player's matches, ≥150 passes per half,
87 players; all metrics team- and role-relative):

| metric | reliability |
|---|---|
| completion rate (reference) | 0.58 |
| regret vs best plausible option (poss) | **0.49** |
| choice percentile, goal10 | 0.38 |
| choice percentile, poss (the pre-specified leaderboard metric) | **0.19** |

**Honest reading:** the pre-specified metric is only weakly stable. 15 of 61 players have CIs excluding zero
(~3 expected by chance), so there is *some* signal. The leaderboard (Rodri, Busquets, Rice, Stones and Kroos
near the top) has face validity, but it's exploratory. Regret is more stable, partly because it tracks risk
appetite, which is itself a stable style trait rather than "decision quality".

### A humbling example

Euro 2024 final, 72': Bellingham's lay-off to Palmer (P 0.96) became England's equaliser. The model
prefers a riskier ball into the box (P 0.36) with higher EV (+0.095 vs +0.027). Both can be true: EV is an
average over many repetitions of that moment, and a single outcome doesn't validate or refute it. This is
exactly how the demo should present recommendations.

## 9. Limitations to state in the README

- Visible options only. Off-camera teammates are never recommended (30% of failed passes go off camera).
- No velocities: completion and value see a snapshot. M3 measures the cost.
- Frame context for a receiver is measured at the moment of the pass, not at arrival.
- Counterfactual completion is unverifiable for never-attempted options. Plausibility filtering limits exposure,
  it doesn't remove it.
- Decision-quality leaderboards are weakly reliable (split-half 0.19–0.49).

## 10. Try this

1. **Horizon sensitivity.** Change `horizon` in `action_labels` to 5 and 25. How does the goal10-style EV's
   preference for risk (median P of best option) move? It should approach the poss behaviour as the horizon grows.
2. **Plausibility threshold.** Re-run `score_passes.py --decisions-only --plausible 0.05` and `0.2`.
   How do leaderboard reliability and "played = best" change?
3. **Physics speeds.** Set `V_PLAYER = 6` and `V_BALL = 20` in `pass_features.py`, rebuild features, and compare the
   physics logistic's AUC. How sensitive is the prior to its constants?
4. **Selection model with value.** Add `ev_poss` as a selection-model feature. Does predicting choices improve?
   (If yes, professionals' choices carry information about value the geometry alone misses.)
5. **Failure location.** Replace the target position with the lane midpoint as the turnover location in
   `option_tables`. Which options gain or lose EV, and does predictive validity change?
