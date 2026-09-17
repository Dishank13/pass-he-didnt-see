# M3 walkthrough: what does a 360 snapshot miss?

**Question:** StatsBomb 360 frames have no velocities and show only players on the broadcast camera.
M1 and M2 are built on them. How much predictive accuracy does that cost?

Read in this order: `data/skillcorner.py` → `features/tracking_passes.py` → `scripts/velocity_study.py` →
`notebooks/03_velocity_study.ipynb`. Figures are in `reports/m3/`.

---

## 1. The dataset choice (and why the plan changed)

The plan said Metrica sample games (true full tracking). Checking access first changed that:

| | Metrica sample | SkillCorner open data |
|---|---|---|
| matches | 2 (+1 in another format) | **20** A-League 2024/25 |
| tracking | full, 25 fps, all 22 players | broadcast, 10 fps, detected **or extrapolated** per player |
| visibility | none (must be simulated) | **real** `is_detected` flag + camera footprint |
| failed passes | `BALL LOST`, **no intended receiver** | **targeted player recorded** for every pass |
| licence | "acknowledge the source" | MIT |

The deciding factor was the **intended target on failed passes**. Without it, the task falls back on pass end
locations, which for interceptions lie on the interceptor, the same leakage M2 was designed to avoid. The
trade-off: SkillCorner's off-screen players are *extrapolated estimates*, so "full information" means the
best available reconstruction, not ground truth.

**Sanity check that "on screen" ≈ 360:** median players visible at a pass is 15 (SkillCorner) vs 16
(StatsBomb 360), and the target is on screen for 82% of passes (StatsBomb: 86% of completed passes' recipients in view).

## 2. Design: same passes, four information conditions

| condition | players | velocities | mimics |
|---|---|---|---|
| full_vel | all 22 | ✓ | full tracking |
| full_pos | all 22 | – | a perfect snapshot |
| vis_vel | on screen | ✓ | broadcast tracking of visible players |
| vis_pos | on screen | – | **StatsBomb 360** |

- **Same passes in every condition** (12,867 where passer and target are on screen). Differences between
  conditions are then caused by information, not by sample composition.
- **The M2 features are reused unchanged**, after converting SkillCorner metres to the StatsBomb frame.
- **Velocities** come from a 0.5 s *backward* difference, which uses only information available at the pass.
- **Leave-one-match-out** prediction and **paired** contrasts. CIs bootstrap over 10-minute match blocks:
  20 matches is too few clusters for a match-level bootstrap, so this is a stated compromise.
- **Two model families.** A physics logistic (interception margins only) and LightGBM (all features). They
  answer different questions: "does velocity improve a *physical* model?" vs "is there *any* usable signal?"

### Pitch control and time-to-intercept, with velocities
The M2 margin assumed defenders start from rest: `t = reaction + distance / v_max`. Pitch-control models
(Spearman 2018; the Friends of Tracking/Metrica tutorial version) let a player keep moving along their current
velocity during the reaction time and then sprint:

```
t_intercept = reaction + | p + v·reaction − point | / v_max
```

That's `dyn_*_margin` in `tracking_passes.py`.

## 3. Results (SkillCorner, 20 matches, 12,867 passes, 85.8% completed)

| model | full_vel | full_pos | vis_vel | vis_pos (360-like) |
|---|---|---|---|---|
| physics logistic, AUC | 0.832 | 0.853 | 0.826 | 0.847 |
| physics, static + dynamic margins, AUC | 0.854 | | 0.847 | |
| LightGBM, AUC | **0.864** | 0.859 | 0.863 | **0.858** |

Paired contrasts (ΔAUC, richer − poorer information, 95% CI):

| contrast | LightGBM | physics |
|---|---|---|
| velocity (all players) | **+0.004** [0.002, 0.007] | −0.021 [−0.025, −0.017] |
| visibility (no velocity) | +0.001 [−0.001, 0.004] | +0.006 [0.003, 0.009] |
| **total: full tracking vs 360-like** | **+0.006** [0.003, 0.009] | −0.015 |
| velocity margins *added to* static margins | | +0.001 [−0.000, 0.002] |

**What velocity information matters** (LightGBM ablation on top of static features):

| added | ΔAUC |
|---|---|
| receiver and passer movement (speed, speed along and across the pass line) | **+0.006** [0.003, 0.008] |
| defender velocity margins and closing speed | +0.000 [−0.001, 0.002] |

Velocity helps most for **fast-moving receivers** (+0.009) and short passes (+0.007), and not measurably for long passes.

### Reading the results

1. **For predicting whether a chosen pass arrives, a 360-style snapshot loses very little.** ΔAUC is 0.006, and
   log loss is 0.296 vs 0.292. That's reassuring for M1/M2: the headline models aren't crippled by missing
   velocities.
2. **What's missing is the receiver's run, not defenders' momentum.** Receiver movement explains the entire
   velocity gain. In football terms, a 360 frame can't tell a player checking towards the ball from one drifting away.
3. **Naive physics with velocities made things *worse*.** Projecting every defender along their velocity for
   0.5 s overshoots: real defenders decelerate and redirect, and broadcast-derived velocities (especially
   extrapolated players') are noisy. Where movement is largest (fast receivers, short passes) the dynamic model
   degrades most (−0.039). Nested inside the static model it adds nothing. **Lesson:** "add more physics"
   isn't automatically better. Better pitch-control models handle this with probabilistic arrival times
   and time-integrated control, and they tune the parameters.
4. **Off-screen players barely matter *for this task*,** given passer and target are on screen. That doesn't mean
   visibility is harmless for M2's *option* evaluation: an off-screen defender near an off-screen teammate can't
   be scored at all, and 18% of targets are off screen. That's limitation D, measured here, not solved.

### External validation of M2

The M2 completion model, trained **only on StatsBomb 360** from other competitions, applied unchanged to
A-League broadcast tracking (a different league, provider and technology):

| | AUC | ECE |
|---|---|---|
| M2 model on vis_pos (360-like) | **0.841** [0.830, 0.850] | 0.010 |
| in-domain LightGBM, leave-one-match-out | 0.858 | 0.005 |

The model transfers with a modest AUC loss and stays well calibrated. That's strong evidence the M2 features
capture football, not StatsBomb-specific artefacts. (On full_pos it over-predicts slightly, ECE 0.025, since it
never saw frames with all 22 players.)

**Reference:** SkillCorner's own `xpass_completion` scores AUC 0.900 on the same passes. Its inputs are
undocumented and may use information our setup deliberately excludes, so it's a ceiling-ish reference, not a
like-for-like competitor.

## 4. Honest limits of this study

- One league, 20 matches, broadcast tracking. Extrapolated positions aren't ground truth.
- The task is completion of the *chosen* pass. Option-level EV (M2c) may be more sensitive to missing players.
- The velocity physics is deliberately simple. A tuned probabilistic pitch-control model might extract more.

## 5. Try this

1. **Better projection.** Replace `drift = p + v·reaction` with a capped version (`v` clipped to 4 yd/s, or
   reaction 0.3 s). Does the dynamic physics model stop underperforming the static one?
2. **Off-screen targets.** Rerun the LightGBM conditions on passes where the *target is off screen*
   (full vs vis, with the target forced in). How much worse is the 360-like condition when the key player isn't visible?
3. **Option ranking.** For each pass, score all teammates with the M2 transfer model under full_pos and vis_pos.
   How often does the best option change? That measures visibility's effect on "the pass he didn't see" directly.
4. **Receiver-run feature for 360.** StatsBomb has no velocities, but consecutive 360 frames of the same possession
   sometimes exist (e.g. carry → pass). Could a coarse movement estimate be recovered?
5. **Metrica check.** Using Metrica's full tracking, simulate visibility with *real StatsBomb camera footprints*
   centred on the ball, and compare with SkillCorner's real detection flags.
