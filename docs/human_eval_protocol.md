# Blind human evaluation: protocol (pre-registered)

Written on 2026-09-17, **before any responses were collected**. Changes after data collection
must be listed under "Deviations" with a reason.

## Question

When experienced football watchers see a real moment frozen at the instant of a pass, do they prefer
the model's best *plausible* option over the pass the professional actually played?

This tests whether the model's "pass he didn't see" suggestions look like better football to humans,
not just to the model. It is **not** a test of whether the suggestions would have worked.

## Materials

- 40 moments exported by `scripts/export_demo.py` (seed 2026) into `app/public/data/eval_set.json`.
  - From non-final matches only, so none overlap the public explorer. At most one moment per match.
  - The best plausible option differs from the played pass, the two targets are ≥10 yd apart, and the EV
    gap is in the top half of such passes.
  - Stratified: 20 in build-up (passer x < 80) and 20 in the final third.
- Two passes are drawn from the passer, labelled A and B, with identical chalk styling. Which label is the
  model's option was randomised once (seed 2026) and balanced (20 / 20). The answer key lives in
  `reports/m4/eval_key.json` and is never served by the app. It is committed to the **private** repository so it
  survives local cleanup; if the repository is ever made public before data collection ends, remove it first
  (`scripts/export_demo.py` regenerates it deterministically from the processed data).
- No outcome, score, names, teams, minute or model numbers are shown.

## Raters

- Target: **≥10 raters** (minimum 5), recruited from friends who watch football regularly. Each self-reports background
  (played/coached, regular watcher, casual fan) and optionally initials. No other personal data.
- Raters are told truthfully that one pass was played and one is a model suggestion, in random order.
- Raters complete the test alone in the web app (`#blind-test`) and send the exported JSON file.

## Primary analysis

- **Outcome per judgment:** 1 if the rater chose the model's option, 0 if they chose the played pass.
  "Can't choose" answers are excluded from the primary analysis and reported as a rate.
- **Estimate:** the proportion of judgments preferring the model's option.
- **Uncertainty:** a two-way cluster bootstrap resampling moments and raters (10,000 draws), 95% percentile CI.
- **Test:** H0: preference = 0.5. The two-sided result counts as significant if the 95% CI excludes 0.5.
- **Interpretation, fixed in advance:**
  - CI above 0.5: humans tend to prefer the suggestion.
  - CI below 0.5: humans prefer the professional's choice. That would support the view that the model's EV
    misses something players and watchers see.
  - CI includes 0.5: the suggestions aren't distinguishable from real choices to these raters. That's
    evidence they're at least plausible, not that they're better.

## Secondary analyses (exploratory, labelled as such)

1. Preference by stratum (build-up vs final third).
2. Preference by rater background.
3. Correlation between the model's EV gap and the share of raters preferring its option, per moment (Spearman).
4. Inter-rater agreement: Fleiss' kappa on A/B choices.
5. "Can't choose" rate and median response time.

## Power note

A simulation run before data collection (5 simulated raters, true preference 60%, 5% "can't choose")
gave an estimate of 0.588 with 95% CI 0.494–0.681. **With 5 raters, even a real 60% preference is not
detectable**, which is why the target is ≥10 raters (the CI narrows by roughly 1/√2 on the rater
dimension; moment-level clustering still limits it). A null result must be reported as "not detectable
with this sample", not as "no effect".

## Deviations

*(none yet)*
