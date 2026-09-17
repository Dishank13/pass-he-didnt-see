# M4 walkthrough: demo app and blind evaluation

**Goal:** make the work explorable by people who won't read a notebook, and set up an honest test of whether
humans agree with the model.

Read in this order: `scripts/export_demo.py` → `app/src/lib/types.ts` → `app/src/components/Pitch.tsx` →
`app/src/views/*` → `docs/human_eval_protocol.md` → `scripts/analyse_human_eval.py`.

---

## 1. Architecture: precompute everything, keep the app thin

The models are Python and some take an hour to run. The demo doesn't need live inference: every moment it shows is
fixed. So `export_demo.py` writes three small JSON files (≈150 KB in total), and the app is a static site with no
backend, no API keys, no cost, and nothing to break.

| file | contents |
|---|---|
| `moments.json` | 38 passes from six major finals: all players, the camera's visible area, every option's P / values / EV / plausibility |
| `eval_set.json` | 40 blind A/B items: players, camera area, the two option locations. **No labels, names or model numbers** |
| `stats.json` | headline numbers from M0–M3 result files |

The price of this design is that new moments need a re-export. That's acceptable for a portfolio demo.

## 2. Design: what the data sees

The visual signature is the **camera footprint**. The pitch is drawn in shadow with a hatch, and only the area the
broadcast camera covered is lit. It isn't decoration: it's limitation D made visible. Every number on the page comes
from the lit area only, and teammates in the shadow are never suggested.

Other decisions:
- **Telestrator vocabulary.** The pass played is a straight chalk line; the model's suggestion is a curved yellow
  telestrator stroke. Match headers use broadcast lower-third styling (minute · team · passer → recipient).
- **Units people can read.** Percentages for P and "+0.095 xG" for value. "Expected value" is always explained as an
  average, never as a verdict.
- **Plausible options by default.** Implausible options are hidden behind a toggle and drawn dashed, matching the
  modelling decision from M2.
- **Real names.** StatsBomb events use legal names ("Ángel Fabián Di María Hernández"); lineups carry nicknames. The
  export maps them. Otherwise the explorer would read like a court record.
- **Phones.** Below 900 px the pitch crops to the half holding all the action (portrait), and the moment list
  becomes a select menu. No horizontal scrolling at 375 px.
- **Accessibility.** Options are keyboard-focusable buttons with descriptive labels, focus rings are visible,
  arrow animation respects `prefers-reduced-motion`, and there's a light/dark theme.

## 3. The blind test: designed to avoid fooling ourselves

A tempting version shows people the model's suggestion next to the real pass with outcomes and names. That measures
reputation and hindsight, not football judgement. The protocol removes both:

- **Blind:** identical styling for A and B, no outcome, names, teams, minute or score. The answer key is written to
  `reports/m4/eval_key.json`, outside the served app, so a curious rater can't find it in the network tab.
- **Balanced randomisation:** which label is the model's option is fixed per item (seed 2026) and balanced 20/20, so
  a rater who always presses A scores exactly 50%.
- **Selection rules fixed in advance:** non-final matches only (no overlap with the explorer), one moment per match,
  options ≥10 yd apart, top-half EV gaps, stratified build-up / final third.
- **Pre-registration:** the question, primary metric, test and *interpretation of each possible result* are written
  down before any data exists. Changes after data collection go under "Deviations".
- **The right uncertainty:** judgments are crossed (each rater × each moment), so the CI comes from a **two-way
  bootstrap** that resamples raters and moments independently. Treating 200 judgments as independent would
  overstate precision.
- **Power, checked by simulation before recruiting:** with 5 raters, a true 60% preference gives CI 0.49–0.68 and
  isn't detectable. Hence the target of ≥10 raters, and the pre-committed wording for a null result: "not detectable
  with this sample", never "no effect".

## 4. What the blind test can and can't show

- **Can:** whether football watchers find the suggestions better, worse or indistinguishable *as decisions*.
- **Can't:** whether the suggested pass would have worked. Humans share the same blind spots as the frame: no runs,
  nothing off camera.
- **The most interesting null:** "indistinguishable" still means something. The model's suggestions look like real
  professional choices, which is exactly what the plausibility filter was designed to achieve.

## 5. Deploying

`npm run build` produces `app/dist` with relative asset paths (`base: "./"`), so it works on GitHub Pages under
`/<repo>/`. StatsBomb data is non-commercial with attribution: the footer and the How it works page credit both providers.

## 6. Try this

1. **Add corners.** Export six Euro 2024 corners with first-touch probabilities and the M1 defensive suggestions,
   and add a Corners view that reuses `Pitch` (the rings become receiver probabilities).
2. **Option explanations.** For the selected option, show which completion features pushed P up or down (a small SHAP
   export per option).
3. **Sensitivity in the UI.** Add the plausibility threshold as a slider (5% / 10% / 20%) using the stored `policy_p`,
   so the "best plausible option" updates live.
4. **Run the test.** Recruit raters, run `analyse_human_eval.py`, and write up the result, especially if it's null.
