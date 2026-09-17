# M1 walkthrough: corners, a TacticAI replication on open data

**Questions:** (1) Who touches a corner first? (2) Does a shot follow? (3) How should the
defence adjust? And for each: does a graph neural network actually beat simpler models?

Read in this order: `features/corner_graph.py` → `models/corner_baselines.py` →
`models/corner_gnn.py` → `eval/metrics.py` → `scripts/train_corners.py` →
`models/defence_search.py` → `notebooks/01_corners.ipynb`. Figures are in `reports/m1/`.

---

## 1. Corners as graphs

Each corner is a **complete graph** over the visible players (median 19). Nodes carry
position, role (attacker / taker / keeper), goal geometry, distance to the posts and the taker,
and local marking density (opponents and teammates within 2 yd). The model computes
**edge features** (relative position, distance, same team) from node positions.

**Only information available at the kick.** Pass end location, height and technique are known
in the event data, but they're *outcomes* of the corner. Using them would leak the answer.

**Canonicalisation over augmentation.** Corners from the far flag are mirrored (y → 80 − y), so
every corner is taken from the same side. With ~1k examples, it's better to remove the symmetry
from the data than to hope the model learns it. `tests/test_corner_graph.py` checks that a corner
and its mirror image produce identical features. (Swapping in flip *augmentation* is exercise 3.)

## 2. The GNN (dense GATv2)

A complete graph with ≤22 nodes can be processed as a dense `[B, N, N]` tensor with a mask. That
*is* message passing on a complete graph, with no graph library required (`corner_gnn.py`,
~100 lines).

- **Message passing:** each player's state is updated from a weighted sum of every other player's
  state. Three layers mean information can travel "defender → attacker he marks → space behind".
- **Attention:** GATv2 scores each pair with `aᵀ LeakyReLU(W_q h_i + W_k h_j + W_e e_ij)`.
  Original GAT applies the nonlinearity *after* combining with `a`, so every node ranks its
  neighbours identically ("static attention"). GATv2 lets each player attend to different others.
- **Permutation equivariance:** shuffling the player order shuffles the node outputs the same way.
  No player ordering is baked in. That's the structural argument for a GNN over a flat MLP.
- **Masking:** padding and the taker get `-inf` logits. A test confirms that changing padded rows
  doesn't change any prediction.
- **Multi-task heads:** node logits (who touches it first) and pooled mean+max → shot probability,
  trained jointly on one trunk.

## 3. Evaluation you can trust

- **Cluster bootstrap by match.** Corners from one match are correlated. The test in
  `test_metrics.py` shows that ignoring this shrinks CIs by more than 2x on correlated data.
- **Paired differences.** "Is A better than B?" is answered with the CI of A − B on the same
  resampled matches, not by eyeballing two overlapping CIs.
- **Tie-aware top-k.** The first version broke ties by array order. Because StatsBomb lists
  teammates first, the *uniform* baseline scored 22% top-3 on attacking touches and 2% on
  defensive ones. Ties are now scored by their expected value. It's a small bug with a big lesson:
  sanity-check baselines by subgroup.
- **Selection bias caught mid-run.** The first protocol early-stopped the GNN on the *validation*
  set that also scored it. That gave the GNN a +4.7-point top-3 "win" over LightGBM, CI [1.4, 8.8].
  Early-stopping on an inner slice of training matches instead shrank it to **+1.0 [−4.2, 6.2]**.
  Any choice made while looking at a split contaminates that split.
- **Test once.** Choices were frozen on validation. Euro 2024 was scored once, after retraining
  on train+val (GNN epochs fixed from validation × 1.2). One later metric fix (tie-aware top-k)
  was applied by re-scoring saved predictions. No model or choice changed.

## 4. Results (Euro 2024 test set, 288 corners, 165 with a confident first-touch label)

### Receiver: who touches it first?

| model | top-1 % | top-3 % | NLL | top-3 attack / defence |
|---|---|---|---|---|
| uniform (chance) | 5.4 | 16.3 | 2.91 | 16 / 16 |
| hotspot (hard) | 3.6 [0.7, 7.0] | 18.2 [11.6, 24.9] | 3.00 | 3 / 34 |
| LightGBM (hard) | 28.5 [21.2, 35.9] | 50.9 [43.6, 58.1] | 2.33 | 36 / 67 |
| **GNN (hard)** | 26.1 [20.1, 32.1] | **55.2 [48.0, 62.7]** | **2.31** | 41 / 71 |
| LightGBM (hybrid) | 26.7 [20.1, 33.1] | 53.3 [46.1, 60.4] | 2.43 | 37 / 71 |
| GNN (hybrid) | 30.3 [23.7, 36.5] | 48.5 [41.2, 56.0] | 2.37 | 30 / 68 |

- Learned models reach **~5x chance on top-1 and ~3x on top-3**. "Stand in the usual spot"
  (hotspot) is barely better than chance, so *relative* positioning is what carries signal.
- **GNN vs LightGBM on the exact player: no significant difference.** Top-3 GNN − LGBM (hard)
  is +4.2 [−0.6, +9.5] on test and +0.5 [−3.9, +5.0] on validation.
- **Defensive first touches are much more predictable** (~70% top-3) than attacking ones (30–41%).
  Keepers and near-post defenders win the predictable first contacts. Which attacker wins it
  depends on runs and delivery, which a static frame can't see.
- **Soft labels did not help.** GNN hybrid − hard: −1.6 [−6.0, +3.8] (val) and −6.7 [−12.3, −1.0]
  (test). With 8 pre-specified comparisons, one borderline result is suggestive rather than
  conclusive. Likely cause: the soft targets inherit the noise that made those corners ambiguous
  in the first place.

### Which *team* wins the first touch?

| model | AUC [95% CI] |
|---|---|
| uniform | 0.47 [0.40, 0.53] |
| LightGBM (hard) | 0.57 [0.50, 0.63] |
| **GNN (hard)** | **0.65 [0.58, 0.71]** |
| GNN (hybrid) | 0.68 [0.61, 0.74] |

**Here the GNN clearly wins:** paired AUC difference GNN − LightGBM = **+0.08 [0.02, 0.14]**
(hard), +0.10 [0.03, 0.16] (hybrid). Summing node probabilities per team rewards a model that has
learned the *contest* (marking, crowding) rather than individual spots. Relational structure
is exactly what message passing is built for. This is the M1 headline for GNNs.

### Shot within 20 seconds

| model | log loss | AUC | ECE |
|---|---|---|---|
| base rate | 0.644 | 0.50 | 0.05 |
| logistic | 0.636 | 0.57 | 0.07 |
| LightGBM | 0.620 | 0.62 [0.55, 0.69] | 0.05 |
| GNN (hard) | 0.624 | 0.62 [0.55, 0.69] | 0.06 |

Weak at best: LightGBM/GNN beat the base rate on test (log loss −0.023 [−0.044, −0.003]) but not
on validation (AUC ~0.55, no log-loss gain). Whether a corner produces a shot depends mostly on
the delivery and the aerial duel, which happen *after* the frame. It's an honest negative result,
and the reason shot probability is **not** used as the defensive objective.

## 5. Defensive suggestions

**Method.** Greedy constrained search on the 40 test corners where the GNN most favours the
attackers. Each defender (keeper fixed) may move ≤3 yd, stay ≥1 yd from others, for ≤6 moves.
The objective is P(attackers win the first touch) under the hard-label GNN ensemble.

**Result.** P(attack first touch): 0.78 → 0.59 on average (−0.19), ~15 yd of total movement.

**But is it real?** An optimiser pointed at a neural network finds inputs the network *likes*
(adversarial examples use the same mechanism). Two controls:

| scorer | before | suggested | random moves (same sizes) | share of drop retained |
|---|---|---|---|---|
| GNN hard (optimised against) | 0.78 | 0.59 | 0.78 | 100% |
| GNN hybrid (independent seeds + targets) | 0.73 | 0.55 | 0.72 | 90% |
| LightGBM (different model family) | 0.62 | 0.55 | 0.61 | **36%** |

Suggested − random drop is positive under all three scorers (LightGBM: +0.06 [0.03, 0.08]).
**About a third of the improvement survives a different model family.** The rest is what one family
of models believes. Two models trained on the same features share biases, so the 90% GNN
transfer is weaker evidence than it looks.

**What the moves do:** 61% shift toward the near-post side and the number of defenders in the
six-yard box rises from 72 to 101 across the 40 corners. Only 54% end closer to an attacker. The
model's advice is "protect the near post and the six-yard box" rather than "mark tighter",
a recognisable zonal principle. It's still a hypothesis for a coach to evaluate, not a causal
claim: the model learned which setups *precede* attacker first touches, not what happens
when defenders move.

## 6. Honest summary for the README

- On ~1.2k training corners, a GNN **matches** LightGBM at naming the first toucher (~55% top-3,
  3x chance) and **beats** it at predicting which team wins the first contact (AUC +0.08).
- Shot outcomes are barely predictable from the setup alone.
- Soft labels for ambiguous corners didn't help.
- Defensive suggestions are consistent across GNNs, partially transfer to LightGBM (~1/3 of the
  effect), and beat random moves. Their tactical content is near-post / six-yard-box protection.
- Not replicated from TacticAI: velocities (unavailable in 360 data, see M3) and the generative
  defensive model (optional stretch below).

## 7. Try this

1. **Team-held-out split.** Remove every team in Euro 2024 from training and re-run
   `train_corners.py --split test`. How much of the receiver accuracy is team-specific routine?
2. **Stricter labels.** Rebuild corners with `min_regret=2.0, margin=3.0`. Does GNN vs LightGBM
   change? (The M0 walkthrough promised this check.)
3. **Augmentation instead of canonicalisation.** Disable the flip in `build_corner_graphs`, and
   instead add a flipped copy of every training corner. Compare validation NLL.
4. **Attack the objective harder.** Raise `max_move` to 6 yd and `max_rounds` to 15. Does the
   LightGBM-retained share of the drop fall? (It should, if larger moves leave the data distribution.)
5. **Robust objective.** Optimise the *average* P(attack) of the GNN and LightGBM together.
   Suggestions that satisfy both families should transfer better to a third model (try logistic
   regression on `receiver_node_table` features).
6. **Stretch: conditional generative model.** Train a small CVAE that generates defender positions
   given attacker positions. Compare its samples' P(attack) with the greedy search's (TacticAI's approach).
