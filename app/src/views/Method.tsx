import { ci } from "../lib/format";
import type { Stats } from "../lib/types";

export function Method({ stats }: { stats: Stats | null }) {
  const s = stats;
  return (
    <article className="method">
      <h2>How it works</h2>
      <p className="lede">
        For every pass with a StatsBomb 360 freeze frame, the model scores every visible teammate as a passing option and compares the
        best one with the pass that was actually played.
      </p>

      <section>
        <h3>What each option is worth</h3>
        <p className="formula">
          expected value = <b>P(arrives)</b> × value if it arrives + (1 − <b>P(arrives)</b>) × value if it's lost
        </p>
        <ul>
          <li><b>P(arrives)</b>: a gradient-boosted model of pass completion, built on how much time defenders need to reach the passing lane and the receiver. It's calibrated, so its percentages mean what they say.</li>
          <li><b>Value</b>: expected goals from the rest of the possession, minus the expected goals the opponent creates if they win the ball there. Losing the ball deep in the opponent's half can score slightly above zero, because teams that win it near their own goal often give it straight back.</li>
          <li><b>Plausible options</b>: a separate model learns which passes professionals actually choose. Suggestions are limited to options it rates at least 10% likely, because the data can't tell us how passes nobody attempts would really go.</li>
        </ul>
      </section>

      {s && (
        <section>
          <h3>What the evidence says</h3>
          <table className="facts">
            <tbody>
              <tr><th>Data</th><td>{s.m0.matches} matches, {s.m0.freeze_frames.toLocaleString()} freeze frames, {s.m0.passes_linked.toLocaleString()} passes with an identified target</td></tr>
              <tr><th>Pass completion (Euro 2024, held out)</th><td>AUC {ci(s.m2.completion_auc)}, calibration error {s.m2.completion_ece.toFixed(3)}</td></tr>
              <tr><th>Predicting pros' choices</th><td>the pass played is in the model's top 3 {Math.round(s.m2.selection_top3 * 100)}% of the time</td></tr>
              <tr><th>Is decision quality a stable trait?</th><td>only weakly: split-half reliability {s.m2.split_half_choice.toFixed(2)}–{s.m2.split_half_regret.toFixed(2)}</td></tr>
              <tr><th>Corners: who touches it first (top 3)</th><td>graph network {ci(s.m1.receiver_top3_gnn, 2)} vs chance {s.m1.receiver_top3_uniform[0].toFixed(2)}</td></tr>
              <tr><th>What a snapshot misses (tracking data)</th><td>AUC {s.m3.lgbm_360like[0].toFixed(3)} from a 360-like snapshot vs {s.m3.lgbm_full[0].toFixed(3)} with full tracking; the gap is mostly the receiver's run</td></tr>
              <tr><th>Transfer to another league</th><td>trained on StatsBomb, tested on A-League broadcast tracking: AUC {ci(s.m3.transfer_auc)}</td></tr>
            </tbody>
          </table>
        </section>
      )}

      <section>
        <h3>What it can't see</h3>
        <ul>
          <li>Only players inside the camera view. Shaded areas on the pitch are unknown, and teammates there are never suggested.</li>
          <li>No player movement. A frame can't tell a player sprinting into space from one standing still.</li>
          <li>Expected value is an average over many replays of a moment. A pass that became a goal can still have had a better option, and vice versa.</li>
          <li>For options nobody attempts, completion chances can't be checked against data. That's why suggestions stay within plausible options.</li>
        </ul>
      </section>

      <section className="credits">
        <h3>Data and code</h3>
        <p>
          Event and 360 data: StatsBomb Open Data (non-commercial use, attribution). Tracking data for the velocity study: SkillCorner
          Open Data (MIT). Method write-ups, notebooks and code are in the project repository under <code>docs/learning</code> and <code>notebooks</code>.
        </p>
      </section>
    </article>
  );
}
