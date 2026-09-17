import { useMemo, useState } from "react";
import { Pitch, type PitchArrow } from "../components/Pitch";
import { pct, shortName, xg } from "../lib/format";
import type { Moment } from "../lib/types";

const KIND_NOTE: Record<Moment["kind"], string> = {
  "goal assist": "This pass led to a goal. Expected value averages over every way this moment could play out, so a gap here isn't a verdict on the goal.",
  "the pass he didn't see": "A completed pass, with a clearly better option that professionals regularly choose in situations like this.",
  "best option, taken under risk": "The riskiest-looking choice was also the best plausible one. The model agrees with the player.",
};

export function Explorer({ moments }: { moments: Moment[] }) {
  const groups = useMemo(() => {
    const m = new Map<string, Moment[]>();
    for (const x of moments) {
      const key = `${x.competition}|${x.match}`;
      m.set(key, [...(m.get(key) ?? []), x]);
    }
    return [...m.entries()];
  }, [moments]);

  const initial = moments.find((m) => m.goal_assist && m.passer.includes("Bellingham")) ?? moments[0];
  const [selectedId, setSelectedId] = useState(initial?.id);
  const [showAll, setShowAll] = useState(false);
  const [optionIdx, setOptionIdx] = useState<number | null>(null);
  const moment = moments.find((m) => m.id === selectedId) ?? moments[0];
  if (!moment) return <p className="empty">No moments found. Run scripts/export_demo.py to generate app/public/data.</p>;

  const played = moment.options.find((o) => o.actual)!;
  const best = moment.options.find((o) => o.idx === moment.best_plausible_idx)!;
  const bestAll = moment.options.find((o) => o.idx === moment.best_all_idx)!;
  const same = played.idx === best.idx;
  const shownOptions = showAll ? moment.options : moment.options.filter((o) => o.plausible);
  const selected = moment.options.find((o) => o.idx === optionIdx) ?? null;

  const arrows: PitchArrow[] = same
    ? [{ from: moment.passer_xy, to: [played.x, played.y], tone: "suggested" }]
    : [
        { from: moment.passer_xy, to: [played.x, played.y], tone: "played" },
        { from: moment.passer_xy, to: [best.x, best.y], tone: "suggested" },
      ];

  const choose = (id: string) => {
    setSelectedId(id);
    setOptionIdx(null);
  };

  return (
    <div className="explorer">
      <nav className="rail" aria-label="Moments from finals">
        <label className="rail-select">
          <span>Moment</span>
          <select value={moment.id} onChange={(e) => choose(e.target.value)}>
            {groups.map(([key, items]) => (
              <optgroup key={key} label={`${key.split("|")[0]} final · ${key.split("|")[1]}`}>
                {items.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.minute}' {shortName(m.passer)} → {shortName(m.recipient)} ({m.kind})
                  </option>
                ))}
              </optgroup>
            ))}
          </select>
        </label>
        {groups.map(([key, items]) => {
          const [comp, match] = key.split("|");
          return (
            <section key={key} className="rail-group">
              <h3>
                <span className="rail-comp">{comp} final</span>
                <span className="rail-score">{match}</span>
              </h3>
              <ul>
                {items.map((m) => (
                  <li key={m.id}>
                    <button className={m.id === moment.id ? "active" : ""} aria-current={m.id === moment.id} onClick={() => choose(m.id)}>
                      <span className="rail-min">{m.minute}'</span>
                      <span className="rail-who">
                        {shortName(m.passer)} <span aria-hidden>→</span> {shortName(m.recipient)}
                      </span>
                      <span className={`tag tag-${m.kind.split(" ")[0].replace("'", "")}`}>{m.kind}</span>
                    </button>
                  </li>
                ))}
              </ul>
            </section>
          );
        })}
      </nav>

      <article className="stage">
        <header className="lower-third">
          <div className="lt-main">
            <span className="lt-minute">{moment.minute}'</span>
            <span className="lt-team">{moment.team}</span>
            <span className="lt-pass">
              {shortName(moment.passer)} <span aria-hidden>→</span> {shortName(moment.recipient)}
            </span>
          </div>
          <div className="lt-sub">
            {moment.competition} final · {moment.match} · pass {moment.outcome === "Complete" ? "completed" : "not completed"}
            {moment.goal_assist ? " · assist" : ""}
          </div>
        </header>

        <div className="pitch-wrap">
          <Pitch
            players={moment.players}
            visibleArea={moment.visible_area}
            arrows={arrows}
            options={shownOptions}
            selectedIdx={optionIdx}
            onSelectOption={setOptionIdx}
            animationKey={moment.id}
            title={`Freeze frame: ${moment.passer} passes to ${moment.recipient ?? "a teammate"}, minute ${moment.minute}`}
          />
          <div className="legend" aria-hidden>
            <span><i className="lg lg-played" />played</span>
            <span><i className="lg lg-suggested" />best plausible option</span>
            <span><i className="lg lg-ring" />ring size = expected value</span>
            <span><i className="lg lg-shadow" />outside camera view</span>
          </div>
        </div>

        <div className="controls">
          <label className="toggle">
            <input type="checkbox" checked={showAll} onChange={(e) => setShowAll(e.target.checked)} />
            Show options professionals rarely choose (dashed)
          </label>
          <span className="hint">Select a ring to see that option's numbers.</span>
        </div>

        <section className="readout" aria-live="polite">
          <div className="read-col">
            <h4>Played</h4>
            <p className="read-where">to {shortName(moment.recipient)}, {moment.actual_zone}</p>
            <dl>
              <div><dt>chance it arrives</dt><dd className="num">{pct(played.p)}</dd></div>
              <div><dt>expected value</dt><dd className="num">{xg(played.ev)}</dd></div>
            </dl>
          </div>
          {same ? (
            <div className="read-col verdict">
              <h4>Best plausible option</h4>
              <p className="read-where">The pass played.</p>
              <p className="read-note">{KIND_NOTE[moment.kind]}</p>
            </div>
          ) : (
            <div className="read-col suggested">
              <h4>Best plausible option</h4>
              <p className="read-where">to the teammate {moment.best_plausible_zone}</p>
              <dl>
                <div><dt>chance it arrives</dt><dd className="num">{pct(best.p)}</dd></div>
                <div><dt>expected value</dt><dd className="num">{xg(best.ev)}</dd></div>
                <div><dt>gap to the pass played</dt><dd className="num strong">{xg(moment.delta_ev_plausible)}</dd></div>
              </dl>
              <p className="read-note">{KIND_NOTE[moment.kind]}</p>
            </div>
          )}
          <div className="read-col detail">
            <h4>{selected ? (selected.actual ? "Selected: the pass played" : "Selected option") : "Option detail"}</h4>
            {selected ? (
              <dl>
                <div><dt>chance it arrives</dt><dd className="num">{pct(selected.p)}</dd></div>
                <div><dt>value if it arrives</dt><dd className="num">{xg(selected.v_success)}</dd></div>
                <div><dt>value if it's lost</dt><dd className="num">{xg(selected.v_fail)}</dd></div>
                <div><dt>expected value</dt><dd className="num strong">{xg(selected.ev)}</dd></div>
                <div><dt>how often pros pick it here</dt><dd className="num">{pct(selected.policy_p)}</dd></div>
              </dl>
            ) : (
              <p className="read-note">
                Each ring is a visible teammate. Expected value = chance it arrives × value if it does + chance it's lost × cost if it isn't,
                in expected goals for the rest of the possession.
                {bestAll.idx !== best.idx && (
                  <> Ignoring what professionals usually choose, the model's top option would be a riskier ball {pct(bestAll.p)} likely to arrive.</>
                )}
              </p>
            )}
          </div>
        </section>
      </article>
    </div>
  );
}
