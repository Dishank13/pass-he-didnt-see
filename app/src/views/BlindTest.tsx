import { useEffect, useRef, useState } from "react";
import { Pitch } from "../components/Pitch";
import {
  clearSession,
  exportFileName,
  loadSession,
  newSession,
  nextIndex,
  recordAnswer,
  saveSession,
  type Choice,
  type EvalSession,
  type Rater,
} from "../lib/evalStore";
import type { EvalItem } from "../lib/types";

const EXPERIENCE: Rater["experience"][] = ["played or coached", "regular watcher", "casual fan"];

export function BlindTest({ items, seed }: { items: EvalItem[]; seed: number }) {
  const [session, setSession] = useState<EvalSession>(() => loadSession(seed) ?? newSession(seed));
  const ids = items.map((i) => i.id);
  const index = nextIndex(session, ids);
  const shownAt = useRef(performance.now());

  useEffect(() => saveSession(session), [session]);
  useEffect(() => {
    shownAt.current = performance.now();
  }, [index]);

  const answer = (choice: Choice) => {
    const item = items[index];
    if (!item) return;
    setSession((s) => recordAnswer(s, item.id, choice, performance.now() - shownAt.current));
  };

  useEffect(() => {
    if (!session.rater || index >= items.length) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.metaKey || e.ctrlKey || e.altKey) return;
      const k = e.key.toLowerCase();
      if (k === "a") answer("A");
      else if (k === "b") answer("B");
      else if (k === "u") answer("unsure");
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  if (!session.rater) return <RaterForm onStart={(rater) => setSession((s) => ({ ...s, rater }))} total={items.length} />;

  if (index >= items.length) {
    const download = () => {
      const blob = new Blob([JSON.stringify(session, null, 1)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = exportFileName(session);
      a.click();
      URL.revokeObjectURL(url);
    };
    return (
      <section className="blind done">
        <h2>All {items.length} moments answered</h2>
        <p>Download your answers and send the file to the person who shared this test. Nothing has been uploaded anywhere.</p>
        <div className="blind-actions">
          <button className="primary" onClick={download}>Download answers</button>
          <button
            className="quiet"
            onClick={() => {
              clearSession();
              setSession(newSession(seed));
            }}
          >
            Start a new test
          </button>
        </div>
      </section>
    );
  }

  const item = items[index];
  return (
    <section className="blind">
      <header className="blind-head">
        <p className="blind-q">You have the ball (gold ring). Which pass do you play?</p>
        <div className="progress" role="progressbar" aria-valuemin={0} aria-valuemax={items.length} aria-valuenow={index}>
          <span style={{ width: `${(index / items.length) * 100}%` }} />
          <em>
            {index + 1} of {items.length}
          </em>
        </div>
      </header>
      <div className="pitch-wrap">
        <Pitch
          players={item.players}
          visibleArea={item.visible_area}
          arrows={[
            { from: item.passer_xy, to: item.option_a, tone: "neutral", label: "A" },
            { from: item.passer_xy, to: item.option_b, tone: "neutral", label: "B" },
          ]}
          animationKey={item.id}
          title={`Blind test moment ${index + 1}: two possible passes, A and B`}
        />
      </div>
      <div className="blind-actions">
        <button className="primary" onClick={() => answer("A")}>
          Play pass A <kbd>A</kbd>
        </button>
        <button className="primary" onClick={() => answer("B")}>
          Play pass B <kbd>B</kbd>
        </button>
        <button className="quiet" onClick={() => answer("unsure")}>
          Can't choose <kbd>U</kbd>
        </button>
      </div>
      <p className="fine">Red: your team, attacking to the right. Blue: opponents. Shaded area: outside the camera view, so players there are missing.</p>
    </section>
  );
}

function RaterForm({ onStart, total }: { onStart: (r: Rater) => void; total: number }) {
  const [initials, setInitials] = useState("");
  const [experience, setExperience] = useState<Rater["experience"]>("regular watcher");
  return (
    <section className="blind intro">
      <h2>Blind test: pick the pass</h2>
      <p>
        You'll see {total} real moments from professional matches, frozen at the instant of a pass. Two passes are drawn. One was
        actually played, the other is the model's suggestion, in random order. Choose the pass you'd play. It takes about 10 minutes.
      </p>
      <p className="fine">Your answers stay in this browser until you download them at the end.</p>
      <form
        onSubmit={(e) => {
          e.preventDefault();
          onStart({ initials: initials.trim(), experience });
        }}
      >
        <label>
          Initials (optional)
          <input value={initials} maxLength={8} onChange={(e) => setInitials(e.target.value)} autoComplete="off" />
        </label>
        <fieldset>
          <legend>Your football background</legend>
          {EXPERIENCE.map((x) => (
            <label key={x} className="radio">
              <input type="radio" name="exp" checked={experience === x} onChange={() => setExperience(x)} />
              {x}
            </label>
          ))}
        </fieldset>
        <button className="primary" type="submit">
          Start the test
        </button>
      </form>
    </section>
  );
}
