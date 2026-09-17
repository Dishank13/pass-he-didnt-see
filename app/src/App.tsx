import { useEffect, useState } from "react";
import type { EvalItem, Moment, Stats } from "./lib/types";
import { BlindTest } from "./views/BlindTest";
import { Explorer } from "./views/Explorer";
import { Method } from "./views/Method";

type Route = "explore" | "blind-test" | "method";
const ROUTES: { id: Route; label: string }[] = [
  { id: "explore", label: "Explore finals" },
  { id: "blind-test", label: "Blind test" },
  { id: "method", label: "How it works" },
];

const routeFromHash = (): Route => {
  const h = window.location.hash.replace("#", "") as Route;
  return ROUTES.some((r) => r.id === h) ? h : "explore";
};

interface Data {
  moments: Moment[];
  evalItems: EvalItem[];
  evalSeed: number;
  stats: Stats | null;
}

async function loadData(): Promise<Data> {
  const base = import.meta.env.BASE_URL;
  const get = async (name: string) => {
    const r = await fetch(`${base}data/${name}`);
    if (!r.ok) throw new Error(`${name}: HTTP ${r.status}`);
    return r.json();
  };
  const [m, e, s] = await Promise.all([get("moments.json"), get("eval_set.json"), get("stats.json").catch(() => null)]);
  return { moments: m.moments, evalItems: e.items, evalSeed: e.seed, stats: s };
}

export default function App() {
  const [route, setRoute] = useState<Route>(routeFromHash);
  const [data, setData] = useState<Data | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const onHash = () => setRoute(routeFromHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  useEffect(() => {
    loadData().then(setData, (e: Error) => setError(e.message));
  }, []);

  return (
    <div className="app">
      <header className="masthead">
        <a className="wordmark" href="#explore">
          The pass <span>he didn't see</span>
        </a>
        <nav className="tabs" aria-label="Sections">
          {ROUTES.map((r) => (
            <a key={r.id} href={`#${r.id}`} aria-current={route === r.id ? "page" : undefined}>
              {r.label}
            </a>
          ))}
        </nav>
      </header>

      <main>
        {error && (
          <p className="empty">
            Couldn't load the demo data ({error}). Run <code>python scripts/export_demo.py</code>, then reload.
          </p>
        )}
        {!error && !data && <p className="loading">Loading moments…</p>}
        {data && route === "explore" && <Explorer moments={data.moments} />}
        {data && route === "blind-test" && <BlindTest items={data.evalItems} seed={data.evalSeed} />}
        {data && route === "method" && <Method stats={data.stats} />}
      </main>

      <footer className="foot">
        Data: StatsBomb Open Data · SkillCorner Open Data. A portfolio project, not affiliated with either.
      </footer>
    </div>
  );
}
