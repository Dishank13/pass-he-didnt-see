import { useEffect, useId, useState } from "react";
import { arrowPath } from "../lib/format";
import type { FramePlayer, PassOption, XY } from "../lib/types";

export interface PitchArrow {
  from: XY;
  to: XY;
  tone: "played" | "suggested" | "neutral";
  label?: string; // e.g. "A" / "B" in the blind test
}

interface PitchProps {
  players: FramePlayer[];
  visibleArea: XY[];
  arrows: PitchArrow[];
  options?: PassOption[];
  selectedIdx?: number | null;
  onSelectOption?: (idx: number | null) => void;
  animationKey?: string; // change to replay the arrow-drawing animation
  title: string;
}

const PAD = 4;

/** True on narrow screens, where a full landscape pitch is too small to read. */
function useNarrow(query = "(max-width: 900px)") {
  const [narrow, setNarrow] = useState(() => typeof window !== "undefined" && window.matchMedia(query).matches);
  useEffect(() => {
    const mq = window.matchMedia(query);
    const on = () => setNarrow(mq.matches);
    mq.addEventListener("change", on);
    return () => mq.removeEventListener("change", on);
  }, [query]);
  return narrow;
}

/** viewBox for the full pitch, or for the half holding all the action (portrait, for phones). */
export function pitchViewBox(xs: number[], cropToHalf: boolean): string {
  const full = `${-PAD} ${-PAD} ${120 + 2 * PAD} ${80 + 2 * PAD}`;
  if (!cropToHalf || xs.length === 0) return full;
  const lo = Math.min(...xs);
  const hi = Math.max(...xs);
  if (lo >= 52) return `${52 - PAD} ${-PAD} ${68 + 2 * PAD} ${80 + 2 * PAD}`;
  if (hi <= 68) return `${-PAD} ${-PAD} ${68 + 2 * PAD} ${80 + 2 * PAD}`;
  return full;
}

/**
 * The pitch in StatsBomb coordinates (120 x 80 yards, y pointing down).
 * Signature: everything outside the broadcast camera's footprint is left in shadow,
 * because the data behind every number here only saw the lit area.
 */
export function Pitch({ players, visibleArea, arrows, options, selectedIdx, onSelectOption, animationKey, title }: PitchProps) {
  const uid = useId().replace(/:/g, "");
  const clipId = `view-${uid}`;
  const hatchId = `hatch-${uid}`;
  const hasView = visibleArea.length >= 3;
  const viewPoints = visibleArea.map(([x, y]) => `${x},${y}`).join(" ");
  const narrow = useNarrow();
  const actionXs = [...players.map((p) => p.x), ...arrows.flatMap((a) => [a.from[0], a.to[0]])];
  const maxEv = options?.length ? Math.max(...options.map((o) => o.ev)) : 0;
  const minEv = options?.length ? Math.min(...options.map((o) => o.ev)) : 0;

  return (
    <svg
      className="pitch"
      viewBox={pitchViewBox(actionXs, narrow)}
      role="img"
      aria-label={title}
      onClick={() => onSelectOption?.(null)}
    >
      <defs>
        <clipPath id={clipId}>{hasView ? <polygon points={viewPoints} /> : <rect x={0} y={0} width={120} height={80} />}</clipPath>
        <pattern id={hatchId} width="2.4" height="2.4" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
          <line x1="0" y1="0" x2="0" y2="2.4" className="pitch-hatch" />
        </pattern>
      </defs>

      {/* shadow pitch: what the camera did not show */}
      <rect x={-PAD} y={-PAD} width={120 + 2 * PAD} height={80 + 2 * PAD} className="pitch-surround" />
      <rect x={0} y={0} width={120} height={80} className="turf-shadow" />
      <rect x={0} y={0} width={120} height={80} fill={`url(#${hatchId})`} />
      {/* lit pitch: the camera footprint */}
      <g clipPath={`url(#${clipId})`}>
        <rect x={0} y={0} width={120} height={80} className="turf-lit" />
        {[...Array(12)].map((_, i) => (i % 2 === 0 ? <rect key={i} x={i * 10} y={0} width={10} height={80} className="turf-stripe" /> : null))}
      </g>
      {hasView && <polygon points={viewPoints} className="view-edge" />}

      <PitchLines />

      {/* option rings: bigger = higher expected value; dashed = rarely chosen by professionals here */}
      {options?.map((o) => {
        const t = maxEv > minEv ? (o.ev - minEv) / (maxEv - minEv) : 0.5;
        const r = 1.9 + 1.8 * t;
        const selected = selectedIdx === o.idx;
        return (
          <g
            key={`opt-${o.idx}`}
            className={`option ${o.plausible ? "plausible" : "implausible"} ${selected ? "selected" : ""}`}
            tabIndex={0}
            role="button"
            aria-label={`Option: ${Math.round(o.p * 100)}% to complete, expected value ${o.ev.toFixed(3)} xG${o.actual ? ", the pass played" : ""}`}
            aria-pressed={selected}
            onClick={(e) => {
              e.stopPropagation();
              onSelectOption?.(selected ? null : o.idx);
            }}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                onSelectOption?.(selected ? null : o.idx);
              }
            }}
          >
            <circle cx={o.x} cy={o.y} r={r + 1.2} className="option-hit" />
            <circle cx={o.x} cy={o.y} r={r} className="option-ring" />
          </g>
        );
      })}

      {players.map((p) => {
        const cls = `player ${p.teammate ? "attack" : "defence"} ${p.actor ? "passer" : ""}`;
        return p.keeper ? (
          <rect key={`pl-${p.idx}`} x={p.x - 1.25} y={p.y - 1.25} width={2.5} height={2.5} rx={0.5} className={cls} />
        ) : (
          <circle key={`pl-${p.idx}`} cx={p.x} cy={p.y} r={1.3} className={cls} />
        );
      })}

      <g key={animationKey} className="arrows">
        {arrows.map((a, i) => {
          const d = arrowPath(a.from, a.to, a.tone === "played" ? 0 : 0.14);
          return (
            <g key={i} className={`arrow ${a.tone}`}>
              <path d={d} className="arrow-casing" />
              <path d={d} className="arrow-stroke" pathLength={1} />
              <circle cx={a.to[0]} cy={a.to[1]} r={0.9} className="arrow-end" />
              {a.label && (
                <text x={a.to[0]} y={a.to[1] - 3.2} className="arrow-label" textAnchor="middle">
                  {a.label}
                </text>
              )}
            </g>
          );
        })}
      </g>
    </svg>
  );
}

function PitchLines() {
  return (
    <g className="lines">
      <rect x={0} y={0} width={120} height={80} />
      <line x1={60} y1={0} x2={60} y2={80} />
      <circle cx={60} cy={40} r={10} />
      <circle cx={60} cy={40} r={0.4} className="spot" />
      {/* penalty areas, six-yard boxes, spots, goals */}
      <rect x={0} y={18} width={18} height={44} />
      <rect x={102} y={18} width={18} height={44} />
      <rect x={0} y={30} width={6} height={20} />
      <rect x={114} y={30} width={6} height={20} />
      <circle cx={12} cy={40} r={0.4} className="spot" />
      <circle cx={108} cy={40} r={0.4} className="spot" />
      <path d="M18,32 A10,10 0 0 1 18,48" />
      <path d="M102,32 A10,10 0 0 0 102,48" />
      <rect x={-2} y={36} width={2} height={8} className="goal" />
      <rect x={120} y={36} width={2} height={8} className="goal" />
    </g>
  );
}
