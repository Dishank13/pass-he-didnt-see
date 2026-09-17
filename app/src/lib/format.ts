import type { CI } from "./types";

/** Signed expected-goal value, e.g. "+0.095 xG". */
export function xg(v: number, digits = 3): string {
  const rounded = Number(v.toFixed(digits)) || 0; // `|| 0` turns -0 into 0, so we never print "-0.000"
  return `${rounded >= 0 ? "+" : ""}${rounded.toFixed(digits)} xG`;
}

/** Probability as a whole percentage, e.g. "36%". */
export function pct(p: number): string {
  return `${Math.round(p * 100)}%`;
}

/** "0.912 (0.903–0.922)" */
export function ci(c: CI, digits = 3): string {
  return `${c[0].toFixed(digits)} (${c[1].toFixed(digits)}–${c[2].toFixed(digits)})`;
}

/** Display name for a player. Names arrive as familiar nicknames from the export ("Rodri", "Ángel Di María"). */
export function shortName(name: string | null): string {
  return name?.trim() || "a teammate";
}

/**
 * A telestrator-style arrow path from a to b: a gentle quadratic curve, shortened at both ends
 * so it starts outside the passer's marker and stops before the receiver's ring.
 */
export function arrowPath(a: [number, number], b: [number, number], bend = 0.12, inset = 1.8): string {
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  const len = Math.hypot(dx, dy);
  if (len < 2 * inset + 0.1) return `M${a[0]},${a[1]} L${b[0]},${b[1]}`;
  const ux = dx / len;
  const uy = dy / len;
  const s: [number, number] = [a[0] + ux * inset, a[1] + uy * inset];
  const e: [number, number] = [b[0] - ux * inset, b[1] - uy * inset];
  // control point offset perpendicular to the pass direction
  const c: [number, number] = [(s[0] + e[0]) / 2 - uy * len * bend, (s[1] + e[1]) / 2 + ux * len * bend];
  const r = (n: number) => Math.round(n * 100) / 100;
  return `M${r(s[0])},${r(s[1])} Q${r(c[0])},${r(c[1])} ${r(e[0])},${r(e[1])}`;
}
