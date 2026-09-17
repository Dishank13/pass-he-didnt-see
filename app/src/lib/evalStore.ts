// Blind-test answers live only in this browser (localStorage) until the rater downloads them.
// Nothing is sent anywhere: raters email or message the exported file.

export type Choice = "A" | "B" | "unsure";

export interface Rater {
  initials: string;
  experience: "played or coached" | "regular watcher" | "casual fan";
}

export interface EvalSession {
  version: 1;
  seed: number;
  rater: Rater | null;
  startedAt: string;
  answers: Record<string, { choice: Choice; ms: number }>;
}

const KEY = "phds-blind-test-v1";

export function newSession(seed: number, now = new Date()): EvalSession {
  return { version: 1, seed, rater: null, startedAt: now.toISOString(), answers: {} };
}

/** Load a saved session. Returns null if storage is unavailable, empty, or from a different eval set. */
export function loadSession(seed: number, storage: Pick<Storage, "getItem"> | undefined = safeStorage()): EvalSession | null {
  try {
    const raw = storage?.getItem(KEY);
    if (!raw) return null;
    const s = JSON.parse(raw) as EvalSession;
    return s.version === 1 && s.seed === seed ? s : null;
  } catch {
    return null;
  }
}

export function saveSession(s: EvalSession, storage: Pick<Storage, "setItem"> | undefined = safeStorage()): void {
  try {
    storage?.setItem(KEY, JSON.stringify(s));
  } catch {
    // Private mode or blocked storage: the session still works in memory for this visit.
  }
}

export function clearSession(storage: Pick<Storage, "removeItem"> | undefined = safeStorage()): void {
  try {
    storage?.removeItem(KEY);
  } catch {
    /* ignore */
  }
}

export function recordAnswer(s: EvalSession, id: string, choice: Choice, ms: number): EvalSession {
  return { ...s, answers: { ...s.answers, [id]: { choice, ms: Math.round(ms) } } };
}

/** Index of the first unanswered item, or items.length when finished. */
export function nextIndex(s: EvalSession, ids: string[]): number {
  const i = ids.findIndex((id) => !(id in s.answers));
  return i === -1 ? ids.length : i;
}

export function exportFileName(s: EvalSession): string {
  const who = (s.rater?.initials || "anon").replace(/[^A-Za-z0-9]/g, "").slice(0, 8) || "anon";
  return `blind-test-${who}-${s.startedAt.slice(0, 10)}.json`;
}

function safeStorage(): Storage | undefined {
  try {
    return typeof window !== "undefined" ? window.localStorage : undefined;
  } catch {
    return undefined;
  }
}
