import { describe, expect, it } from "vitest";
import { arrowPath, pct, shortName, xg } from "./format";
import { pitchViewBox } from "../components/Pitch";
import { exportFileName, loadSession, newSession, nextIndex, recordAnswer, saveSession } from "./evalStore";

describe("format", () => {
  it("formats signed xG and percentages", () => {
    expect(xg(0.0951)).toBe("+0.095 xG");
    expect(xg(-0.02)).toBe("-0.020 xG");
    expect(xg(-0.0001)).toBe("+0.000 xG"); // no "-0.000"
    expect(pct(0.364)).toBe("36%");
  });

  it("keeps full names and falls back for unknown recipients", () => {
    expect(shortName("Randal Kolo Muani")).toBe("Randal Kolo Muani");
    expect(shortName(null)).toBe("a teammate");
    expect(shortName("  ")).toBe("a teammate");
  });

  it("draws an inset curve between two points", () => {
    const d = arrowPath([0, 0], [20, 0], 0.1, 2);
    expect(d.startsWith("M2,0 Q10,")).toBe(true);
    expect(d.endsWith(" 18,0")).toBe(true);
  });
});

describe("blind test session", () => {
  const memory = () => {
    const m = new Map<string, string>();
    return { getItem: (k: string) => m.get(k) ?? null, setItem: (k: string, v: string) => void m.set(k, v) };
  };

  it("records answers immutably and finds the next unanswered item", () => {
    const s0 = newSession(2026, new Date("2026-09-17T10:00:00Z"));
    const s1 = recordAnswer(s0, "b", "A", 1234.6);
    expect(s0.answers).toEqual({});
    expect(s1.answers.b).toEqual({ choice: "A", ms: 1235 });
    expect(nextIndex(s1, ["a", "b", "c"])).toBe(0);
    expect(nextIndex(recordAnswer(s1, "a", "B", 1), ["a", "b"])).toBe(2);
  });

  it("round-trips through storage and rejects a different eval set", () => {
    const store = memory();
    const s = recordAnswer(newSession(2026), "x", "unsure", 10);
    saveSession(s, store);
    expect(loadSession(2026, store)?.answers.x.choice).toBe("unsure");
    expect(loadSession(1, store)).toBeNull();
  });

  it("builds a safe export file name", () => {
    const s = { ...newSession(2026, new Date("2026-09-17T10:00:00Z")), rater: { initials: "d.s!", experience: "casual fan" as const } };
    expect(exportFileName(s)).toBe("blind-test-ds-2026-09-17.json");
  });
});

describe("pitch crop", () => {
  it("crops to the half holding every player and arrow, only when asked", () => {
    expect(pitchViewBox([60, 90, 118], true)).toBe("48 -4 76 88");
    expect(pitchViewBox([2, 30, 66], true)).toBe("-4 -4 76 88");
    expect(pitchViewBox([10, 110], true)).toBe("-4 -4 128 88");
    expect(pitchViewBox([60, 90], false)).toBe("-4 -4 128 88");
  });
});
