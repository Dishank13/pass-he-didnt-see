// Shapes of the JSON written by scripts/export_demo.py.
// Coordinates are StatsBomb yards: 120 x 80, origin top-left, the passing team attacks towards x = 120.

export type XY = [number, number];

export interface FramePlayer {
  idx: number;
  x: number;
  y: number;
  teammate: boolean; // same team as the passer
  actor: boolean; // the passer
  keeper: boolean;
}

export interface PassOption {
  idx: number; // player_idx of the teammate this option passes to
  x: number;
  y: number;
  p: number; // P(pass completes)
  v_success: number; // possession value if completed (xG difference)
  v_fail: number; // value if it fails (negative = costly)
  ev: number; // p * v_success + (1 - p) * v_fail
  policy_p: number; // how often a typical professional picks this option here
  plausible: boolean; // policy_p >= 0.10, or it's the pass actually played
  actual: boolean;
}

export type MomentKind = "goal assist" | "the pass he didn't see" | "best option, taken under risk";

export interface Moment {
  id: string;
  kind: MomentKind;
  competition: string;
  stage: string;
  match: string;
  date: string;
  minute: number;
  team: string;
  passer: string;
  recipient: string | null;
  outcome: "Complete" | "Incomplete" | "Out";
  goal_assist: boolean;
  passer_xy: XY;
  actual_idx: number;
  best_plausible_idx: number;
  best_all_idx: number;
  ev_actual: number;
  ev_best_plausible: number;
  delta_ev_plausible: number;
  best_plausible_zone: string;
  actual_zone: string;
  options: PassOption[];
  players: FramePlayer[];
  visible_area: XY[];
}

export interface EvalItem {
  id: string;
  players: FramePlayer[];
  visible_area: XY[];
  passer_xy: XY;
  option_a: XY;
  option_b: XY;
}

export type CI = [number, number, number]; // point, low, high

export interface Stats {
  m0: { matches: number; freeze_frames: number; corners_labelled: number; passes_linked: number };
  m1: { receiver_top3_gnn: CI; receiver_top3_lgbm: CI; receiver_top3_uniform: CI; team_auc_gnn: number; team_auc_lgbm: number };
  m2: {
    completion_auc: CI;
    completion_ece: number;
    physics_auc: CI;
    value_r2: CI;
    selection_top3: number;
    split_half_choice: number;
    split_half_regret: number;
  };
  m3: { lgbm_full: CI; lgbm_360like: CI; total_gap: CI; transfer_auc: CI };
}
