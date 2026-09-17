"""Pitch coordinate conventions.

StatsBomb uses a 120 x 80 yard pitch with its origin at the TOP-LEFT corner:
x runs along the length (0 -> 120) and y runs across the width (0 -> 80), pointing
*down* when you draw the pitch the conventional way.

Important property of StatsBomb event data: every event (and its 360 freeze frame)
is already expressed from the perspective of the team performing the event, which
attacks towards x = 120. So in practice there's no need to "normalise attacking
direction" for events. It only matters when you mix both teams into one frame,
e.g. a defensive action's frame versus the preceding pass.

Symmetry used for augmentation
------------------------------
Football is (approximately) invariant to reflecting the pitch across its long axis
(left wing <-> right wing): y -> 80 - y. Reflecting across the halfway line
(x -> 120 - x) is NOT a valid augmentation, because it reverses which goal is attacked.
TacticAI used both reflections because corners were canonicalised first. We only
rely on the y-reflection.
"""

from __future__ import annotations

import numpy as np

PITCH_LENGTH = 120.0
PITCH_WIDTH = 80.0
GOAL_CENTER = np.array([PITCH_LENGTH, PITCH_WIDTH / 2])


def flip_y(xy: np.ndarray) -> np.ndarray:
    """Reflect points across the pitch's long axis (y -> 80 - y).

    Accepts any array whose last dimension is 2 (a single point, an (N, 2) set of
    players, or a polygon ring). Returns a new array. The input is not modified.
    """
    out = np.array(xy, dtype=float, copy=True)
    out[..., 1] = PITCH_WIDTH - out[..., 1]
    return out


def flip_x(xy: np.ndarray) -> np.ndarray:
    """Reflect across the halfway line (x -> 120 - x).

    Only for switching between team perspectives (the opponent's view of the
    same frame), never as augmentation.
    """
    out = np.array(xy, dtype=float, copy=True)
    out[..., 0] = PITCH_LENGTH - out[..., 0]
    return out


def to_opponent_perspective(xy: np.ndarray) -> np.ndarray:
    """Express locations from the other team's point of view (a 180-degree rotation)."""
    return flip_y(flip_x(xy))


def distance_to_goal(xy: np.ndarray) -> np.ndarray:
    """Euclidean distance (yards) from each point to the centre of the attacked goal."""
    xy = np.asarray(xy, dtype=float)
    return np.linalg.norm(xy - GOAL_CENTER, axis=-1)


def angle_to_goal(xy: np.ndarray) -> np.ndarray:
    """Visible angle (radians) of the goal mouth from each point. A standard xG feature.

    Posts are at y = 36 and y = 44 (8-yard goal centred on y = 40).
    """
    xy = np.asarray(xy, dtype=float)
    p1 = np.array([PITCH_LENGTH, 36.0]) - xy
    p2 = np.array([PITCH_LENGTH, 44.0]) - xy
    cos = (p1 * p2).sum(-1) / (np.linalg.norm(p1, axis=-1) * np.linalg.norm(p2, axis=-1) + 1e-9)
    return np.arccos(np.clip(cos, -1.0, 1.0))
