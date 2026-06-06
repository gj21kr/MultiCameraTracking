"""Visualization helpers — draw tracks/detections onto frames (no GUI)."""

from typing import Dict, List, Optional

import cv2
import numpy as np


def color_for_id(idx: int) -> tuple:
    """Deterministic distinct BGR color for an integer id."""
    rng = np.random.RandomState(int(idx) * 977 + 13)
    return tuple(int(v) for v in rng.randint(40, 230, 3))


def draw_tracks(
    image: np.ndarray,
    tracks: List,
    use_global_id: bool = True,
    fps: Optional[float] = None,
    cam_id: Optional[int] = None,
) -> np.ndarray:
    """Return a copy of ``image`` with track boxes + IDs drawn.

    Color follows the global (cross-camera) id when available so the same
    person shares a color across views; otherwise the per-camera track id.
    """
    out = image.copy()
    for track in tracks:
        x1, y1, x2, y2 = track.bbox.astype(int)
        gid = getattr(track, "global_id", None)
        color_id = gid if (use_global_id and gid is not None) else track.track_id
        color = color_for_id(color_id)

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        label = f"ID{track.track_id}"
        if use_global_id and gid is not None:
            label = f"G{gid}|{track.track_id}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            out, label, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA,
        )

    # HUD
    hud = []
    if cam_id is not None:
        hud.append(f"Cam {cam_id}")
    hud.append(f"tracks: {len(tracks)}")
    if fps is not None:
        hud.append(f"{fps:.1f} FPS")
    text = "  ".join(hud)
    cv2.rectangle(out, (0, 0), (8 + 9 * len(text), 26), (0, 0, 0), -1)
    cv2.putText(
        out, text, (5, 18),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA,
    )
    return out
