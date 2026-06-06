"""GT-based BEV and pseudo-3D trajectory visualisation for WILDTRACK.

Reads ``annotations_positions/*.json`` directly — no detector, no tracker,
no calibration required.  Ground-truth personIDs and positionIDs are the
single source of truth (ADR-006, Domain Knowledge First).

Outputs
-------
outputs/traj_bev.png  : Bird's-eye-view (X, Y) in metres, one colour per person.
outputs/traj_3d.png   : Pseudo-3D (X, Y, t) scatter + line. t is the frame
                        index (0, 1, 2, ...).  This is NOT true (X, Y, Z);
                        height is not reconstructed — see ADR-006.

Usage
-----
    python scripts/plot_trajectory.py \\
        --root data/Wildtrack_dataset \\
        [--person-ids 122,200,300] \\
        [--max-frames 100] \\
        [--output outputs]
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # headless — no GUI required
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# Allow running as ``python scripts/plot_trajectory.py`` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.wildtrack_grid import (
    position_id_to_world,
    ORIGIN_X,
    ORIGIN_Y,
    GRID_W,
    GRID_H,
    CELL_M,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Physical extents of the WILDTRACK scene in metres.
SCENE_X_MIN: float = ORIGIN_X                          # -3.0 m
SCENE_X_MAX: float = ORIGIN_X + CELL_M * GRID_W       #  9.0 m
SCENE_Y_MIN: float = ORIGIN_Y                          # -9.0 m
SCENE_Y_MAX: float = ORIGIN_Y + CELL_M * GRID_H       # 27.0 m


# ── Data loading ──────────────────────────────────────────────────────────────

def load_gt_trajectories(
    annotations_dir: Path,
    max_frames: Optional[int] = None,
) -> Dict[int, List[Tuple[int, float, float]]]:
    """Parse all annotation JSON files and build per-person trajectories.

    Args:
        annotations_dir: Directory containing ``00000000.json``, ``00000005.json``, ...
        max_frames: Load at most this many annotation files (None = all 400).

    Returns:
        Mapping ``{personID: [(frame_index, X_m, Y_m), ...]}``, time-sorted.
        ``frame_index`` is 0-based (0, 1, 2, ...) — one unit = 5 raw frames
        = 0.5 s at 2 fps annotation rate.
        A person is present only in frames where they appear.
    """
    json_files = sorted(annotations_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(
            f"No JSON files found in {annotations_dir}. "
            "Expected WILDTRACK annotations_positions/*.json."
        )

    if max_frames is not None:
        json_files = json_files[:max_frames]

    logger.info(
        "Loading %d annotation files from %s", len(json_files), annotations_dir
    )

    trajectories: Dict[int, List[Tuple[int, float, float]]] = {}

    for frame_idx, json_path in enumerate(json_files):
        with open(json_path, "r", encoding="utf-8") as fh:
            entries = json.load(fh)

        for entry in entries:
            person_id: int = int(entry["personID"])
            position_id: int = int(entry["positionID"])
            x_m, y_m = position_id_to_world(position_id)

            if person_id not in trajectories:
                trajectories[person_id] = []
            trajectories[person_id].append((frame_idx, x_m, y_m))

    # Ensure time-sorted (files already sorted, but make it explicit).
    for pid in trajectories:
        trajectories[pid].sort(key=lambda t: t[0])

    total_persons = len(trajectories)
    total_observations = sum(len(v) for v in trajectories.values())
    logger.info(
        "Loaded %d persons, %d total observations across %d frames.",
        total_persons,
        total_observations,
        len(json_files),
    )

    return trajectories


def select_person_ids(
    trajectories: Dict[int, List[Tuple[int, float, float]]],
    requested: Optional[List[int]],
    top_n: int = 3,
) -> List[int]:
    """Return person IDs to plot.

    If ``requested`` is given, validate each ID exists.
    Otherwise return the ``top_n`` persons with the most observations.
    """
    if requested:
        missing = [pid for pid in requested if pid not in trajectories]
        if missing:
            raise ValueError(
                f"Requested person IDs not found in dataset: {missing}. "
                f"Available IDs (first 20): {sorted(trajectories)[:20]}"
            )
        return requested

    # Auto-select: top-N by observation count.
    ranked = sorted(trajectories, key=lambda pid: len(trajectories[pid]), reverse=True)
    chosen = ranked[:top_n]
    logger.info(
        "Auto-selected top-%d persons by observation count: %s "
        "(observation counts: %s)",
        top_n,
        chosen,
        [len(trajectories[p]) for p in chosen],
    )
    return chosen


# ── Colour assignment ─────────────────────────────────────────────────────────

def _person_colours(person_ids: List[int]) -> Dict[int, tuple]:
    """Assign a distinct colour to each person from a qualitative colour map."""
    cmap = matplotlib.colormaps.get_cmap("tab10").resampled(max(len(person_ids), 1))
    return {pid: cmap(i) for i, pid in enumerate(person_ids)}


# ── Plot 1: BEV (X, Y) ───────────────────────────────────────────────────────

def plot_bev(
    trajectories: Dict[int, List[Tuple[int, float, float]]],
    person_ids: List[int],
    output_path: Path,
) -> None:
    """Bird's-eye-view trajectory plot on the ground plane.

    Axes: X east [m], Y north [m].  Scene boundary drawn as a rectangle.
    Each person: line + start marker (circle) + end marker (cross).
    """
    colours = _person_colours(person_ids)

    fig, ax = plt.subplots(figsize=(8, 10))

    # Scene boundary.
    rect = plt.Rectangle(
        (SCENE_X_MIN, SCENE_Y_MIN),
        SCENE_X_MAX - SCENE_X_MIN,
        SCENE_Y_MAX - SCENE_Y_MIN,
        linewidth=1.5,
        edgecolor="black",
        facecolor="whitesmoke",
        zorder=0,
    )
    ax.add_patch(rect)

    for pid in person_ids:
        traj = trajectories[pid]
        xs = [pt[1] for pt in traj]
        ys = [pt[2] for pt in traj]
        colour = colours[pid]

        ax.plot(xs, ys, color=colour, linewidth=1.4, alpha=0.85, zorder=2)
        ax.scatter(xs[0], ys[0], color=colour, marker="o", s=60, zorder=3,
                   label=f"Person {pid} (start)")
        ax.scatter(xs[-1], ys[-1], color=colour, marker="x", s=80, zorder=3,
                   linewidths=2)

    ax.set_xlim(SCENE_X_MIN - 0.5, SCENE_X_MAX + 0.5)
    ax.set_ylim(SCENE_Y_MIN - 0.5, SCENE_Y_MAX + 0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("X [m] (east)", fontsize=11)
    ax.set_ylabel("Y [m] (north)", fontsize=11)
    ax.set_title(
        "WILDTRACK GT trajectories — BEV (X, Y)\n"
        f"(positionID -> world, 2.5 cm/cell, {len(person_ids)} person(s))",
        fontsize=12,
    )
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    logger.info("BEV figure saved -> %s", output_path)


# ── Plot 2: Pseudo-3D (X, Y, t) ──────────────────────────────────────────────

def plot_pseudo3d(
    trajectories: Dict[int, List[Tuple[int, float, float]]],
    person_ids: List[int],
    output_path: Path,
) -> None:
    """Pseudo-3D trajectory plot: axes are X [m], Y [m], frame index t.

    This is NOT true (X, Y, Z): height is not reconstructed.
    The vertical axis is time (frame index), which gives an informative
    space-time view of pedestrian motion (ADR-006).

    Title and axis labels explicitly state "frame index (pseudo-3D)" to
    avoid claiming true 3-D reconstruction.
    """
    colours = _person_colours(person_ids)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    for pid in person_ids:
        traj = trajectories[pid]
        ts = np.array([pt[0] for pt in traj], dtype=float)
        xs = np.array([pt[1] for pt in traj], dtype=float)
        ys = np.array([pt[2] for pt in traj], dtype=float)
        colour = colours[pid]

        ax.plot(xs, ys, ts, color=colour, linewidth=1.4, alpha=0.85)
        ax.scatter(xs, ys, ts, color=colour, s=12, alpha=0.7,
                   label=f"Person {pid}")
        # Start and end markers.
        ax.scatter(xs[0], ys[0], ts[0], color=colour, s=80, marker="o",
                   depthshade=False)
        ax.scatter(xs[-1], ys[-1], ts[-1], color=colour, s=80, marker="x",
                   depthshade=False, linewidths=2)

    ax.set_xlabel("X [m] (east)", fontsize=10, labelpad=8)
    ax.set_ylabel("Y [m] (north)", fontsize=10, labelpad=8)
    ax.set_zlabel("t [frame index]", fontsize=10, labelpad=8)
    ax.set_title(
        "WILDTRACK GT trajectories — BEV (X, Y) + time axis (pseudo-3D)\n"
        "(vertical axis = frame index; NOT true X,Y,Z — height not reconstructed)",
        fontsize=11,
    )
    ax.legend(fontsize=9, loc="upper left")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    logger.info("Pseudo-3D figure saved -> %s", output_path)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualise WILDTRACK GT trajectories using positionID -> world "
            "conversion.  No detector or tracker used."
        )
    )
    parser.add_argument(
        "--root",
        default="data/Wildtrack_dataset",
        help="Path to the WILDTRACK dataset root (must contain annotations_positions/).",
    )
    parser.add_argument(
        "--person-ids",
        default=None,
        type=str,
        help=(
            "Comma-separated personIDs to plot, e.g. '122,200,300'.  "
            "If omitted, the 3 persons with the most observations are chosen."
        ),
    )
    parser.add_argument(
        "--max-frames",
        default=None,
        type=int,
        help="Maximum number of annotation files to load (default: all 400).",
    )
    parser.add_argument(
        "--output",
        default="outputs",
        help="Directory for output PNG files (default: outputs/).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    root = Path(args.root)
    annotations_dir = root / "annotations_positions"
    if not annotations_dir.is_dir():
        logger.error(
            "annotations_positions not found under %s. "
            "Expected: %s",
            root,
            annotations_dir,
        )
        sys.exit(1)

    # Parse requested person IDs.
    requested_ids: Optional[List[int]] = None
    if args.person_ids:
        try:
            requested_ids = [int(s.strip()) for s in args.person_ids.split(",")]
        except ValueError as exc:
            logger.error("--person-ids parse error: %s", exc)
            sys.exit(1)

    output_dir = Path(args.output)

    # ── A1.2: Load GT time-series ─────────────────────────────────────────────
    trajectories = load_gt_trajectories(
        annotations_dir=annotations_dir,
        max_frames=args.max_frames,
    )

    # ── Select persons to plot ────────────────────────────────────────────────
    person_ids = select_person_ids(trajectories, requested_ids, top_n=3)

    # ── A1.3: Generate figures ────────────────────────────────────────────────
    plot_bev(
        trajectories=trajectories,
        person_ids=person_ids,
        output_path=output_dir / "traj_bev.png",
    )
    plot_pseudo3d(
        trajectories=trajectories,
        person_ids=person_ids,
        output_path=output_dir / "traj_3d.png",
    )

    logger.info(
        "Done. Outputs: %s/traj_bev.png, %s/traj_3d.png",
        output_dir,
        output_dir,
    )


if __name__ == "__main__":
    main()
