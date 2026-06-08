"""GT vs predicted trajectory comparison (FEAT-11, A2.3).

Loads GT trajectories from WILDTRACK ``annotations_positions/*.json`` using
``positionID -> world`` (metres, same as FEAT-10 / plot_trajectory.py) and
predicted trajectories from ``outputs/pred_trajectory.json`` (produced by the
make_demo.py trajectory hook, A2.2).

Matching strategy
-----------------
Global assignment (Hungarian / linear_sum_assignment) over the full sequence,
not per-frame.  Cost matrix: mean Euclidean distance between GT and predicted
trajectories after temporal alignment at their common frame indices.

Temporal alignment
------------------
GT annotations exist every 5 raw frames (~0.5 s, 2 fps).  The prediction hook
records one entry per processed frame.  Alignment: for each GT frame index,
find the nearest predicted frame index and use that predicted position.

Coordinate system
-----------------
GT (positionID -> world) = metres.  foot_to_ground output = centimetres.
make_demo.py hook divides by 100 before saving, so pred_trajectory.json is
already in metres.  Both datasets share the same WILDTRACK world frame.

Outputs
-------
outputs/traj_gt_vs_pred.png  BEV overlay: GT dashed, pred solid, same colour
                              per matched pair, unmatched grey.
outputs/compare_metrics.json mean/median distance error (m) + match table.
stdout                        summary line with mean & median error.

Fallback (pseudo-prediction)
-----------------------------
If ``pred_trajectory.json`` is absent but ``--pseudo`` is set, the script
projects GT bbox foot-points via calibration to produce a "pseudo-prediction"
that validates the projection pipeline.  The figure subtitle identifies which
mode was used so there is no ambiguity (ADR-006 honesty requirement).

Top-K visualisation (--top-k)
------------------------------
Plotting all matched pairs (default 57) creates an unreadable spaghetti figure.
``--top-k N`` restricts the BEV overlay to the N pairs with the smallest mean
distance error (best-aligned), keeping the figure clean and legible.
Metrics (mean/median error) are still computed over **all** matched pairs so the
headline numbers are not distorted.  The figure title explicitly states how many
pairs are shown vs. the total.  Pass ``--top-k 0`` to plot all pairs (original
spaghetti mode, useful for debugging).

Usage
-----
    python scripts/compare_trajectory.py \\
        --root data/Wildtrack_dataset \\
        [--pred outputs/pred_trajectory.json] \\
        [--max-frames 100] \\
        [--output outputs] \\
        [--pseudo] \\
        [--top-k 3]
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment

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

SCENE_X_MIN = ORIGIN_X
SCENE_X_MAX = ORIGIN_X + CELL_M * GRID_W
SCENE_Y_MIN = ORIGIN_Y
SCENE_Y_MAX = ORIGIN_Y + CELL_M * GRID_H


# ── GT loading (reuses FEAT-10 logic) ─────────────────────────────────────────

def load_gt_trajectories(
    annotations_dir: Path,
    max_frames: Optional[int],
) -> Dict[int, List[Tuple[int, float, float]]]:
    """Parse WILDTRACK annotation JSONs.

    Returns {personID: [(frame_idx, X_m, Y_m), ...]} time-sorted.
    frame_idx is 0-based annotation index (1 unit = 5 raw frames = 0.5 s).
    """
    json_files = sorted(annotations_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(
            f"No annotation JSONs in {annotations_dir}. "
            "Expected annotations_positions/*.json from WILDTRACK dataset."
        )
    if max_frames is not None:
        json_files = json_files[:max_frames]

    logger.info("Loading %d GT annotation files", len(json_files))

    traj: Dict[int, List[Tuple[int, float, float]]] = {}
    for frame_idx, jf in enumerate(json_files):
        with open(jf, "r", encoding="utf-8") as fh:
            entries = json.load(fh)
        for entry in entries:
            pid = int(entry["personID"])
            x_m, y_m = position_id_to_world(int(entry["positionID"]))
            traj.setdefault(pid, []).append((frame_idx, x_m, y_m))

    for pid in traj:
        traj[pid].sort(key=lambda t: t[0])

    logger.info(
        "GT: %d persons, %d total observations across %d frames",
        len(traj),
        sum(len(v) for v in traj.values()),
        len(json_files),
    )
    return traj


# ── Predicted trajectory loading ──────────────────────────────────────────────

def load_pred_trajectories(
    pred_path: Path,
) -> Dict[int, List[Tuple[int, float, float]]]:
    """Load pred_trajectory.json -> {global_id: [(t, X_m, Y_m), ...]}."""
    data = json.loads(pred_path.read_text(encoding="utf-8"))
    traj: Dict[int, List[Tuple[int, float, float]]] = {}
    for k, v in data.items():
        gid = int(k)
        traj[gid] = [(int(row[0]), float(row[1]), float(row[2])) for row in v]
        traj[gid].sort(key=lambda t: t[0])
    logger.info("Predicted: %d tracks from %s", len(traj), pred_path)
    return traj


# ── Pseudo-prediction fallback ────────────────────────────────────────────────

def build_pseudo_predictions(
    annotations_dir: Path,
    calib_root: str,
    cameras: List[int],
    max_frames: Optional[int],
) -> Tuple[Dict[int, List[Tuple[int, float, float]]], str]:
    """Project GT bbox foot-points via calibration as pseudo-predictions.

    This validates the projection pipeline independently of the tracker.
    Returns (traj_dict, mode_label).
    mode_label is included in the figure to signal pseudo mode.
    """
    from src.tracking.calibration import WildtrackCalibration

    calib = WildtrackCalibration.from_dir(calib_root)

    json_files = sorted(annotations_dir.glob("*.json"))
    if max_frames is not None:
        json_files = json_files[:max_frames]

    logger.info(
        "Building pseudo-predictions from GT bboxes (%d files, cameras %s)",
        len(json_files),
        cameras,
    )

    cam_set = set(cameras)
    traj: Dict[int, List[Tuple[int, float, float]]] = defaultdict(list)

    for frame_idx, jf in enumerate(json_files):
        with open(jf, "r", encoding="utf-8") as fh:
            entries = json.load(fh)
        for entry in entries:
            pid = int(entry["personID"])
            pts: List[np.ndarray] = []
            for v in entry.get("views", []):
                cam_id = v["viewNum"] + 1
                if cam_id not in cam_set:
                    continue
                if v["xmin"] < 0 or v["ymin"] < 0:
                    continue
                bbox = np.array(
                    [v["xmin"], v["ymin"], v["xmax"], v["ymax"]], dtype=float
                )
                g_cm = calib.foot_to_ground(cam_id, bbox)
                if g_cm is not None and np.all(np.isfinite(g_cm)):
                    pts.append(g_cm)
            if pts:
                med_cm = np.median(np.array(pts), axis=0)
                x_m = float(med_cm[0]) / 100.0
                y_m = float(med_cm[1]) / 100.0
                traj[pid].append((frame_idx, x_m, y_m))

    result = {}
    for pid, obs in traj.items():
        obs.sort(key=lambda t: t[0])
        result[pid] = obs

    logger.info("Pseudo-predictions: %d person-tracks", len(result))
    return result, "pseudo-prediction (GT bbox reprojection)"


# ── Temporal alignment ─────────────────────────────────────────────────────────

def _traj_dict(obs: List[Tuple[int, float, float]]) -> Dict[int, np.ndarray]:
    """Convert [(t, x, y), ...] to {t: array([x, y])}."""
    return {t: np.array([x, y]) for t, x, y in obs}


def align_trajectories(
    gt_obs: List[Tuple[int, float, float]],
    pred_obs: List[Tuple[int, float, float]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Align GT and predicted observations at common frame indices.

    GT frames are a subset of all frames (annotations every 5 raw frames).
    For each GT frame index, find the nearest predicted frame index.
    Returns two arrays of shape (N, 2) in metres: gt_pts, pred_pts.
    """
    gt_map = _traj_dict(gt_obs)
    pred_map = _traj_dict(pred_obs)
    pred_times = np.array(sorted(pred_map.keys()))

    if len(pred_times) == 0:
        return np.zeros((0, 2)), np.zeros((0, 2))

    gt_pts = []
    pred_pts = []
    for t_gt, gx, gy in sorted(gt_obs, key=lambda r: r[0]):
        # Nearest predicted frame.
        idx = int(np.argmin(np.abs(pred_times - t_gt)))
        t_pred = pred_times[idx]
        if abs(t_pred - t_gt) > 30:
            # Skip if more than 30 frames apart (unmatched time window).
            continue
        gt_pts.append([gx, gy])
        pred_pts.append(pred_map[t_pred])

    if not gt_pts:
        return np.zeros((0, 2)), np.zeros((0, 2))
    return np.array(gt_pts), np.array(pred_pts)


# ── Global Hungarian matching ─────────────────────────────────────────────────

def global_hungarian_match(
    gt_traj: Dict[int, List[Tuple[int, float, float]]],
    pred_traj: Dict[int, List[Tuple[int, float, float]]],
    max_cost_m: float = 10.0,
) -> Tuple[
    List[Tuple[int, int]],      # matched (gt_pid, pred_gid) pairs
    List[int],                  # unmatched GT pids
    List[int],                  # unmatched pred global_ids
    Dict[Tuple[int, int], float],  # mean_dist_m per pair
]:
    """One-shot global assignment: minimise mean trajectory distance.

    Cost matrix: mean Euclidean distance (m) between aligned GT and predicted
    trajectory points for every (person_id, global_id) pair.
    Pairs with no temporal overlap get cost = max_cost_m (effectively inf).
    """
    gt_ids = list(gt_traj.keys())
    pred_ids = list(pred_traj.keys())

    if not gt_ids or not pred_ids:
        return [], gt_ids, pred_ids, {}

    n_gt, n_pred = len(gt_ids), len(pred_ids)
    cost = np.full((n_gt, n_pred), max_cost_m, dtype=float)
    dist_cache: Dict[Tuple[int, int], float] = {}

    for i, gpid in enumerate(gt_ids):
        for j, ppid in enumerate(pred_ids):
            gt_pts, pred_pts = align_trajectories(gt_traj[gpid], pred_traj[ppid])
            if len(gt_pts) == 0:
                continue
            dists = np.linalg.norm(gt_pts - pred_pts, axis=1)
            mean_d = float(np.mean(dists))
            cost[i, j] = mean_d
            dist_cache[(gpid, ppid)] = mean_d

    ri, ci = linear_sum_assignment(cost)

    matched: List[Tuple[int, int]] = []
    mean_dist: Dict[Tuple[int, int], float] = {}
    matched_gt = set()
    matched_pred = set()

    for r, c in zip(ri, ci):
        gpid = gt_ids[r]
        ppid = pred_ids[c]
        d = cost[r, c]
        if d < max_cost_m:
            matched.append((gpid, ppid))
            mean_dist[(gpid, ppid)] = dist_cache.get((gpid, ppid), d)
            matched_gt.add(gpid)
            matched_pred.add(ppid)

    unmatched_gt = [g for g in gt_ids if g not in matched_gt]
    unmatched_pred = [p for p in pred_ids if p not in matched_pred]

    return matched, unmatched_gt, unmatched_pred, mean_dist


# ── Colour helpers ─────────────────────────────────────────────────────────────

def _colour_map(n: int) -> List:
    cmap = matplotlib.colormaps.get_cmap("tab10").resampled(max(n, 1))
    return [cmap(i) for i in range(n)]


# ── BEV overlay figure ────────────────────────────────────────────────────────

def _select_top_k_pairs(
    matched: List[Tuple[int, int]],
    mean_dist: Dict[Tuple[int, int], float],
    top_k: int,
) -> List[Tuple[int, int]]:
    """Return the top-K matched pairs with the lowest mean distance error.

    If top_k <= 0 or top_k >= len(matched), all pairs are returned (spaghetti mode).
    Pairs are sorted ascending by mean distance so the best-aligned are first.
    """
    if top_k <= 0 or top_k >= len(matched):
        return matched
    ranked = sorted(
        matched,
        key=lambda pair: mean_dist.get(pair, float("inf")),
    )
    return ranked[:top_k]


def plot_overlay(
    gt_traj: Dict[int, List[Tuple[int, float, float]]],
    pred_traj: Dict[int, List[Tuple[int, float, float]]],
    matched: List[Tuple[int, int]],
    unmatched_gt: List[int],
    unmatched_pred: List[int],
    mean_dist: Dict[Tuple[int, int], float],
    output_path: Path,
    mode_label: str,
    mean_err_m: float,
    median_err_m: float,
    top_k: int = 3,
) -> None:
    """BEV overlay: GT dashed, pred solid, matched pairs same colour.

    Only the top-K best-aligned pairs (lowest mean distance error) are drawn in
    colour.  The remaining matched pairs and all unmatched tracks are drawn in
    light grey so the scene boundary is still visible without the spaghetti.
    Metrics (mean/median) are computed over ALL matched pairs; the figure title
    makes the shown/total distinction explicit.

    Pass top_k=0 to draw all pairs in colour (original spaghetti debug mode).

    Unmatched GT: grey dashed.  Unmatched pred: grey solid.
    Figure subtitle declares the mode (real inference vs pseudo-prediction)
    per ADR-006 honesty requirement.
    """
    pairs_to_highlight = _select_top_k_pairs(matched, mean_dist, top_k)
    highlight_set = set(pairs_to_highlight)
    total_matched = len(matched)
    n_shown = len(pairs_to_highlight)

    fig, ax = plt.subplots(figsize=(8, 10))

    rect = plt.Rectangle(
        (SCENE_X_MIN, SCENE_Y_MIN),
        SCENE_X_MAX - SCENE_X_MIN,
        SCENE_Y_MAX - SCENE_Y_MIN,
        linewidth=1.5, edgecolor="black", facecolor="whitesmoke", zorder=0,
    )
    ax.add_patch(rect)

    grey_dim = (0.75, 0.75, 0.75, 0.30)

    # Draw non-highlighted matched pairs dimly first (background layer).
    for gt_pid, pred_gid in matched:
        if (gt_pid, pred_gid) in highlight_set:
            continue
        gt_obs = gt_traj[gt_pid]
        xs_gt = [pt[1] for pt in gt_obs]
        ys_gt = [pt[2] for pt in gt_obs]
        ax.plot(xs_gt, ys_gt, color=grey_dim, linestyle="--",
                linewidth=0.8, zorder=1)

        pred_obs = pred_traj[pred_gid]
        xs_pr = [pt[1] for pt in pred_obs]
        ys_pr = [pt[2] for pt in pred_obs]
        ax.plot(xs_pr, ys_pr, color=grey_dim, linestyle="-",
                linewidth=0.8, zorder=1)

    # Draw unmatched tracks dimly.
    grey_unmatched = (0.6, 0.6, 0.6, 0.25)
    for gt_pid in unmatched_gt:
        obs = gt_traj[gt_pid]
        xs = [pt[1] for pt in obs]
        ys = [pt[2] for pt in obs]
        ax.plot(xs, ys, color=grey_unmatched, linestyle="--",
                linewidth=0.7, zorder=1)

    for pred_gid in unmatched_pred:
        obs = pred_traj[pred_gid]
        xs = [pt[1] for pt in obs]
        ys = [pt[2] for pt in obs]
        ax.plot(xs, ys, color=grey_unmatched, linestyle="-",
                linewidth=0.7, zorder=1)

    # Draw highlighted pairs in colour (foreground layer).
    colours = _colour_map(n_shown)
    for idx, (gt_pid, pred_gid) in enumerate(pairs_to_highlight):
        colour = colours[idx]
        dist_label = f"{mean_dist.get((gt_pid, pred_gid), float('nan')):.2f} m"

        gt_obs = gt_traj[gt_pid]
        xs_gt = [pt[1] for pt in gt_obs]
        ys_gt = [pt[2] for pt in gt_obs]
        ax.plot(xs_gt, ys_gt, color=colour, linestyle="--", linewidth=1.8,
                alpha=0.9, zorder=3,
                label=f"GT person {gt_pid} (dashed)")
        # Start = circle, end = cross
        ax.scatter(xs_gt[0], ys_gt[0], color=colour, marker="o",
                   s=70, zorder=4)
        ax.scatter(xs_gt[-1], ys_gt[-1], color=colour, marker="x",
                   s=90, linewidths=2, zorder=4)

        pred_obs = pred_traj[pred_gid]
        xs_pr = [pt[1] for pt in pred_obs]
        ys_pr = [pt[2] for pt in pred_obs]
        ax.plot(xs_pr, ys_pr, color=colour, linestyle="-", linewidth=2.2,
                alpha=0.95, zorder=3,
                label=f"Pred id {pred_gid} (solid, err {dist_label})")
        ax.scatter(xs_pr[0], ys_pr[0], color=colour, marker="o",
                   s=70, zorder=4)
        ax.scatter(xs_pr[-1], ys_pr[-1], color=colour, marker="x",
                   s=90, linewidths=2, zorder=4)

    ax.set_xlim(SCENE_X_MIN - 0.5, SCENE_X_MAX + 0.5)
    ax.set_ylim(SCENE_Y_MIN - 0.5, SCENE_Y_MAX + 0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("X [m] (east)", fontsize=11)
    ax.set_ylabel("Y [m] (north)", fontsize=11)

    # Title explicitly states shown/total so metrics are not misread.
    if top_k > 0 and n_shown < total_matched:
        shown_label = (
            f"showing top-{n_shown} of {total_matched} matched pairs "
            "(metrics over all)"
        )
    else:
        shown_label = f"all {total_matched} matched pairs shown"

    ax.set_title(
        "WILDTRACK GT (dashed) vs Predicted (solid) — BEV overlay\n"
        f"Mode: {mode_label}\n"
        f"{shown_label}\n"
        f"Mean err: {mean_err_m:.3f} m  Median err: {median_err_m:.3f} m",
        fontsize=10,
    )

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, fontsize=8, loc="upper right", ncol=1)
    ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    logger.info("Overlay figure saved -> %s", output_path)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare WILDTRACK GT trajectories with predicted trajectories. "
            "Produces BEV overlay PNG and distance error metrics."
        )
    )
    parser.add_argument(
        "--root", default="data/Wildtrack_dataset",
        help="WILDTRACK dataset root (must contain annotations_positions/).",
    )
    parser.add_argument(
        "--pred", default="outputs/pred_trajectory.json",
        help="Predicted trajectory JSON from make_demo.py hook.",
    )
    parser.add_argument(
        "--max-frames", default=None, type=int,
        help="Cap GT annotation files to load (default: all 400).",
    )
    parser.add_argument(
        "--cameras", default="1,2,3,4,5,6,7",
        help="Comma-separated camera IDs (used for pseudo-prediction fallback).",
    )
    parser.add_argument(
        "--output", default="outputs",
        help="Output directory for PNG and JSON results.",
    )
    parser.add_argument(
        "--pseudo", action="store_true",
        help=(
            "Use GT bbox reprojection as pseudo-prediction "
            "(validates projection pipeline when real pred_trajectory.json "
            "is absent or when --force-pseudo is set)."
        ),
    )
    parser.add_argument(
        "--max-match-dist", type=float, default=10.0,
        help="Maximum mean trajectory distance (m) to accept a GT-pred match.",
    )
    parser.add_argument(
        "--top-k", type=int, default=3,
        dest="top_k",
        help=(
            "Number of best-aligned matched pairs to highlight in colour on the "
            "BEV figure (default: 3).  Remaining pairs and unmatched tracks are "
            "drawn in dim grey.  Pass 0 to draw all pairs in colour (original "
            "spaghetti debug mode).  Metrics are always computed over ALL "
            "matched pairs — top-k is visualisation only."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    root = Path(args.root)
    annotations_dir = root / "annotations_positions"
    if not annotations_dir.is_dir():
        logger.error(
            "annotations_positions not found under %s. Check --root.", root
        )
        sys.exit(1)

    cameras = [int(c) for c in args.cameras.split(",")]
    output_dir = Path(args.output)
    pred_path = Path(args.pred)

    # ── Determine mode: real inference or pseudo-prediction ──────────────────
    mode_label: str
    pred_traj: Dict[int, List[Tuple[int, float, float]]]

    if not args.pseudo and pred_path.exists():
        pred_traj = load_pred_trajectories(pred_path)
        mode_label = f"real inference (pred_trajectory.json)"
        logger.info("Using real predicted trajectories from %s", pred_path)
    else:
        if not args.pseudo:
            logger.warning(
                "%s not found. Falling back to pseudo-prediction "
                "(GT bbox reprojection via calibration). "
                "Run make_demo.py first for real predictions, "
                "or pass --pseudo to suppress this warning.",
                pred_path,
            )
        calib_root = str(root / "calibrations")
        pred_traj, mode_label = build_pseudo_predictions(
            annotations_dir=annotations_dir,
            calib_root=calib_root,
            cameras=cameras,
            max_frames=args.max_frames,
        )
        logger.info("Mode: %s", mode_label)

    # ── Load GT trajectories ─────────────────────────────────────────────────
    gt_traj = load_gt_trajectories(
        annotations_dir=annotations_dir,
        max_frames=args.max_frames,
    )

    if not gt_traj:
        logger.error("No GT trajectories loaded. Exiting.")
        sys.exit(1)
    if not pred_traj:
        logger.error("No predicted trajectories available. Exiting.")
        sys.exit(1)

    # ── Global Hungarian matching ────────────────────────────────────────────
    matched, unmatched_gt, unmatched_pred, mean_dist = global_hungarian_match(
        gt_traj=gt_traj,
        pred_traj=pred_traj,
        max_cost_m=args.max_match_dist,
    )
    logger.info(
        "Matching: %d matched pairs, %d unmatched GT, %d unmatched pred",
        len(matched),
        len(unmatched_gt),
        len(unmatched_pred),
    )

    # ── Compute distance errors across all matched pairs ────────────────────
    all_point_dists: List[float] = []
    for gt_pid, pred_gid in matched:
        gt_pts, pred_pts = align_trajectories(gt_traj[gt_pid], pred_traj[pred_gid])
        if len(gt_pts) > 0:
            dists = np.linalg.norm(gt_pts - pred_pts, axis=1)
            all_point_dists.extend(dists.tolist())

    if all_point_dists:
        mean_err_m = float(np.mean(all_point_dists))
        median_err_m = float(np.median(all_point_dists))
    else:
        mean_err_m = float("nan")
        median_err_m = float("nan")
        logger.warning("No aligned point pairs found — check temporal overlap.")

    # ── stdout summary ───────────────────────────────────────────────────────
    summary = (
        f"[FEAT-11] Mode: {mode_label} | "
        f"Matched: {len(matched)} pairs | "
        f"Mean dist: {mean_err_m:.3f} m | "
        f"Median dist: {median_err_m:.3f} m | "
        f"N points: {len(all_point_dists)}"
    )
    print(summary)
    logger.info(summary)

    # ── Save metrics JSON ────────────────────────────────────────────────────
    metrics = {
        "mode": mode_label,
        "matched_pairs": len(matched),
        "unmatched_gt": len(unmatched_gt),
        "unmatched_pred": len(unmatched_pred),
        "n_aligned_points": len(all_point_dists),
        "mean_dist_m": round(mean_err_m, 4) if not np.isnan(mean_err_m) else None,
        "median_dist_m": round(median_err_m, 4) if not np.isnan(median_err_m) else None,
        "matched_detail": [
            {
                "gt_person_id": int(gpid),
                "pred_global_id": int(ppid),
                "mean_dist_m": round(mean_dist.get((gpid, ppid), float("nan")), 4),
            }
            for gpid, ppid in matched
        ],
    }
    metrics_path = output_dir / "compare_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2))
    logger.info("Metrics saved -> %s", metrics_path)

    # ── BEV overlay figure ───────────────────────────────────────────────────
    plot_overlay(
        gt_traj=gt_traj,
        pred_traj=pred_traj,
        matched=matched,
        unmatched_gt=unmatched_gt,
        unmatched_pred=unmatched_pred,
        mean_dist=mean_dist,
        output_path=output_dir / "traj_gt_vs_pred.png",
        mode_label=mode_label,
        mean_err_m=mean_err_m,
        median_err_m=median_err_m,
        top_k=args.top_k,
    )

    logger.info(
        "Done. Outputs: %s/traj_gt_vs_pred.png, %s/compare_metrics.json",
        output_dir,
        output_dir,
    )


if __name__ == "__main__":
    main()
