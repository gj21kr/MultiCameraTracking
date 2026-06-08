"""Find the best 3-second (6-frame) window in the 400-frame WILDTRACK inference.

Scoring strategy: combined track-density + accuracy.
  - Accuracy (GT-vs-pred): per-frame greedy matching on the ground plane.
    GT: positionID -> world (m) from annotations_positions/*.json
    Pred: pred_trajectory.json {global_id: [[t, X_m, Y_m], ...]}
    Metric: mean distance of matched pairs (lower = better),
    filtered to pairs within 3 m threshold.
  - Track density: number of global IDs that appear in >= density_min_frames
    of the window (default 3/6). This measures how many *confirmed* tracks
    are visible, which drives the richness of the GIF (boxes + trajectories).
  - Combined score = confirmed_tracks / (1 + avg_dist_m)  [higher = better]

Warmup skip: early frames (< warmup_frames) are excluded because
  trackers are still initialising and produce few visible boxes/trajectories
  even if the positional accuracy happens to be good for the few detections
  that do exist. Default warmup_frames=15.

Sliding window (window_size=6 frames):
  - All 6 frames must have valid pred data.
  - Average matched pairs >= min_matches (default 6, lower than before
    because density metric already guards against sparse windows).
  - Best window = maximum combined score.

Outputs
-------
  stdout: selected window + per-frame breakdown
  outputs/best_window.json: {t_start, t_end, avg_dist_m, avg_n_matched,
                              confirmed_tracks, score, frame_details,
                              selection_reason}

Usage
-----
    python scripts/find_best_window.py
    python scripts/find_best_window.py --root data/Wildtrack_dataset \\
        --pred outputs/pred_trajectory.json --min-matches 6 --warmup 15
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.wildtrack_grid import position_id_to_world


def _load_gt_by_raw_frame(
    annotations_dir: Path,
) -> Dict[int, Dict[int, Tuple[float, float]]]:
    """Load GT annotations keyed by raw frame number (filename stem as int).

    Returns {raw_frame_number: {person_id: (X_m, Y_m)}}
    """
    gt: Dict[int, Dict[int, Tuple[float, float]]] = {}
    for jf in sorted(annotations_dir.glob("*.json")):
        raw_frame = int(jf.stem)
        entries = json.loads(jf.read_text(encoding="utf-8"))
        gt[raw_frame] = {}
        for e in entries:
            x, y = position_id_to_world(int(e["positionID"]))
            gt[raw_frame][int(e["personID"])] = (x, y)
    logger.info(
        "GT loaded: %d annotation frames (raw frames %s..%s)",
        len(gt),
        min(gt) if gt else "?",
        max(gt) if gt else "?",
    )
    return gt


def _load_pred_by_raw_frame(
    pred_path: Path,
) -> Dict[int, Dict[int, Tuple[float, float]]]:
    """Load pred_trajectory.json keyed by raw frame index.

    Returns {raw_frame: {global_id: (X_m, Y_m)}}
    """
    data = json.loads(pred_path.read_text(encoding="utf-8"))
    pred: Dict[int, Dict[int, Tuple[float, float]]] = {}
    for gid_str, obs in data.items():
        gid = int(gid_str)
        for row in obs:
            t = int(row[0])
            x, y = float(row[1]), float(row[2])
            if t not in pred:
                pred[t] = {}
            pred[t][gid] = (x, y)
    logger.info(
        "Pred loaded: %d global IDs, raw frames %s..%s",
        len(data),
        min(pred) if pred else "?",
        max(pred) if pred else "?",
    )
    return pred


def _frame_quality(
    raw_t: int,
    gt_by_raw: Dict[int, Dict[int, Tuple[float, float]]],
    gt_raw_frames: np.ndarray,
    pred_by_raw: Dict[int, Dict[int, Tuple[float, float]]],
    match_radius_m: float = 3.0,
    max_gt_gap: int = 3,
) -> Tuple[Optional[float], int]:
    """Compute per-frame tracking quality.

    Finds the nearest GT annotation frame (within max_gt_gap raw frames),
    then greedily matches each pred position to the nearest GT within match_radius_m.

    Returns:
        (mean_dist_m, n_matched) — (None, 0) if no valid data.
    """
    if raw_t not in pred_by_raw:
        return None, 0

    idx = int(np.argmin(np.abs(gt_raw_frames - raw_t)))
    gt_rf = int(gt_raw_frames[idx])
    if abs(gt_rf - raw_t) > max_gt_gap:
        return None, 0

    gt = gt_by_raw.get(gt_rf, {})
    pred = pred_by_raw[raw_t]
    if not gt or not pred:
        return None, 0

    gt_positions = np.array(list(gt.values()), dtype=float)
    pred_positions = np.array(list(pred.values()), dtype=float)

    matched_dists: List[float] = []
    for ppos in pred_positions:
        dists = np.linalg.norm(gt_positions - ppos, axis=1)
        min_dist = float(np.min(dists))
        if min_dist < match_radius_m:
            matched_dists.append(min_dist)

    if not matched_dists:
        return None, 0
    return float(np.mean(matched_dists)), len(matched_dists)


def _count_confirmed_tracks(
    pred_by_raw: Dict[int, Dict[int, Tuple[float, float]]],
    window: List[int],
    density_min_frames: int = 3,
) -> int:
    """Count global IDs that appear in >= density_min_frames of the window.

    A 'confirmed' track has been seen across multiple frames of the window,
    meaning it has an established trajectory and will produce visible boxes
    and trajectory lines in the rendered video.
    """
    from collections import Counter
    appearance: Counter = Counter()
    for t in window:
        for gid in pred_by_raw.get(t, {}):
            appearance[gid] += 1
    return sum(1 for cnt in appearance.values() if cnt >= density_min_frames)


def find_best_window(
    gt_by_raw: Dict[int, Dict[int, Tuple[float, float]]],
    pred_by_raw: Dict[int, Dict[int, Tuple[float, float]]],
    n_frames: int = 400,
    window_size: int = 6,
    min_matches: int = 6,
    match_radius_m: float = 3.0,
    warmup_frames: int = 15,
    density_min_frames: int = 3,
) -> Tuple[int, float, float, int, float, List[dict]]:
    """Find the best sliding window using combined track-density + accuracy score.

    The score is: confirmed_tracks / (1 + avg_dist_m)  [higher = better].

    'confirmed_tracks' = number of global IDs visible in >= density_min_frames
    of the window. This correlates with visual richness (box + trajectory count)
    and filters out the tracker warmup period.

    Windows starting before warmup_frames are excluded to avoid immature-tracker
    segments where tracks are still initialising and GIF would look sparse.

    Args:
        warmup_frames: Skip windows whose t_start < warmup_frames.
        density_min_frames: Minimum frame appearances to count a track as confirmed.

    Returns:
        (t_start, avg_dist_m, avg_n_matched, confirmed_tracks, score, frame_details_list)
    """
    gt_raw_frames = np.array(sorted(gt_by_raw.keys()))

    # Per-frame quality (accuracy)
    frame_scores: Dict[int, Tuple[Optional[float], int]] = {}
    for t in range(n_frames):
        frame_scores[t] = _frame_quality(
            t, gt_by_raw, gt_raw_frames, pred_by_raw,
            match_radius_m=match_radius_m,
        )

    candidates = []
    for t_start in range(warmup_frames, n_frames - window_size + 1):
        window = list(range(t_start, t_start + window_size))
        valid = [(t, frame_scores[t]) for t in window if frame_scores[t][0] is not None]
        if len(valid) < window_size:
            continue
        avg_match = float(np.mean([s[1] for _, s in valid]))
        if avg_match < min_matches:
            continue
        avg_dist = float(np.mean([s[0] for _, s in valid]))
        # Track density: confirmed global IDs across the window
        confirmed = _count_confirmed_tracks(pred_by_raw, window, density_min_frames)
        # Combined score: density / (1 + error) — higher is better
        score = confirmed / (1.0 + avg_dist)
        candidates.append((score, avg_match, avg_dist, confirmed, t_start))

    if not candidates:
        raise RuntimeError(
            f"No valid {window_size}-frame window found (warmup={warmup_frames}, "
            f"min_matches={min_matches}). Try lowering --min-matches or --warmup."
        )

    # Sort by score descending
    candidates.sort(reverse=True)
    best_score, best_match, best_dist, best_confirmed, best_t = candidates[0]

    frame_details = []
    for t in range(best_t, best_t + window_size):
        d, n = frame_scores[t]
        frame_details.append({
            "frame": t,
            "mean_dist_m": round(d, 4) if d is not None else None,
            "n_matched": n,
        })

    return best_t, best_dist, best_match, best_confirmed, best_score, frame_details


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--root", default="data/Wildtrack_dataset",
        help="WILDTRACK dataset root (must contain annotations_positions/).",
    )
    p.add_argument(
        "--pred", default="outputs/pred_trajectory.json",
        help="Predicted trajectory JSON from make_demo.py.",
    )
    p.add_argument(
        "--output", default="outputs",
        help="Directory for best_window.json.",
    )
    p.add_argument(
        "--window-size", type=int, default=6,
        help="Sliding window size in frames (default: 6 = 3 s at 2 fps).",
    )
    p.add_argument(
        "--min-matches", type=int, default=6,
        help="Minimum average matched pairs per frame (default: 6).",
    )
    p.add_argument(
        "--match-radius", type=float, default=3.0,
        help="Maximum GT-pred distance to count as a match (default: 3.0 m).",
    )
    p.add_argument(
        "--warmup", type=int, default=15,
        help=(
            "Skip windows starting before this frame index (default: 15). "
            "Avoids immature-tracker segments where tracks are still initialising."
        ),
    )
    p.add_argument(
        "--density-min-frames", type=int, default=3,
        help=(
            "A global ID counts as 'confirmed' only if it appears in at least "
            "this many frames of the window (default: 3 out of 6). "
            "Confirmed-track count is the density component of the scoring metric."
        ),
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    root = Path(args.root)
    annotations_dir = root / "annotations_positions"
    if not annotations_dir.is_dir():
        logger.error("annotations_positions not found under %s", root)
        sys.exit(1)

    pred_path = Path(args.pred)
    if not pred_path.exists():
        logger.error("pred_trajectory.json not found at %s", pred_path)
        sys.exit(1)

    gt_by_raw = _load_gt_by_raw_frame(annotations_dir)
    pred_by_raw = _load_pred_by_raw_frame(pred_path)

    t_start, avg_dist, avg_matches, confirmed, score, frame_details = find_best_window(
        gt_by_raw=gt_by_raw,
        pred_by_raw=pred_by_raw,
        window_size=args.window_size,
        min_matches=args.min_matches,
        match_radius_m=args.match_radius,
        warmup_frames=args.warmup,
        density_min_frames=args.density_min_frames,
    )
    t_end = t_start + args.window_size - 1

    result = {
        "t_start": t_start,
        "t_end": t_end,
        "window_size": args.window_size,
        "avg_dist_m": round(avg_dist, 4),
        "avg_n_matched": round(avg_matches, 1),
        "confirmed_tracks": confirmed,
        "score": round(score, 4),
        "selection_reason": (
            f"Maximum combined score (confirmed_tracks / (1 + avg_dist_m)) = "
            f"{score:.2f}: {confirmed} confirmed tracks (appearing in "
            f">= {args.density_min_frames}/{args.window_size} frames), "
            f"avg GT-vs-pred distance {avg_dist:.3f} m, "
            f"avg {avg_matches:.1f} matched pairs/frame. "
            f"Warmup skip: frames < {args.warmup} excluded."
        ),
        "frame_details": frame_details,
    }

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "best_window.json"
    out_path.write_text(json.dumps(result, indent=2))

    summary = (
        f"Best 3-second window: frames {t_start}-{t_end} | "
        f"score={score:.2f} | confirmed_tracks={confirmed} | "
        f"avg_dist={avg_dist:.3f} m | avg_matched={avg_matches:.1f} pairs"
    )
    print(summary)
    print("Frame breakdown:")
    for fd in frame_details:
        d_str = f"{fd['mean_dist_m']:.3f} m" if fd["mean_dist_m"] is not None else "N/A"
        print(f"  frame {fd['frame']:3d}: dist={d_str}, matched={fd['n_matched']}")
    print(
        f"\nSelection rationale: picked window with highest combined score "
        f"(track density / accuracy) after skipping warmup frames 0-{args.warmup - 1}."
    )
    print(f"Result saved -> {out_path}")


if __name__ == "__main__":
    main()
