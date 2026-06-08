"""Extract the best 3-second (6-frame) window from outputs/grid.mp4 and
save as demo/wildtrack_demo.gif for README embedding.

Steps
-----
1. Run find_best_window logic to determine t_start (or read best_window.json).
2. Extract 6 consecutive frames from outputs/grid.mp4 at the chosen window.
3. Resize to target width (~800 px, maintaining aspect ratio).
4. Save as GIF with PIL (palette-optimised, looping).

GIF parameters
--------------
- Width: 1050 px (height auto-scaled; source is 1278x716)
- Duration per frame: 500 ms (2 fps preserved — 6 frames = 3 s loop)
- Loop: infinite (loop=0 in PIL)
- Color quantization: 256 colours, fast palette via PIL ADAPTIVE

Usage
-----
    python scripts/make_demo_gif.py
    python scripts/make_demo_gif.py --video outputs/grid.mp4 \\
        --best-window outputs/best_window.json \\
        --output demo/wildtrack_demo.gif --width 1050
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

try:
    from PIL import Image
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False


def _extract_frames(video_path: Path, t_start: int, t_end: int) -> List[np.ndarray]:
    """Extract frames [t_start, t_end] (inclusive) from video as BGR numpy arrays."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    logger.info(
        "Video: %s | total_frames=%d fps=%.1f", video_path.name, total, fps
    )

    if t_end >= total:
        raise ValueError(
            f"t_end={t_end} >= total_frames={total}. Check best_window.json."
        )

    frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, t_start)
    for t in range(t_start, t_end + 1):
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError(f"Failed to read frame {t} from {video_path}")
        frames.append(frame)
        logger.debug("Extracted frame %d/%d", t, t_end)

    cap.release()
    logger.info("Extracted %d frames (%d..%d)", len(frames), t_start, t_end)
    return frames


def _resize_frame(frame_bgr: np.ndarray, target_width: int) -> np.ndarray:
    """Resize frame to target_width, preserving aspect ratio."""
    h, w = frame_bgr.shape[:2]
    scale = target_width / w
    target_h = int(round(h * scale))
    if target_h % 2 != 0:
        target_h += 1
    return cv2.resize(frame_bgr, (target_width, target_h), interpolation=cv2.INTER_AREA)


def _save_gif_pil(
    frames_bgr: List[np.ndarray],
    output_path: Path,
    duration_ms: int = 500,
    target_width: int = 800,
) -> int:
    """Save frames as a looping GIF using PIL. Returns file size in bytes."""
    if not _PIL_AVAILABLE:
        raise ImportError(
            "Pillow is required for GIF output. Install with: pip install Pillow"
        )

    pil_frames = []
    for frame_bgr in frames_bgr:
        resized = _resize_frame(frame_bgr, target_width)
        # BGR -> RGB for PIL
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        # Convert to palette mode for smaller GIF size
        img_p = img.convert("P", palette=Image.ADAPTIVE, colors=256)
        pil_frames.append(img_p)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pil_frames[0].save(
        str(output_path),
        format="GIF",
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )

    size_bytes = output_path.stat().st_size
    return size_bytes


def _load_or_compute_best_window(
    best_window_path: Path,
    root: str,
    pred: str,
) -> dict:
    """Load best_window.json if it exists, otherwise run find_best_window logic."""
    if best_window_path.exists():
        data = json.loads(best_window_path.read_text(encoding="utf-8"))
        logger.info(
            "Loaded best_window.json: t_start=%d t_end=%d avg_dist=%.3f m",
            data["t_start"], data["t_end"], data["avg_dist_m"],
        )
        return data

    logger.info(
        "best_window.json not found at %s. Running find_best_window...",
        best_window_path,
    )
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from scripts.find_best_window import (
        _load_gt_by_raw_frame,
        _load_pred_by_raw_frame,
        find_best_window,
    )

    annotations_dir = Path(root) / "annotations_positions"
    gt_by_raw = _load_gt_by_raw_frame(annotations_dir)
    pred_by_raw = _load_pred_by_raw_frame(Path(pred))
    t_start, avg_dist, avg_matches, confirmed, score, frame_details = find_best_window(
        gt_by_raw=gt_by_raw,
        pred_by_raw=pred_by_raw,
    )
    t_end = t_start + 5
    result = {
        "t_start": t_start,
        "t_end": t_end,
        "window_size": 6,
        "avg_dist_m": round(avg_dist, 4),
        "avg_n_matched": round(avg_matches, 1),
        "confirmed_tracks": confirmed,
        "score": round(score, 4),
        "selection_reason": (
            f"Maximum combined score (track_density / (1 + avg_dist_m)) = "
            f"{score:.2f}: {confirmed} confirmed tracks, "
            f"avg GT-vs-pred distance {avg_dist:.3f} m, "
            f"avg {avg_matches:.1f} matched pairs/frame."
        ),
        "frame_details": frame_details,
    }
    best_window_path.parent.mkdir(parents=True, exist_ok=True)
    best_window_path.write_text(json.dumps(result, indent=2))
    return result


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--video", default="outputs/grid.mp4",
        help="Source video (default: outputs/grid.mp4).",
    )
    p.add_argument(
        "--best-window", default="outputs/best_window.json",
        help="best_window.json from find_best_window.py (computed if missing).",
    )
    p.add_argument(
        "--root", default="data/Wildtrack_dataset",
        help="WILDTRACK root (only needed if best_window.json is missing).",
    )
    p.add_argument(
        "--pred", default="outputs/pred_trajectory.json",
        help="Predicted trajectory JSON (only needed if best_window.json is missing).",
    )
    p.add_argument(
        "--output", default="demo/wildtrack_demo.gif",
        help="Output GIF path (default: demo/wildtrack_demo.gif).",
    )
    p.add_argument(
        "--width", type=int, default=1050,
        help="Target GIF width in pixels (default: 1050).",
    )
    p.add_argument(
        "--duration-ms", type=int, default=500,
        help="Duration per frame in ms (default: 500 = 2 fps).",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    video_path = Path(args.video)
    if not video_path.exists():
        logger.error("Video not found: %s", video_path)
        sys.exit(1)

    window = _load_or_compute_best_window(
        best_window_path=Path(args.best_window),
        root=args.root,
        pred=args.pred,
    )
    t_start = window["t_start"]
    t_end = window["t_end"]
    avg_dist = window["avg_dist_m"]
    avg_matches = window["avg_n_matched"]

    frames_bgr = _extract_frames(video_path, t_start, t_end)

    output_path = Path(args.output)
    size_bytes = _save_gif_pil(
        frames_bgr=frames_bgr,
        output_path=output_path,
        duration_ms=args.duration_ms,
        target_width=args.width,
    )
    size_mb = size_bytes / (1024 * 1024)

    summary_lines = [
        f"GIF saved -> {output_path}",
        f"  Frames: {t_start}-{t_end} ({t_end - t_start + 1} frames)",
        f"  Avg GT-vs-pred dist: {avg_dist:.3f} m | avg matched: {avg_matches:.1f} pairs",
        f"  Size: {size_bytes:,} bytes ({size_mb:.2f} MB)",
        f"  Width: {args.width} px | Duration/frame: {args.duration_ms} ms",
    ]
    for line in summary_lines:
        print(line)
    logger.info("Done.")


if __name__ == "__main__":
    main()
