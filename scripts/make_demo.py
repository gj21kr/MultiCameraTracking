"""Generate a multi-camera tracking demo video from a dataset (FEAT-9).

This is the single reproducible entry point for the project's goal:

    clone -> python scripts/make_demo.py [...] -> outputs/grid.mp4

Sources
-------
  --self-test                 synthetic multi-cam set (no download; CI/smoke)
  --dataset wildtrack --root <DIR>     WILDTRACK image-sequence dataset
  --dataset video --videos a.mp4,b.mp4,...   one video file per camera
  --dataset images --root <DIR>        <DIR>/C1,<DIR>/C2,... image folders

Examples
--------
  python scripts/make_demo.py --self-test
  python scripts/make_demo.py --dataset wildtrack --root data/wildtrack \
      --cameras 1,2,3,4 --max-frames 100 --model yolo26s --conf 0.4
"""

import sys
import json
import time
import argparse
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from src.utils.config import DetectorConfig, TrackerConfig, ReIDConfig, resolve_device
from src.utils.visualization import draw_tracks
from src.tracking.detector import Detector
from src.tracking.tracker import MultiCameraTracker
from src.tracking.video_writer import MultiCameraVideoWriter
from src.data.sources import (
    from_wildtrack, VideoFileSource, ImageSequenceSource, make_synthetic_source,
)

logger = logging.getLogger("make_demo")


def build_source(args):
    """Construct a FrameSource from CLI args."""
    cameras = (
        [int(c) for c in args.cameras.split(",")] if args.cameras else None
    )

    if args.self_test:
        # Use the ultralytics sample image (people in a street scene) as the
        # synthetic base so the real detector finds real people.
        sample = Path("outputs/_bus.jpg")
        if not sample.exists():
            import urllib.request
            sample.parent.mkdir(parents=True, exist_ok=True)
            logger.info("Downloading sample image for self-test...")
            urllib.request.urlretrieve(
                "https://ultralytics.com/images/bus.jpg", str(sample)
            )
        base = cv2.imread(str(sample))
        return make_synthetic_source(
            base, num_cameras=args.synthetic_cameras,
            num_frames=args.max_frames or 40, fps=args.fps,
        )

    if args.dataset == "wildtrack":
        if not args.root:
            raise SystemExit("--root is required for --dataset wildtrack")
        return from_wildtrack(
            args.root, cameras=cameras, max_frames=args.max_frames, step=args.step
        )

    if args.dataset == "images":
        if not args.root:
            raise SystemExit("--root is required for --dataset images")
        root = Path(args.root)
        cam_dirs = {}
        for d in sorted(root.iterdir()):
            if d.is_dir() and d.name[:1].lower() == "c":
                try:
                    cam_dirs[int(d.name[1:])] = str(d)
                except ValueError:
                    continue
        if cameras:
            cam_dirs = {c: cam_dirs[c] for c in cameras if c in cam_dirs}
        if not cam_dirs:
            raise SystemExit(f"No C<N> camera folders under {root}")
        return ImageSequenceSource(
            cam_dirs, max_frames=args.max_frames, step=args.step, fps=args.fps
        )

    if args.dataset == "video":
        if not args.videos:
            raise SystemExit("--videos is required for --dataset video")
        paths = [p for p in args.videos.split(",") if p]
        video_paths = {i + 1: p for i, p in enumerate(paths)}
        return VideoFileSource(
            video_paths, max_frames=args.max_frames, step=args.step
        )

    raise SystemExit(f"Unknown dataset: {args.dataset}")


def main():
    ap = argparse.ArgumentParser(description="Multi-camera tracking demo video generator")
    ap.add_argument("--self-test", action="store_true", help="Synthetic source (no download)")
    ap.add_argument("--dataset", choices=["wildtrack", "video", "images"], default="wildtrack")
    ap.add_argument("--root", help="Dataset root (wildtrack/images)")
    ap.add_argument("--videos", help="Comma-separated video files (dataset=video)")
    ap.add_argument("--cameras", help="Comma-separated camera ids to use")
    ap.add_argument("--synthetic-cameras", type=int, default=4, help="self-test camera count")
    ap.add_argument("--max-frames", type=int, help="Cap synchronized frames")
    ap.add_argument("--step", type=int, default=1, help="Take every Nth frame")
    ap.add_argument("--model", default="yolo26s", help="YOLO model type")
    ap.add_argument("--model-path", help="Explicit model weight path")
    ap.add_argument("--conf", type=float, default=0.4, help="Detection confidence")
    ap.add_argument("--imgsz", type=int, default=640, help="Detector input size")
    ap.add_argument("--tile", default="", help="SAHI-style tiling 'RxC' e.g. 2x3 (empty=off)")
    ap.add_argument("--classes", default="0", help="Comma-separated COCO class ids (0=person)")
    ap.add_argument("--device", default="cuda:0", help="cuda:0 / cpu (auto-falls back to cpu)")
    ap.add_argument("--reid", action="store_true", help="Enable cross-camera ReID association")
    ap.add_argument("--no-grid", action="store_true", help="Skip grid montage output")
    ap.add_argument("--no-per-camera", action="store_true", help="Skip per-camera mp4s")
    ap.add_argument("--fps", type=float, default=10.0, help="Output video FPS")
    ap.add_argument("--output", default="outputs", help="Output directory")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    t_start = time.time()
    device = resolve_device(args.device)
    classes = [int(c) for c in args.classes.split(",")]

    # --- Source ---
    source = build_source(args)
    camera_ids = source.camera_ids
    fps = getattr(source, "fps", args.fps) or args.fps
    logger.info("Source ready: cameras=%s fps=%.1f", camera_ids, fps)

    # --- Components ---
    det_cfg = DetectorConfig(
        model_type=args.model, model_path=args.model_path,
        confidence_threshold=args.conf, classes=classes, device=device, fp16=True,
        input_size=(args.imgsz, args.imgsz),
    )
    detector = Detector(det_cfg)
    tile_rc = None
    if args.tile:
        _r, _c = args.tile.lower().split("x")
        tile_rc = (int(_r), int(_c))

    trk_cfg = TrackerConfig(track_thresh=args.conf, use_reid=args.reid)
    tracker = MultiCameraTracker(trk_cfg, num_cameras=len(camera_ids))

    reid = None
    if args.reid:
        from src.tracking.reid import ReIDExtractor
        reid = ReIDExtractor(ReIDConfig(device=device))
        if getattr(reid, "is_dummy", False):
            logger.warning("ReID is DUMMY -> cross-camera ids will be unreliable.")

    writer = MultiCameraVideoWriter(
        output_dir=args.output, camera_ids=camera_ids, fps=fps,
        write_per_camera=not args.no_per_camera, write_grid=not args.no_grid,
    )

    # --- Trajectory accumulation state (A2.2, FEAT-11) ---
    # Accumulates (t, X_m, Y_m) per global_id using ground-plane projection.
    # Only active when source is WILDTRACK (calibration available).
    # Non-destructive: mp4 output is completely unaffected.
    pred_trajectories: Dict[int, List] = defaultdict(list)
    _calib_for_hook: Optional[object] = None
    _hook_global_id_map: Dict[Tuple[int, int], int] = {}   # (cam_id, track_id) -> global_id  [geometric]
    _hook_next_id: List = [0]        # mutable next-id counter [geometric]
    if not args.self_test and args.dataset == "wildtrack" and args.root:
        try:
            from src.tracking.calibration import WildtrackCalibration
            _calib_root = str(Path(args.root) / "calibrations")
            _calib_for_hook = WildtrackCalibration.from_dir(_calib_root)
            logger.info("Trajectory hook: calibration loaded from %s", _calib_root)
        except Exception as exc:
            logger.warning(
                "Trajectory hook: calibration load failed (%s) — hook disabled.", exc
            )
            _calib_for_hook = None

    # --- Process loop ---
    n = 0
    det_total = 0
    with source, writer:
        while True:
            frames = source.read()
            if frames is None:
                break
            images = {cid: f.image for cid, f in frames.items()}

            # Detection (batch) keeping camera order.
            cids = list(images.keys())
            if tile_rc:
                det_lists = [detector.detect_tiled(images[c], tile_rc[0], tile_rc[1]) for c in cids]
            else:
                det_lists = detector.detect_batch([images[c] for c in cids])
            detections = {c: d for c, d in zip(cids, det_lists)}
            det_total += sum(len(d) for d in detections.values())

            # Optional ReID embeddings for cross-camera association.
            if reid is not None:
                for c, dets in detections.items():
                    if dets:
                        embs = reid.extract_batch(images[c], [d.bbox for d in dets])
                        for d, e in zip(dets, embs):
                            d.embedding = e

            tracks = tracker.update(detections, images)

            # --- Trajectory hook (A2.2): accumulate ground-plane positions ---
            # Append (frame_index, X_m, Y_m) per global_id.
            # foot_to_ground returns cm; /100 converts to metres for GT parity.
            #
            # global_id assignment strategy:
            #   - If ReID is active, tracks already have global_id from
            #     associate_cross_camera.
            #   - If ReID is inactive (default), we fall back to geometric
            #     cross-camera association via assign_global_ids_geometric
            #     (calibration.py) so the trajectory hook always has IDs.
            #   Per-camera track_id is used as a fallback key when geometry
            #   cannot link across cameras (single-view global_id).
            if _calib_for_hook is not None:
                if not args.reid:
                    # Geometric global_id assignment using ground-plane proximity.
                    from src.tracking.calibration import assign_global_ids_geometric
                    assign_global_ids_geometric(
                        all_tracks=tracks,
                        calib=_calib_for_hook,
                        global_id_map=_hook_global_id_map,
                        next_id_box=_hook_next_id,
                        dist_thresh_cm=100.0,
                    )
                for cam_id, cam_tracks in tracks.items():
                    for trk in cam_tracks:
                        gid = trk.global_id
                        if gid is None:
                            continue
                        ground_cm = _calib_for_hook.foot_to_ground(cam_id, trk.bbox)
                        if ground_cm is None or not np.all(np.isfinite(ground_cm)):
                            continue
                        x_m = float(ground_cm[0]) / 100.0
                        y_m = float(ground_cm[1]) / 100.0
                        pred_trajectories[gid].append([n, x_m, y_m])

            annotated = {
                c: draw_tracks(images[c], tracks.get(c, []),
                               use_global_id=args.reid, cam_id=c)
                for c in cids
            }
            writer.write(annotated)

            n += 1
            if n % 25 == 0:
                logger.info("processed %d frames...", n)

    elapsed = time.time() - t_start

    # --- Save predicted trajectories (A2.2) ---
    if pred_trajectories:
        pred_traj_path = Path(args.output) / "pred_trajectory.json"
        pred_traj_path.parent.mkdir(parents=True, exist_ok=True)
        # JSON keys must be strings; values are [[t, X_m, Y_m], ...]
        pred_traj_path.write_text(
            json.dumps({str(k): v for k, v in pred_trajectories.items()}, indent=2)
        )
        logger.info(
            "Predicted trajectories: %d tracks -> %s",
            len(pred_trajectories),
            pred_traj_path,
        )
    else:
        logger.info("No predicted trajectories accumulated (hook inactive or no tracks).")

    # --- Run metadata ---
    meta = {
        "frames": n,
        "cameras": camera_ids,
        "detections_total": det_total,
        "avg_detections_per_frame": round(det_total / max(1, n * len(camera_ids)), 2),
        "model": args.model,
        "weight_name": getattr(detector, "weight_name", args.model),
        "device": device,
        "reid": args.reid,
        "fps_out": fps,
        "elapsed_sec": round(elapsed, 1),
        "proc_fps": round(n / elapsed, 2) if elapsed > 0 else None,
        "outputs": writer.paths,
        "torch_version": __import__("torch").__version__,
    }
    meta_path = Path(args.output) / "run_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    logger.info("DONE: %d frames in %.1fs (%.2f fps)", n, elapsed, meta["proc_fps"] or 0)
    logger.info("Outputs: %s", writer.paths)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
