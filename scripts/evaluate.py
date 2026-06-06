"""Quantitative tracking evaluation on WILDTRACK ground truth.

Computes per-camera and overall MOTA / IDF1 / ID-switches by matching tracker
output against the official WILDTRACK 2D per-view annotations
(annotations_positions/*.json). This is the baseline harness every quality
change should be measured against.

Usage:
    python scripts/evaluate.py --root data/Wildtrack_dataset \
        --cameras 1,2,3,4,5,6,7 --max-frames 100 --conf 0.3
"""

import sys
import json
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

# motmetrics 1.4 still calls numpy APIs removed in NumPy 2.0; shim them.
if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=np.float64: np.asarray(a, dtype=dtype)
for _name, _alias in (("float", float), ("int", int), ("bool", bool), ("object", object)):
    if not hasattr(np, _name):
        setattr(np, _name, _alias)

import motmetrics as mm

from src.utils.config import DetectorConfig, TrackerConfig, ReIDConfig, resolve_device
from src.tracking.detector import Detector
from src.tracking.tracker import MultiCameraTracker
from src.data.sources import from_wildtrack

logger = logging.getLogger("evaluate")


def load_gt(root: str, cameras, n_frames):
    """Return gt[cam][frame_idx] = (ids, boxes_xywh) from WILDTRACK JSON.

    Annotation files are sorted; index i aligns with source frame i.
    viewNum (0-based) == camera_id - 1.
    """
    ann_dir = Path(root) / "annotations_positions"
    files = sorted(ann_dir.glob("*.json"))[:n_frames]
    gt = {c: [] for c in cameras}
    for f in files:
        persons = json.loads(f.read_text())
        per_cam = {c: ([], []) for c in cameras}
        for p in persons:
            pid = p["personID"]
            for v in p["views"]:
                cam = v["viewNum"] + 1
                if cam not in per_cam:
                    continue
                if v["xmin"] < 0 or v["ymin"] < 0 or v["xmax"] < 0 or v["ymax"] < 0:
                    continue  # not visible in this view
                x, y = v["xmin"], v["ymin"]
                w, h = v["xmax"] - v["xmin"], v["ymax"] - v["ymin"]
                if w <= 0 or h <= 0:
                    continue
                per_cam[cam][0].append(pid)
                per_cam[cam][1].append([x, y, w, h])
        for c in cameras:
            ids, boxes = per_cam[c]
            gt[c].append((ids, np.array(boxes, dtype=float).reshape(-1, 4)))
    return gt, len(files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/Wildtrack_dataset")
    ap.add_argument("--cameras", default="1,2,3,4,5,6,7")
    ap.add_argument("--max-frames", type=int, default=100)
    ap.add_argument("--model", default="yolo26s")
    ap.add_argument("--model-path", default="models/yolo26s.pt")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--track-buffer", type=int, default=30, help="max frames a lost track coasts")
    ap.add_argument("--match-thresh", type=float, default=0.3, help="min IoU to accept a match")
    ap.add_argument("--iou", type=float, default=0.5, help="IoU match threshold for MOT")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--reid", action="store_true")
    ap.add_argument("--output", default="outputs/eval_metrics.json")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cameras = [int(c) for c in args.cameras.split(",")]
    device = resolve_device(args.device)

    source = from_wildtrack(args.root, cameras=cameras, max_frames=args.max_frames)
    n_req = args.max_frames or len(source)
    gt, n_gt = load_gt(args.root, cameras, n_req)
    logger.info("GT frames: %d, cameras: %s", n_gt, cameras)

    detector = Detector(DetectorConfig(
        model_type=args.model, model_path=args.model_path,
        confidence_threshold=args.conf, classes=[0], device=device, fp16=True))
    tracker = MultiCameraTracker(
        TrackerConfig(track_thresh=args.conf, match_thresh=args.match_thresh,
                      track_buffer=args.track_buffer, use_reid=args.reid),
        num_cameras=len(cameras))
    reid = None
    if args.reid:
        from src.tracking.reid import ReIDExtractor
        reid = ReIDExtractor(ReIDConfig(device=device))

    accs = {c: mm.MOTAccumulator(auto_id=True) for c in cameras}

    fidx = 0
    with source:
        while fidx < n_gt:
            frames = source.read()
            if frames is None:
                break
            images = {c: f.image for c, f in frames.items()}
            cids = list(images.keys())
            det_lists = detector.detect_batch([images[c] for c in cids])
            detections = {c: d for c, d in zip(cids, det_lists)}
            if reid is not None:
                for c, dets in detections.items():
                    if dets:
                        embs = reid.extract_batch(images[c], [d.bbox for d in dets])
                        for d, e in zip(dets, embs):
                            d.embedding = e
            tracks = tracker.update(detections, images)

            for c in cameras:
                gt_ids, gt_boxes = gt[c][fidx]
                cam_tracks = tracks.get(c, [])
                hyp_ids = [t.track_id for t in cam_tracks]
                hyp_boxes = np.array(
                    [[t.bbox[0], t.bbox[1], t.bbox[2] - t.bbox[0], t.bbox[3] - t.bbox[1]]
                     for t in cam_tracks], dtype=float).reshape(-1, 4)
                dists = mm.distances.iou_matrix(gt_boxes, hyp_boxes, max_iou=args.iou)
                accs[c].update(gt_ids, hyp_ids, dists)

            fidx += 1
            if fidx % 25 == 0:
                logger.info("evaluated %d/%d frames", fidx, n_gt)

    metrics = ["mota", "idf1", "num_switches", "num_false_positives",
               "num_misses", "mostly_tracked", "mostly_lost", "precision", "recall"]
    mh = mm.metrics.create()
    names = [f"C{c}" for c in cameras]
    summary = mh.compute_many(
        [accs[c] for c in cameras], metrics=metrics, names=names, generate_overall=True)

    print("\n" + mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap={k: k for k in metrics},
    ))

    out = {row: {m: (None if (v != v) else float(v)) for m, v in summary.loc[row].items()}
           for row in summary.index}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(
        {"frames": n_gt, "cameras": cameras, "conf": args.conf, "reid": args.reid,
         "device": device, "metrics": out}, indent=2))
    logger.info("saved %s", args.output)


if __name__ == "__main__":
    main()
