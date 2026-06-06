"""Ground-plane (BEV) evaluation on WILDTRACK — the dataset's intended metric.

WILDTRACK 2D per-view boxes are projected ground-plane cylinders, not tight
detection boxes, so per-camera 2D MOTA is misleading (huge IoU-mismatch FP).
The correct space is the common ground plane: project each detection's foot
point to world coords, fuse detections of the same person across the 7 cameras
(geometric cross-camera association, ADR-003), and match to GT ground positions.

Reports MODA / MODP / precision / recall / F1, and runs the fuse ON vs OFF
ablation in one pass so the value of cross-camera fusion is explicit.

Usage:
    python scripts/evaluate_ground.py --root data/Wildtrack_dataset --max-frames 100
"""

import sys
import json
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.utils.config import DetectorConfig, resolve_device
from src.tracking.detector import Detector
from src.tracking.calibration import WildtrackCalibration
from src.data.sources import from_wildtrack

logger = logging.getLogger("eval_ground")


def gt_ground_positions(root, calib, cameras, n_frames):
    """GT world position per person/frame = median of visible-view foot projections."""
    ann_dir = Path(root) / "annotations_positions"
    files = sorted(ann_dir.glob("*.json"))[:n_frames]
    frames = []
    for f in files:
        persons = json.loads(f.read_text())
        pos = []
        for p in persons:
            pts = []
            for v in p["views"]:
                cam = v["viewNum"] + 1
                if cam not in cameras or v["xmin"] < 0 or v["ymin"] < 0:
                    continue
                bbox = np.array([v["xmin"], v["ymin"], v["xmax"], v["ymax"]], float)
                g = calib.foot_to_ground(cam, bbox)
                if g is not None and np.all(np.isfinite(g)):
                    pts.append(g)
            if pts:
                pos.append(np.median(np.array(pts), axis=0))
        frames.append(np.array(pos).reshape(-1, 2))
    return frames, len(files)


def fuse_points(points, dist_thresh):
    """Cluster ground points within dist_thresh (union-find); return centroids."""
    n = len(points)
    if n == 0:
        return np.zeros((0, 2))
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(points[i] - points[j]) <= dist_thresh:
                a, b = find(i), find(j)
                if a != b:
                    parent[a] = b
    from collections import defaultdict
    comps = defaultdict(list)
    for i in range(n):
        comps[find(i)].append(i)
    return np.array([np.mean([points[i] for i in m], axis=0) for m in comps.values()])


def match_score(gt, pred, radius):
    """Greedy optimal matching within radius. Returns (tp, fp, fn, mean_tp_dist)."""
    if len(gt) == 0 and len(pred) == 0:
        return 0, 0, 0, []
    if len(gt) == 0:
        return 0, len(pred), 0, []
    if len(pred) == 0:
        return 0, 0, len(gt), []
    D = np.linalg.norm(gt[:, None, :] - pred[None, :, :], axis=2)
    big = radius * 10
    cost = np.where(D <= radius, D, big)
    ri, ci = linear_sum_assignment(cost)
    tp = 0
    dists = []
    for r, c in zip(ri, ci):
        if D[r, c] <= radius:
            tp += 1
            dists.append(D[r, c])
    fn = len(gt) - tp
    fp = len(pred) - tp
    return tp, fp, fn, dists


def run(frames_gt, det_per_frame, radius, fuse_dist):
    out = {}
    for mode in ("nofuse", "fuse"):
        TP = FP = FN = 0
        all_d = []
        for gt, preds in zip(frames_gt, det_per_frame):
            pred = fuse_points(preds, fuse_dist) if mode == "fuse" else np.array(preds).reshape(-1, 2)
            tp, fp, fn, dists = match_score(gt, pred, radius)
            TP += tp; FP += fp; FN += fn; all_d += dists
        n_gt = TP + FN
        moda = 1 - (FN + FP) / n_gt if n_gt else 0.0
        modp = (1 - np.mean(all_d) / radius) if all_d else 0.0
        prec = TP / (TP + FP) if (TP + FP) else 0.0
        rec = TP / (TP + FN) if (TP + FN) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        out[mode] = dict(MODA=moda, MODP=modp, precision=prec, recall=rec, f1=f1,
                         TP=TP, FP=FP, FN=FN)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/Wildtrack_dataset")
    ap.add_argument("--cameras", default="1,2,3,4,5,6,7")
    ap.add_argument("--max-frames", type=int, default=100)
    ap.add_argument("--model", default="yolo26s")
    ap.add_argument("--model-path", default="models/yolo26s.pt")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--tile", default="", help="SAHI-style tiling 'RxC' e.g. 2x3 (empty=off)")
    ap.add_argument("--radius", type=float, default=50.0, help="match radius (cm)")
    ap.add_argument("--mask-region", action="store_true",
                    help="drop predictions outside the GT-annotated ground region")
    ap.add_argument("--region-margin", type=float, default=200.0,
                    help="margin (cm) added to the GT region bbox when masking")
    ap.add_argument("--fuse-dist", type=float, default=100.0, help="cross-cam fuse dist (cm)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output", default="outputs/eval_ground.json")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cameras = [int(c) for c in args.cameras.split(",")]
    device = resolve_device(args.device)
    tile_rc = None
    if args.tile:
        r, c = args.tile.lower().split("x")
        tile_rc = (int(r), int(c))
    calib = WildtrackCalibration.from_dir(str(Path(args.root) / "calibrations"))

    source = from_wildtrack(args.root, cameras=cameras, max_frames=args.max_frames)
    frames_gt, n = gt_ground_positions(args.root, calib, set(cameras), args.max_frames)
    logger.info("GT frames: %d", n)

    det_cfg = DetectorConfig(model_type=args.model, model_path=args.model_path,
                             confidence_threshold=args.conf, classes=[0],
                             device=device, fp16=True, input_size=(args.imgsz, args.imgsz))
    detector = Detector(det_cfg)

    det_per_frame = []
    fidx = 0
    with source:
        while fidx < n:
            frames = source.read()
            if frames is None:
                break
            cids = list(frames.keys())
            if tile_rc:
                det_lists = [detector.detect_tiled(frames[c].image, tile_rc[0], tile_rc[1])
                             for c in cids]
            else:
                det_lists = detector.detect_batch([frames[c].image for c in cids])
            pts = []
            for c, dets in zip(cids, det_lists):
                for d in dets:
                    g = calib.foot_to_ground(c, d.bbox)
                    if g is not None and np.all(np.isfinite(g)):
                        pts.append(g)
            det_per_frame.append(np.array(pts).reshape(-1, 2))
            fidx += 1
            if fidx % 25 == 0:
                logger.info("processed %d/%d", fidx, n)

    # Optional: drop predictions outside the GT-annotated ground region. The
    # detector fires on people/spectators outside the WILDTRACK area (structural
    # false positives); masking to the GT region converts recall gains from
    # higher imgsz / tiling into precision instead of FP.
    if args.mask_region:
        allg = np.vstack([g for g in frames_gt if len(g)])
        lo = allg.min(0) - args.region_margin
        hi = allg.max(0) + args.region_margin
        def inside(pts):
            if len(pts) == 0:
                return pts
            m = np.all((pts >= lo) & (pts <= hi), axis=1)
            return pts[m]
        det_per_frame = [inside(p) for p in det_per_frame]
        logger.info("region mask: x[%.0f,%.0f] y[%.0f,%.0f] cm", lo[0], hi[0], lo[1], hi[1])

    res = run(frames_gt, det_per_frame, args.radius, args.fuse_dist)

    print(f"\n=== Ground-plane eval (WILDTRACK, {n} frames, conf={args.conf}, "
          f"imgsz={args.imgsz}, radius={args.radius}cm) ===")
    print(f"{'mode':10s} {'MODA':>8s} {'MODP':>8s} {'prec':>8s} {'recall':>8s} "
          f"{'F1':>8s} {'TP':>7s} {'FP':>7s} {'FN':>7s}")
    for mode in ("nofuse", "fuse"):
        m = res[mode]
        print(f"{mode:10s} {m['MODA']:8.3f} {m['MODP']:8.3f} {m['precision']:8.3f} "
              f"{m['recall']:8.3f} {m['f1']:8.3f} {m['TP']:7d} {m['FP']:7d} {m['FN']:7d}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(
        {"frames": n, "cameras": cameras, "conf": args.conf, "imgsz": args.imgsz,
         "radius": args.radius, "fuse_dist": args.fuse_dist, "device": device,
         "results": res}, indent=2))
    logger.info("saved %s", args.output)


if __name__ == "__main__":
    main()
