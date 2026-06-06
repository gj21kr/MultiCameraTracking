# Performance — measurement, results, and honest limits

This documents how tracking quality is measured on WILDTRACK and what the
quality changes actually moved. Everything here is reproducible with the
`scripts/evaluate*.py` harnesses; all numbers are CPU runs (`torch 2.x+cpu`,
`yolo26s`).

## 1. Why two metrics

WILDTRACK ships two kinds of ground truth:

- **Per-view 2D boxes** (`annotations_positions/*.json`) — but these are
  *projected ground-plane cylinders assuming a fixed human size*, **not** tight
  visual detection boxes. Evaluating tight YOLO boxes against them at IoU≥0.5
  produces large IoU-mismatch false positives, and YOLO also fires on people
  outside the annotated ground region. **Per-camera 2D MOTA is therefore
  misleading on WILDTRACK** — useful only for *relative* before/after signals.
- **Ground-plane positions** — the dataset's intended evaluation space. We
  project each detection's foot point (bbox bottom-center) to world coordinates
  via the camera calibration and match to GT positions. This is the metric to
  trust (MODA / MODP / precision / recall).

Harnesses:
- `scripts/evaluate.py` — per-camera 2D MOTA/IDF1/ID-switches (relative signal).
- `scripts/evaluate_ground.py` — ground-plane MODA/MODP with a cross-camera
  fusion on/off ablation.

## 2. Calibration sanity check

Projecting each GT person's *visible-view* foot points to the ground plane and
measuring their spread (should be ~0 for a perfect system):

| metric | value |
|---|---|
| mean intra-person ground spread | 82 cm |
| median | 61 cm |
| 90th pct | 171 cm |

The ~80 cm spread is the floor set by estimating a world position from a 2D
bbox bottom-center. It is the dominant error source for ground-plane matching
and bounds how tight the match radius / fusion distance can be.

## 3. ② Cross-camera geometric fusion (ground plane)

Project all 7 cameras' detections to the ground, then fuse detections within
`--fuse-dist` into one world detection (union-find), vs. no fusion.
(20 frames, conf 0.3, radius 50 cm, fuse 100 cm.)

| mode | MODA | precision | recall | FP |
|---|---|---|---|---|
| no fusion | −1.89 | 0.195 | 0.603 | 1585 |
| **fusion (②)** | **−0.87** | **0.223** | 0.352 | **779** |

At 100 frames (conf 0.4, radius 75 cm, fuse 100 cm) the result holds:

| mode | MODA | precision | recall | F1 | FP |
|---|---|---|---|---|---|
| no fusion | −2.02 | 0.209 | 0.725 | 0.324 | 5915 |
| **fusion (②)** | **−0.96** | **0.255** | 0.502 | **0.338** | **3154** |

Fusion **roughly halves false positives** (5915→3154) and improves MODA/F1,
because the same person seen by 7 cameras collapses to one world detection
instead of seven. The recall drop is the known tension: at the ~80 cm
projection-noise scale, a fuse distance large enough to merge a person's
cross-camera points also merges genuinely distinct nearby people. Better foot
estimation is the lever (see §6).

## 4. ③ Real Kalman filter + association fix

Two defects fixed in `src/tracking/tracker.py` / `config.py`:
- the "Kalman filter" was a fixed-gain α=0.5 smoother → replaced with a proper
  constant-velocity Kalman filter (state `[cx,cy,w,h,vx,vy,vw,vh]`, covariance).
- `match_thresh` (min IoU to accept a match) was **0.8**, far too strict →
  set to **0.3** (ByteTrack-style). At 0.8, any real motion broke association.

**Controlled test** (single object moving 12 px/frame) — unambiguous correctness:

| | before | after (③) |
|---|---|---|
| track identity | new id every frame | **single stable id** |
| velocity estimate | n/a | **vx ≈ 12.0 (exact)** |

**Per-camera 2D, 100 frames** (noisy metric — relative only). Three configs:
`baseline` (match 0.8, buffer 30), `③` (match 0.3, buffer 30, conf 0.3),
`tuned` (conf 0.5, match 0.5, buffer 8):

| metric | baseline | ③ | **tuned** |
|---|---|---|---|
| recall | 0.177 | **0.449** | 0.233 |
| precision | 0.108 | 0.122 | **0.265** |
| IDF1 | 0.091 | 0.106 | **0.180** |
| num_false_positives | 11204 | 24752 | **4947** |
| num_switches | 165 | 1090 | **233** |
| MODA | −1.31 | −2.93 | **−0.445** |

Honest reading:
- ③ alone gives **2.5× recall** (tracks persist instead of dying every frame),
  but inflates FP / ID-switches because more hypotheses are scored against
  projected GT boxes they cannot tightly overlap — a metric artifact, not a
  tracker defect (the controlled test shows the tracker is correct).
- **Tuning detection + track lifecycle** (conf↑, match↑, buffer↓) cuts FP **−80%**
  (24752→4947) and switches **−79%** (1090→233), and **doubles IDF1** vs
  baseline (0.091→0.180) — at the cost of recall (high conf drops weak
  detections). Clear precision↔recall trade-off.
- To get **both**: put appearance (ReID) into the association cost so IDs hold
  without needing high conf, and/or raise `imgsz` to recover small-pedestrian
  recall (§6).

**Appearance in association (D).** Fusing OSNet embeddings into the matching
cost (accept a match when IoU *or* appearance is convincing, `--reid`),
conf 0.3, 100 frames:

| metric | IoU-only (③) | +appearance (D) |
|---|---|---|
| num_switches | 1090 | **820** (−25%) |
| recall | 0.449 | 0.421 |
| IDF1 | 0.106 | **0.117** |
| num_false_positives | 24752 | **21209** |

Honest reading: appearance cuts ID-switches ~25% at flat recall — a real but
**marginal** gain. It does not change the verdict: recall ~0.42 means ~58% of
pedestrians are still missed. The ceiling of association-level tuning is low;
the bottleneck is **detection recall** and the **detection-level (vs
feature-level) fusion architecture** — see §6.

Reproduce:
```bash
python scripts/evaluate.py --root data/Wildtrack_dataset --max-frames 100 \
    --conf 0.5 --match-thresh 0.5 --track-buffer 8
```

## 5. Speed

CPU, `yolo26s`, 1080×1920: ~0.9 fps end-to-end (7 cameras). Offline demo
generation only; not real-time. Levers: GPU (≫10×), OpenVINO/ONNX CPU export
(2–4×), smaller model (`yolo26n`), lower `imgsz`.

## 6. What would move the trustworthy (ground-plane) metric next

1. **Better foot-point estimation** — bottom-center has ~80 cm world error.
   Use the calibration to back-project a foot-line / person-height prior, or a
   pose keypoint (ankles). This is the single highest-leverage fix.
2. **Restrict to the annotated ground region** — mask detections outside the
   WILDTRACK area to cut false positives that are real people but un-annotated.
3. **Higher `--imgsz` (1280)** — recovers small/distant pedestrians (recall),
   at a speed cost.
4. **Ground-plane tracking + IDF1** — track fused world detections over time
   (Kalman on the ground plane) and report ground IDF1, so ③'s temporal value
   is measured in the correct space.
5. **BoxMOT (BoT-SORT)** per camera as a stronger drop-in if per-camera quality
   is the bottleneck.

All numbers above are reproducible:

```bash
python scripts/evaluate.py        --root data/Wildtrack_dataset --max-frames 100 --conf 0.3
python scripts/evaluate_ground.py --root data/Wildtrack_dataset --max-frames 100 --conf 0.4 --radius 75
```
