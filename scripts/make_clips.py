#!/usr/bin/env python3
"""
Synthetic camera motion over still instrument photos, and temporal evaluation.

Why bother: per-frame accuracy is not the metric that matters for a robot. A
classifier that alternates between "Hemostat" and "Mayo" as the camera drifts
is unusable even at 92% per-frame. This measures FLICKER -- how often the
prediction changes while the ground truth does not -- and shows how much
temporal majority voting recovers.

What this is NOT: real camera motion. It is a 2D warp of a fixed photograph,
so there is no parallax, no genuine change of viewpoint, no rolling shutter,
and the motion blur is synthetic. It tests robustness to framing, scale and
blur. It does not test robustness to seeing the instrument from a new angle.

    # one clip, single instrument, constant ground truth
    python make_clips.py --data <root> --mode single --out clip.mp4

    # pan across a tiled tray, label changes as the camera moves
    python make_clips.py --data <root> --mode tray --out tray.mp4

    # generate many clips and report flicker / smoothing gain
    python make_clips.py --data <root> --weights classifier.pt --evaluate 40
"""

import argparse
import random
import re
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
OUT = 640          # output frame size, matches the classifier's training size


# ------------------------------------------------------------------ model

class PartAwareNet(nn.Module):
    def __init__(self, n_classes, n_views):
        super().__init__()
        self.trunks = nn.ModuleList()
        for _ in range(n_views):
            m = resnet18(weights=None)
            m.fc = nn.Identity()
            self.trunks.append(m)
        self.head = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(512 * n_views, 256),
            nn.ReLU(inplace=True), nn.Dropout(0.3), nn.Linear(256, n_classes))

    def forward(self, x):
        return self.head(torch.cat([t(x[:, i]) for i, t in enumerate(self.trunks)], 1))


class Classifier:
    def __init__(self, weights, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ck = torch.load(weights, map_location=self.device, weights_only=False)
        self.classes = ck["classes"]
        self.n_views = 3 if ck.get("mode") == "cascade" else 1
        self.model = PartAwareNet(len(self.classes), self.n_views).to(self.device)
        self.model.load_state_dict(ck["state_dict"])
        self.model.eval()
        self.tf = transforms.Compose([
            transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

    def predict_batch(self, rgb_frames, batch=32):
        out = []
        for i in range(0, len(rgb_frames), batch):
            xs = torch.stack([self.tf(Image.fromarray(f))
                              for f in rgb_frames[i:i + batch]])
            xs = xs.unsqueeze(1).repeat(1, self.n_views, 1, 1, 1)
            with torch.no_grad():
                p = torch.softmax(self.model(xs.to(self.device)), 1).cpu()
            out.append(p)
        probs = torch.cat(out)
        return [(self.classes[i], float(probs[j, i]))
                for j, i in enumerate(probs.argmax(1))]


# ------------------------------------------------------------- motion

def camera_path(n_frames, w, h, kind, rng):
    """Yield (cx, cy, scale, angle) per frame.

    scale < 1 means the crop window is smaller than the source, i.e. zoomed in.
    """
    if kind == "single":
        # drift across the middle of the frame with a slow zoom and tilt
        x0, x1 = w * rng.uniform(0.35, 0.45), w * rng.uniform(0.55, 0.65)
        y0, y1 = h * rng.uniform(0.40, 0.50), h * rng.uniform(0.50, 0.60)
        s0, s1 = rng.uniform(0.80, 0.95), rng.uniform(0.55, 0.75)
        a0, a1 = rng.uniform(-8, 8), rng.uniform(-8, 8)
    else:                                   # pan left to right over a tray
        x0, x1 = w * 0.08, w * 0.92
        y0 = y1 = h * 0.5
        s0 = s1 = min(1.0, h / max(h, 1)) * 0.95
        a0 = a1 = 0.0

    for i in range(n_frames):
        t = i / max(1, n_frames - 1)
        e = t * t * (3 - 2 * t)             # ease in/out, less robotic
        yield (x0 + (x1 - x0) * e,
               y0 + (y1 - y0) * e,
               s0 + (s1 - s0) * e,
               a0 + (a1 - a0) * e)


def render(src, cx, cy, scale, angle, out=OUT, blur_px=0, jitter=0.0, rng=None):
    """Crop a rotated, scaled window centred at (cx, cy) and resize to out."""
    h = src.shape[0]
    win = max(32, int(h * scale))

    M = cv2.getRotationMatrix2D((cx, cy), angle, out / win)
    M[0, 2] += out / 2 - cx
    M[1, 2] += out / 2 - cy
    frame = cv2.warpAffine(src, M, (out, out), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)

    if blur_px >= 3:                        # directional blur along the motion
        k = np.zeros((blur_px, blur_px), np.float32)
        k[blur_px // 2, :] = 1.0 / blur_px
        frame = cv2.filter2D(frame, -1, k)

    if jitter and rng:
        frame = cv2.convertScaleAbs(frame,
                                    alpha=1 + rng.uniform(-jitter, jitter),
                                    beta=rng.uniform(-12, 12) * jitter * 10)
    return frame


def numeric_prefix(p: Path) -> int:
    m = re.match(r"(\d+)", p.name)
    return int(m.group(1)) if m else -1


class Source:
    """Held-out stills, same split the classifier was validated on."""
    def __init__(self, root: Path, lo=0.65, hi=0.80):
        self.classes = sorted(d.name for d in root.iterdir()
                              if d.is_dir() and d.name.lower() != "overlapping")
        self.files = {}
        for c in self.classes:
            fs = sorted((root / c).glob("*.jpg"), key=numeric_prefix)
            n = len(fs)
            self.files[c] = fs[int(n * lo):int(n * hi)]   # test block
        print(f"source: {sum(len(v) for v in self.files.values())} test-block images")

    def single(self, rng, cls=None):
        cls = cls or rng.choice(self.classes)
        img = cv2.imread(str(rng.choice(self.files[cls])))
        return cv2.resize(img, (OUT, OUT)), cls

    def tray(self, rng, n=5):
        picks = [rng.choice(self.classes) for _ in range(n)]
        canvas = np.zeros((OUT, OUT * n, 3), np.uint8)
        for i, c in enumerate(picks):
            img = cv2.imread(str(rng.choice(self.files[c])))
            canvas[:, i * OUT:(i + 1) * OUT] = cv2.resize(img, (OUT, OUT))
        return canvas, picks


def make_clip(src_bgr, kind, n_frames, rng, blur=True, jitter=0.06):
    """Return a list of BGR frames."""
    h, w = src_bgr.shape[:2]
    path = list(camera_path(n_frames, w, h, kind, rng))
    frames = []
    for i, (cx, cy, s, a) in enumerate(path):
        if blur and 0 < i < len(path) - 1:
            dx = abs(path[i + 1][0] - path[i - 1][0]) / 2
            bp = int(np.clip(dx / 6, 0, 11)) | 1      # odd kernel
        else:
            bp = 0
        frames.append(render(src_bgr, cx, cy, s, a, blur_px=bp,
                             jitter=jitter, rng=rng))
    return frames


def write_video(frames, path, fps=20):
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        vw.write(f)
    vw.release()


# ---------------------------------------------------------- evaluation

def smooth(labels, k):
    """Majority vote over a centred window of k frames."""
    out, half = [], k // 2
    for i in range(len(labels)):
        window = labels[max(0, i - half): i + half + 1]
        out.append(Counter(window).most_common(1)[0][0])
    return out


def evaluate(src: Source, clf: Classifier, n_clips, n_frames, seed=0):
    rng = random.Random(seed)
    per_frame_hits = total = 0
    switches = transitions = 0
    smoothed_hits = {k: 0 for k in (5, 9, 15)}

    for _ in range(n_clips):
        bgr, truth = src.single(rng)
        frames = make_clip(bgr, "single", n_frames, rng)
        rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
        preds = [p for p, _ in clf.predict_batch(rgb)]

        per_frame_hits += sum(p == truth for p in preds)
        total += len(preds)
        switches += sum(a != b for a, b in zip(preds, preds[1:]))
        transitions += len(preds) - 1
        for k in smoothed_hits:
            smoothed_hits[k] += sum(p == truth for p in smooth(preds, k))

    print(f"\n{n_clips} clips x {n_frames} frames, ground truth constant "
          f"within each clip\n")
    print(f"  per-frame accuracy   {per_frame_hits / total:.3f}")
    print(f"  flicker rate         {switches / transitions:.3f}"
          f"   (fraction of consecutive frames where the label changes)")
    for k, hits in smoothed_hits.items():
        print(f"  majority vote k={k:<3d}    {hits / total:.3f}")
    print("\nFlicker matters more than per-frame accuracy for a robot: a label "
          "that oscillates cannot be acted on.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--mode", choices=["single", "tray"], default="tray")
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--out", default="clip.mp4")
    ap.add_argument("--evaluate", type=int, default=0,
                    help="run the flicker experiment over this many clips")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    src = Source(args.data)
    rng = random.Random(args.seed)

    if args.evaluate:
        if not args.weights:
            raise SystemExit("--evaluate needs --weights")
        evaluate(src, Classifier(args.weights), args.evaluate, args.frames,
                 args.seed)
        return

    if args.mode == "single":
        bgr, truth = src.single(rng)
        print(f"clip of: {truth}")
    else:
        bgr, picks = src.tray(rng)
        print("tray: " + ", ".join(picks))

    frames = make_clip(bgr, args.mode, args.frames, rng)

    if args.weights:                      # annotate with live predictions
        clf = Classifier(args.weights)
        preds = clf.predict_batch([cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                                   for f in frames])
        raw = [p for p, _ in preds]
        sm = smooth(raw, 9)
        for f, (p, c), s in zip(frames, preds, sm):
            cv2.rectangle(f, (0, 0), (OUT, 92), (20, 20, 20), -1)
            cv2.putText(f, f"{p}  {c:.0%}", (16, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2,
                        cv2.LINE_AA)
            cv2.putText(f, f"smoothed: {s}", (16, 74),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 230, 140), 2,
                        cv2.LINE_AA)

    write_video(frames, args.out)
    print(f"wrote {args.out}  ({len(frames)} frames)")


if __name__ == "__main__":
    main()