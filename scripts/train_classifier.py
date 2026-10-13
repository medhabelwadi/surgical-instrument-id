#!/usr/bin/env python3
"""
Instrument classifier: fine-tuned ResNet-18, with an optional part-aware
cascade head.

Splits are CONTIGUOUS ID RANGES, not random. The Mendeley images come from one
capture session with per-class sequential numbering, so neighbouring files are
near-duplicate frames. A random split puts those neighbours on both sides and
inflates accuracy -- this is the most likely explanation for the 99.3% mAP
reported in the SID-RAS paper. Contiguous blocks keep near-duplicates together:

    [  0% ------- 65% ] train
    [ 65% ------- 80% ] test   (held out, reported once at the end)
    [ 80% ------ 100% ] val    (model selection; furthest from train, strictest)

    python train_classifier.py --data <root> --mode full --epochs 40
    python train_classifier.py --data <root> --mode cascade --boxes boxes.json

Note on --mode cascade: measured worse than --mode full on this data (see
docs/findings.md experiment 4). The part crops land on drape unless --boxes is
supplied, and even with boxes, cropping cost ~20 points -- apparent size
appears to carry class information.
"""

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ----------------------------------------------------------- part extraction

def instrument_mask(bgr: np.ndarray):
    """Foreground mask via Otsu, largest connected component.

    Works poorly on this dataset -- the instruments are thin and specular and
    the drape has its own lighting gradient -- which is why --boxes exists.
    Returns None when the result is degenerate so callers can fall back.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if mask.mean() > 127:
        mask = 255 - mask
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return None
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    mask = (labels == largest).astype(np.uint8) * 255
    frac = mask.mean() / 255.0
    return None if frac < 0.005 or frac > 0.95 else mask


def end_crops(bgr: np.ndarray, mask: np.ndarray, win_frac=0.35):
    """Square crops at each extreme of the principal axis, ordered by the mask
    area inside them -- a deterministic proxy for head/tail, not a semantic
    label."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 50:
        return None
    pts = np.stack([xs, ys], 1).astype(np.float32)
    centre = pts.mean(0)
    centred = pts - centre
    eigvals, eigvecs = np.linalg.eigh(np.cov(centred.T))
    axis = eigvecs[:, np.argmax(eigvals)]

    proj = centred @ axis
    lo, hi = np.percentile(proj, 5), np.percentile(proj, 95)
    if hi - lo < 20:
        return None

    win = max(32, int((hi - lo) * win_frac))
    H, W = mask.shape
    out = []
    for t in (lo, hi):
        cx, cy = centre + axis * t
        x0 = int(np.clip(cx - win / 2, 0, W - 1))
        y0 = int(np.clip(cy - win / 2, 0, H - 1))
        x1 = int(np.clip(x0 + win, 0, W))
        y1 = int(np.clip(y0 + win, 0, H))
        if x1 - x0 < 16 or y1 - y0 < 16:
            return None
        out.append((bgr[y0:y1, x0:x1], mask[y0:y1, x0:x1].mean()))
    out.sort(key=lambda p: -p[1])
    return out[0][0], out[1][0]


def pad_to_square(img: Image.Image, fill=(0, 0, 0)):
    """Pad rather than stretch -- instruments are elongated, and a square
    resize distorts them differently depending on their angle."""
    w, h = img.size
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), fill)
    canvas.paste(img, ((side - w) // 2, (side - h) // 2))
    return canvas


def to_pil(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


# --------------------------------------------------------------------- data

def numeric_prefix(path: Path) -> int:
    m = re.match(r"(\d+)", path.name)
    return int(m.group(1)) if m else -1


class InstrumentSet(Dataset):
    def __init__(self, items, classes, mode, train, dump_dir=None, boxes=None):
        self.items, self.classes, self.mode = items, classes, mode
        self.boxes = boxes or {}
        self.dump_dir = dump_dir
        self.dumped = 0
        aug = [transforms.RandomHorizontalFlip(),
               transforms.RandomVerticalFlip(),
               transforms.RandomRotation(180),
               transforms.ColorJitter(0.3, 0.3, 0.2)] if train else []
        self.tf = transforms.Compose(
            [transforms.Resize((224, 224))] + aug +
            [transforms.ToTensor(),
             transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        bgr = cv2.imread(str(path))
        if bgr is None:
            bgr = np.zeros((224, 224, 3), np.uint8)

        # crop before anything else, so cascade end-crops come from the
        # instrument rather than the full frame
        key = f"{path.parent.name}/{path.name}"
        if key in self.boxes:
            x0, y0, x1, y1 = self.boxes[key]
            if x1 - x0 > 16 and y1 - y0 > 16:
                bgr = bgr[y0:y1, x0:x1]

        views = [to_pil(bgr)]
        if self.mode == "cascade":
            mask = instrument_mask(bgr)
            ends = end_crops(bgr, mask) if mask is not None else None
            if ends is None:
                ends = (bgr, bgr)      # segmentation failed; don't drop the sample
            views += [to_pil(ends[0]), to_pil(ends[1])]

            if self.dump_dir and self.dumped < 6:
                d = Path(self.dump_dir); d.mkdir(parents=True, exist_ok=True)
                stem = f"{self.classes[label]}_{path.stem[:14]}"
                for name, v in zip(("full", "endA", "endB"), views):
                    v.save(d / f"{stem}_{name}.png")
                self.dumped += 1

        return torch.stack([self.tf(v) for v in views]), label


def build_splits(root: Path, mode, val_frac=0.2, test_frac=0.15,
                 dump_dir=None, boxes=None):
    classes = sorted(d.name for d in root.iterdir()
                     if d.is_dir() and d.name.lower() != "overlapping")
    train_items, test_items, val_items = [], [], []
    for ci, c in enumerate(classes):
        files = sorted((root / c).glob("*.jpg"), key=numeric_prefix)
        n = len(files)
        a = int(n * (1 - val_frac - test_frac))     # 65%
        b = int(n * (1 - val_frac))                 # 80%
        train_items += [(f, ci) for f in files[:a]]
        test_items += [(f, ci) for f in files[a:b]]
        val_items += [(f, ci) for f in files[b:]]

    print(f"{len(classes)} classes | {len(train_items)} train | "
          f"{len(test_items)} test | {len(val_items)} val")
    print("  contiguous ID ranges, so near-duplicate frames stay on one side "
          "of each boundary")
    print("  val is furthest from train and is the strictest of the three")

    def mk(items, train):
        return InstrumentSet(items, classes, mode, train, dump_dir, boxes)

    return mk(train_items, True), mk(test_items, False), mk(val_items, False), classes


# -------------------------------------------------------------------- model

class PartAwareNet(nn.Module):
    """One ResNet-18 trunk per view; concatenated features -> classifier.

    With n_views == 1 this is just a fine-tuned ResNet-18 with a two-layer head.
    """

    def __init__(self, n_classes, n_views):
        super().__init__()
        self.trunks = nn.ModuleList()
        for _ in range(n_views):
            m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            m.fc = nn.Identity()
            self.trunks.append(m)
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512 * n_views, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):                       # x: (B, V, 3, 224, 224)
        feats = [t(x[:, i]) for i, t in enumerate(self.trunks)]
        return self.head(torch.cat(feats, dim=1))


# --------------------------------------------------------------- evaluation

def evaluate(model, loader, classes, device):
    model.eval()
    cm = np.zeros((len(classes), len(classes)), int)
    with torch.no_grad():
        for x, y in loader:
            p = model(x.to(device)).argmax(1).cpu()
            for t, q in zip(y.tolist(), p.tolist()):
                cm[t, q] += 1
    return cm.trace() / cm.sum(), cm


def report(acc, cm, classes, title=""):
    print(f"\n{title}top-1 accuracy: {acc:.3f}   (chance {1/len(classes):.3f})")
    print("\nper-class:")
    for i, c in enumerate(classes):
        tot = cm[i].sum()
        print(f"  {cm[i, i]/tot:.3f}  ({cm[i, i]:4d}/{tot:4d})  {c}")
    w = max(len(c) for c in classes)
    print("\nconfusion matrix (rows = truth):")
    print(" " * (w + 2) + "".join(f"{c[:6]:>8}" for c in classes))
    for i, c in enumerate(classes):
        print(f"{c:>{w}}  " + "".join(f"{v:>8d}" for v in cm[i]))
    off = [(cm[i, j], classes[i], classes[j])
           for i in range(len(classes)) for j in range(len(classes)) if i != j]
    print("\nmost-confused pairs:")
    for n, t, p in sorted(off, reverse=True)[:5]:
        if n:
            print(f"  {n:4d}  {t}  ->  {p}")


# -------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--mode", choices=["full", "cascade"], default="full")
    ap.add_argument("--boxes", type=Path, default=None,
                    help="boxes.json from localize_boxes.py")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--dump-crops", default=None,
                    help="write a few example crops here and stop")
    ap.add_argument("--out", default="classifier.pt")
    args = ap.parse_args()

    torch.manual_seed(0); random.seed(0); np.random.seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device} | mode: {args.mode}")

    boxes = json.loads(args.boxes.read_text()) if args.boxes else None
    if boxes:
        print(f"cropping with {len(boxes)} cached boxes")

    train_set, test_set, val_set, classes = build_splits(
        args.data, args.mode, args.val_frac, args.test_frac,
        args.dump_crops, boxes)
    n_views = 3 if args.mode == "cascade" else 1

    train_loader = DataLoader(train_set, args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(test_set, args.batch, shuffle=False,
                             num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_set, args.batch, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    if args.dump_crops:
        for _ in range(2):
            next(iter(train_loader))
        print(f"\nwrote sample crops to {args.dump_crops}/ -- LOOK AT THEM.")
        print("End crops must land on the jaws/handles. If they are blank "
              "drape, segmentation failed and the cascade cannot help.")
        return

    model = PartAwareNet(len(classes), n_views).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)

    best = 0.0
    best_cm = None
    for ep in range(1, args.epochs + 1):
        model.train()
        tot = n = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            tot += loss.item() * y.size(0); n += y.size(0)
        sched.step()

        acc, cm = evaluate(model, val_loader, classes, device)
        print(f"epoch {ep:3d}  loss {tot/n:.4f}  val {acc:.4f}")
        if acc > best:
            best, best_cm = acc, cm
            torch.save({"state_dict": model.state_dict(),
                        "classes": classes, "mode": args.mode}, args.out)

    # evaluate the SELECTED checkpoint on the held-out test block, not
    # whatever the last epoch happened to produce
    ck = torch.load(args.out, map_location=device, weights_only=False)
    model.load_state_dict(ck["state_dict"])
    test_acc, test_cm = evaluate(model, test_loader, classes, device)

    print(f"\n{'='*60}")
    print(f"val  (used for model selection) : {best:.4f}")
    print(f"test (held out, reported once)  : {test_acc:.4f}")
    print(f"{'='*60}")
    report(test_acc, test_cm, classes, title="TEST — ")
    print(f"\nweights: {args.out}")


if __name__ == "__main__":
    main()