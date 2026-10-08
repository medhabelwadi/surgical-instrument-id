#!/usr/bin/env python3
"""Fine-tune an ImageNet-pretrained ResNet-18 to classify surgical instruments.

Expects class folders of images:
    root/
      Forceps/*.jpg
      Hemostat/*.jpg
      ...

    python train_classifier.py --data <root>
"""

import argparse
import json
import random
import re
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


def to_pil(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------

def numeric_prefix(path: Path) -> int:
    """Leading integer in the filename.

    The Mendeley images are named with contiguous per-class ID ranges from one
    capture session, so a RANDOM split would put near-duplicate frames on both
    sides and inflate accuracy. Split on this instead.
    """
    m = re.match(r"(\d+)", path.name)
    return int(m.group(1)) if m else -1


class InstrumentSet(Dataset):
    def __init__(self, items, train, boxes=None):
        self.items = items
        self.boxes = boxes or {}
        aug = [transforms.RandomHorizontalFlip(),
               transforms.RandomVerticalFlip(),
               transforms.RandomRotation(180),
               transforms.ColorJitter(0.3, 0.3, 0.2)] if train else []
        self.tf = transforms.Compose(
            [transforms.Resize((224, 224))] + aug +
            [transforms.ToTensor(),
             transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
        )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        bgr = cv2.imread(str(path))
        if bgr is None:
            bgr = np.zeros((224, 224, 3), np.uint8)
        key = f"{path.parent.name}/{path.name}"
        if key in self.boxes:
            x0, y0, x1, y1 = self.boxes[key]
            if x1 - x0 > 16 and y1 - y0 > 16:
                bgr = bgr[y0:y1, x0:x1]
        return self.tf(to_pil(bgr)), label


def build_splits(root: Path, val_frac=0.2, boxes=None):
    classes = sorted(d.name for d in root.iterdir()
                     if d.is_dir() and d.name.lower() != "overlapping")
    train_items, val_items = [], []
    for ci, c in enumerate(classes):
        files = sorted((root / c).glob("*.jpg"), key=numeric_prefix)
        cut = int(len(files) * (1 - val_frac))
        train_items += [(f, ci) for f in files[:cut]]
        val_items += [(f, ci) for f in files[cut:]]
    print(f"{len(classes)} classes | {len(train_items)} train | {len(val_items)} val")
    print("  (split by filename ID range, not randomly -- these images come "
          "from one capture session and random splits leak near-duplicates)")
    return (InstrumentSet(train_items, True, boxes),
            InstrumentSet(val_items, False, boxes),
            classes)


# ----------------------------------------------------------------------------
# model
# ----------------------------------------------------------------------------

class ResNet18Classifier(nn.Module):
    """Single-view ResNet-18 classifier."""

    def __init__(self, n_classes):
        super().__init__()
        # Keep the single-trunk checkpoint keys compatible with existing
        # full-image checkpoints and the Gradio demo.
        m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Identity()
        self.trunks = nn.ModuleList([m])
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):  # Training: (B, 3, 224, 224); demo: (B, 1, 3, 224, 224)
        if x.ndim == 5:
            x = x[:, 0]
        return self.head(self.trunks[0](x))


# ----------------------------------------------------------------------------

def evaluate(model, loader, classes, device):
    model.eval()
    cm = np.zeros((len(classes), len(classes)), int)
    with torch.no_grad():
        for x, y in loader:
            p = model(x.to(device)).argmax(1).cpu()
            for t, q in zip(y.tolist(), p.tolist()):
                cm[t, q] += 1
    return cm.trace() / cm.sum(), cm


def report(acc, cm, classes):
    print(f"\ntop-1 accuracy: {acc:.3f}   (chance {1/len(classes):.3f})")
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="classifier.pt")
    ap.add_argument("--boxes", type=Path, default=None,
                    help="boxes.json from localize_boxes.py")
    args = ap.parse_args()

    torch.manual_seed(0); random.seed(0); np.random.seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    boxes = json.loads(args.boxes.read_text()) if args.boxes else None
    if boxes:
        print(f"cropping with {len(boxes)} cached boxes")
    train_set, val_set, classes = build_splits(args.data, boxes=boxes)

    train_loader = DataLoader(train_set, args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_set, args.batch, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    model = ResNet18Classifier(len(classes)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)

    best = 0.0
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
            best = acc
            torch.save({"state_dict": model.state_dict(),
                        "classes": classes, "mode": "full"}, args.out)
            best_cm = cm

    print(f"\nbest val accuracy: {best:.4f}  -> {args.out}")
    report(best, best_cm, classes)


if __name__ == "__main__":
    main()
