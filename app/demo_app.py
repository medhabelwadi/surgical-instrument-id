#!/usr/bin/env python3
"""
Click-to-identify demo on a synthetic tray.

A tray is built by tiling N single-instrument images side by side. The point of
tiling rather than compositing instruments onto a shared drape: each tile stays
a complete, unmodified training-distribution frame. Clicking a tile crops back
to exactly that frame, so the classifier sees what it was trained on.

This matters because cropping tightly around an instrument cost ~20 points of
accuracy (docs/findings.md, experiment 4) -- apparent size appears to carry
class information, so any reframing degrades the model.

Because tile identity is known, the demo shows predicted AND true labels.

    pip install gradio
    python demo_app.py --data /path/to/mendley-tools --weights classifier.pt

Honest limitation to state when presenting: a tiled composite is not a
cluttered tray. Instruments do not overlap and each keeps its own background.
This demonstrates the click-to-identify interaction and the classifier, not
cluttered-scene handling. Real overlapping trays are the `Overlapping` split,
and are a separate experiment.
"""

import argparse
import random
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
TILE = 640          # the frame size the classifier was trained on


class PartAwareNet(nn.Module):
    """Must match train_classifier.py."""

    def __init__(self, n_classes, n_views):
        super().__init__()
        self.trunks = nn.ModuleList()
        for _ in range(n_views):
            m = resnet18(weights=None)
            m.fc = nn.Identity()
            self.trunks.append(m)
        self.head = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(512 * n_views, 256),
            nn.ReLU(inplace=True), nn.Dropout(0.3), nn.Linear(256, n_classes),
        )

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
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        print(f"{len(self.classes)} classes | {self.n_views} view(s) | {self.device}")

    def predict(self, rgb_tile):
        v = self.tf(Image.fromarray(rgb_tile))
        x = v.unsqueeze(0).repeat(1, self.n_views, 1, 1, 1) if self.n_views > 1 \
            else v.unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            p = torch.softmax(self.model(x.to(self.device)), 1)[0].cpu()
        order = p.argsort(descending=True)
        return [(self.classes[i], float(p[i])) for i in order]


class TrayBuilder:
    """Tiles single-instrument frames into a synthetic tray."""

    def __init__(self, root: Path):
        self.root = root
        self.classes = sorted(d.name for d in root.iterdir()
                              if d.is_dir() and d.name.lower() != "overlapping")
        self.files = {c: sorted((root / c).glob("*.jpg")) for c in self.classes}
        print(f"tray source: {len(self.classes)} classes, "
              f"{sum(len(v) for v in self.files.values())} images")

    def build(self, n_tools, rows, seed=None):
        rng = random.Random(seed)
        picks = rng.sample(self.classes, min(n_tools, len(self.classes)))
        while len(picks) < n_tools:                 # allow repeats if asked
            picks.append(rng.choice(self.classes))
        rng.shuffle(picks)

        cols = int(np.ceil(len(picks) / rows))
        canvas = np.zeros((rows * TILE, cols * TILE, 3), np.uint8)
        tiles = []                                   # (x0, y0, x1, y1, label)

        for i, cls in enumerate(picks):
            r, c = divmod(i, cols)
            img = cv2.imread(str(rng.choice(self.files[cls])))
            if img is None:
                continue
            img = cv2.resize(img, (TILE, TILE))
            y0, x0 = r * TILE, c * TILE
            canvas[y0:y0 + TILE, x0:x0 + TILE] = img
            tiles.append((x0, y0, x0 + TILE, y0 + TILE, cls))

        return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), tiles


def build_ui(clf: Classifier, trays: TrayBuilder):
    def new_tray(n_tools, rows, st):
        rgb, tiles = trays.build(int(n_tools), int(rows))
        st = {"tiles": tiles, "image": rgb}
        return rgb, st, "Click an instrument."

    def on_click(st, evt: gr.SelectData):
        if not st or st["image"] is None:
            return None, "Generate a tray first."
        x, y = evt.index
        rgb = st["image"]

        hit = next((t for t in st["tiles"]
                    if t[0] <= x < t[2] and t[1] <= y < t[3]), None)
        if hit is None:
            return rgb, "Clicked outside the tray."

        x0, y0, x1, y1, truth = hit
        ranked = clf.predict(rgb[y0:y1, x0:x1])
        name, conf = ranked[0]
        ok = name == truth

        vis = rgb.copy()
        colour = (60, 220, 60) if ok else (240, 80, 80)
        cv2.rectangle(vis, (x0 + 4, y0 + 4), (x1 - 4, y1 - 4), colour, 10)
        cv2.circle(vis, (x, y), 14, (255, 255, 255), -1)
        cv2.circle(vis, (x, y), 10, (40, 40, 40), -1)

        lines = [f"## {name}", f"**{conf:.1%} confident**", "",
                 f"{'correct' if ok else 'INCORRECT'} — ground truth: *{truth}*",
                 "", "Runners-up:"]
        lines += [f"- {n} — {p:.1%}" for n, p in ranked[1:4]]
        return vis, "\n".join(lines)

    with gr.Blocks(title="Surgical Instrument ID") as ui:
        state = gr.State({"tiles": [], "image": None})
        gr.Markdown(
            "# Surgical Instrument Identification\n"
            "Generate a tray, then **click an instrument** to identify it.\n\n"
            "*Tiled composite — each tile is an unmodified frame from the "
            "held-out set, so the classifier sees its training distribution. "
            "Overlapping instruments are a separate, harder problem.*"
        )
        with gr.Row():
            n_tools = gr.Slider(2, 7, value=4, step=1, label="Instruments")
            rows = gr.Slider(1, 3, value=2, step=1, label="Rows")
            go = gr.Button("Generate tray", variant="primary")
        with gr.Row():
            with gr.Column(scale=3):
                img = gr.Image(type="numpy", label="Tray", interactive=False)
            with gr.Column(scale=2):
                out = gr.Markdown("Generate a tray to begin.")

        go.click(new_tray, [n_tools, rows, state], [img, state, out])
        img.select(on_click, [state], [img, out])
        ui.load(new_tray, [n_tools, rows, state], [img, state, out])

    return ui


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--weights", default="classifier.pt")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true",
                    help="public gradio tunnel; prefer an ssh -L tunnel instead")
    args = ap.parse_args()

    ui = build_ui(Classifier(args.weights), TrayBuilder(args.data))
    ui.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()