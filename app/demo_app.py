#!/usr/bin/env python3
"""
Surgical instrument identification demo.

Two interactions on one synthetic tray of five instruments:

  1. Click an instrument  -> the classifier names it
  2. Request a tool by name -> the system highlights every instance it finds,
     or reports that the tool is not on this tray

The "find by name" path runs the classifier over all five tiles and highlights
by PREDICTION, not by ground truth -- otherwise the demo would be showing off
the tray builder rather than the model. Ground truth is used only to mark
whether each decision was right.

Tiles are drawn from the VALIDATION split only (the same 80/20 split by
filename ID range that training used), so nothing on screen was trained on.
Note this is a held-out split of the same capture session -- same drape, same
lighting, same physical instruments -- not an independent test set.

Tiling rather than compositing keeps each tile an unmodified training-
distribution frame: cropping tightly around an instrument cost ~20 points of
accuracy (docs/findings.md, experiment 4), so any reframing would handicap the
model for reasons unrelated to the task being demonstrated.

Works on Gradio 5 and 6 (theme/css moved to launch() in 6).

    python demo_app.py --data /path/to/mendley-tools --weights classifier.pt
"""

import argparse
import random
import re
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

TILE = 640          # frame size the classifier was trained on
N_TOOLS = 5

# The dataset description lists nine categories but ships seven; gloves and
# cotton were never uploaded, so the classifier cannot predict them. They are
# offered here anyway -- asking for one is a legitimate way to exercise the
# "not on this tray" path.
UNTRAINED_CLASSES = ["Surgical Gloves", "Medical Cotton"]

GREEN = (60, 200, 90)
RED = (235, 85, 85)

CSS = """
.gradio-container { max-width: 100% !important; }
#tray img { border-radius: 8px; }
#readout { min-height: 170px; font-size: 1.02rem; }
"""


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


def numeric_prefix(path: Path) -> int:
    m = re.match(r"(\d+)", path.name)
    return int(m.group(1)) if m else -1


class TrayBuilder:
    """Tiles held-out single-instrument frames into a synthetic tray."""

    def __init__(self, root: Path, val_frac=0.2):
        self.classes = sorted(d.name for d in root.iterdir()
                              if d.is_dir() and d.name.lower() != "overlapping")
        self.files = {}
        for c in self.classes:
            fs = sorted((root / c).glob("*.jpg"), key=numeric_prefix)
            cut = int(len(fs) * (1 - val_frac))
            self.files[c] = fs[cut:]        # validation split only
        total = sum(len(v) for v in self.files.values())
        print(f"tray source: {len(self.classes)} classes, {total} held-out images")

    def build(self, n=N_TOOLS):
        rng = random.Random()
        # sample with repetition so "find all instances" is a real query
        picks = [rng.choice(self.classes) for _ in range(n)]
        if len(set(picks)) < 3:                       # avoid a dull tray
            picks = rng.sample(self.classes, min(n, len(self.classes)))
            while len(picks) < n:
                picks.append(rng.choice(self.classes))
            rng.shuffle(picks)

        canvas = np.zeros((TILE, TILE * n, 3), np.uint8)
        tiles = []
        for i, cls in enumerate(picks):
            img = cv2.imread(str(rng.choice(self.files[cls])))
            if img is None:
                continue
            canvas[:, i * TILE:(i + 1) * TILE] = cv2.resize(img, (TILE, TILE))
            tiles.append({"x0": i * TILE, "x1": (i + 1) * TILE, "truth": cls})
        return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), tiles


def draw(rgb, tiles, highlight=None, click_xy=None):
    """highlight: dict tile_index -> (colour, caption)"""
    vis = rgb.copy()
    for i, t in enumerate(tiles):
        if highlight and i in highlight:
            colour, caption = highlight[i]
            cv2.rectangle(vis, (t["x0"] + 8, 8), (t["x1"] - 8, TILE - 8), colour, 14)
            if caption:
                (tw, th), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 1.6, 4)
                cv2.rectangle(vis, (t["x0"] + 8, 8), (t["x0"] + 28 + tw, 36 + th),
                              colour, -1)
                cv2.putText(vis, caption, (t["x0"] + 18, 28 + th),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 4,
                            cv2.LINE_AA)
    if click_xy:
        cv2.circle(vis, click_xy, 18, (255, 255, 255), -1)
        cv2.circle(vis, click_xy, 13, (30, 30, 30), -1)
    return vis


def build_ui(clf: Classifier, trays: TrayBuilder):
    options = sorted(clf.classes + UNTRAINED_CLASSES)

    def new_tray():
        rgb, tiles = trays.build()
        # classify every tile up front -- the "find" path reads these
        for t in tiles:
            ranked = clf.predict(rgb[:, t["x0"]:t["x1"]])
            t["pred"], t["conf"] = ranked[0]
            t["ranked"] = ranked
        st = {"rgb": rgb, "tiles": tiles}
        return rgb, st, ("New tray of five instruments. Click one, or request "
                         "a tool by name.")

    def on_click(st, evt: gr.SelectData):
        if not st:
            return None, "Generate a tray first."
        x, y = evt.index
        tiles, rgb = st["tiles"], st["rgb"]
        idx = next((i for i, t in enumerate(tiles) if t["x0"] <= x < t["x1"]), None)
        if idx is None:
            return rgb, "Clicked outside the tray."

        t = tiles[idx]
        ok = t["pred"] == t["truth"]
        vis = draw(rgb, tiles, {idx: (GREEN if ok else RED, t["pred"])}, (x, y))

        runners = "".join(f"\n- {n} — {p:.0%}" for n, p in t["ranked"][1:4])
        verdict = "correct" if ok else f"**incorrect** — actually *{t['truth']}*"
        return vis, (f"### {t['pred']}\n**{t['conf']:.0%} confidence** · {verdict}\n"
                     f"\nNext most likely:{runners}")

    def on_find(st, wanted):
        if not st:
            return None, "Generate a tray first."
        tiles, rgb = st["tiles"], st["rgb"]

        found = [i for i, t in enumerate(tiles) if t["pred"] == wanted]
        actual = [i for i, t in enumerate(tiles) if t["truth"] == wanted]

        if not found:
            vis = draw(rgb, tiles)
            msg = f"### Not on this tray\nNo *{wanted}* detected."
            if wanted in UNTRAINED_CLASSES:
                msg += ("\n\n_This class is listed in the dataset description "
                        "but was never published, so the model cannot predict "
                        "it._")
            elif actual:
                msg += (f"\n\n**Missed.** A *{wanted}* is present at position "
                        f"{', '.join(str(i + 1) for i in actual)} but was not "
                        f"recognised.")
            return vis, msg

        highlight = {i: (GREEN if tiles[i]["truth"] == wanted else RED,
                         f"{tiles[i]['conf']:.0%}") for i in found}
        vis = draw(rgb, tiles, highlight)

        n_right = sum(1 for i in found if tiles[i]["truth"] == wanted)
        msg = [f"### Found {len(found)} × {wanted}",
               f"Position{'s' if len(found) > 1 else ''}: "
               f"{', '.join(str(i + 1) for i in found)}"]
        if n_right != len(found):
            n_fp = len(found) - n_right
            msg.append(f"\n**{n_fp} false positive{'s' if n_fp > 1 else ''}** "
                       f"(red).")
        missed = [i for i in actual if i not in found]
        if missed:
            msg.append(f"\n**Missed {len(missed)}** at position "
                       f"{', '.join(str(i + 1) for i in missed)}.")
        return vis, "\n".join(msg)

    with gr.Blocks(title="Surgical Instrument ID") as ui:
        state = gr.State(None)

        gr.Markdown("## Surgical Instrument Identification")

        with gr.Row():
            tool = gr.Dropdown(options, value=options[0], label="Request a tool",
                               scale=3)
            find = gr.Button("Find on tray", variant="primary", scale=1)
            regen = gr.Button("New tray", scale=1)

        img = gr.Image(type="numpy", interactive=False, elem_id="tray",
                       show_label=False)
        out = gr.Markdown("Loading…", elem_id="readout")

        gr.Markdown(
            "<sub>Five instruments tiled from held-out frames. Highlighting "
            "uses the model's predictions, not ground truth: green means the "
            "prediction matches the true label, red means it does not.</sub>"
        )

        regen.click(new_tray, None, [img, state, out])
        find.click(on_find, [state, tool], [img, out])
        img.select(on_click, [state], [img, out])
        ui.load(new_tray, None, [img, state, out])

    return ui


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--weights", default="classifier.pt")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()

    ui = build_ui(Classifier(args.weights), TrayBuilder(args.data))

    # theme and css live on Blocks() in gradio 5 and on launch() in 6
    kwargs = dict(server_name="0.0.0.0", server_port=args.port, share=args.share)
    try:
        ui.launch(theme=gr.themes.Soft(), css=CSS, **kwargs)
    except TypeError:
        ui.launch(**kwargs)


if __name__ == "__main__":
    main()