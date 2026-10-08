#!/usr/bin/env python3
"""
Surgical instrument identification demo.

Three ways to query one synthetic tray of five instruments:

  1. Click an instrument      -> the classifier names it
  2. Pick a tool from the list -> highlights every instance it finds
  3. Say the tool out loud     -> Whisper transcribes, the name is resolved,
                                  then the same search runs

The "find" path highlights by PREDICTION, not ground truth -- otherwise the
demo would be showing off the tray builder rather than the model. Ground truth
is used only to mark whether each decision was right.

Tiles come from the VALIDATION split only (same 80/20 split by filename ID
range that training used), so nothing on screen was trained on. It is a
held-out split of the same capture session, not an independent test set.

    pip install gradio openai-whisper
    python demo_app.py --data /path/to/mendley-tools --weights classifier.pt
    python demo_app.py ... --whisper-model small     # slower, more accurate
    python demo_app.py ... --no-voice                # skip Whisper entirely
"""

import argparse
import difflib
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

# Listed in the dataset description but never published, so the classifier
# cannot predict them. Offered anyway -- a clean way to exercise "not on tray".
UNTRAINED_CLASSES = ["Surgical Gloves", "Medical Cotton"]

GREEN = (60, 200, 90)
RED = (235, 85, 85)

CSS = """
.gradio-container { max-width: 100% !important; }
#tray img { border-radius: 8px; }
#readout { min-height: 170px; font-size: 1.02rem; }
"""

# Spoken requests are not class names. "Hand me the mayos", "clamp",
# "suture scissors" all have to resolve to a label the model knows.
ALIASES = {
    "Mayo": ["mayo", "mayos", "mayo scissors", "mayo scissor"],
    "Stitch Scissors": ["stitch scissors", "stitch scissor", "suture scissors",
                        "stitch", "suture scissor"],
    "Episiotomy Scissors": ["episiotomy scissors", "episiotomy", "episiotomy scissor",
                            "epis", "episiotomy shears"],
    "Hemostat": ["hemostat", "hemostats", "haemostat", "clamp", "mosquito",
                 "kelly", "artery forceps"],
    "Forceps": ["forceps", "forcep", "tweezers", "pickups", "pick ups",
                "tissue forceps"],
    "Scalpel": ["scalpel", "blade", "knife", "scalpel blade", "ten blade"],
    "Syringe": ["syringe", "needle", "injection"],
    "Surgical Gloves": ["gloves", "glove", "surgical gloves"],
    "Medical Cotton": ["cotton", "gauze", "swab", "medical cotton", "sponge"],
}

# Filler that surrounds the actual request.
FILLER = re.compile(
    r"\b(please|can i (have|get)|could i (have|get)|hand me|pass me|give me|"
    r"i need|i want|get me|bring me|the|a|an|some|now|thanks|thank you)\b")


def resolve_tool(text, valid):
    """Map a spoken phrase to one of `valid`, or None.

    Alias substring match first, then fuzzy match on the cleaned phrase.
    Returns (label, how_it_matched) so the UI can be honest about guessing.
    """
    if not text:
        return None, None
    clean = FILLER.sub(" ", text.lower().strip(" .,!?"))
    clean = re.sub(r"\s+", " ", clean).strip()

    # longest alias first so "stitch scissors" beats "scissors"
    pairs = [(label, a) for label, al in ALIASES.items() for a in al
             if label in valid]
    for label, alias in sorted(pairs, key=lambda p: -len(p[1])):
        if alias in clean:
            return label, f"matched “{alias}”"

    candidates = [a for _, a in pairs]
    close = difflib.get_close_matches(clean, candidates, n=1, cutoff=0.6)
    if close:
        label = next(l for l, a in pairs if a == close[0])
        return label, f"closest match to “{close[0]}”"
    return None, None


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


class Transcriber:
    """Lazy Whisper wrapper -- the model loads on first use, not at startup."""

    def __init__(self, name="base"):
        self.name, self.model = name, None

    def __call__(self, path):
        if self.model is None:
            import whisper
            print(f"loading whisper '{self.name}' …")
            self.model = whisper.load_model(self.name)
        # fp16 is a GPU path; force it off so CPU runs cleanly
        return self.model.transcribe(path, fp16=torch.cuda.is_available())["text"].strip()


def build_ui(clf: Classifier, trays: TrayBuilder, asr: Transcriber | None):
    options = sorted(clf.classes + UNTRAINED_CLASSES)

    def new_tray():
        rgb, tiles = trays.build()
        for t in tiles:
            ranked = clf.predict(rgb[:, t["x0"]:t["x1"]])
            t["pred"], t["conf"] = ranked[0]
            t["ranked"] = ranked
        st = {"rgb": rgb, "tiles": tiles}
        return rgb, st, ("New tray of five instruments. Click one, or request "
                         "a tool by name or voice.")

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

    def search(st, wanted, prefix=""):
        tiles, rgb = st["tiles"], st["rgb"]
        found = [i for i, t in enumerate(tiles) if t["pred"] == wanted]
        actual = [i for i, t in enumerate(tiles) if t["truth"] == wanted]

        if not found:
            msg = f"{prefix}### Not on this tray\nNo *{wanted}* detected."
            if wanted in UNTRAINED_CLASSES:
                msg += ("\n\n_This class is listed in the dataset description "
                        "but was never published, so the model cannot predict it._")
            elif actual:
                msg += (f"\n\n**Missed.** A *{wanted}* is present at position "
                        f"{', '.join(str(i + 1) for i in actual)} but was not "
                        f"recognised.")
            return draw(rgb, tiles), msg

        highlight = {i: (GREEN if tiles[i]["truth"] == wanted else RED,
                         f"{tiles[i]['conf']:.0%}") for i in found}
        vis = draw(rgb, tiles, highlight)

        n_right = sum(1 for i in found if tiles[i]["truth"] == wanted)
        msg = [f"{prefix}### Found {len(found)} × {wanted}",
               f"Position{'s' if len(found) > 1 else ''}: "
               f"{', '.join(str(i + 1) for i in found)}"]
        if n_right != len(found):
            n_fp = len(found) - n_right
            msg.append(f"\n**{n_fp} false positive{'s' if n_fp > 1 else ''}** (red).")
        missed = [i for i in actual if i not in found]
        if missed:
            msg.append(f"\n**Missed {len(missed)}** at position "
                       f"{', '.join(str(i + 1) for i in missed)}.")
        return vis, "\n".join(msg)

    def on_find(st, wanted):
        if not st:
            return None, "Generate a tray first."
        return search(st, wanted)

    def on_voice(st, audio_path, current):
        if not st:
            return None, "Generate a tray first.", current
        if asr is None:
            return st["rgb"], "Voice input is disabled (`--no-voice`).", current
        if not audio_path:
            return st["rgb"], "No audio recorded.", current

        try:
            heard = asr(audio_path)
        except Exception as exc:
            return st["rgb"], f"Transcription failed: {exc}", current

        wanted, how = resolve_tool(heard, options)
        quoted = f'> heard: *“{heard}”*\n\n'
        if wanted is None:
            return (st["rgb"],
                    quoted + "### Didn't catch a tool name\n"
                    "Try naming one of the instruments in the list.",
                    current)

        vis, msg = search(st, wanted, prefix=quoted)
        return vis, msg + f"\n\n<sub>{how}</sub>", wanted

    with gr.Blocks(title="Surgical Instrument ID") as ui:
        state = gr.State(None)

        gr.Markdown("## Surgical Instrument Identification")

        with gr.Row():
            tool = gr.Dropdown(options, value=options[0], label="Request a tool",
                               scale=3)
            find = gr.Button("Find on tray", variant="primary", scale=1)
            regen = gr.Button("New tray", scale=1)

        if asr is not None:
            mic = gr.Audio(sources=["microphone"], type="filepath",
                           label="…or say it out loud (e.g. “hand me the mayo scissors”)")

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
        if asr is not None:
            mic.stop_recording(on_voice, [state, mic, tool], [img, out, tool])
        ui.load(new_tray, None, [img, state, out])

    return ui


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--weights", default="classifier.pt")
    ap.add_argument("--whisper-model", default="base",
                    help="tiny | base | small | medium")
    ap.add_argument("--no-voice", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()

    asr = None if args.no_voice else Transcriber(args.whisper_model)
    ui = build_ui(Classifier(args.weights), TrayBuilder(args.data), asr)

    # theme and css live on Blocks() in gradio 5 and on launch() in 6
    kwargs = dict(server_name="0.0.0.0", server_port=args.port, share=args.share)
    try:
        ui.launch(theme=gr.themes.Soft(), css=CSS, **kwargs)
    except TypeError:
        ui.launch(**kwargs)


if __name__ == "__main__":
    main()