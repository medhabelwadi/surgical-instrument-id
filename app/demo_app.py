#!/usr/bin/env python3
"""
Surgical instrument identification demo.

Two tabs:

  Tray   -- a still tray of five instruments. Click one, pick one from the
            list, or say its name out loud.
  Video  -- a synthetic camera pan across the same five instruments. Scrub or
            play, pause anywhere, click a point; a tile-sized box is cropped
            around the cursor and classified. No predictions during playback --
            inference runs only on click.

All frames come from the TEST block (65-80% of each class's ID range), the
contiguous split the classifier never trained on. The 80-100% block is avoided
deliberately: those images are mostly stock product photos rather than
photographed trays, so they are a domain-shift set, not a like-for-like one.

PERFORMANCE: playback streams one rendered frame per tick from the server, so
it is bandwidth-bound. Over an SSH tunnel this will stutter. Run locally for a
smooth demo. Frames are shipped at DISPLAY_SCALE while crops are taken at full
resolution, which keeps classification unaffected.

    pip install gradio openai-whisper
    python demo_app.py --data /path/to/mendley-tools --weights classifier.pt
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

TILE = 640                      # frame size the classifier was trained on
N_TOOLS = 5
VIEW_W, VIEW_H = 2 * TILE, TILE  # viewport shows exactly two tiles
BOX = TILE                       # click crop = one tile-equivalent
DISPLAY_SCALE = 0.75             # ship smaller frames; crop at full res

DEFAULT_FRAMES = 120             # with a 0.1 s tick, a 12 s pan
TICK = 0.1

TEST_LO, TEST_HI = 0.65, 0.80    # the held-out contiguous block

UNTRAINED_CLASSES = ["Surgical Gloves", "Medical Cotton"]

GREEN = (60, 200, 90)
RED = (235, 85, 85)
BLUE = (90, 160, 250)

CSS = """
.gradio-container { max-width: 100% !important; }
#tray img, #vid img { border-radius: 8px; }
#readout, #vreadout { min-height: 150px; font-size: 1.02rem; }
#transport { margin-top: -8px; }
"""

ALIASES = {
    "Mayo": ["mayo", "mayos", "mayo scissors", "mayo scissor"],
    "Stitch Scissors": ["stitch scissors", "stitch scissor", "suture scissors",
                        "stitch", "suture scissor"],
    "Episiotomy Scissors": ["episiotomy scissors", "episiotomy",
                            "episiotomy scissor", "epis", "episiotomy shears"],
    "Hemostat": ["hemostat", "hemostats", "haemostat", "clamp", "mosquito",
                 "kelly", "artery forceps"],
    "Forceps": ["forceps", "forcep", "tweezers", "pickups", "pick ups",
                "tissue forceps"],
    "Scalpel": ["scalpel", "blade", "knife", "scalpel blade", "ten blade"],
    "Syringe": ["syringe", "needle", "injection"],
    "Surgical Gloves": ["gloves", "glove", "surgical gloves"],
    "Medical Cotton": ["cotton", "gauze", "swab", "medical cotton", "sponge"],
}

FILLER = re.compile(
    r"\b(please|can i (have|get)|could i (have|get)|hand me|pass me|give me|"
    r"i need|i want|get me|bring me|the|a|an|some|now|thanks|thank you)\b")


def resolve_tool(text, valid):
    """Map a spoken phrase to one of `valid`, or None."""
    if not text:
        return None, None
    clean = FILLER.sub(" ", text.lower().strip(" .,!?"))
    clean = re.sub(r"\s+", " ", clean).strip()

    pairs = [(label, a) for label, al in ALIASES.items() for a in al
             if label in valid]
    for label, alias in sorted(pairs, key=lambda p: -len(p[1])):
        if alias in clean:
            return label, f"matched “{alias}”"

    close = difflib.get_close_matches(clean, [a for _, a in pairs], n=1, cutoff=0.6)
    if close:
        return next(l for l, a in pairs if a == close[0]), \
               f"closest match to “{close[0]}”"
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
        print(f"{len(self.classes)} classes | {self.n_views} view(s) | {self.device}")

    def predict(self, rgb):
        v = self.tf(Image.fromarray(rgb))
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

    def __init__(self, root: Path, lo=TEST_LO, hi=TEST_HI):
        self.classes = sorted(d.name for d in root.iterdir()
                              if d.is_dir() and d.name.lower() != "overlapping")
        self.files = {}
        for c in self.classes:
            fs = sorted((root / c).glob("*.jpg"), key=numeric_prefix)
            n = len(fs)
            self.files[c] = fs[int(n * lo):int(n * hi)]
        total = sum(len(v) for v in self.files.values())
        print(f"tray source: {len(self.classes)} classes, {total} test-block "
              f"images ({lo:.0%}-{hi:.0%} of each ID range)")

    def build(self, n=N_TOOLS):
        rng = random.Random()
        picks = [rng.choice(self.classes) for _ in range(n)]
        if len(set(picks)) < 3:
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


class Clip:
    """A straight horizontal pan across the tray canvas.

    The viewport is exactly two tiles wide, so the camera starts with tile 1
    fully in frame and ends with tile 5 fully in frame. Frames are rendered on
    demand -- one is a few ms, and caching hundreds of them would put hundreds
    of MB into session state.
    """

    def __init__(self, canvas_rgb, tiles, n_frames=DEFAULT_FRAMES):
        self.canvas = canvas_rgb
        self.tiles = tiles
        self.n = max(2, int(n_frames))
        self.x_start = VIEW_W // 2
        self.x_end = canvas_rgb.shape[1] - VIEW_W // 2

    def centre(self, i):
        t = int(np.clip(i, 0, self.n - 1)) / (self.n - 1)
        e = t * t * (3 - 2 * t)                   # ease in/out
        return self.x_start + (self.x_end - self.x_start) * e

    def frame(self, i):
        """Full-resolution VIEW_W x VIEW_H crop of the canvas."""
        cx = self.centre(i)
        x0 = int(np.clip(cx - VIEW_W / 2, 0, self.canvas.shape[1] - VIEW_W))
        return self.canvas[:, x0:x0 + VIEW_W]


def to_display(frame):
    return cv2.resize(frame, (int(VIEW_W * DISPLAY_SCALE),
                              int(VIEW_H * DISPLAY_SCALE)),
                      interpolation=cv2.INTER_AREA)


def crop_box(frame, x, y, size=BOX):
    """Square box of `size` centred on (x, y), clipped to the frame."""
    H, W = frame.shape[:2]
    half = size // 2
    x0 = int(np.clip(x - half, 0, max(0, W - size)))
    y0 = int(np.clip(y - half, 0, max(0, H - size)))
    return (x0, y0, min(W, x0 + size), min(H, y0 + size)), \
        frame[y0:min(H, y0 + size), x0:min(W, x0 + size)]


def draw(rgb, tiles, highlight=None, click_xy=None):
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
    """Lazy Whisper wrapper -- loads on first use, not at startup."""

    def __init__(self, name="base"):
        self.name, self.model = name, None

    def __call__(self, path):
        if self.model is None:
            import whisper
            print(f"loading whisper '{self.name}' …")
            self.model = whisper.load_model(self.name)
        return self.model.transcribe(
            path, fp16=torch.cuda.is_available())["text"].strip()


def build_ui(clf: Classifier, trays: TrayBuilder, asr):
    options = sorted(clf.classes + UNTRAINED_CLASSES)

    # ---------------------------------------------------------- still tab

    def new_tray():
        rgb, tiles = trays.build()
        for t in tiles:
            ranked = clf.predict(rgb[:, t["x0"]:t["x1"]])
            t["pred"], t["conf"] = ranked[0]
            t["ranked"] = ranked
        return rgb, {"rgb": rgb, "tiles": tiles}, \
            "New tray. Click an instrument, or request one by name or voice."

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
        return vis, (f"### {t['pred']}\n**{t['conf']:.0%} confidence** · {verdict}"
                     f"\n\nNext most likely:{runners}")

    def search(st, wanted, prefix=""):
        tiles, rgb = st["tiles"], st["rgb"]
        found = [i for i, t in enumerate(tiles) if t["pred"] == wanted]
        actual = [i for i, t in enumerate(tiles) if t["truth"] == wanted]

        if not found:
            msg = f"{prefix}### Not on this tray\nNo *{wanted}* detected."
            if wanted in UNTRAINED_CLASSES:
                msg += ("\n\n_Listed in the dataset description but never "
                        "published, so the model cannot predict it._")
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
        return (None, "Generate a tray first.") if not st else search(st, wanted)

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
            return (st["rgb"], quoted + "### Didn't catch a tool name\n"
                    "Try naming one of the instruments in the list.", current)
        vis, msg = search(st, wanted, prefix=quoted)
        return vis, msg + f"\n\n<sub>{how}</sub>", wanted

    # ---------------------------------------------------------- video tab

    def new_clip(n_frames):
        rgb, tiles = trays.build()
        clip = Clip(rgb, tiles, n_frames)
        return (to_display(clip.frame(0)), {"clip": clip},
                gr.update(maximum=clip.n - 1, value=0),
                "Scrub or press play. Pause anywhere and click a point to "
                "identify what is under the cursor.")

    def show_frame(st, i):
        return None if not st else to_display(st["clip"].frame(int(i)))

    def tick(st, i):
        return 0 if not st else (int(i) + 1) % st["clip"].n

    def on_video_click(st, i, evt: gr.SelectData):
        if not st:
            return None, "Generate a clip first."
        frame = st["clip"].frame(int(i))          # full resolution
        dx, dy = evt.index                        # display coordinates
        x, y = int(dx / DISPLAY_SCALE), int(dy / DISPLAY_SCALE)

        (x0, y0, x1, y1), patch = crop_box(frame, x, y)
        if patch.shape[0] < 32 or patch.shape[1] < 32:
            return to_display(frame), "Click nearer the middle of the frame."

        ranked = clf.predict(patch)
        name, conf = ranked[0]

        vis = frame.copy()
        cv2.rectangle(vis, (x0, y0), (x1 - 1, y1 - 1), BLUE, 8)
        cv2.circle(vis, (x, y), 14, (255, 255, 255), -1)
        cv2.circle(vis, (x, y), 9, (30, 30, 30), -1)
        label = f"{name}  {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 3)
        ty = y0 if y0 > th + 24 else y1
        cv2.rectangle(vis, (x0, ty - th - 22), (x0 + tw + 24, ty), BLUE, -1)
        cv2.putText(vis, label, (x0 + 12, ty - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3, cv2.LINE_AA)

        runners = "".join(f"\n- {n} — {p:.0%}" for n, p in ranked[1:4])
        return to_display(vis), (f"### {name}\n**{conf:.0%} confidence** · "
                                 f"frame {int(i)}\n\nNext most likely:{runners}")

    # ------------------------------------------------------------------ ui

    with gr.Blocks(title="Surgical Instrument ID") as ui:
        gr.Markdown("## Surgical Instrument Identification")

        with gr.Tabs():
            with gr.Tab("Tray"):
                state = gr.State(None)
                with gr.Row():
                    tool = gr.Dropdown(options, value=options[0],
                                       label="Request a tool", scale=3)
                    find = gr.Button("Find on tray", variant="primary", scale=1)
                    regen = gr.Button("New tray", scale=1)
                if asr is not None:
                    mic = gr.Audio(sources=["microphone"], type="filepath",
                                   label="…or say it out loud "
                                         "(e.g. “hand me the mayo scissors”)")
                img = gr.Image(type="numpy", interactive=False, elem_id="tray",
                               show_label=False)
                out = gr.Markdown("Loading…", elem_id="readout")
                gr.Markdown(
                    "<sub>Highlighting uses the model's predictions, not "
                    "ground truth: green means the prediction matches the true "
                    "label, red means it does not.</sub>")

                regen.click(new_tray, None, [img, state, out])
                find.click(on_find, [state, tool], [img, out])
                img.select(on_click, [state], [img, out])
                if asr is not None:
                    mic.stop_recording(on_voice, [state, mic, tool],
                                       [img, out, tool])
                ui.load(new_tray, None, [img, state, out])

            with gr.Tab("Video"):
                vstate = gr.State(None)
                playing = gr.State(False)

                vimg = gr.Image(type="numpy", interactive=False, elem_id="vid",
                                show_label=False)

                # transport controls sit directly under the frame, like a player
                with gr.Row(elem_id="transport", equal_height=True):
                    play = gr.Button("▶", variant="primary", scale=0,
                                     min_width=64)
                    scrub = gr.Slider(0, DEFAULT_FRAMES - 1, value=0, step=1,
                                      show_label=False, container=False,
                                      scale=20)

                vout = gr.Markdown("Loading…", elem_id="vreadout")

                with gr.Accordion("Clip settings", open=False):
                    with gr.Row():
                        nframes = gr.Slider(60, 300, value=DEFAULT_FRAMES,
                                            step=20,
                                            label="Clip length (frames)",
                                            scale=3)
                        vregen = gr.Button("New clip", scale=1)

                gr.Markdown(
                    "<sub>No predictions during playback. Clicking crops a "
                    "tile-sized box around the cursor and runs inference on "
                    "that box only.</sub>")

                try:
                    timer = gr.Timer(TICK, active=False)

                    def toggle(is_playing):
                        now = not is_playing
                        return (now, gr.update(active=now),
                                gr.update(value="❚❚" if now else "▶"))

                    play.click(toggle, [playing], [playing, timer, play])
                    timer.tick(tick, [vstate, scrub], [scrub])
                except AttributeError:        # gradio without gr.Timer
                    gr.Markdown("<sub>Playback needs a newer Gradio; use the "
                                "frame slider.</sub>")

                vregen.click(new_clip, [nframes], [vimg, vstate, scrub, vout])
                scrub.change(show_frame, [vstate, scrub], [vimg])
                vimg.select(on_video_click, [vstate, scrub], [vimg, vout])
                ui.load(new_clip, [nframes], [vimg, vstate, scrub, vout])

    return ui


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--weights", default="classifier.pt")
    ap.add_argument("--whisper-model", default="base")
    ap.add_argument("--no-voice", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()

    asr = None if args.no_voice else Transcriber(args.whisper_model)
    ui = build_ui(Classifier(args.weights), TrayBuilder(args.data), asr)

    kwargs = dict(server_name="0.0.0.0", server_port=args.port, share=args.share)
    try:
        ui.launch(theme=gr.themes.Soft(), css=CSS, **kwargs)
    except TypeError:
        ui.launch(**kwargs)


if __name__ == "__main__":
    main()