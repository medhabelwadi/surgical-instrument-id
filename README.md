# Surgical Instrument Identification

This repository explores image classification for seven types of surgical
instruments. The classifier is based on an ImageNet-pretrained ResNet-18
fine-tuned on the instrument images. It predicts one instrument class for an
image; it is not a detector that localizes multiple instruments in a scene.

The dataset used here is the [SID-RAS / SURGICAL TOOLS dataset on
Mendeley Data](https://data.mendeley.com/datasets/cyghvmjrt3/1). The dataset is
not included in this repository: download it separately and follow its
license and attribution requirements.

## What is in this repository?

- `scripts/train_classifier.py` — prepares filename-range train/validation
  splits and trains a whole-image ResNet-18 classifier
- `app/demo_app.py` — Gradio demo. It tiles individual instrument images into
  a synthetic "tray"; clicking a tile classifies that image and shows the
  prediction and its known class.
- `docs/findings.md` — experiment notes and results.
- `artifacts/classifier.pt` — trained checkpoint used by the demo. This file
  is allowed in Git so it can be committed with the project.

The demo's tiled tray is not a cluttered or overlapping surgical scene. It
demonstrates the classifier and click interaction, not overlapping-object
detection.

## Dataset setup

First clone the repository:

```bash
git clone https://github.com/medhabelwadi/surgical-instrument-id.git
cd surgical-instrument-id
```

Then download and extract the dataset from [Mendeley
Data](https://data.mendeley.com/datasets/cyghvmjrt3/1). Set the data path to
the directory containing the class folders. The training script and demo
expect JPEGs directly inside each class folder:

```text
data/sid-ras/
├── Episiotomy Scissors/
│   └── *.jpg
├── Forceps/
│   └── *.jpg
├── Hemostat/
│   └── *.jpg
└── ... other class folders ...
```

If extraction creates an extra enclosing directory, use the inner directory
that contains the class folders as `--data`. The scripts exclude a folder
named `Overlapping` from classification classes. The demo samples from the
non-overlapping class folders.


## Run the demo

Use Python 3.10 or 3.11. From the repository root, create and activate a
virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install PyTorch and torchvision for your machine using the
[official PyTorch install selector](https://pytorch.org/get-started/locally/)
(choose CPU or the appropriate CUDA build). Then install the project
dependencies:

```bash
python -m pip install -r requirements.txt
```

```bash
python app/demo_app.py \
  --data data/sid-ras \
  --weights artifacts/classifier.pt \
  --port 7860
```

Open [http://localhost:7860](http://localhost:7860), generate a tray, and
click an instrument. To use a different port, change `--port`. The app binds
to `0.0.0.0` so it can be reached through a local network or SSH tunnel; do
not use `--share` unless you intentionally want Gradio to create a public
tunnel.

The demo needs **both** the checkpoint and the dataset images: the checkpoint
provides the learned model, while the app uses the dataset to assemble its
sample trays. Neither is automatically downloaded by the app.

## Train a checkpoint

From the repository root, install the dependencies as above, download and
extract the dataset, then run the whole-image model:

```bash
python scripts/train_classifier.py \
  --data data/sid-ras \
  --mode full \
  --epochs 40 \
  --out artifacts/classifier.pt
```

The script splits each class by filename ID range rather than randomly. The
dataset images are sequential captures, so a random split could put
near-duplicate frames in both training and validation sets. The recorded
whole-image experiment reached 0.935 top-1 validation accuracy at 40 epochs;
see [the experiment notes](docs/findings.md) for class-level results and
limitations. Results can vary with the exact data files and environment.



