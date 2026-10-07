# Findings

Running log of experiments for surgical instrument identification. 

---

## Context

The challenge spec requires the system to "identify a subset of specific
surgical tools presented or requested by name by the proctor (tool list TBD)."
Two constraints:

- **The tool list is TBD**, so a fixed-class detector is a liability. Adding a
  tool should not require retraining.
- **No cloud compute.** All inference must run inside the 12 ft x 12 ft robot
  cell, so hosted VLM APIs are ruled out at deployment.

---

## Dataset: Mendeley "SURGICAL TOOLS" / SID-RAS

- https://data.mendeley.com/datasets/cyghvmjrt3/1 (DOI 10.17632/cyghvmjrt3.1)
- Paper: *A robust dataset for surgical instrument detection to aid autonomous
  robotic surgery*, Data in Brief 2025 (PubMed 40677265)
- 6,000 images, 148 MB, **CC BY-ND 4.0**


### Licence note

CC BY-**ND** 4.0 — NoDerivatives. Training models and publishing results is
fine. *Redistributing* an annotated version of the images is not, without
permission from the authors. 
---

## Experiment 1 — Frozen DINOv2 prototype bank (no training)

**Setup.** `facebook/dinov2-base`, frozen. Five reference images per class,
embeddings averaged into a class prototype. 100 held-out queries per class
classified by cosine nearest neighbour. CLS token, L2-normalised.

**Motivation.** If frozen features separate the classes, a new tool costs five
photographs instead of a retraining run — which is what the TBD tool list
needs.

**Result: 0.670 top-1** (chance 0.143)

| Class | Accuracy |
|---|---|
| Scalpel | 0.990 |
| Syringe | 0.870 |
| Forceps | 0.850 |
| Mayo | 0.600 |
| Episiotomy Scissors | 0.500 |
| Hemostat | 0.450 |
| Stitch Scissors | 0.430 |

**Reading.** The split is by silhouette. Scalpel, syringe and forceps have
distinctive outlines and score well. The four weak classes are all
ring-handled hinged instruments that differ mainly at the jaws.

**Conclusion.** Frozen self-supervised features capture coarse instrument
*type* but not the fine distinctions a surgical tray actually requires. A
prototype bank alone is not sufficient.

### Supporting negative results

- **Rotation augmentation of prototypes: no effect.** On an earlier 4-class
  Roboflow set, 0.836 → 0.828 with rotated reference copies. Orientation
  variance was not the bottleneck. Ruled out cheaply.
- **Severe failure on micro-instruments.** On an ophthalmic 5-class set the
  same method scored 0.363 (chance 0.200) with an unstructured confusion
  matrix — every class bleeding into every other.

---

## Experiment 2 — Fine-tuned ResNet-18, whole image

**Setup.** ImageNet-pretrained ResNet-18, fully fine-tuned. 3,584 train /
896 val. AdamW, lr 1e-4, cosine schedule, label smoothing 0.05. Augmentation:
flips, 180° rotation, colour jitter.

**Split: by filename ID range, not randomly.** The images come from one
capture session with contiguous per-class numbering, so a random split leaks
near-duplicate frames across train and test. Our numbers are lower than the
paper's and more honest.

| Epochs | Top-1 |
|---|---|
| 15 | 0.884 |
| 40 | **0.935** |

At 15 epochs validation was still climbing — the model was underfit.

### Per-class at 40 epochs

| Class | Frozen (exp 1) | Fine-tuned |
|---|---|---|
| Scalpel | 0.990 | 0.984 |
| Forceps | 0.850 | 0.969 |
| Stitch Scissors | 0.430 | **0.961** |
| Syringe | 0.870 | 0.945 |
| Episiotomy Scissors | 0.500 | **0.938** |
| Hemostat | 0.450 | **0.883** |
| Mayo | 0.600 | 0.867 |

### Confusion matrix (rows = truth)

```
                     Episio  Forcep  Hemost    Mayo  Scalpe  Stitch  Syring
Episiotomy Scissors     120       0       0       7       0       1       0
            Forceps       0     124       0       0       4       0       0
           Hemostat       3       0     113      12       0       0       0
               Mayo      10       0       7     111       0       0       0
            Scalpel       0       2       0       0     126       0       0
    Stitch Scissors       0       0       0       5       0     123       0
            Syringe       0       0       0       0       7       0     121
```

**Reading.** The three classes that failed under frozen features improved
most; stitch scissors more than doubled. Scalpel dipped slightly, consistent
with capacity reallocating toward the hard classes.

**What remains.** 36 of 58 errors sit in one bidirectional cluster:
Mayo ↔ Hemostat ↔ Episiotomy Scissors — three ring-handled hinged
instruments. Mayo is the weakest class and the sink everything leaks into.

**Open oddity.** Syringe → Scalpel, 7 errors. These should not resemble each
other. Worth eyeballing the misclassified images; suspect label noise or
near-invisible instruments rather than a modelling failure.

---

## Next

1. **SAM-based cropping.** Run SAM once over all 4,480 images, cache a tight
   box per image, retrain off the crops. Thin low-contrast structures on
   uniform backgrounds are SAM's strength and it will not be fooled by the
   drape gradient. Prompt with a centre point rather than automatic mask
   generation — much faster, and the instruments are roughly centred.
2. **Part-aware cascade** on top of those crops, targeting the
   Mayo/Hemostat/Episiotomy cluster.
3. **Open-set evaluation.** Everything published on this data is fixed-class.
   Measure: hold out one class entirely, add it back with *k* reference
   photographs, and report accuracy as a function of *k*. That is the question
   the TBD tool list poses and nobody has asked it on this dataset.

### Not doing, and why

- **YOLO detection on SID-RAS.** Already published — six YOLO variants
  (v10n/s/m, v11n/s/m) at a reported average 99.3% mAP. Reproducing it adds
  nothing. Cite instead.
- **More epochs.** Gains have flattened; cropping is the bigger lever.


---

## References

- SID-RAS dataset paper — https://pubmed.ncbi.nlm.nih.gov/40677265/
- Part-aware cascade — https://arxiv.org/pdf/2602.01679
- RoboNurse-VLA (100 images/instrument) — https://arxiv.org/pdf/2409.19590
- Mayo Clinic instrument counting (~90 images/class) —
  https://link.springer.com/article/10.1186/s13037-024-00406-y
- Gestonurse (Wachs lab prior art) —
  https://web.ics.purdue.edu/~jpwachs/papers/gestonurse.PDF

**Data-scale consensus across published systems: 90–250 images per
instrument.** This is the basis for the ask when the real tool set is
available — one afternoon of filming per instrument set.