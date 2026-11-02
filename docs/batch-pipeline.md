# Carotid segmentation (CCA / ICA / ECA) via TotalSegmentator + rule-based post-processing

## Why this exists

TotalSegmentator does **not** segment all three carotid segments out of the box:

| Segment | TotalSegmentator support |
|---|---|
| Common carotid artery (CCA) | `total` task, classes `common_carotid_artery_left/right` |
| Internal carotid artery (ICA) | `headneck_bones_vessels` task, classes `internal_carotid_artery_left/right` |
| External carotid artery (ECA) | **not present in any TotalSegmentator task** |

(Confirmed directly against the installed `TotalSegmentator==2.16.0` class maps in this env — not from
memory/docs, which can drift between versions.)

So this pipeline:
1. Runs the two open-source TotalSegmentator tasks above to get trusted CCA and ICA seed masks (no
   license key needed — both tasks are Apache-2.0).
2. Reconstructs the missing ECA, and refines all three segments to native CT resolution, using a
   rule-based vessel-tree analysis: Frangi vesselness filtering → connect CCA+ICA into one lumen mask →
   skeletonize → find the real bifurcation in the centerline graph → label the three branches →
   propagate those labels back to a full-resolution volumetric mask.

## Status

The image-processing logic (bifurcation detection, cycle-artifact cleanup at the junction, branch
classification, volumetric label propagation, left/right merging, NIfTI I/O with orientation handling)
is implemented and validated against synthetic Y-shaped vessel geometry (see reasoning below) — the
algorithm is verified correct on controlled data. **It has not yet been run on a real CT angiography
case**, since none was available in this environment. Before trusting output clinically or for
downstream modeling (e.g. CFD, stenosis measurement), run it on a handful of your own cases and check
the QA numbers in `summary.json` plus a visual review in ITK-SNAP / 3D Slicer / your SimVascular
workflow.

## Setup

You already have a `carotid` conda env with everything needed (`TotalSegmentator`, `torch`, `nibabel`,
`scikit-image`, `scipy`, `networkx`, `dicom2nifti`, `SimpleITK`, `nnunetv2`). Nothing new to install.

```bash
conda activate carotid
```

First run of each TotalSegmentator task downloads its model weights automatically (needs internet
once, then cached).

## Usage

```bash
python run_batch.py --input /path/to/cases --output /path/to/results
```

`--input` is a folder where each case is either:
- a single `case.nii.gz` file, or
- a subfolder of DICOM files for that case's CT series (converted via `dicom2nifti` automatically).

Output per case: `<output>/<case_id>/carotid_segments.nii.gz`, a single multilabel NIfTI:

```
1 = CCA right   2 = CCA left
3 = ICA right   4 = ICA left
5 = ECA right   6 = ECA left
```

`<output>/summary.json` has per-case, per-side QA: whether the bifurcation was resolved, segment
volumes, and connectivity checks (each label should be exactly 1 connected component — if
`*_multi_component: true`, review that case).

Options:
- `--device {gpu,cpu,mps}` — default auto-detects (this Mac has MPS available).
- `--cap-ica-at-skull-base` — optionally trims the ICA at the styloid-process landmark + margin. Off
  by default, since `headneck_bones_vessels`'s ICA model has its own learned extent and this landmark
  is only an approximation of true skull-base (carotid canal) entry. Turn on and tune
  `--skull-base-margin-mm` if you find the raw ICA output runs further than you want.
- `--keep-intermediate` — keeps the raw per-class TotalSegmentator outputs instead of discarding them.

## Requirements on the input data

- **Contrast-enhanced CT angiography (CTA)**, not plain CT — the whole approach (adaptive HU banding,
  Frangi vesselness) depends on the carotid lumen being bright relative to surrounding tissue. Plain CT
  will give unreliable results and there's no way around that without contrast.
- FOV should cover from below the CCA origin (roughly aortic-arch level) up through the skull base.
  If the scan doesn't include neck vertebrae C1–C7, `summary.json`'s `vertebrae_found` list for that
  case will be short/empty — a quick way to flag insufficient FOV.

## Known limitations / tuning knobs

- **Bifurcation ambiguity in tortuous or diseased vessels.** The rule (one branch continues to the
  known ICA seed, one to the known CCA seed, the rest is ECA) assumes a fairly standard bifurcation
  anatomy. Severe tortuosity, occlusion, or a very high/low bifurcation may make it fail — in that case
  the case falls back to just the raw TotalSegmentator CCA/ICA masks with `bifurcation_found: false`
  and no ECA, rather than guessing.
- **HU band and Frangi threshold are adaptive but not bulletproof** — see `vesselness.py`,
  `adaptive_lumen_hu_band()` and `frangi_thresh` in `vessel_lumen_mask()`. If a case's ECA comes out
  under- or over-segmented, that threshold and the morphological closing radius (`ball(1)`) are the
  first things to tune.
- **Bifurcation-apex convention**: the single voxel where CCA/ICA/ECA meet is counted as the end of the
  CCA (a convention, not a universal standard — clinically the "carotid bulb" spans this zone loosely).
  Change the `label_vol[bn] = 1` line in `treeanalysis.py` if you need a different convention.
- No fine-tuned model is involved — if this rule-based approach isn't accurate enough after tuning,
  the natural next step (given `nnunetv2` is already in this env) is to hand-label a training set from
  this pipeline's output (after manual correction) and fine-tune a dedicated 3-class carotid model.

## Layout

```
carotid_seg/
  totalseg_steps.py   # TotalSegmentator task wrappers (CCA, ICA, styloid, C1-C7 landmarks)
  vesselness.py        # per-scan adaptive HU banding + Frangi vessel lumen extraction
  treeanalysis.py       # skeleton graph, cycle-artifact cleanup, bifurcation search, branch labeling
  volumize.py            # label propagation back to full volume + QA metrics
  pipeline.py             # per-case orchestration, left/right merge, NIfTI I/O
run_batch.py              # CLI entrypoint
```
