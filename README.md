# A deep learning-enhanced PTV framework for rod-like particles in non-Newtonian fluids

Code and trajectory data supporting the paper:

> Wolff L., Maggi C., Toro J.P., Delpiano J., Gómez J., Paul A., Moreno-Casas P.A. (2026).
> *A deep learning-enhanced PTV framework for simultaneous translational and rotational
> tracking of rod-like particles in non-Newtonian fluids.*

The framework combines **YOLOv11 instance segmentation** with a customised
**alpha-beta-gamma predictive filter** to reconstruct the simultaneous
translational and rotational trajectories of individual steel fibers suspended
in a Carbopol yield-stress gel flowing through an L-shaped casting device.

The measurement conditions are the hard part: fiber widths of about two pixels,
spatially non-uniform surface reflectivity along each fiber, and progressive
inter-particle occlusion as the fiber loading increases from 25 to 800 fibers.

---

## Dataset

The annotated training dataset is hosted on Roboflow:

**https://app.roboflow.com/particle-tracking-velocimetry/deep-learning-ptv-wolff-et-al**

673 images, one class (`Fiber`), instance-segmentation masks. Built from 267
manually annotated frames expanded by horizontal and vertical flips, 90 degree
rotations, random rotations of ±15 degrees and shear of ±15 %.

The raw image sequences that the tracking notebook consumes (6 experiments ×
600 frames at 1024 × 1024, roughly 27 GB) are **not** distributed here. The
trajectory files they produce are included in `results/`, so every downstream
analysis can be reproduced without re-running the segmentation. Contact the
corresponding author for access to the raw frames.

---

## Repository layout

```
.
├── notebooks/
│   ├── 01_train_segmentation_model.ipynb   Fine-tune YOLOv11 on the Roboflow dataset
│   └── 02_run_ptv_tracking.ipynb           Segment, track and post-process one experiment
├── model/
│   └── best.pt                             Fine-tuned YOLOv11-seg weights (19.6 MB)
├── results/                                Trajectory files, one pair per experiment
│   ├── trajectories_<N>_raw.json           All tracks
│   └── trajectories_<N>_filtered.json      Tracks of at least 20 frames
├── requirements.txt
└── README.md
```

Large binaries (`*.json`, `*.pt`) are stored with **Git LFS**. Install it before
cloning, otherwise you get pointer files instead of data:

```bash
git lfs install
git clone https://github.com/patricioamoreno/PaperCode-A-DL-enhanced-PTV-framework
```

---

## Requirements

Python 3.12, and a CUDA-capable GPU for training. Tracking runs on CPU but is
slow.

```bash
pip install -r requirements.txt
```

The Ultralytics version is pinned to the one used to produce the published
results. The notebooks use the current argument names (`show_labels`,
`line_width`, `max_det`), so they also work with newer 8.3.x releases, but the
segmentation output can differ slightly between versions.

---

## Reproducing the pipeline

### 1. Train the segmentation model — `notebooks/01_train_segmentation_model.ipynb`

Downloads the Roboflow dataset and fine-tunes `yolo11s-seg.pt` for 40 epochs at
1024 × 1024. The Roboflow API key is read from the environment:

```bash
export ROBOFLOW_API_KEY="your_key"
```

Training took about 5.9 h on a single NVIDIA RTX 4060 Ti (8 GB). The weights
produced by this step are the ones already shipped in `model/best.pt`, so this
notebook only needs to be run to retrain from scratch.

### 2. Track the fibers — `notebooks/02_run_ptv_tracking.ipynb`

Expects the raw frames laid out as:

```
images/
├── 25 Fibras/Cam 1/*.bmp
├── 50 Fibras/Cam 1/*.bmp
└── ...
```

Set `IMAGES_ROOT` in the configuration cell if they live elsewhere. For each
experiment the notebook:

1. **Segments** every frame with the fine-tuned model (`conf = 0.25`).
2. **Characterises** each detection from its bounding box: centroid, projected
   length (longer side) and orientation angle.
3. **Associates** detections with the tracks seen in the previous frame, using
   the alpha-beta-gamma prediction as the search gate
   (±10 px in x and y, ±5 degrees in angle).
4. **Post-processes**: discards tracks shorter than 20 frames, which are
   dominated by elongated bubbles misclassified as fibers.

Both the raw and the filtered trajectories are written, so the effect of the
post-processing step can be audited.

The final section reproduces two diagnostics — cumulative tracked fibers versus
detections per frame, and the distribution of the instantaneous velocity
components — directly from the files in `results/`, without needing the images.

---

## Trajectory file format

Each file is a single JSON object.

| Key | Type | Description |
|---|---|---|
| `fibras_por_frame` | list of 600 int | Detections returned by the segmentation model in each frame |
| `"1"`, `"2"`, … | object | One tracked trajectory; the key is the track identifier |

Every track holds five parallel arrays, all of the same length *N* (the number
of frames in which that fiber was tracked):

| Field | Shape | Units | Description |
|---|---|---|---|
| `frame` | `[[n], …]` | frame index, 1-based | Frames in which the fiber was detected |
| `centroide` | `[[x, y], …]` | px | Bounding-box centre, image coordinates, y downwards |
| `largo_maximo` | `[[l], …]` | px | Longer side of the bounding box |
| `angulo` | `[[θ], …]` | deg | Orientation, `atan2(box height, box width)` |
| `kalman` | `[state, …]` | mixed | Internal filter state, see below |

The field names are kept in their original Spanish so the files stay compatible
with the dataset deposited alongside the paper.

`kalman` stores the predictor state used for association, in the layout
`[[x, y], [vx, vy], [ax, ay], [angle], [omega], [angular_acceleration], [length]]`
with positions in px, velocities in px/s and angles in degrees. It is kept for
completeness and is **not** used by any analysis reported in the paper: the
published velocities and angular velocities are plain finite differences of
`centroide` and `angulo`.

### Contents

| Experiment | Raw tracks | Filtered tracks | Raw file | Filtered file |
|---|---|---|---|---|
| 25 fibers | 916 | 32 | 2.2 MB | 1.7 MB |
| 50 fibers | 729 | 28 | 1.8 MB | 1.4 MB |
| 100 fibers | 2 468 | 176 | 6.9 MB | 5.5 MB |
| 200 fibers | 3 274 | 219 | 10.1 MB | 8.3 MB |
| 400 fibers | 12 853 | 642 | 24.5 MB | 16.6 MB |
| 800 fibers | 33 339 | 1 450 | 47.4 MB | 26.5 MB |

The 50-fiber experiment is included here for completeness; the paper reports the
five loadings 25, 100, 200, 400 and 800.

---

## Notes on the method

Three properties of the implementation are worth knowing before reusing the
data. All three are discussed in the paper.

**The orientation angle is folded into [0°, 90°].** It is derived from the
bounding box as `atan2(height, width)` with both sides positive, so it measures
the inclination with respect to the horizontal and does not distinguish +30°
from −30°. Every orientation statistic in the paper follows this convention.

**Detections per frame are capped at 300.** That is the Ultralytics default for
`max_det`, and it is reached in 564 of the 600 frames of the 800-fiber
experiment. The value is set explicitly in the configuration cell so the
behaviour is visible; raise it to re-run that case without the cap.

**Tracks are not bridged across missed detections.** Association only considers
the tracks present in the immediately preceding frame, so a single missed
detection terminates a track. This is the origin of the trajectory
fragmentation quantified in the paper, and it means the frames of any track are
always contiguous — finite differences over `centroide` need no gap handling.

---

## Citation

If you use this code or the trajectory data, please cite both the repository
and the article. GitHub reads `CITATION.cff`, so the formatted citation is also
available from the *Cite this repository* button on the project page.

```bibtex
@article{wolff2026ptv,
  title   = {A deep learning-enhanced PTV framework for simultaneous
             translational and rotational tracking of rod-like particles in
             non-Newtonian fluids},
  author  = {Wolff, Lukas and Maggi, Crist\'obal and Toro, J. P. and
             Delpiano, Jos\'e and G\'omez, Jorge and Paul, \'Alvaro and
             Moreno-Casas, Patricio A.},
  journal = {Engineering Research Express},
  year    = {2026},
  note    = {In review}
}
```

The DOI, volume and pages will be added to `CITATION.cff` once the article is
accepted.

## License

Released under the [MIT License](LICENSE).

The annotated training dataset is distributed separately through Roboflow under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); see the dataset link
above for its terms.

## Contact

Patricio A. Moreno-Casas — patriciomoreno@miuandes.cl
Universidad de los Andes, Santiago, Chile
