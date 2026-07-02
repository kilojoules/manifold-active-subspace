# Manifold–Active-Subspace Alignment

Do trained neural networks align the geometry of their layer representations
(the data manifold, measured by TwoNN intrinsic dimension) with the directions
their loss actually depends on (the active subspace / AGOP)?

At each probed layer of VGG-16 and ResNet-18 we measure:

1. **TwoNN intrinsic dimension** of the activation cloud (Facco et al. 2017
   CDF-fit procedure, with subsample error bars and an MLE cross-check),
2. **AGOP** `E[∇loss ∇lossᵀ]` w.r.t. the layer activations, built from
   per-sample gradients,
3. **Principal-angle alignment** between the top-k PCA directions and the
   top-k AGOP eigenvectors, reported as mean cos²θ and as a ratio over the
   `k/d` random-subspace baseline.

Controls: random-init networks, split-half estimation (PCA from one half of
the images, AGOP from the other), within-class alignment (class means removed
from both objects, additionally crossed with split-half), centered-AGOP
robustness check, and a k-sensitivity sweep.

## Run

```bash
pip install -r requirements.txt
python experiment.py            # full run: downloads Imagenette (~98 MB), ~4 models
python experiment.py --smoke    # fast sanity check (100 images, 2 models)
python experiment.py --plot-only  # regenerate figures from results/results.json
```

Runs on CPU, CUDA, or Apple MPS. The full run finishes in well under an hour
on a laptop.

## Layout

- `experiment.py` — data collection (single forward/backward pass per batch,
  per-sample gradients retained) and all analyses/controls
- `twonn.py` — TwoNN estimator (Facco CDF fit + MLE variant + subsample error bars)
- `results/results.json` — all numbers behind the paper's tables (committed)
- `results/alignment_experiment.{png,pdf}` — main figure
- `results/alignment_controls.{png,pdf}` — controls figure
- `paper/` — TMLR manuscript (`main.tex`); `python paper/gen_table.py`
  regenerates the results table from `results/results.json`

## Data caveats (disclosed in the paper)

- Imagenette-160 images are upsampled to 224px, degrading top-1 accuracy to
  ~66%; models are ImageNet-pretrained and never fine-tuned on Imagenette.
- Imagenette's val split is drawn ~97% from the original ImageNet *training*
  split, so the pretrained models saw most evaluation images during training.
