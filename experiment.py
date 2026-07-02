"""
Manifold-active subspace alignment experiment.

For each layer of each network:
  1. TwoNN ID          — intrinsic dimension of the data cloud (Facco et al. CDF fit,
                         with subsample error bars and MLE cross-check)
  2. AGOP              — E[∇loss ∇loss^T] w.r.t. that layer's activations, built from
                         per-sample gradients (uniform sample weighting)
  3. Principal angles  — between top-k PCA directions and top-k AGOP eigenvectors
  4. Random baseline   — expected alignment for two random k-dim subspaces in R^d = k/d

Controls (per layer):
  - Split-half         — PCA from one half of the images, AGOP from the other half;
                         removes same-sample coupling between activations and gradients
  - Within-class       — class means removed from both activations and gradients;
                         tests whether alignment reduces to shared class-mean structure
  - Class-mean overlap — fraction of each subspace lying in the span of the 9 centred
                         class-mean directions
  - Centered AGOP      — gradient covariance instead of second moment (mean-gradient
                         robustness check)
  - Subsample error bars and a k-sensitivity sweep for the alignment ratio

Conv layers are globally-average-pooled (activations) / spatially-averaged (gradients)
so every layer gives a consistent (N, d) matrix. The spatial mean of the gradient is
exactly the sensitivity of the loss to spatially-uniform per-channel perturbations,
which is the natural pairing with GAP activations (and exactly equals the GAP-gradient
when downstream ops are GAP-invariant, e.g. layer4 -> avgpool in ResNet).

Controls: random-init networks (same architectures, untrained weights, fixed seed).
"""
import argparse
import json
import tarfile
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torchvision import models

from twonn import twonn_id, twonn_id_mle, twonn_id_subsampled

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEED       = 0
N_IMAGES   = 500
BATCH_SIZE = 32
N_SUBSAMPLE_REPS = 20   # error-bar repetitions (80% subsamples)
N_SPLITS         = 10   # split-half repetitions
K_SWEEP          = [2, 3, 5, 8, 10, 15, 20, 25, 30, 40, 50]

DATA_DIR   = Path("./imagenette2-160")
RESULTS    = Path("./results")
FIG_DIRS   = [RESULTS, Path("./paper/figures")]

DEVICE = (
    "mps"  if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available()         else
    "cpu"
)

IMAGENETTE_TO_IMAGENET = [0, 217, 482, 491, 497, 566, 569, 571, 574, 701]

VGG_LAYERS = {
    "pool1": "features.4",
    "pool2": "features.9",
    "pool3": "features.16",
    "pool4": "features.23",
    "pool5": "features.30",
    "fc1":   "classifier.1",   # post-ReLU output of the first FC layer
    "fc2":   "classifier.4",   # post-ReLU output of the second FC layer
    "out":   "classifier.6",
}
RESNET_LAYERS = {
    "layer1":  "layer1",
    "layer2":  "layer2",
    "layer3":  "layer3",
    "layer4":  "layer4",
    "avgpool": "avgpool",      # identical to GAP(layer4) by construction: sanity check
    "fc":      "fc",
}

# Each model entry: arch name, layer map, pretrained flag, display label
MODEL_CONFIGS = {
    "vgg16":           ("vgg16",    VGG_LAYERS,    True,  "VGG-16 (trained)"),
    "vgg16_random":    ("vgg16",    VGG_LAYERS,    False, "VGG-16 (random init)"),
    "resnet18":        ("resnet18", RESNET_LAYERS, True,  "ResNet-18 (trained)"),
    "resnet18_random": ("resnet18", RESNET_LAYERS, False, "ResNet-18 (random init)"),
}


def load_model(arch, pretrained):
    torch.manual_seed(SEED)   # reproducible random-init controls
    if arch == "vgg16":
        w = models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        return models.vgg16(weights=w)
    if arch == "resnet18":
        w = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        return models.resnet18(weights=w)
    raise ValueError(arch)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def download_imagenette():
    url  = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
    dest = Path("imagenette2-160.tgz")
    if DATA_DIR.exists():
        return
    if not dest.exists():
        print("Downloading Imagenette2-160 (~98 MB)…")
        urllib.request.urlretrieve(url, dest)
    print("Extracting…")
    with tarfile.open(dest, "r:gz") as tf:
        tf.extractall(".")
    dest.unlink()
    print("Done.")


def build_loader(n_images):
    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_dir = DATA_DIR / "val"
    dataset = torchvision.datasets.ImageFolder(val_dir, transform=transform)
    per_class = n_images // len(dataset.classes)
    counts    = defaultdict(int)
    indices   = []
    for i, (_, lbl) in enumerate(dataset.samples):
        if counts[lbl] < per_class:
            indices.append(i)
            counts[lbl] += 1
    subset = torch.utils.data.Subset(dataset, indices)
    return torch.utils.data.DataLoader(
        subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

# ---------------------------------------------------------------------------
# Module accessor
# ---------------------------------------------------------------------------

def get_module(model, dotted_path):
    m = model
    for part in dotted_path.split("."):
        m = getattr(m, part)
    return m

# ---------------------------------------------------------------------------
# Core collection — single forward+backward pass, per-sample gradients kept
# ---------------------------------------------------------------------------

def collect(model, layer_path_map, loader):
    """Return per-layer activations and per-sample gradients, plus labels/accuracy.

    Uses reduction='sum' so each row of out.grad is exactly ∇_{a_i} ℓ_i with
    uniform per-sample weighting (mean reduction would scale rows by 1/B and
    over-weight samples in smaller final batches).
    """
    model.eval()

    act_lists  = {k: [] for k in layer_path_map}
    grad_lists = {k: [] for k in layer_path_map}
    cur        = {}
    handles    = []

    for name, path in layer_path_map.items():
        module = get_module(model, path)

        def make_hook(lname):
            def hook(mod, inp, out):
                # retain_grad on the module's actual output tensor: it is on the
                # loss path, so out.grad is populated by backward().
                out.retain_grad()
                cur[lname] = out
            return hook

        handles.append(module.register_forward_hook(make_hook(name)))

    criterion = nn.CrossEntropyLoss(reduction="sum")
    n_correct = n_total = 0
    labels = []

    for x, y_local in loader:
        x      = x.to(DEVICE)
        y_inet = torch.tensor(
            [IMAGENETTE_TO_IMAGENET[yi.item()] for yi in y_local],
            dtype=torch.long, device=DEVICE,
        )
        model.zero_grad()
        logits = model(x)
        loss   = criterion(logits, y_inet)
        loss.backward()

        n_correct += (logits.argmax(1) == y_inet).sum().item()
        n_total   += x.size(0)
        labels.extend(y_local.tolist())

        for name in layer_path_map:
            out = cur.get(name)
            if out is None:
                raise RuntimeError(f"Hook for layer '{name}' did not fire")
            if out.grad is None:
                raise RuntimeError(
                    f"No gradient reached layer '{name}' — hook tensor fell "
                    f"off the autograd graph"
                )

            # Activations: GAP for conv → (batch, C); FC → (batch, d)
            act = out.mean(dim=tuple(range(2, out.dim()))) if out.dim() > 2 else out
            act_lists[name].append(
                act.detach().float().cpu().numpy().astype(np.float32)
            )

            # Gradients: spatial mean for conv (sensitivity to uniform
            # per-channel perturbations; exact GAP-gradient when downstream
            # ops are GAP-invariant)
            g = out.grad
            if g.dim() > 2:
                g = g.mean(dim=tuple(range(2, g.dim())))
            grad_lists[name].append(
                g.detach().float().cpu().numpy().astype(np.float32)
            )

        cur.clear()

    for h in handles:
        h.remove()

    activations = {k: np.concatenate(v, 0) for k, v in act_lists.items()}
    gradients   = {k: np.concatenate(v, 0) for k, v in grad_lists.items()}
    accuracy    = n_correct / n_total if n_total else 0.0
    return activations, gradients, np.array(labels), accuracy

# ---------------------------------------------------------------------------
# Geometry helpers (all via thin SVD of (N, d) matrices — never d×d eigh)
# ---------------------------------------------------------------------------

def pca_basis(X, k):
    """Top-k PCA directions (d, k) of the centred rows of X."""
    Xc = X - X.mean(0, keepdims=True)
    _, sv, Vt = np.linalg.svd(Xc, full_matrices=False)
    k = min(k, int(np.sum(sv > 1e-12 * sv[0])) if sv[0] > 0 else 0)
    return Vt[:k].T if k > 0 else None


def grad_basis(G, k):
    """Top-k AGOP eigenvectors (d, k): right singular vectors of the (N, d)
    per-sample gradient matrix (AGOP = G^T G / N shares them)."""
    _, sv, Vt = np.linalg.svd(G, full_matrices=False)
    k = min(k, int(np.sum(sv > 1e-12 * sv[0])) if sv[0] > 0 else 0)
    return Vt[:k].T if k > 0 else None


def agop_eigenvalues(G):
    sv = np.linalg.svd(G, compute_uv=False)
    return sv ** 2 / len(G)


def effective_rank(eigvals, threshold):
    if len(eigvals) == 0 or eigvals[0] <= 0:
        return 0
    return int(np.sum(eigvals > threshold * eigvals[0]))


def mean_cos2(U, V):
    """Mean squared cosine of principal angles between the column spaces."""
    if U is None or V is None or U.shape[1] == 0 or V.shape[1] == 0:
        return None
    sv = np.clip(np.linalg.svd(U.T @ V, compute_uv=False), 0.0, 1.0)
    return float(np.mean(sv ** 2))


def class_center(X, y):
    """Remove the per-class mean from each row."""
    Xc = X.copy()
    for c in np.unique(y):
        m = y == c
        Xc[m] -= X[m].mean(0, keepdims=True)
    return Xc


def classmean_basis(X, y):
    """Orthonormal basis (d, r) of the span of centred class means (r ≤ C-1)."""
    means = np.stack([X[y == c].mean(0) for c in np.unique(y)])
    means -= X.mean(0, keepdims=True)
    _, sv, Vt = np.linalg.svd(means, full_matrices=False)
    r = int(np.sum(sv > 1e-10 * sv[0])) if sv[0] > 0 else 0
    return Vt[:r].T if r > 0 else None


def subspace_fraction_in(U, B):
    """Mean squared projection of U's columns onto span(B) ∈ [0, 1]."""
    if U is None or B is None:
        return None
    P = B.T @ U
    return float(np.sum(P ** 2) / U.shape[1])


def stratified_halves(y, rng):
    h1 = []
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        h1.extend(idx[: len(idx) // 2])
    mask = np.zeros(len(y), bool)
    mask[np.array(h1)] = True
    return np.where(mask)[0], np.where(~mask)[0]

# ---------------------------------------------------------------------------
# Per-layer analysis
# ---------------------------------------------------------------------------

def analyse_layer(A, G, y, rng, n_reps=N_SUBSAMPLE_REPS, n_splits=N_SPLITS):
    A = A.astype(np.float64)
    G = G.astype(np.float64)
    N, d = A.shape
    res = {"embed_dim": int(d), "n_samples": int(N)}

    # --- intrinsic dimension -------------------------------------------------
    id_val = twonn_id(A)
    id_mean, id_std = twonn_id_subsampled(A, n_rep=n_reps, random_state=rng.integers(2**31))
    res["twonn_id"]     = id_val
    res["twonn_id_std"] = id_std
    res["twonn_id_mle"] = twonn_id_mle(A)

    # --- AGOP spectrum & effective rank at several thresholds ---------------
    eig = agop_eigenvalues(G)
    res["agop_spectrum_top50"]  = [float(v) for v in eig[:50]]
    res["agop_rank_0.1pct"]     = effective_rank(eig, 0.001)
    res["agop_effective_rank"]  = effective_rank(eig, 0.01)
    res["agop_rank_5pct"]       = effective_rank(eig, 0.05)

    # --- main alignment ------------------------------------------------------
    k = max(1, int(round(id_val))) if np.isfinite(id_val) else 0
    k_use = max(1, min(k, res["agop_effective_rank"], d - 1, N - 1)) if k > 0 else 0
    res["k_used"] = int(k_use)
    if k_use == 0:
        return res

    U = pca_basis(A, k_use)
    V = grad_basis(G, k_use)
    c2 = mean_cos2(U, V)
    res["mean_cos2"]       = c2
    res["random_baseline"] = k_use / d
    res["ratio"]           = c2 / (k_use / d) if c2 is not None else None

    # --- subsample error bars (same-k, 80% of images) ------------------------
    c2_reps = []
    m = int(0.8 * N)
    for _ in range(n_reps):
        idx = rng.choice(N, m, replace=False)
        ks = min(k_use, m - 1)
        c2_s = mean_cos2(pca_basis(A[idx], ks), grad_basis(G[idx], ks))
        if c2_s is not None:
            c2_reps.append(c2_s)
    if c2_reps:
        res["mean_cos2_subsample_mean"] = float(np.mean(c2_reps))
        res["mean_cos2_std"]            = float(np.std(c2_reps))
        res["ratio_std"]                = float(np.std(c2_reps) / (k_use / d))

    # --- split-half control (PCA and AGOP from disjoint image halves) --------
    sh = []
    for _ in range(n_splits):
        h1, h2 = stratified_halves(y, rng)
        ks = min(k_use, len(h1) - 1)
        a = mean_cos2(pca_basis(A[h1], ks), grad_basis(G[h2], ks))
        b = mean_cos2(pca_basis(A[h2], ks), grad_basis(G[h1], ks))
        if a is not None and b is not None:
            sh.append(0.5 * (a + b))
    if sh:
        res["split_half_cos2"]     = float(np.mean(sh))
        res["split_half_cos2_std"] = float(np.std(sh))
        res["split_half_ratio"]    = float(np.mean(sh) / (k_use / d))

    # --- within-class control (class means removed from both objects) --------
    Aw = class_center(A, y)
    Gw = class_center(G, y)
    idw = twonn_id(Aw)
    res["twonn_id_within"] = idw
    kw = max(1, int(round(idw))) if np.isfinite(idw) else 0
    eig_w = agop_eigenvalues(Gw)
    kw_use = max(1, min(kw, effective_rank(eig_w, 0.01), d - 1, N - 1)) if kw > 0 else 0
    if kw_use > 0:
        c2w = mean_cos2(pca_basis(Aw, kw_use), grad_basis(Gw, kw_use))
        res["within_k"]        = int(kw_use)
        res["within_cos2"]     = c2w
        res["within_baseline"] = kw_use / d
        res["within_ratio"]    = c2w / (kw_use / d) if c2w is not None else None

        # Crossed control: within-class AND split-half. Same-sample functional
        # coupling (gradient residuals ≈ linear map of activation residuals,
        # exact at the logits layer where softmax ≈ affine locally) inflates
        # the plain within-class score; disjoint halves remove it.
        shw = []
        for _ in range(n_splits):
            h1, h2 = stratified_halves(y, rng)
            ks = min(kw_use, len(h1) - 1)
            a = mean_cos2(pca_basis(Aw[h1], ks), grad_basis(Gw[h2], ks))
            b = mean_cos2(pca_basis(Aw[h2], ks), grad_basis(Gw[h1], ks))
            if a is not None and b is not None:
                shw.append(0.5 * (a + b))
        if shw:
            res["within_split_cos2"]     = float(np.mean(shw))
            res["within_split_cos2_std"] = float(np.std(shw))
            res["within_split_ratio"]    = float(np.mean(shw) / (kw_use / d))

    # --- class-mean structure diagnostics ------------------------------------
    Bm = classmean_basis(A, y)
    if Bm is not None:
        r = Bm.shape[1]
        res["classmean_dim"]            = int(r)
        res["classmean_agop_cos2"]      = mean_cos2(Bm, grad_basis(G, r))
        res["classmean_baseline"]       = r / d
        res["pca_frac_in_classmean"]    = subspace_fraction_in(U, Bm)
        res["agop_frac_in_classmean"]   = subspace_fraction_in(V, Bm)

    # --- centered-AGOP robustness check --------------------------------------
    Gc = G - G.mean(0, keepdims=True)
    c2c = mean_cos2(U, grad_basis(Gc, k_use))
    res["centered_agop_cos2"]  = c2c
    res["centered_agop_ratio"] = c2c / (k_use / d) if c2c is not None else None

    # --- k-sensitivity sweep --------------------------------------------------
    sweep = {}
    for kk in K_SWEEP:
        if kk >= min(d, N):
            continue
        c2k = mean_cos2(pca_basis(A, kk), grad_basis(G, kk))
        if c2k is not None:
            sweep[str(kk)] = float(c2k / (kk / d))
    res["k_sweep_ratio"] = sweep

    return res

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _savefig(fig, stem):
    for base in FIG_DIRS:
        base.mkdir(parents=True, exist_ok=True)
        for ext in ("png", "pdf"):
            fig.savefig(base / f"{stem}.{ext}", dpi=200, bbox_inches="tight")
    print(f"Saved figure → {stem}.png/.pdf")


def plot_results(all_results, n_images):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    archs = ["vgg16", "resnet18"]
    arch_labels = {"vgg16": "VGG-16", "resnet18": "ResNet-18"}

    # ---- Figure 1: ID / AGOP rank / alignment, trained vs random ------------
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    for row, arch in enumerate(archs):
        trained = all_results.get(arch, {})
        random_ = all_results.get(f"{arch}_random", {})
        layers  = list(trained.get("layers", {}).keys())
        n       = len(layers)
        x_pos   = np.arange(n) / max(n - 1, 1)

        def vals(res, key):
            return [res.get("layers", {}).get(l, {}).get(key) for l in layers]

        def arr(res, key, fill=np.nan):
            return np.array([v if v is not None else fill for v in vals(res, key)],
                            dtype=float)

        # ── Left: TwoNN ID with subsample error bars ────────────────────────
        ax = axes[row, 0]
        ax.errorbar(x_pos, arr(trained, "twonn_id"), yerr=arr(trained, "twonn_id_std"),
                    fmt="o-", color="tab:blue", capsize=3,
                    label=f"trained (acc={trained.get('accuracy', 0):.1%})")
        ax.errorbar(x_pos, arr(random_, "twonn_id"), yerr=arr(random_, "twonn_id_std"),
                    fmt="o--", color="tab:blue", alpha=0.5, capsize=3,
                    label="random init")
        ax.set_ylabel("TwoNN ID")
        ax.set_title(f"{arch_labels[arch]} — intrinsic dimension")

        # ── Middle: AGOP effective rank, 1% line + 0.1–5% threshold band ────
        ax = axes[row, 1]
        for res, style, alpha, lbl in ((trained, "s-", 1.0, "trained (1% thr.)"),
                                       (random_, "s--", 0.5, "random init")):
            r1  = arr(res, "agop_effective_rank")
            r01 = arr(res, "agop_rank_0.1pct")
            r5  = arr(res, "agop_rank_5pct")
            ax.plot(x_pos, r1, style, color="tab:orange", alpha=alpha, label=lbl)
            ax.fill_between(x_pos, r5, r01, color="tab:orange", alpha=0.12 * alpha)
        ax.set_ylabel("AGOP effective rank")
        ax.set_title(f"{arch_labels[arch]} — active subspace size\n(band: 5%–0.1% thresholds)")

        # ── Right: alignment with error bars + random baseline ──────────────
        ax = axes[row, 2]
        ax.errorbar(x_pos, arr(trained, "mean_cos2"), yerr=arr(trained, "mean_cos2_std"),
                    fmt="^-", color="tab:green", lw=2, capsize=3, label="trained")
        ax.errorbar(x_pos, arr(random_, "mean_cos2"), yerr=arr(random_, "mean_cos2_std"),
                    fmt="^--", color="tab:green", lw=2, alpha=0.5, capsize=3,
                    label="random init")
        ax.plot(x_pos, arr(trained, "random_baseline"), ":", color="gray", lw=1.2,
                label="random baseline (k/d)")
        ax.plot(x_pos, arr(random_, "random_baseline"), ":", color="gray", lw=1.2,
                alpha=0.5)
        ax.set_ylabel("mean cos²θ  (1 = perfect)")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{arch_labels[arch]} — PCA vs AGOP alignment")

        for col in range(3):
            ax = axes[row, col]
            ax.set_xticks(x_pos)
            ax.set_xticklabels(layers, rotation=45, ha="right", fontsize=8)
            ax.set_xlabel("Layer")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

    plt.suptitle(
        f"Data manifold geometry vs. active subspace alignment\n"
        f"Imagenette ({n_images} images, GAP representations) — "
        f"trained vs. random-init control",
        fontsize=11,
    )
    plt.tight_layout()
    _savefig(fig, "alignment_experiment")
    plt.close(fig)

    # ---- Figure 2: controls (trained nets), alignment ratio on log scale ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    for col, arch in enumerate(archs):
        trained = all_results.get(arch, {})
        layers  = list(trained.get("layers", {}).keys())
        n       = len(layers)
        x_pos   = np.arange(n, dtype=float)
        L       = trained.get("layers", {})

        def get(key):
            return np.array(
                [L.get(l, {}).get(key) if L.get(l, {}).get(key) is not None
                 else np.nan for l in layers], dtype=float)

        ax = axes[col]
        ax.errorbar(x_pos, get("ratio"), yerr=get("ratio_std"), fmt="o-",
                    color="tab:green", capsize=3, label="full (same samples)")
        ax.plot(x_pos, get("split_half_ratio"), "s--", color="tab:purple",
                label="split-half (disjoint samples)")
        ax.plot(x_pos, get("within_ratio"), "^:", color="tab:red",
                label="within-class (class means removed)")
        ax.plot(x_pos, get("within_split_ratio"), "v:", color="tab:brown",
                label="within-class + split-half")
        ax.plot(x_pos, get("centered_agop_ratio"), "x-", color="tab:gray",
                alpha=0.7, label="centered AGOP")
        ax.axhline(1.0, color="k", lw=1, ls=":", label="chance (ratio = 1)")
        ax.set_yscale("log")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(layers, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("alignment ratio  (mean cos²θ ÷ k/d)")
        ax.set_title(f"{arch_labels[arch]} (trained) — alignment controls")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, which="both")

    plt.tight_layout()
    _savefig(fig, "alignment_controls")
    plt.close(fig)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(smoke=False):
    n_images = 100 if smoke else N_IMAGES
    configs = ({k: MODEL_CONFIGS[k] for k in ("vgg16", "resnet18_random")}
               if smoke else MODEL_CONFIGS)

    print(f"Device: {DEVICE}")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    download_imagenette()
    loader = build_loader(n_images)
    print(f"Dataset: {len(loader.dataset)} images")

    RESULTS.mkdir(exist_ok=True)
    all_results = {"config": {
        "seed": SEED, "n_images": n_images, "batch_size": BATCH_SIZE,
        "torch": torch.__version__, "torchvision": torchvision.__version__,
        "device": DEVICE,
    }}

    for mi, (model_key, (arch, layer_paths, pretrained, label)) in enumerate(
            configs.items()):
        print(f"\n{'='*60}\nModel: {label}\n{'='*60}")

        model = load_model(arch, pretrained).to(DEVICE)
        model.eval()

        print("Collecting activations + per-sample gradients…")
        acts, grads, y, accuracy = collect(model, layer_paths, loader)
        print(f"Top-1 accuracy: {accuracy:.1%}")

        layer_results = {}
        for li, layer_name in enumerate(layer_paths):
            rng = np.random.default_rng([SEED, mi, li])
            metrics = analyse_layer(acts[layer_name], grads[layer_name], y, rng)
            layer_results[layer_name] = metrics

            id_s = (f"{metrics['twonn_id']:.1f}±{metrics.get('twonn_id_std', 0):.1f}"
                    if np.isfinite(metrics["twonn_id"]) else "nan")
            def fmt(key, nd=3):
                v = metrics.get(key)
                return f"{v:.{nd}f}" if isinstance(v, float) else "n/a"
            def fmt_ratio(key):
                v = metrics.get(key)
                return f"{v:.1f}x" if isinstance(v, float) else "n/a"
            print(
                f"  {layer_name:10s} d={metrics['embed_dim']:5d} ID={id_s:12s} "
                f"rank={metrics['agop_effective_rank']:4d} "
                f"align={fmt('mean_cos2')} ratio={fmt_ratio('ratio'):8s} "
                f"split={fmt_ratio('split_half_ratio'):8s} "
                f"within={fmt_ratio('within_ratio'):8s} "
                f"w+s={fmt_ratio('within_split_ratio'):8s} "
                f"pca_cm={fmt('pca_frac_in_classmean', 2)} "
                f"agop_cm={fmt('agop_frac_in_classmean', 2)}"
            )

        all_results[model_key] = {"accuracy": accuracy, "layers": layer_results}

        del model, acts, grads
        if DEVICE == "mps":
            torch.mps.empty_cache()

    out_json = RESULTS / ("results_smoke.json" if smoke else "results.json")
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved results → {out_json}")

    if not smoke:
        try:
            plot_results(all_results, n_images)
        except ImportError as e:
            print(f"Plotting skipped ({e}); results.json is saved — "
                  f"rerun with --plot-only once matplotlib is available.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="fast sanity run: 100 images, 2 models, no plots")
    ap.add_argument("--plot-only", action="store_true",
                    help="regenerate figures from results/results.json")
    args = ap.parse_args()
    if args.plot_only:
        with open(RESULTS / "results.json") as f:
            saved = json.load(f)
        plot_results(saved, saved.get("config", {}).get("n_images", N_IMAGES))
    else:
        main(smoke=args.smoke)
