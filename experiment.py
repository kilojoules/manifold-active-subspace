"""
Manifold-active subspace alignment experiment.

For each layer of each network:
  1. TwoNN ID          — intrinsic dimension of the data cloud (geometry)
  2. AGOP              — E[∇loss ∇loss^T] w.r.t. that layer's activations (gradients)
  3. Principal angles  — between top-k PCA directions and top-k AGOP eigenvectors
  4. Random baseline   — expected alignment for two random k-dim subspaces in R^d = k/d

Conv layer activations are globally-averaged-pooled so every layer gives a
consistent (N, d) matrix (d = channels for conv, d = width for FC).

Controls: random-init networks (same architectures, untrained weights) test whether
the alignment structure is a consequence of training or just architectural shape.
"""
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

from twonn import twonn_id

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
N_IMAGES   = 500
BATCH_SIZE = 32
DATA_DIR   = Path("./imagenette2-160")
RESULTS    = Path("./results")
RESULTS.mkdir(exist_ok=True)

DEVICE = (
    "mps"  if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available()         else
    "cpu"
)
print(f"Device: {DEVICE}")

IMAGENETTE_TO_IMAGENET = [0, 217, 482, 491, 497, 566, 569, 571, 574, 701]

VGG_LAYERS = {
    "pool1": "features.4",
    "pool2": "features.9",
    "pool3": "features.16",
    "pool4": "features.23",
    "pool5": "features.30",
    "fc1":   "classifier.1",
    "fc2":   "classifier.4",
    "out":   "classifier.6",
}
RESNET_LAYERS = {
    "layer1":  "layer1",
    "layer2":  "layer2",
    "layer3":  "layer3",
    "layer4":  "layer4",
    "avgpool": "avgpool",
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


def build_loader():
    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_dir = DATA_DIR / "val"
    dataset = torchvision.datasets.ImageFolder(val_dir, transform=transform)
    per_class = N_IMAGES // len(dataset.classes)
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
# Core collection — single forward+backward pass for all layers
# ---------------------------------------------------------------------------

def collect(model, layer_path_map, loader):
    model.eval()

    act_lists = {k: [] for k in layer_path_map}
    G_accum   = {k: None for k in layer_path_map}
    cur       = {}
    handles   = []

    for name, path in layer_path_map.items():
        module = get_module(model, path)

        def make_hook(lname):
            def hook(mod, inp, out):
                out_f = out.float()
                out_f.retain_grad()   # grad on the raw output, not a detached copy
                cur[lname] = out_f
            return hook

        handles.append(module.register_forward_hook(make_hook(name)))

    criterion = nn.CrossEntropyLoss()
    n_correct = n_total = 0

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

        for name in layer_path_map:
            out = cur.get(name)
            if out is None:
                continue

            # Activations: GAP for conv → (batch, C); FC → (batch, d)
            act = out.mean(dim=tuple(range(2, out.dim()))) if out.dim() > 2 else out
            act_lists[name].append(act.detach().cpu().numpy().astype(np.float32))

            # AGOP: spatial-mean of gradient (Grad-CAM style for conv)
            if out.grad is not None:
                g = out.grad
                if g.dim() > 2:
                    g = g.mean(dim=tuple(range(2, g.dim())))
                g_np = g.detach().cpu().numpy().astype(np.float64)
                if G_accum[name] is None:
                    G_accum[name] = np.zeros((g_np.shape[1], g_np.shape[1]))
                G_accum[name] += g_np.T @ g_np

        cur.clear()

    for h in handles:
        h.remove()

    activations = {k: np.concatenate(v, 0) if v else None for k, v in act_lists.items()}
    G_matrices  = {k: (G / n_total if G is not None else None) for k, G in G_accum.items()}
    accuracy    = n_correct / n_total if n_total else 0.0
    return activations, G_matrices, accuracy

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def pca_subspace(X, k):
    X_c = X - X.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(X_c, full_matrices=False)
    return Vt[:k].T

def agop_subspace(G, k):
    if G is None:
        return None
    eigvals, eigvecs = np.linalg.eigh(G)
    idx = np.argsort(eigvals)[::-1]
    return eigvecs[:, idx][:, :k]

def principal_angle_cosines(U, V):
    sv = np.linalg.svd(U.T @ V, compute_uv=False)
    return np.clip(sv, 0.0, 1.0)

def effective_rank(G, threshold=0.01):
    if G is None:
        return 0
    ev = np.sort(np.linalg.eigvalsh(G))[::-1]
    ev = np.maximum(ev, 0.0)
    return 0 if ev[0] == 0 else int(np.sum(ev > threshold * ev[0]))

# ---------------------------------------------------------------------------
# Per-layer analysis
# ---------------------------------------------------------------------------

def analyse_layer(activations, G):
    result = {}
    d      = activations.shape[1]
    N      = activations.shape[0]

    id_val = twonn_id(activations.astype(np.float64))
    result["twonn_id"]           = id_val
    result["agop_effective_rank"] = effective_rank(G)
    result["embed_dim"]           = d

    k = max(1, int(round(id_val))) if not np.isnan(id_val) else 0
    if k > 0 and G is not None:
        k_use = max(1, min(k, result["agop_effective_rank"], d - 1, N - 1))
        U = pca_subspace(activations.astype(np.float64), k_use)
        V = agop_subspace(G, k_use)
        if V is not None:
            cosines             = principal_angle_cosines(U, V)
            result["mean_cos2"] = float(np.mean(cosines**2))
            result["k_used"]    = k_use
            # Expected alignment for two random k-dim subspaces in R^d
            result["random_baseline"] = k_use / d
        else:
            result["mean_cos2"] = result["random_baseline"] = None
            result["k_used"]    = k_use
    else:
        result["mean_cos2"] = result["random_baseline"] = None
        result["k_used"]    = k

    return result

# ---------------------------------------------------------------------------
# Plotting  (2 rows × 3 cols, one row per architecture)
# ---------------------------------------------------------------------------

def plot_results(all_results):
    import matplotlib.pyplot as plt

    archs = ["vgg16", "resnet18"]
    arch_labels = {"vgg16": "VGG-16", "resnet18": "ResNet-18"}

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    for row, arch in enumerate(archs):
        trained_key = arch
        random_key  = f"{arch}_random"

        trained = all_results.get(trained_key, {})
        random  = all_results.get(random_key, {})

        layers  = list(trained.get("layers", {}).keys())
        n       = len(layers)
        x_pos   = np.arange(n) / max(n - 1, 1)

        def vals(res, key):
            return [res.get("layers", {}).get(l, {}).get(key) for l in layers]

        # ── Left: TwoNN ID ──────────────────────────────────────────────────
        ax = axes[row, 0]
        ids_t = vals(trained, "twonn_id")
        ids_r = vals(random,  "twonn_id")
        ax.plot(x_pos, ids_t, "o-",  color="tab:blue",   label=f"trained (acc={trained.get('accuracy',0):.1%})")
        ax.plot(x_pos, ids_r, "o--", color="tab:blue",   alpha=0.5, label="random init")
        ax.set_ylabel("TwoNN ID")
        ax.set_title(f"{arch_labels[arch]} — intrinsic dimension")

        # ── Middle: AGOP effective rank ──────────────────────────────────────
        ax = axes[row, 1]
        ranks_t = vals(trained, "agop_effective_rank")
        ranks_r = vals(random,  "agop_effective_rank")
        ax.plot(x_pos, ranks_t, "s-",  color="tab:orange", label="trained")
        ax.plot(x_pos, ranks_r, "s--", color="tab:orange", alpha=0.5, label="random init")
        ax.set_ylabel("AGOP effective rank")
        ax.set_title(f"{arch_labels[arch]} — active subspace size")

        # ── Right: Alignment + random baseline ──────────────────────────────
        ax = axes[row, 2]
        aligns_t   = vals(trained, "mean_cos2")
        aligns_r   = vals(random,  "mean_cos2")
        baselines_t = vals(trained, "random_baseline")
        baselines_r = vals(random,  "random_baseline")

        def masked(x_pos, ys):
            xv = [x_pos[i] for i, v in enumerate(ys) if v is not None]
            yv = [v         for v     in ys            if v is not None]
            return xv, yv

        xt, yt = masked(x_pos, aligns_t)
        xr, yr = masked(x_pos, aligns_r)
        xbt, ybt = masked(x_pos, baselines_t)
        xbr, ybr = masked(x_pos, baselines_r)

        ax.plot(xt, yt, "^-",  color="tab:green", lw=2,   label="trained")
        ax.plot(xr, yr, "^--", color="tab:green", lw=2, alpha=0.5, label="random init")
        ax.plot(xbt, ybt, ":",  color="gray", lw=1.2, label="random baseline (k/d)")
        ax.plot(xbr, ybr, ":",  color="gray", lw=1.2, alpha=0.5)
        ax.set_ylabel("mean cos²θ  (1 = perfect)")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{arch_labels[arch]} — PCA vs AGOP alignment")

        # x-axis labels, legend, grid for this row
        for col in range(3):
            ax = axes[row, col]
            ax.set_xticks(x_pos)
            ax.set_xticklabels(layers, rotation=45, ha="right", fontsize=8)
            ax.set_xlabel("Layer")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

    plt.suptitle(
        f"Data manifold geometry vs. active subspace alignment\n"
        f"Imagenette ({N_IMAGES} images, GAP representations) — "
        f"trained vs. random-init control",
        fontsize=11,
    )
    plt.tight_layout()
    out = RESULTS / "alignment_experiment.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.close()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    download_imagenette()
    loader = build_loader()
    print(f"Dataset: {len(loader.dataset)} images")

    all_results = {}

    for model_key, (arch, layer_paths, pretrained, label) in MODEL_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"Model: {label}")
        print(f"{'='*60}")

        model = load_model(arch, pretrained).to(DEVICE)
        model.eval()

        print("Collecting activations + AGOP…")
        activations, G_matrices, accuracy = collect(model, layer_paths, loader)
        print(f"Top-1 accuracy: {accuracy:.1%}")

        layer_results = {}
        for layer_name in layer_paths:
            acts = activations.get(layer_name)
            G    = G_matrices.get(layer_name)
            if acts is None:
                continue
            metrics = analyse_layer(acts, G)
            layer_results[layer_name] = metrics

            id_s  = f"{metrics['twonn_id']:.1f}" if not np.isnan(metrics['twonn_id']) else "nan"
            al_s  = f"{metrics['mean_cos2']:.3f}" if metrics.get("mean_cos2") is not None else "n/a"
            rb_s  = f"{metrics['random_baseline']:.3f}" if metrics.get("random_baseline") is not None else "n/a"
            print(
                f"  {layer_name:10s}  d={metrics['embed_dim']:5d}  "
                f"ID={id_s:6s}  AGOP-rank={metrics['agop_effective_rank']:4d}  "
                f"align={al_s}  baseline={rb_s}  "
                f"ratio={str(round(metrics['mean_cos2']/metrics['random_baseline'], 1))+'x' if metrics.get('mean_cos2') and metrics.get('random_baseline') else 'n/a'}"
            )

        all_results[model_key] = {"accuracy": accuracy, "layers": layer_results}

        del model
        if DEVICE == "mps":
            torch.mps.empty_cache()

    out_json = RESULTS / "results.json"
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2, default=lambda x: None)
    print(f"\nSaved results → {out_json}")

    plot_results(all_results)


if __name__ == "__main__":
    main()
