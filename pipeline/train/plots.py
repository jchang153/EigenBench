"""Embedding visualization helpers for trained BT/BTD models."""

from __future__ import annotations

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA


def _to_2d_with_pca(u: np.ndarray, v: np.ndarray):
    """Project embeddings to 2D via PCA (with 1D fallback)."""

    all_embeddings = np.vstack([u, v])
    dim = all_embeddings.shape[1]

    if dim >= 2:
        pca = PCA(n_components=2)
        all_2d = pca.fit_transform(all_embeddings)
        variance = pca.explained_variance_ratio_
    elif dim == 1:
        # PCA cannot produce 2 components from 1-D input; keep a zero y-axis.
        x = all_embeddings[:, 0]
        all_2d = np.column_stack([x, np.zeros_like(x)])
        variance = np.array([1.0, 0.0], dtype=float)
    else:
        all_2d = np.zeros((all_embeddings.shape[0], 2), dtype=float)
        variance = np.array([0.0, 0.0], dtype=float)

    u_2d = all_2d[: len(u)]
    v_2d = all_2d[len(u) :]
    return u_2d, v_2d, variance


def _build_u_color_index(num_u: int, num_models: int):
    """Map each u-row to a model index color.

    For criterion-conditioned models, u rows are packed as
    row_idx = criterion * num_models + judge_idx, so modulo by num_models
    recovers judge_idx.
    """

    if num_models <= 0:
        return [0] * num_u
    return [idx % num_models for idx in range(num_u)]


def save_uv_embedding_plot(
    model: torch.nn.Module,
    model_names: list[str],
    save_path: str,
    *,
    eigentrust_scores: np.ndarray | None = None,
    eigentrust_elo: np.ndarray | None = None,
    figsize=(20, 12),
):
    """Save side-by-side PCA visualization of u and v embeddings.

    Left: judge lenses (u, triangles), optionally sized by inverse tie propensity.
    Right: model dispositions (v, circles), legend on right side.
    """

    if not hasattr(model, "u") or not hasattr(model, "v"):
        raise ValueError("Model does not expose both 'u' and 'v' embeddings.")

    with torch.no_grad():
        u = model.u.weight.detach().cpu().numpy()
        v = model.v.weight.detach().cpu().numpy()

        log_lambda = None
        if hasattr(model, "log_lambda"):
            log_lambda = model.log_lambda.weight.detach().cpu().numpy().reshape(-1)

    if len(model_names) < v.shape[0]:
        extra = [f"Model {i}" for i in range(len(model_names), v.shape[0])]
        model_names = list(model_names) + extra
    else:
        model_names = list(model_names[: v.shape[0]])

    u_2d, v_2d, variance = _to_2d_with_pca(u, v)

    fig, (ax_u, ax_v) = plt.subplots(1, 2, figsize=figsize)

    colors = cm.tab20(np.linspace(0, 1, max(1, len(model_names))))
    u_color_idx = _build_u_color_index(len(u_2d), len(model_names))

    # Judge sizes: inverse lambda when available, else fixed.
    if log_lambda is not None and len(log_lambda) == len(u_2d):
        lambda_vals = np.exp(log_lambda)
        lam_min = float(np.min(lambda_vals))
        lam_max = float(np.max(lambda_vals))
        if lam_max > lam_min:
            scale = (lambda_vals - lam_min) / (lam_max - lam_min)
        else:
            scale = np.ones_like(lambda_vals)
        u_sizes = 120 + (1.0 - scale) * 240.0
    else:
        u_sizes = np.full(len(u_2d), 160.0, dtype=float)

    v_size = 170.0

    for idx, (x, y) in enumerate(u_2d):
        color_idx = u_color_idx[idx]
        ax_u.scatter(
            x,
            y,
            c=[colors[color_idx]],
            alpha=0.72,
            s=float(u_sizes[idx]),
            marker="^",
            edgecolors="black",
            linewidth=0.55,
        )

    for idx, (x, y) in enumerate(v_2d):
        ax_v.scatter(
            x,
            y,
            c=[colors[idx]],
            alpha=0.78,
            s=v_size,
            marker="o",
            edgecolors="black",
            linewidth=0.55,
        )
        # Show model index next to each disposition point for quick lookup.
        ax_v.annotate(
            str(idx),
            (x, y),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=8,
            color="black",
        )

    var_text = f"PC1 {variance[0]:.1%}, PC2 {variance[1]:.1%}"
    ax_u.set_title(f"Judge Lenses (u)\n{var_text}", fontsize=14)
    ax_v.set_title("Model Dispositions (v)", fontsize=14)

    for ax in (ax_u, ax_v):
        ax.axhline(y=0, color="black", linestyle="-", alpha=0.3, linewidth=0.8)
        ax.axvline(x=0, color="black", linestyle="-", alpha=0.3, linewidth=0.8)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

    legend_elements = []
    for idx, model_name in enumerate(model_names):
        label = f"[{idx}] {model_name}"
        if eigentrust_elo is not None and idx < len(eigentrust_elo):
            label = f"{label} | EB={eigentrust_elo[idx]:.1f}"
        elif eigentrust_scores is not None and idx < len(eigentrust_scores):
            label = f"{label} | t={eigentrust_scores[idx]:.4f}"

        legend_elements.append(
            Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                markerfacecolor=colors[idx],
                markeredgecolor="black",
                markersize=8,
                label=label,
                linestyle="None",
            )
        )

    ax_v.legend(
        handles=legend_elements,
        fontsize=10,
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        frameon=False,
    )

    plt.tight_layout()
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
