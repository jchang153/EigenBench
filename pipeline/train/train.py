"""Training routines and dataset adapters."""

from __future__ import annotations

from collections import defaultdict
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset


class Comparisons(Dataset):
    """Legacy comparison format: [l, i, j, k, r]."""

    def __init__(self, comparisons):
        self.data = comparisons

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        l, i, j, k, r = self.data[idx]
        return (
            torch.tensor(i, dtype=torch.long),
            torch.tensor(j, dtype=torch.long),
            torch.tensor(k, dtype=torch.long),
            torch.tensor(r, dtype=torch.float32),
        )


class CriteriaComparisons(Dataset):
    """Criterion comparison format: [c, l, i, j, k, r]."""

    def __init__(self, comparisons):
        self.data = comparisons

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        c, l, i, j, k, r = self.data[idx]
        return (
            torch.tensor(c, dtype=torch.long),
            torch.tensor(i, dtype=torch.long),
            torch.tensor(j, dtype=torch.long),
            torch.tensor(k, dtype=torch.long),
            torch.tensor(r, dtype=torch.float32),
        )


def _loss_has_plateaued(
    loss_history,
    *,
    window: int,
    relative_tolerance: float | None,
) -> bool:
    """Check whether the recent loss trajectory has plateaued."""

    if window <= 1 or len(loss_history) < window:
        return False

    recent = np.asarray(loss_history[-window:], dtype=float)
    avg_abs_diff = float(np.mean(np.abs(np.diff(recent))))

    if relative_tolerance is not None:
        baseline = max(float(np.mean(np.abs(recent[:-1]))), 1e-12)
        rel_change = avg_abs_diff / baseline
        if rel_change <= relative_tolerance:
            return True

    return False


def build_model_labels(num_models: int, spec_models: dict, extracted_name_map: dict[int, str]) -> list[str]:
    """Build display labels for model indices from spec and extracted records."""

    labels = [f"Model {i}" for i in range(num_models)]

    spec_names = list(spec_models.keys())
    for i in range(min(num_models, len(spec_names))):
        labels[i] = spec_names[i]

    for idx, name in extracted_name_map.items():
        if 0 <= idx < num_models and isinstance(name, str) and name.strip():
            labels[idx] = name.strip()

    return labels


def eigentrust_to_elo(trust_scores, num_models: int) -> np.ndarray:
    """Convert EigenTrust scores to the display Elo scale used by EigenBench."""

    trust_np = np.asarray(trust_scores, dtype=float).reshape(-1)
    trust_safe = np.clip(trust_np, 1e-12, None)
    return 1500.0 + 400.0 * np.log10(num_models * trust_safe)


def train_vector_bt(
    model,
    dataloader,
    lr,
    weight_decay,
    max_epochs,
    device,
    save_path=None,
    normalize=False,
    use_btd=False,
    criterion_mode=False,
    plateau_window: int = 5,
    plateau_relative_tolerance: float | None = 1e-3,
    verbose: bool = False,
):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss() if use_btd else nn.BCELoss()

    loss_history = []

    for epoch in range(1, max_epochs + 1):
        total_loss = 0.0
        model.train()

        for batch in dataloader:
            if criterion_mode:
                c, i, j, k, r = batch
                c, i, j, k = c.to(device), i.to(device), j.to(device), k.to(device)
            else:
                i, j, k, r = batch
                i, j, k = i.to(device), j.to(device), k.to(device)

            r = r.to(device)

            if use_btd:
                r = r.long()
                logits = model(c, i, j, k) if criterion_mode else model(i, j, k)
                loss = loss_fn(logits, r)
            else:
                p = model(i, j, k)
                loss = loss_fn(p, r)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if normalize and hasattr(model, "v"):
                with torch.no_grad():
                    model.v.weight.data = F.normalize(model.v.weight.data, p=2, dim=1)

            total_loss += loss.item() * r.size(0)

        avg_loss = total_loss / len(dataloader.dataset)
        loss_history.append(avg_loss)

        if _loss_has_plateaued(
            loss_history,
            window=int(plateau_window),
            relative_tolerance=plateau_relative_tolerance,
        ):
            print("loss plateaued, breaking")
            break

        print(f"Epoch {epoch:>3d}, Loss = {avg_loss:.4f}")

    plt.figure()
    plt.plot(range(1, len(loss_history) + 1), loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.title("Training Loss over Epochs")
    plt.tight_layout()

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        loss_path = os.path.join(save_path, "training_loss.png")
        plt.savefig(loss_path)
        print(f"Loss plot saved to {loss_path}")

        model_path = os.path.join(save_path, "model.pt")
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    return loss_history


def group_split_comparisons(comparisons, test_size=0.2, random_state=42, verbose: bool = False):
    """Grouped train/test split to reduce leakage.

    Groups by (criterion, scenario, judge, unordered evaluee pair), then keeps
    each full group entirely in either train or test.
    """

    groups = defaultdict(list)

    for comp in comparisons:
        c, l, i, j, k, r = comp
        eval_pair = tuple(sorted((j, k)))
        group_key = (l, i, eval_pair)
        groups[group_key].append(comp)

    group_keys = list(groups.keys())

    if random_state is not None:
        random.seed(random_state)
    random.shuffle(group_keys)

    n_test_groups = int(len(group_keys) * test_size)
    test_keys = set(group_keys[:n_test_groups])
    train_keys = set(group_keys[n_test_groups:])

    train_comps = []
    for key in train_keys:
        train_comps.extend(groups[key])

    test_comps = []
    for key in test_keys:
        test_comps.extend(groups[key])

    if verbose:
        print(f"Split {len(group_keys)} groups into {len(train_keys)} train groups and {len(test_keys)} test groups")
        print(f"Train comparisons: {len(train_comps)}, Test comparisons: {len(test_comps)}")

    return train_comps, test_comps
