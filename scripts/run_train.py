"""Run BT/BTD training from a Python run spec.

Usage:
    python scripts/run_train.py runs.example_custom
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader

from pipeline.config import load_run_spec
from pipeline.io import (
    load_records,
    extract_comparisons_with_ties_criteria,
    handle_inconsistencies_with_ties_criteria,
)
from pipeline.train import (
    CriteriaComparisons,
    Comparisons,
    CriteriaVectorBTD,
    VectorBT,
    train_vector_bt,
    group_split_comparisons,
)
from pipeline.trust import compute_trust_matrix_ties, compute_trust_matrix, row_normalize, eigentrust


def _resolve_output_root(evaluations_path: str, train_cfg: dict) -> str:
    configured = train_cfg.get("output_dir")
    if configured:
        return configured
    evaluations_dir = os.path.dirname(evaluations_path) or "."
    return os.path.join(evaluations_dir, "train")


def main(spec_ref: str):
    _spec, run_dir = load_run_spec(spec_ref)

    train_cfg = _spec["training"]
    if not train_cfg.get("enabled", True):
        print("Training disabled in run spec.")
        return

    collection_cfg = _spec["collection"]
    evaluations_path = collection_cfg.get("evaluations_path")
    if not evaluations_path:
        raise SystemExit("Set collection.evaluations_path in your run spec.")

    print(f"Run: {_spec['name']}")
    print(f"Run folder: {run_dir}")
    print(f"Evaluations file: {evaluations_path}")

    data = load_records(evaluations_path)

    num_criteria = int(train_cfg.get("num_criteria", 8))
    comparisons, _ = extract_comparisons_with_ties_criteria(data, num_criteria=num_criteria)
    comparisons = handle_inconsistencies_with_ties_criteria(comparisons)

    if not train_cfg.get("separate_criteria", False):
        comparisons = [[0] + i[1:] for i in comparisons]

    num_models = len(set([i[2] for i in comparisons] + [i[3] for i in comparisons] + [i[4] for i in comparisons]))
    num_criteria_eff = len(set([i[0] for i in comparisons]))

    model_kind = train_cfg.get("model", "btd_ties")

    batch_size = int(train_cfg.get("batch_size", 32))
    lr = float(train_cfg.get("lr", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    max_epochs = int(train_cfg.get("max_epochs", 100))
    device = train_cfg.get("device", "cpu")
    dims = list(train_cfg.get("dims", [2]))

    out_root = _resolve_output_root(evaluations_path, train_cfg)
    os.makedirs(out_root, exist_ok=True)
    print(f"Training outputs root: {out_root}")

    if train_cfg.get("group_split", False):
        print("Using grouped split: keeps the same (scenario, judge, evaluee-pair) group entirely in train or test.")
        train_comps, test_comps = group_split_comparisons(
            comparisons,
            test_size=float(train_cfg.get("test_size", 0.2)),
            random_state=42,
        )
    else:
        print("Using random row split over comparisons.")
        train_comps, test_comps = train_test_split(comparisons, test_size=float(train_cfg.get("test_size", 0.2)), random_state=42, shuffle=True)

    for d in dims:
        out_dir = os.path.join(out_root, f"btd_d{d}")
        os.makedirs(out_dir, exist_ok=True)

        if model_kind == "btd_ties":
            model = CriteriaVectorBTD(num_criteria_eff, num_models, d)
            train_loader = DataLoader(CriteriaComparisons(train_comps), batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(CriteriaComparisons(test_comps), batch_size=batch_size, shuffle=False)
            criterion_mode = True
            use_btd = True
        else:
            model = VectorBT(num_models, d)
            train_loader = DataLoader(Comparisons([[0] + c[1:] for c in train_comps]), batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(Comparisons([[0] + c[1:] for c in test_comps]), batch_size=batch_size, shuffle=False)
            criterion_mode = False
            use_btd = False

        loss_history = train_vector_bt(
            model=model,
            dataloader=train_loader,
            lr=lr,
            weight_decay=weight_decay,
            max_epochs=max_epochs,
            device=device,
            save_path=out_dir,
            normalize=False,
            use_btd=use_btd,
            criterion_mode=criterion_mode,
        )

        model.eval()
        loss_fn = nn.CrossEntropyLoss() if use_btd else nn.BCELoss()
        total_test_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                if criterion_mode:
                    c, i, j, k, r = batch
                    c, i, j, k = c.to(device), i.to(device), j.to(device), k.to(device)
                else:
                    i, j, k, r = batch
                    i, j, k = i.to(device), j.to(device), k.to(device)
                r = r.to(device)

                if use_btd:
                    r = r.long()
                    logits = model(c, i, j, k)
                    loss = loss_fn(logits, r)
                else:
                    p = model(i, j, k)
                    loss = loss_fn(p, r)
                total_test_loss += loss.item() * r.size(0)

        avg_test_loss = total_test_loss / max(1, len(test_loader.dataset))

        # Always compute EigenTrust at the end of every training run.
        if use_btd:
            T = compute_trust_matrix_ties(model, device)
            t = eigentrust(T, alpha=0)
        else:
            S = compute_trust_matrix(model, device)
            C = row_normalize(S)
            t = eigentrust(C, alpha=0)

        trust_path = os.path.join(out_dir, "eigentrust.txt")
        with open(trust_path, "w", encoding="utf-8") as f:
            f.write("EigenTrust scores:\n")
            f.write(np.array2string(t.cpu().numpy(), separator=", "))
            f.write("\n")

        log_path = os.path.join(out_dir, "log_train.txt")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"train_datasize = {len(train_comps)}\n")
            f.write(f"test_datasize = {len(test_comps)}\n")
            f.write(f"num_models = {num_models}\n")
            f.write(f"num_criteria = {num_criteria_eff}\n")
            f.write(f"dim = {d}\n")
            f.write(f"lr = {lr}\n")
            f.write(f"epochs = {max_epochs}\n")
            f.write(f"min_train_loss = {np.round(min(loss_history), 6)}\n")
            f.write(f"test_loss = {np.round(avg_test_loss, 6)}\n")

        print(f"Finished dim={d}; outputs in {out_dir}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python scripts/run_train.py <spec_module_or_path>")
    main(sys.argv[1])
