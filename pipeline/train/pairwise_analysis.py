"""Shared pairwise fitting used by native Inspect metrics and artifact analysis.

The legacy comparison cleaning, split, fit, EigenTrust and bootstrap order are
unchanged. Artifact output is optional so Inspect can consume the same result.
"""
from __future__ import annotations

import os

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader

from pipeline.utils import (
    extract_comparisons_with_ties_criteria,
    handle_inconsistencies_with_ties_criteria,
)
from pipeline.train import (
    CriteriaComparisons,
    Comparisons,
    CriteriaVectorBTD,
    VectorBT,
    build_model_labels,
    eigentrust_to_elo,
    train_vector_bt,
    group_split_comparisons,
    save_uv_embedding_plot,
    save_eigenbench_plot,
)
from pipeline.trust import compute_trust_matrix_ties, compute_trust_matrix, row_normalize, eigentrust


def validate_pairwise_ranking(training_cfg: dict, num_criteria: int) -> None:
    """Reject unsupported trust matrices before fitting or collecting."""
    if training_cfg.get("separate_criteria", False) and num_criteria > 1:
        raise ValueError(
            "Multi-criterion separate_criteria produces a rectangular trust matrix; "
            "EigenTrust requires an approved square aggregation. Set separate_criteria=False."
        )
    dims = training_cfg.get("dims", [2])
    if not dims or any(type(dim) is not int or dim <= 0 for dim in dims):
        raise ValueError("training.dims must contain positive integers")
    if int(training_cfg.get("max_epochs", 100)) <= 0:
        raise ValueError("training.max_epochs must be positive")
    if training_cfg.get("model", "btd_ties") != "btd_ties":
        raise ValueError("Native EigenBench rankings support training.model='btd_ties'; "
                         "the legacy BT comparison adapter is unresolved (#41).")


def run_pairwise_analysis(*, data: list[dict], models: dict, num_criteria: int,
                          train_cfg: dict, output_root=None, verbose: bool = False,
                          model_order: list[str] | None = None) -> dict:
    validate_pairwise_ranking(train_cfg, num_criteria)
    comparisons, _, extracted_name_map, groups = extract_comparisons_with_ties_criteria(
        data,
        num_criteria=num_criteria,
        verbose=verbose,
        return_name_map=True,
        return_groups=True,
    )
    parsed_count = len(comparisons)
    comparisons = handle_inconsistencies_with_ties_criteria(comparisons, groups)
    diagnostics = {"comparisons_parsed": parsed_count,
                   "comparisons_retained": len(comparisons),
                   "comparisons_dropped": parsed_count - len(comparisons)}
    if not comparisons:
        return {"diagnostics": diagnostics, "dimensions": {},
                "unavailable_reason": "Legacy comparison cleaning retained no rows (#41)."}

    if not train_cfg.get("separate_criteria", False):
        comparisons = [[0] + i[1:] for i in comparisons]

    # Historical records keep their numeric IDs and the baseline label/count
    # resolution. Native scoring passes an explicit order after remapping IDs.
    num_models = (len(model_order) if model_order is not None else
                  len(set([i[2] for i in comparisons] + [i[3] for i in comparisons]
                          + [i[4] for i in comparisons])))
    num_criteria_eff = len(set([i[0] for i in comparisons]))
    if model_order is None:
        largest = max(max(row[2:5]) for row in comparisons)
        if largest >= num_models:
            raise ValueError(
                f"records use model index {largest} but only {num_models} models appear; "
                "model indices must run from 0 without gaps"
            )
    model_labels = (list(model_order) if model_order is not None else
                    build_model_labels(num_models, models, extracted_name_map))

    model_kind = train_cfg.get("model", "btd_ties")

    batch_size = int(train_cfg.get("batch_size", 32))
    lr = float(train_cfg.get("lr", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    max_epochs = int(train_cfg.get("max_epochs", 100))
    device = train_cfg.get("device", "cpu")
    dims = list(train_cfg.get("dims", [2]))

    out_root = str(output_root) if output_root is not None else None
    if out_root is not None:
        os.makedirs(out_root, exist_ok=True)
    if verbose:
        print(f"Training outputs root: {out_root}")

    if train_cfg.get("group_split", False):
        if verbose:
            print("Using grouped split: keeps the same (scenario, judge, evaluee-pair) group entirely in train or test.")
        train_comps, test_comps = group_split_comparisons(
            comparisons,
            test_size=float(train_cfg.get("test_size", 0.2)),
            random_state=42,
            verbose=verbose,
        )
    else:
        if verbose:
            print("Using random row split over comparisons.")
        train_comps, test_comps = train_test_split(comparisons, test_size=float(train_cfg.get("test_size", 0.2)), random_state=42, shuffle=True)

    dimensions = {}
    for d in dims:
        out_dir = os.path.join(out_root, f"btd_d{d}") if out_root is not None else None
        if out_dir is not None:
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
            verbose=verbose,
            plot=out_dir is not None,
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

        avg_test_loss = total_test_loss / len(test_loader.dataset) if len(test_loader.dataset) else None

        # Always compute EigenTrust at the end of every training run.
        if use_btd:
            T = compute_trust_matrix_ties(model, device)
            t = eigentrust(T, alpha=0, verbose=verbose)
        else:
            S = compute_trust_matrix(model, device)
            C = row_normalize(S)
            t = eigentrust(C, alpha=0, verbose=verbose)

        # Convert trust scores to Elo for display in embedding plot legend.
        elo_np = eigentrust_to_elo(t.detach().cpu().numpy(), num_models)

        if not np.isfinite(t.detach().cpu().numpy()).all() or not np.isfinite(elo_np).all():
            raise ValueError("Pairwise analysis produced non-finite ranking values (#41).")

        if out_dir is not None:
            # Save u/v embedding visualization (2D PCA) with model names + EB scores.
            try:
                uv_plot_path = os.path.join(out_dir, "uv_embeddings_pca.png")
                save_uv_embedding_plot(
                    model=model,
                    model_names=model_labels,
                    save_path=uv_plot_path,
                )
                print(f"u/v PCA plot saved to {uv_plot_path}")
            except Exception as e:
                print(f"Skipping u/v PCA plot due to error: {e}")

            # Save separate EigenBench Elo plot (sorted highest -> lowest).
            try:
                eigenbench_plot_path = os.path.join(out_dir, "eigenbench.png")
                save_eigenbench_plot(
                    model_names=model_labels,
                    eigentrust_elo=elo_np,
                    save_path=eigenbench_plot_path,
                )
                print(f"EigenBench plot saved to {eigenbench_plot_path}")
            except Exception as e:
                print(f"Skipping EigenBench plot due to error: {e}")

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
                if avg_test_loss is not None:
                    f.write(f"test_loss = {np.round(avg_test_loss, 6)}\n")

            print(f"Finished dim={d}; outputs in {out_dir}")

        dimensions[d] = {"eigentrust": t.detach().cpu().numpy(), "elo": elo_np}

        # Bootstrap resampling (optional)
        bootstrap_cfg = train_cfg.get("bootstrap")
        if bootstrap_cfg and bootstrap_cfg.get("enabled", False):
            from pipeline.train import run_bootstrap

            bootstrap_dir = os.path.join(out_dir, "bootstrap") if out_dir is not None else None
            print(f"Stage: bootstrap resampling ({bootstrap_cfg.get('n_bootstraps', 100)} samples)")
            dimensions[d]["bootstrap"] = run_bootstrap(
                comparisons=comparisons,
                num_models=num_models,
                num_criteria=num_criteria_eff,
                model_kind=model_kind,
                dim=d,
                model_labels=model_labels,
                output_dir=bootstrap_dir,
                n_bootstraps=int(bootstrap_cfg.get("n_bootstraps", 100)),
                random_seed=int(bootstrap_cfg.get("random_seed", 42)),
                batch_size=batch_size,
                lr=lr,
                weight_decay=weight_decay,
                max_epochs=max_epochs,
                device=device,
                save_models=bool(bootstrap_cfg.get("save_models", False)),
                save_trust_matrices=bool(bootstrap_cfg.get("save_trust_matrices", True)),
                verbose=verbose,
            )
            print(f"Bootstrap outputs in {bootstrap_dir}")
    return {"diagnostics": diagnostics, "dimensions": dimensions, "unavailable_reason": None}
