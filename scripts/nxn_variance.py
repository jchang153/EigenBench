#!/usr/bin/env python3
"""Variance decomposition for the n×n prompt-vs-train experiment.

Pulls the 3 nxn summaries from HF (invi-bhagyesh/ValueArena), builds the
3×3 Elo matrix for each eval constitution, and runs two-way ANOVA-style
sum-of-squares decomposition:

    Y[i,j] = μ + α_i (train) + β_j (prompt) + ε_ij (interaction/residual)

Reports %SS_train, %SS_prompt, %SS_residual per eval and pooled across
the triple. Bootstrap CIs via resampling the bootstrap draws from HF's
samples.json (when available).

Usage:
    .venv/bin/python scripts/nxn_variance.py
    .venv/bin/python scripts/nxn_variance.py --triple loving sarcasm misalignment
    .venv/bin/python scripts/nxn_variance.py --no-bootstrap
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HF_REPO = "invi-bhagyesh/ValueArena"
DEFAULT_TRIPLE = ["loving", "sarcasm", "misalignment"]


def fetch_summary(constitution: str) -> dict:
    """Fetch nxn/<c>/summary.json as {nick: entry} dict."""
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(
        repo_id=HF_REPO,
        filename=f"runs/nxn/{constitution}/summary.json",
        repo_type="dataset",
        force_download=True,
    )
    with open(path) as f:
        data = json.load(f)
    return {entry["model_name"]: entry for entry in data}


def parametric_bootstrap_samples(summary: dict, triple: list[str], B: int,
                                 rng: np.random.Generator) -> list[np.ndarray]:
    """Parametric bootstrap: draw each cell's Elo from Normal(elo_mean, elo_std).
    Returns a list of B 3×3 matrices."""
    n = len(triple)
    mats = []
    for _ in range(B):
        Y = np.full((n, n), np.nan)
        for i, ci in enumerate(triple):
            for j, cj in enumerate(triple):
                nick = f"trained_{ci}__prompt_{cj}"
                if nick in summary:
                    mu = summary[nick]["elo_mean"]
                    sd = summary[nick].get("elo_std", 0.0) or 0.0
                    Y[i, j] = rng.normal(mu, sd) if sd > 0 else mu
        mats.append(Y)
    return mats


def build_matrix(summary: dict, triple: list[str]) -> np.ndarray:
    """Return 3×3 matrix Y[i,j] of Elos for trained_Ci__prompt_Cj."""
    n = len(triple)
    Y = np.full((n, n), np.nan)
    for i, ci in enumerate(triple):
        for j, cj in enumerate(triple):
            nick = f"trained_{ci}__prompt_{cj}"
            if nick in summary:
                Y[i, j] = summary[nick]["elo_mean"]
    return Y


def decompose(Y: np.ndarray) -> dict:
    """Two-way ANOVA sum-of-squares decomposition.

    Y[i,j] = μ + α_i + β_j + ε_ij

    α_i = row_mean_i - grand_mean       (training main effect)
    β_j = col_mean_j - grand_mean       (prompt  main effect)
    ε_ij = Y_ij - row_mean_i - col_mean_j + grand_mean  (residual/interaction)
    """
    mu = np.nanmean(Y)
    row_means = np.nanmean(Y, axis=1)
    col_means = np.nanmean(Y, axis=0)

    alpha = row_means - mu
    beta = col_means - mu

    n, m = Y.shape
    SS_total = np.nansum((Y - mu) ** 2)
    SS_train = m * np.sum(alpha ** 2)
    SS_prompt = n * np.sum(beta ** 2)

    resid = np.zeros_like(Y)
    for i in range(n):
        for j in range(m):
            resid[i, j] = Y[i, j] - row_means[i] - col_means[j] + mu
    SS_resid = np.nansum(resid ** 2)

    return {
        "SS_total": float(SS_total),
        "SS_train": float(SS_train),
        "SS_prompt": float(SS_prompt),
        "SS_resid": float(SS_resid),
        "pct_train": 100 * SS_train / SS_total if SS_total > 0 else 0.0,
        "pct_prompt": 100 * SS_prompt / SS_total if SS_total > 0 else 0.0,
        "pct_resid": 100 * SS_resid / SS_total if SS_total > 0 else 0.0,
        "alpha": alpha.tolist(),
        "beta": beta.tolist(),
    }


def build_matrix_from_samples(samples: dict, triple: list[str], b: int) -> np.ndarray | None:
    """Build 3×3 matrix from bootstrap replicate b. samples is {nick: [elo_0, elo_1, ...]}."""
    n = len(triple)
    Y = np.full((n, n), np.nan)
    ok = True
    for i, ci in enumerate(triple):
        for j, cj in enumerate(triple):
            nick = f"trained_{ci}__prompt_{cj}"
            if nick in samples and b < len(samples[nick]):
                Y[i, j] = samples[nick][b]
            else:
                ok = False
    return Y if ok else None


def _extract_sample_dict(raw: dict | list, triple: list[str]) -> dict:
    """Normalize bootstrap samples to {nick: [elo_draws...]}.

    Format observed on HF varies — handle both list-of-dicts and dict forms.
    """
    if isinstance(raw, dict):
        # already {nick: [...]}
        if all(isinstance(v, list) for v in raw.values()):
            return raw
        # {bootstrap_idx: {nick: elo}} — transpose
        out: dict[str, list[float]] = {}
        for _, per_run in raw.items():
            if not isinstance(per_run, dict):
                continue
            for nick, elo in per_run.items():
                out.setdefault(nick, []).append(float(elo))
        return out
    if isinstance(raw, list):
        # [{nick: elo, ...}, ...] one dict per replicate
        out: dict[str, list[float]] = {}
        for per_run in raw:
            if not isinstance(per_run, dict):
                continue
            for nick, elo in per_run.items():
                # Value may itself be a dict like {"elo": 1500, ...}
                if isinstance(elo, dict):
                    elo = elo.get("elo_mean") or elo.get("elo") or elo.get("value")
                if elo is None:
                    continue
                out.setdefault(nick, []).append(float(elo))
        return out
    return {}


def pooled_decomp(matrices: list[np.ndarray]) -> dict:
    """Pool SS across eval constitutions (k matrices) — simplest pooling:
    compute SS per matrix, then sum. pct = SS_source / sum(SS_total)."""
    totals = {"SS_total": 0.0, "SS_train": 0.0, "SS_prompt": 0.0, "SS_resid": 0.0}
    for Y in matrices:
        d = decompose(Y)
        for k in totals:
            totals[k] += d[k]
    t = totals["SS_total"]
    return {
        **totals,
        "pct_train": 100 * totals["SS_train"] / t if t > 0 else 0.0,
        "pct_prompt": 100 * totals["SS_prompt"] / t if t > 0 else 0.0,
        "pct_resid": 100 * totals["SS_resid"] / t if t > 0 else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--triple", nargs=3, default=DEFAULT_TRIPLE)
    ap.add_argument("--no-bootstrap", action="store_true")
    ap.add_argument("--bootstrap-reps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    triple = args.triple
    print(f"Triple: {triple}\n")

    # Fetch summaries
    summaries: dict[str, dict] = {}
    for c in triple:
        s = fetch_summary(c)
        summaries[c] = s
        print(f"  {c}: {len(s)} models")
    print()

    # Helper: normalize a decomposition to a 2-way split (drop residual).
    def two_way(d: dict) -> tuple[float, float]:
        denom = d["SS_prompt"] + d["SS_train"]
        if denom <= 0:
            return 0.0, 0.0
        return 100 * d["SS_prompt"] / denom, 100 * d["SS_train"] / denom

    # Point estimates per eval (raw 3-way and prompt-vs-train 2-way).
    matrices = []
    print(f"{'Eval':<18} {'%prompt':>9} {'%train':>9}   (2-way, residual dropped)")
    print("-" * 60)
    for c in triple:
        Y = build_matrix(summaries[c], triple)
        matrices.append(Y)
        d = decompose(Y)
        pp, pt = two_way(d)
        print(f"{c:<18} {pp:>8.1f}% {pt:>8.1f}%")

    pooled = pooled_decomp(matrices)
    pp, pt = two_way(pooled)
    print("-" * 60)
    print(f"{'POOLED':<18} {pp:>8.1f}% {pt:>8.1f}%")
    print()

    # Parametric bootstrap CIs — HF doesn't store per-replicate samples,
    # but summary.json has elo_mean + elo_std per model, so draw from those.
    if not args.no_bootstrap:
        B = args.bootstrap_reps
        rng = np.random.default_rng(args.seed)
        print(f"Parametric bootstrap: B={B} draws from Normal(elo_mean, elo_std)\n")

        # Per-constitution draws
        samples_per_c = {c: parametric_bootstrap_samples(summaries[c], triple, B, rng)
                         for c in triple}

        pct_p, pct_t = [], []
        for b in range(B):
            d = pooled_decomp([samples_per_c[c][b] for c in triple])
            pp_b, pt_b = two_way(d)
            pct_p.append(pp_b)
            pct_t.append(pt_b)

        def ci(arr):
            lo, hi = np.percentile(arr, [2.5, 97.5])
            return f"[{lo:5.1f}, {hi:5.1f}]"

        print(f"Pooled 95% CIs (B={B}, 2-way prompt vs train):")
        print(f"  %prompt: {np.mean(pct_p):5.1f}%  CI {ci(pct_p)}")
        print(f"  %train : {np.mean(pct_t):5.1f}%  CI {ci(pct_t)}")

        # Per-eval CIs too
        print()
        print("Per-eval 95% CIs:")
        for c in triple:
            pct_p_c, pct_t_c = [], []
            for Y_b in samples_per_c[c]:
                d_b = decompose(Y_b)
                pp_b, pt_b = two_way(d_b)
                pct_p_c.append(pp_b)
                pct_t_c.append(pt_b)
            print(f"  {c:<14}  %prompt {np.mean(pct_p_c):5.1f}% CI {ci(pct_p_c)}   "
                  f"%train {np.mean(pct_t_c):5.1f}% CI {ci(pct_t_c)}")


if __name__ == "__main__":
    # Strip SOCKS proxies — break huggingface_hub
    import os
    for k in list(os.environ):
        if k.lower() in ("all_proxy", "ftp_proxy", "grpc_proxy", "rsync_proxy"):
            os.environ.pop(k, None)
    main()
