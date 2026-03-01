"""Sampling policies for judge/evaluee selection.

Practical defaults for most runs:
- mode: random_judge_group
- group_size: 4

Coverage guidance:
- Increase `groups` in collection first (e.g. 2-3) before increasing
  `group_size`.

Adaptive guidance:
- Use adaptive_inverse_count when you want balancing toward under-sampled
  judges/evaluees.
- `alpha` controls aggressiveness:
  - alpha = 0 is equivalent to uniform weighting.
  - 1.0-2.0 is a practical range for most runs.
"""

from __future__ import annotations

import random
import numpy as np


def random_groups(total_models: int, group_size: int):
    indices = list(range(total_models))
    random.shuffle(indices)
    groups = [indices[i : i + group_size] for i in range(0, len(indices), group_size)]

    if groups and len(groups[-1]) < group_size:
        used = [item for group in groups[:-1] for item in group]
        available = [idx for idx in used if idx not in groups[-1]]
        needed = group_size - len(groups[-1])
        if len(available) >= needed:
            padding = random.sample(available, needed)
            groups[-1].extend(padding)
    return groups


def sampler_random_judge_group(num_models: int, group_size: int, **kwargs):
    """Random judge + one random group of evaluees."""

    group = random_groups(num_models, group_size)[0]
    judge_idx = random.randint(0, num_models - 1)
    return int(judge_idx), list(group)


def sampler_uniform(num_models: int, group_size: int, **kwargs):
    """Uniform random judge and uniform random evaluee subset."""

    judge_idx = random.randint(0, num_models - 1)
    eval_idxs = random.sample(list(range(num_models)), k=group_size)
    return int(judge_idx), list(eval_idxs)


def sampler_adaptive_inverse_count(num_models: int, group_size: int, judge_counts, eval_counts, alpha: float = 2.0, **kwargs):
    """Inverse-count weighted sampling for judge and evaluees."""

    judge_arr = np.array(judge_counts, dtype=float)
    j_weights = 1.0 / (1.0 + judge_arr) ** alpha
    j_probs = j_weights / j_weights.sum()
    judge_idx = int(np.random.choice(num_models, p=j_probs))

    eval_arr = np.array(eval_counts, dtype=float)
    e_weights = 1.0 / (1.0 + eval_arr) ** alpha
    e_probs = e_weights / e_weights.sum()
    eval_idxs = np.random.choice(num_models, size=group_size, replace=False, p=e_probs).tolist()

    return judge_idx, eval_idxs


def select_sampler(mode: str):
    mode = (mode or "random_judge_group").strip().lower()
    if mode == "random_judge_group":
        return sampler_random_judge_group
    if mode == "adaptive_inverse_count":
        return sampler_adaptive_inverse_count
    if mode == "uniform":
        return sampler_uniform
    raise ValueError(f"Unknown sampler mode: {mode}")
