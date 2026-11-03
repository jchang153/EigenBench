"""Seeded judge/evaluee sampling for the grouped pairwise protocol."""

from __future__ import annotations

import random


def random_groups(total_models: int, group_size: int, *, rng=random):
    indices = list(range(total_models))
    rng.shuffle(indices)
    groups = [indices[i : i + group_size] for i in range(0, len(indices), group_size)]

    if groups and len(groups[-1]) < group_size:
        used = [item for group in groups[:-1] for item in group]
        available = [idx for idx in used if idx not in groups[-1]]
        needed = group_size - len(groups[-1])
        if len(available) >= needed:
            padding = rng.sample(available, needed)
            groups[-1].extend(padding)
    return groups


def sampler_random_judge_group(num_models: int, group_size: int, *, rng=random, **kwargs):
    """Random judge + one random group of evaluees."""

    group = random_groups(num_models, group_size, rng=rng)[0]
    judge_idx = rng.randint(0, num_models - 1)
    return int(judge_idx), list(group)


def sampler_uniform(num_models: int, group_size: int, *, rng=random, **kwargs):
    """Uniform random judge and uniform random evaluee subset."""

    judge_idx = rng.randint(0, num_models - 1)
    eval_idxs = rng.sample(list(range(num_models)), k=group_size)
    return int(judge_idx), list(eval_idxs)


def select_sampler(mode: str):
    mode = (mode or "random_judge_group").strip().lower()
    if mode == "random_judge_group":
        return sampler_random_judge_group
    if mode == "uniform":
        return sampler_uniform
    raise ValueError(f"Unknown sampler mode: {mode}")
