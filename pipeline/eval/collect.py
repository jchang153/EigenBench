"""Core evaluation collection orchestration."""

from __future__ import annotations

from pipeline.utils import extract_comparisons_with_ties_criteria
from .criteria_collectors import collect_group_criteria_evaluations
from .samplers import select_sampler


def build_judge_and_eval_counts(comparisons, num_models: int):
    judge_counts = [0] * num_models
    eval_counts = [0] * num_models

    for _, _, judge, eval1, eval2, _ in comparisons:
        if 0 <= judge < num_models:
            judge_counts[judge] += 1
        if 0 <= eval1 < num_models:
            eval_counts[eval1] += 1
        if 0 <= eval2 < num_models:
            eval_counts[eval2] += 1

    return judge_counts, eval_counts


def collect_core_evaluations(
    criteria,
    scenario,
    scenario_index,
    models,
    evaluations,
    sampler_mode="random_judge_group",
    allow_ties=True,
    group_size=4,
    groups=1,
    alpha=2.0,
    cached_responses_by_scenario=None,
    judge_prompt_prefix_fn=None,
    max_tokens=4096,
    verbose: bool = False,
):
    """Collect one scenario's criterion-wise evaluations.

    Args:
        sampler_mode:
            - random_judge_group: recommended default.
            - adaptive_inverse_count: balances under-sampled judges/evaluees.
            - uniform: baseline.
        group_size: Number of evaluees judged together in each sampled group.
            Recommended default is 4 for most populations.
        groups: Number of sampled judge+group batches to run for this scenario.
            If you need more coverage, increase this before increasing group_size.
        alpha: In adaptive inverse-count sampling, larger alpha increases
            preference for under-sampled judges/evaluees. alpha=0 is uniform.
            Practical range is usually 1.0-2.0.
    """

    num_models = len(models)
    mode = (sampler_mode or "random_judge_group").strip().lower()

    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if group_size > num_models:
        group_size = num_models
    group_count = max(1, int(groups))

    sampler = select_sampler(mode)
    new_evaluations = []

    for round_idx in range(group_count):
        if mode in {"adaptive_inverse_count", "uniform"}:
            all_evals = list(evaluations) + list(new_evaluations)
            if all_evals:
                comparisons, _ = extract_comparisons_with_ties_criteria(
                    all_evals,
                    num_criteria=len(criteria),
                    verbose=verbose,
                )
                judge_counts, eval_counts = build_judge_and_eval_counts(
                    comparisons,
                    num_models=num_models,
                )
            else:
                judge_counts = [0] * num_models
                eval_counts = [0] * num_models

            adaptive_alpha = 0.0 if mode == "uniform" else alpha
            selected_judge, eval_idxs = sampler(
                num_models=num_models,
                group_size=group_size,
                judge_counts=judge_counts,
                eval_counts=eval_counts,
                alpha=adaptive_alpha,
            )
        else:
            selected_judge, eval_idxs = sampler(
                num_models=num_models,
                group_size=group_size,
            )

        if verbose:
            print(f"Group round {round_idx + 1}/{group_count}")
        batch_evaluations = collect_group_criteria_evaluations(
            criteria=criteria,
            scenario=scenario,
            scenario_index=scenario_index,
            models=models,
            judge_idx=selected_judge,
            eval_idxs=eval_idxs,
            allow_ties=allow_ties,
            max_tokens=max_tokens,
            cached_responses_by_scenario=cached_responses_by_scenario,
            judge_prompt_prefix_fn=judge_prompt_prefix_fn,
            verbose=verbose,
        )
        new_evaluations.extend(batch_evaluations)

    return new_evaluations
