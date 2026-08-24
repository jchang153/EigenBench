"""Core evaluation collection orchestration."""

from __future__ import annotations

import random

from pipeline.utils import extract_comparisons_with_ties_criteria
from .criteria_collectors import collect_group_criteria_evaluations
from .samplers import select_sampler


HISTORY_FREE_SAMPLER_MODES = frozenset({"random_judge_group"})


def sampler_needs_history(sampler_mode: str | None) -> bool:
    """True when the sampler reads existing evaluations to pick its next draw."""

    mode = (sampler_mode or "random_judge_group").strip().lower()
    return mode not in HISTORY_FREE_SAMPLER_MODES


def plan_group_assignments(
    *,
    selected_scenarios,
    num_models: int,
    sampler_mode: str = "random_judge_group",
    group_size: int = 4,
    groups: int = 1,
    seed: int | None = None,
):


    mode = (sampler_mode or "random_judge_group").strip().lower()
    if sampler_needs_history(mode):
        raise ValueError(
            f"sampler_mode={mode!r} depends on collected evaluations and cannot "
            "be planned up front; collect it sequentially instead."
        )

    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if group_size > num_models:
        group_size = num_models
    group_count = max(1, int(groups))

    sampler = select_sampler(mode)

    # The samplers use the module-level `random`, so seed it here to make the
    # planned selections reproducible. Restore the caller's state afterwards.
    state = random.getstate()
    if seed is not None:
        random.seed(seed)
    try:
        assignments = []
        for scenario_index, scenario in selected_scenarios:
            for round_idx in range(group_count):
                judge_idx, eval_idxs = sampler(
                    num_models=num_models,
                    group_size=group_size,
                )
                assignments.append(
                    {
                        "scenario_index": scenario_index,
                        "scenario": scenario,
                        "judge_idx": judge_idx,
                        "eval_idxs": eval_idxs,
                        "round_idx": round_idx,
                    }
                )
    finally:
        random.setstate(state)

    return assignments


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
    extra_body_for=None,
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
            extra_body_for=extra_body_for,
            verbose=verbose,
        )
        new_evaluations.extend(batch_evaluations)

    return new_evaluations


def collect_planned_evaluations(
    *,
    assignments,
    criteria,
    models,
    allow_ties: bool = True,
    max_tokens: int = 4096,
    cached_responses_by_scenario=None,
    judge_prompt_prefix_fn=None,
    max_workers: int = 1,
    on_batch=None,
    extra_body_for=None,
    skip_failed_groups: bool = False,
    max_failed_groups: int | None = None,
    verbose: bool = False,
):
    """Collect pre-planned assignments, in parallel when max_workers > 1.

    Assignments from `plan_group_assignments` are mutually independent, so they
    can run concurrently. `on_batch(assignment, evaluations)` is invoked from
    the calling thread as each assignment completes, letting callers append
    incrementally so partial progress survives a failure.

    skip_failed_groups: when True, an assignment whose provider calls fail is
        dropped and collection continues. One flaky endpoint in a large panel
        otherwise discards an entire expensive run. Returns the failures so the
        caller can report which models lost coverage.
    max_failed_groups: abort once this many assignments have failed, so a
        systematically broken model (bad id, revoked key) still stops the run
        instead of quietly producing a panel with a missing participant.

    Returns (evaluations, failures) where failures is a list of
    (assignment, exception).
    """

    workers = max(1, int(max_workers))

    def run_one(assignment):
        return collect_group_criteria_evaluations(
            criteria=criteria,
            scenario=assignment["scenario"],
            scenario_index=assignment["scenario_index"],
            models=models,
            judge_idx=assignment["judge_idx"],
            eval_idxs=assignment["eval_idxs"],
            allow_ties=allow_ties,
            max_tokens=max_tokens,
            cached_responses_by_scenario=cached_responses_by_scenario,
            judge_prompt_prefix_fn=judge_prompt_prefix_fn,
            extra_body_for=extra_body_for,
            verbose=verbose,
        )

    collected = []
    failures: list[tuple[dict, BaseException]] = []

    def note_failure(assignment, exc):
        """Record a failed assignment. Returns True if the run should abort."""

        failures.append((assignment, exc))
        judge_nick = list(models.keys())[assignment["judge_idx"]]
        print(
            f"[collect] SKIPPED scenario {assignment['scenario_index']} "
            f"(judge {judge_nick}): {type(exc).__name__}: {exc}"
        )
        return max_failed_groups is not None and len(failures) >= int(max_failed_groups)

    if workers == 1:
        for assignment in assignments:
            try:
                batch = run_one(assignment)
            except Exception as exc:
                if not skip_failed_groups:
                    raise
                if note_failure(assignment, exc):
                    break
                continue
            if on_batch is not None:
                on_batch(assignment, batch)
            collected.extend(batch)
        return collected, failures

    from concurrent.futures import FIRST_COMPLETED, FIRST_EXCEPTION, ThreadPoolExecutor, wait

    return_when = FIRST_COMPLETED if skip_failed_groups else FIRST_EXCEPTION

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_one, a): a for a in assignments}
        pending = set(futures)
        while pending:
            done, pending = wait(pending, return_when=return_when)
            error = None
            abort = False
            for future in done:
                assignment = futures[future]
                exc = future.exception()
                if exc is not None:
                    if not skip_failed_groups:
                        error = error or exc
                        continue
                    if note_failure(assignment, exc):
                        abort = True
                    continue
                batch = future.result()
                if on_batch is not None:
                    on_batch(assignment, batch)
                collected.extend(batch)
            if error is not None or abort:
                # Hand over every batch that did succeed before aborting, so a
                # single failing assignment does not discard the whole wave.
                for future in pending:
                    future.cancel()
                if error is not None:
                    raise error
                print(
                    f"[collect] aborting: {len(failures)} failed groups "
                    f"reached max_failed_groups={max_failed_groups}"
                )
                break

    return collected, failures
