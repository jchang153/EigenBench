"""EigenBench direct rating as a native Inspect AI task.

    inspect eval inspect_pipeline/eigenbench.py -T spec=runs/my_run/spec.py

One sample per directed judge->evaluee edge. The sampling plan, prompts, and
rating validation come from ``pipeline.eval.direct_rating``; export the
resulting log to the legacy ``evaluations.jsonl`` with
``scripts/export_evaluations.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Inspect loads this file standalone (`inspect eval .../eigenbench.py`), so the
# repo root must be importable before the absolute imports below.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import Model, get_model
from inspect_ai.viewer import (
    TaskSamplesColumn,
    TaskSamplesSort,
    TaskSamplesView,
    ViewerConfig,
)

from pipeline.config import (
    get_criteria_from_spec,
    load_dataset_scenarios_from_spec,
    load_run_spec,
    select_scenarios,
)
from pipeline.eval.direct_rating import (
    _load_cached_responses,  # noqa: PLC2701 - reuse the legacy cache reader
    build_direct_assignments,
    resolve_direct_generation_settings,
    resolve_direct_sampling_settings,
)

from inspect_pipeline.model_mapping import to_inspect_model
from inspect_pipeline.phases import (
    ResponsePool,
    criterion_key,
    criterion_label,
    direct_rating_scorer,
    direct_rating_solver,
)

DEFAULT_MAX_ATTEMPTS = 4


def resolve_spec_ref(spec: str) -> str:
    """Resolve a relative spec path against the repo root.

    Inspect runs a task file with its own directory as the working directory,
    so `-T spec=runs/...` would otherwise resolve under inspect_pipeline/.
    Dotted module refs are passed through untouched.
    """

    looks_like_path = spec.endswith(".py") or "/" in spec or "\\" in spec
    if not looks_like_path:
        return spec
    path = Path(spec).expanduser()
    if path.is_absolute() or path.exists():
        return spec
    candidate = _REPO_ROOT / path
    return str(candidate) if candidate.exists() else spec


def load_selection(spec: dict, run_dir: Path):
    """Scenario + criteria selection, mirroring scripts/run_collect.py."""

    ds = spec["dataset"]
    scenarios = load_dataset_scenarios_from_spec(ds, run_dir=run_dir)
    count = ds.get("count")
    shuffle_seed = ds.get("shuffle_seed")
    selected = select_scenarios(
        scenarios,
        start=int(ds.get("start", 0)),
        count=None if count is None else int(count),
        shuffle=bool(ds.get("shuffle", False)),
        shuffle_seed=None if shuffle_seed is None else int(shuffle_seed),
    )

    constitution = spec["constitution"]
    if "num_criteria" not in constitution:
        raise ValueError(
            "Set constitution.num_criteria in your run spec. "
            "This controls criterion truncation during collection."
        )
    requested = int(constitution["num_criteria"])
    if requested <= 0:
        raise ValueError("constitution.num_criteria must be a positive integer.")
    criteria = get_criteria_from_spec(constitution, run_dir=run_dir)
    if requested < len(criteria):
        criteria = criteria[:requested]
    elif requested > len(criteria):
        raise ValueError(
            f"constitution.num_criteria={requested} exceeds criteria found in "
            f"constitution file ({len(criteria)})."
        )
    return selected, criteria


def build_edge_samples(assignments: list[dict]) -> list[Sample]:
    """Expand assignments into one sample per directed edge.

    ``edge_index`` preserves legacy record ordering through the log.
    """

    samples: list[Sample] = []
    for assignment in assignments:
        s_idx = int(assignment["scenario_index"])
        for eval_idx, eval_nick in zip(
            assignment["eval_idxs"], assignment["eval_nicks"]
        ):
            samples.append(
                Sample(
                    input=assignment["scenario"],
                    id=(
                        f"s{s_idx:04d} r{assignment.get('sampling_round', 0)} · "
                        f"{assignment['judge_nick']} → {eval_nick}"
                    ),
                    metadata={
                        "edge_index": len(samples),
                        "scenario_index": s_idx,
                        "scenario": assignment["scenario"],
                        "judge_idx": int(assignment["judge_idx"]),
                        "judge_nick": assignment["judge_nick"],
                        "eval_idx": int(eval_idx),
                        "eval_nick": eval_nick,
                        "sampler_mode": assignment.get("sampler_mode"),
                        "sampling_round": int(assignment.get("sampling_round", 0)),
                        "group_index": int(assignment.get("group_index", 0)),
                    },
                )
            )
    return samples


def _model_resolver(models: dict[str, object]):
    """Resolve nick -> Model lazily so tasks build without provider API keys."""

    # Validate the mapping eagerly; only client creation is deferred.
    refs = {
        nick: (value if isinstance(value, Model) else to_inspect_model(value))
        for nick, value in models.items()
    }
    cache: dict[str, Model] = {}

    def resolve(nick: str) -> Model:
        model = cache.get(nick)
        if model is None:
            ref = refs[nick]
            model = ref if isinstance(ref, Model) else get_model(ref.name, **ref.model_args)
            cache[nick] = model
        return model

    return resolve


@task
def eigenbench(
    spec: str,
    *,
    models: dict[str, object] | None = None,
    cache: bool | None = None,
) -> Task:
    """Direct-rating EigenBench task.

    Args:
        spec: run spec module or path, e.g. ``runs/my_run/spec.py``.
        models: optional override of the spec's models (Python callers only).
        cache: override ``collection.inspect.cache``.
    """

    run_spec, run_dir = load_run_spec(resolve_spec_ref(spec))
    if run_spec.get("evaluation", {}).get("mode") != "direct_rating":
        raise ValueError(
            "inspect_pipeline.eigenbench supports evaluation.mode='direct_rating' "
            "only; use scripts/run.py for pairwise BTD runs."
        )

    spec_models = models if models is not None else run_spec["models"]
    if not spec_models:
        raise ValueError("spec models must not be empty")

    evaluation_cfg = run_spec.get("evaluation", {})
    collection_cfg = run_spec.get("collection", {})
    direct_cfg = evaluation_cfg.get("direct_rating", {}) or {}
    include_self = bool(direct_cfg.get("include_self", True))
    if not include_self and len(spec_models) < 2:
        raise ValueError("include_self=False requires at least two models")
    scale_min = int(direct_cfg.get("scale_min", 1))
    scale_max = int(direct_cfg.get("scale_max", 10))
    if (scale_min, scale_max) != (1, 10):
        raise ValueError("direct rating collection currently uses the fixed 1-10 scale")

    selected, criteria = load_selection(run_spec, run_dir)
    sampling = resolve_direct_sampling_settings(
        collection_cfg, num_models=len(spec_models), include_self=include_self
    )
    generation = resolve_direct_generation_settings(collection_cfg)
    assignments = build_direct_assignments(
        selected, spec_models, include_self=include_self, **sampling
    )

    inspect_cfg = collection_cfg.get("inspect", {}) or {}
    cache_enabled = bool(inspect_cfg.get("cache", True)) if cache is None else bool(cache)
    max_attempts = int(
        (collection_cfg.get("openrouter", {}) or {}).get(
            "max_attempts", DEFAULT_MAX_ATTEMPTS
        )
    )

    # Seed the pool so cached responses are never regenerated.
    cached = _load_cached_responses(collection_cfg.get("cached_responses_path"))
    seed = {
        (int(s_idx), nick): text
        for s_idx, responses in cached.items()
        for nick, text in responses.items()
        if nick in spec_models
    }

    resolve_model = _model_resolver(spec_models)
    pool = ResponsePool(seed=seed)

    scorer_name = "direct_rating_scorer"
    samples_view = TaskSamplesView(
        name="Judgments",
        columns=[
            TaskSamplesColumn(id="sampleId"),
            TaskSamplesColumn(id="answer"),
            TaskSamplesColumn.score(scorer_name, "mean"),
            *[
                TaskSamplesColumn.score(scorer_name, criterion_key(i))
                for i in range(len(criteria))
            ],
            TaskSamplesColumn(id="input", visible=False),
            TaskSamplesColumn(id="tokens", visible=False),
        ],
        # Weakest judgments first: the point of reading these is finding where a
        # judge broke from the pack, not admiring the middle of the scale.
        sort=[TaskSamplesSort.score(scorer_name, "mean", dir="asc")],
        multiline=False,
        compact_scores=True,
        color_scales_enabled=True,
        score_labels={
            "mean": "Mean",
            **{criterion_key(i): criterion_label(i, c) for i, c in enumerate(criteria)},
        },
        score_color_scales={
            "mean": "good-high",
            **{criterion_key(i): "good-high" for i in range(len(criteria))},
        },
    )

    return Task(
        dataset=MemoryDataset(
            samples=build_edge_samples(assignments), name=f"eigenbench_{run_spec['name']}"
        ),
        solver=direct_rating_solver(
            criteria=criteria,
            resolve_model=resolve_model,
            response_pool=pool,
            generation=generation,
            max_attempts=max_attempts,
            cache_enabled=cache_enabled,
            scale_min=scale_min,
            scale_max=scale_max,
        ),
        scorer=direct_rating_scorer(),
        viewer=ViewerConfig(task_samples_view=samples_view),
        name=f"eigenbench_{run_spec['name']}",
        display_name=f"EigenBench direct rating — {run_spec['name']}",
        metadata={
            "eigenbench": {
                "run_name": run_spec["name"],
                "criteria": criteria,
                "model_order": list(spec_models),
                "include_self": include_self,
                "scale_min": scale_min,
                "scale_max": scale_max,
                "sampler_mode": sampling["sampler_mode"],
                "evaluations_path": collection_cfg.get("evaluations_path"),
                "cached_responses_path": collection_cfg.get("cached_responses_path"),
                "num_scenarios": len(selected),
            }
        },
    )
