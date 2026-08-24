from runs.direct_rating_frontier_200.common import build_spec


RUN_SPEC = build_spec(
    name="direct_rating_frontier_200_kindness",
    constitution_path="data/constitutions/kindness.json",
    num_criteria=8,
    valuearena_slug="kindness",
)
