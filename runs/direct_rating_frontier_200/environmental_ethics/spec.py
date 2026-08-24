from runs.direct_rating_frontier_200.common import build_spec


RUN_SPEC = build_spec(
    name="direct_rating_frontier_200_environmental_ethics",
    constitution_path="data/constitutions/deep_ecology.json",
    num_criteria=12,
)
