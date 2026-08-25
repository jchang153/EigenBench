from runs.direct_rating_frontier_200.common import build_spec


RUN_SPEC = build_spec(
    name="direct_rating_frontier_200_conservatism",
    constitution_path="data/constitutions/conservatism.json",
    num_criteria=10,
)
