# Native pairwise protocol

The compatibility reference is Jonathan Chang's OpenRouter group collector at
EigenBench commit [`38dcca0fc50815eb273cf548f1a4855d265b70f9`](https://github.com/jchang153/EigenBench/tree/38dcca0fc50815eb273cf548f1a4855d265b70f9).
The implementation ports `pipeline/eval/criteria_collectors.py`,
`pipeline/eval/collect.py` and `pipeline/eval/samplers.py` into Inspect execution.
The later mixed local collector is a different collection protocol.

The [paper v4](https://arxiv.org/html/2509.01938v4), section 3.4 and appendices C
and D, describes the response, individual reflection and comparison scaffold,
including comparisons in both response orders. Appendix D says several
collection algorithms were used. It does not specify decoding settings or map
each reported result to a collector commit. Historical evaluation outputs are
not tracked in this repository. These fixtures therefore establish code-level
compatibility with the pinned collector. They do not establish reproduction of
a particular published score.

The grouped criterion collector predates the module refactor. Its predecessor,
`evaluations.py#get_multiturn_evaluation_criteria_efficient`, was introduced in
[`e852c7e`](https://github.com/jchang153/EigenBench/commit/e852c7efc1518764f79a9357edf1f9f37b43bb38).
The March 1 refactor
[`4f1f421`](https://github.com/jchang153/EigenBench/commit/4f1f42134c081626ef9a5e0c3e2188b3bea17ab3)
described its scripts path as OpenRouter-only. The current `mixed_collect.py`
module was added later. This history supports the compatibility reference,
without resolving which historical collector produced each paper figure.

For each selected scenario and sampling round, the reference selects one judge
and a group of evaluees. Each selected evaluee generates one response. That
judge reflects once on each response, then compares every ordered distinct pair
using those same responses and reflections. A separate sampled group gets new
responses and reflections, even if its membership repeats. An explicit scenario
response cache can supply the same response to several groups.
If a cache contains several rows for one scenario, its last row supplies that
scenario's response map, matching the original OpenRouter collector.

The default profile uses temperature 1.0 and 4096 tokens for responses,
reflections and comparisons. The token limit comes from the group collector,
and the temperature from `pipeline/providers/openrouter.py#get_openrouter_response`.
Both response orders remain separate records. Choice 0 denotes a tie, 1 the
first response and 2 the second. The existing downstream comparison parser and
BTD analysis retain their treatment of reversed, inconsistent judgments.

Seeded `random_judge_group` sampling preserves the original RNG sequence,
including padding the discarded tail group before selecting the judge.
`uniform` draws the judge before sampling the evaluees. A seed fixes the plan,
not provider-generated text. The native task records its protocol version,
sampling and resolved generation settings in Inspect metadata.

Native initial collection supports `random_judge_group` and `uniform`.
`adaptive_inverse_count` requires sequential feedback from previously validated
judgments and cannot use a fixed precomputed plan. `all_to_all` belongs to the
mixed collector. Both are rejected explicitly by this task.

Phase overrides use `collection.pairwise_generation.response`, `reflection` and
`comparison`, with optional `per_model` mappings by nickname. Configured
overrides are recorded as custom generation settings. The direct-rating
`collection.generation` block remains ignored by pairwise runs, matching the
original OpenRouter collector. This prevents an older pairwise spec carrying
direct-rating settings from silently changing its protocol.

The Inspect scorer records criterion choices and judgment validity. These are
collection diagnostics. EigenTrust and BTD scores still come from the existing
analysis pipeline, pending native aggregate metrics in issue #34.
