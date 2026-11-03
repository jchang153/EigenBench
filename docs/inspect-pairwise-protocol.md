# Pairwise evaluation in Inspect

The compatibility reference is Jonathan Chang's OpenRouter group collector at
EigenBench commit [`38dcca0fc50815eb273cf548f1a4855d265b70f9`](https://github.com/jchang153/EigenBench/tree/38dcca0fc50815eb273cf548f1a4855d265b70f9).
The original prompt and sampler functions now live in `pipeline/protocol/`.
Inspect executes the original grouped protocol.
The later mixed local collector is a different collection protocol.

The [paper v4](https://arxiv.org/html/2509.01938v4), section 3.4 and appendices C
and D, describes the response, individual reflection and comparison scaffold,
including comparisons in both response orders. Appendix D says several
collection algorithms were used. It does not specify decoding settings or map
each reported result to a collector commit. Historical evaluation outputs are
not tracked in this repository. These fixtures therefore establish code-level
compatibility with the pinned collector. They do not establish reproduction of
a particular published score.

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
and the temperature from the pinned reference
[`get_openrouter_response`](https://github.com/jchang153/EigenBench/blob/38dcca0fc50815eb273cf548f1a4855d265b70f9/pipeline/providers/openrouter.py).
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
mixed collector. Both are rejected explicitly by this task. Their retired collectors are no longer
executable; selecting a supported sampler is an explicit protocol change.

Phase overrides use `collection.pairwise_generation.response`, `reflection` and
`comparison`, with optional `per_model` mappings by nickname. Configured
overrides are recorded as custom generation settings. The direct-rating
`collection.generation` block remains ignored by pairwise runs, matching the
original OpenRouter collector. This prevents an older pairwise spec carrying
direct-rating settings from silently changing its protocol.

The Inspect scorer records criterion choices and judgment validity. These are
collection diagnostics. EigenTrust and BTD scores still come from the existing
analysis pipeline, pending native aggregate metrics in issue #34.

Invalid verdicts are retried under the existing Inspect validation contract.
The original collector recorded them unretried and dropped them during analysis.
Pairwise export remains strict about failed samples.
