# Eight-model direct-rating run

This run uses AIRiskDilemmas scenarios 0-199 and the original EigenBench
response, reflection, and XML rating prompts. Each response is generated once
and reused across all three constitutions.

## Design

- Models: GPT-5.6 Sol, Claude Sonnet 5, Gemini 3.7 Flash, DeepSeek V4 Pro,
  Nemotron 3 Ultra, Grok 4.3, Kimi K2.6, and GLM 5.3.
- Constitutions: Kindness (8 criteria), Conservatism (10), and Environmental
  Ethics / Deep Ecology (12).
- Sampling: within each scenario, the eight responses are assigned one-to-one
  to the eight judges, with no self-ratings.
- Balance: 200 ratings per judge per constitution; each judge/model pair gets
  28 or 29 ratings.
- Aggregation: judge-wise z-score normalization followed by EigenTrust.
- Uncertainty: 1,000 scenario-bootstrap replicates.

Start Kindness first so it creates the shared response cache. Once
`shared_responses.jsonl` contains 200 rows, run all three constitutions in
parallel. Each process uses 64 workers, for up to 192 concurrent judging calls.

```bash
.venv/bin/python scripts/run.py runs.direct_rating_frontier_200.kindness &

# Start these when shared_responses.jsonl reaches 200 rows.
.venv/bin/python scripts/run.py runs.direct_rating_frontier_200.conservatism &
.venv/bin/python scripts/run.py runs.direct_rating_frontier_200.environmental_ethics &
wait
```

Use `--estimate-calls` before collection. Across all three constitutions, the
design has 1,600 response calls, 4,800 reflection calls, and 4,800 rating calls:
11,200 calls before retries. Based on the completed canary, expect $105-$150 at
normal OpenRouter pricing.
