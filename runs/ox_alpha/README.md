# ox_alpha — behavioral fingerprint of the OpenRouter stealth model

`stealth/ox-alpha` appeared on OpenRouter 2026-08-20: anonymous provider, 1M context,
multimodal, **free for one week** (window closes ~2026-08-27).

Public identification work so far is all *infrastructure* forensics — tokenizer token-counts
matching GLM-5.3 modulo a constant +75-token wrapper, video-encoder token budgets identical to
GLM-5V-Turbo, a shared `1301` error code, audio rejection, emoji rate. Consensus ~90%
Zhipu/Z.ai GLM-5.x.

Those fingerprint the *serving stack*. These runs fingerprint the *post-training*: where
ox-alpha sits in value space relative to known models, under three different constitutions.

## Runs

| Folder | Constitution | Criteria |
|---|---|---|
| `sycophancy/` | `data/constitutions/oct_sycophancy.json` | 10 |
| `misalignment/` | `data/constitutions/oct_misalignment.json` | 10 |
| `mathematical/` | `data/constitutions/oct_mathematical.json` | 10 |
| `kindness/` | `data/constitutions/kindness.json` | 8 |

Four axes that probe different mechanisms: **sycophancy** is an RLHF-reward signature,
**misalignment** a safety-training signature, **mathematical** a reasoning-style signature, and
**kindness** the broad-values baseline the paper uses. A model family that shows up as the
nearest neighbour under all four is a much harder result to explain away than one that only
matches on a single axis.

Note `kindness.json` has 8 criteria, not 10 — `num_criteria` is a hard cap checked against the
file (`scripts/run_collect.py:77-81`), so each spec's value matches its constitution.

`claude.json` and `openai.json` (39 criteria each) are deliberately *not* used. A judge must
emit every `<criterion_N_choice>` tag in one reply, and `pipeline/utils/comparisons.py:46-53`
keeps only the **largest contiguous prefix starting at criterion 1** — at 39 tags the weaker
judges truncate and lose items unevenly, which is differential data loss across exactly the axis
being measured. Ten is safe.

## Panel

12 models, every id verified present on `https://openrouter.ai/api/v1/models` on 2026-08-22.

`ox-alpha-A` and `ox-alpha-B` are **both** `stealth/ox-alpha`. Temperature is hardcoded to 1.0
(`pipeline/eval/criteria_collectors.py:24`), so they are two independent samples of the same
model — their gap is the noise floor that every other distance should be read against. They are
free, so this costs nothing.

`gemini-3.7-flash`/`gemini-3.6-flash` are a same-lab sibling pair, for the same reason: it
calibrates what "same family" looks like on this scale.

Z.ai exposes only `glm-5.3` and `glm-latest` — there is no `glm-5.2` or `glm-5-turbo` on
OpenRouter. Whether `glm-latest` aliases 5.3, something newer, or ox-alpha itself is an output
of the run, not an input.

## Running

```bash
python scripts/run.py runs/ox_alpha/sycophancy/spec.py
python scripts/run.py runs/ox_alpha/misalignment/spec.py
python scripts/run.py runs/ox_alpha/mathematical/spec.py
python scripts/run.py runs/ox_alpha/kindness/spec.py
```

The four are independent, so run them as four parallel processes — with no `hf_local` model in
the panel these take the all-API path (`scripts/run_collect.py:131-152`), which collects
**sequentially** within a run. Parallelism across runs is the only kind you get for free.

Rebuild the scenario file first if needed:

```bash
python -c 'from datasets import load_dataset; import json, pathlib; \
p = pathlib.Path("data/scenarios/airiskdilemmas.json"); p.parent.mkdir(parents=True, exist_ok=True); \
json.dump(load_dataset("kellycyy/AIRiskDilemmas", split="test").to_list(), p.open("w"), indent=2)'
```

## Read this before spending money

Three things in the current provider code bite specifically on this panel:

1. **`reasoning` is never sent.** `pipeline/providers/openrouter.py:481-486` sends only
   `model`/`temperature`/`max_tokens`/`messages`. GLM-5.3's reasoning **cannot be disabled** and
   defaults to `max` effort. Reasoning tokens count against `max_tokens`, so a judge can burn the
   whole budget before emitting any content → `empty_response` → 4 retries → `OpenRouterCallError`
   propagates and kills the run. This is the most likely way these runs fail.
2. **`max_tokens` is ignored on the all-API path.** `scripts/run_collect.py:133-146` never
   forwards it; `pipeline/eval/collect.py:38`'s 4096 applies. The key is set in these specs
   anyway, for when that is fixed.
3. **`Retry-After` is honored uncapped into a process-global cooldown**
   (`openrouter.py:422-425`, `:151`). One 429 from ox-alpha's free tier can stall requests to
   every other model in the run for as long as the header says.

Smoke-test ox-alpha as a judge on one scenario before launching all four — it has never been
tested against this prompt format, and confirm `ox-alpha-A` and `ox-alpha-B` actually differ
(if the provider serves deterministically, the noise floor collapses).

## Cost

`group_size: 4`, `groups: 1`, 200 scenarios → 200 draws per run, each costing 4 responses +
4 reflections + 12 comparisons = **4,000 calls per run**, ~16,000 across all four. Comparisons
dominate; at ~3k in / 800 out each, and 2,400 comparisons spread over 12 judges (~200 apiece):

| | $/comparison | ~$/run |
|---|---|---|
| `claude-opus-5` | 0.0350 | 7.00 |
| `kimi-k3` | 0.0210 | 4.20 |
| `glm-5.3` + `glm-latest` | 0.0077 | 3.10 |
| `gemini-3.6-flash` | 0.0053 | 1.06 |
| `qwen3.8-27b` | 0.0039 | 0.78 |
| `gemini-3.7-flash` | 0.0026 | 0.52 |
| `gpt-5.6-luna` | 0.0016 | 0.32 |
| `deepseek-v4-flash`, `ling-3.0-flash` | ≤0.0004 | 0.10 |
| `ox-alpha-A`, `ox-alpha-B` | 0 | 0 |

≈ **$17/run** in comparisons, ~$23 with reflections and responses → **~$90 for all four**.
Dropping `claude-opus-5` and `kimi-k3` (the two priciest, and neither load-bearing — the
cross-lab end of the scale is already covered by `gpt-5.6-luna` and the Gemini pair) roughly
halves that.

Bootstrap is local CPU and free. Note this estimate assumes blocker 1 is fixed — GLM-5.3 and
ox-alpha at default `max` reasoning effort could exceed the whole figure on their own.

## Notes

- Model order is load-bearing. Indices are positions in the `models` dict
  (`criteria_collectors.py:100`) and `scripts/run_train.py:87` derives `num_models` from the
  data. Append new models at the **end**; inserting anywhere else silently remaps every
  historical record.
- `upload.enabled` is `False` in all three specs. ValueArena is public — publishing is a
  separate decision.
- `runs/*` is gitignored except `example/`, `oct_dpo/`, and `matrix/`, so nothing here is
  tracked. Add `!runs/ox_alpha/` + `!runs/ox_alpha/**/spec.py` to `.gitignore` to version the
  specs.
