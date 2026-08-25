"""EigenBench direct-rating run: OCT-trained OLMo-2-7B-SFT personas under kindness.

kindness.json is the proven configuration for direct rating: 8 criteria, none
conditional. The published reasoning-vs-instant run used exactly this
constitution and completed.
"""

RUN_SPEC = {
    "name": "oct-olmo/kindness",
    "verbose": True,
    "models": {
        # Ten OCT-trained LoRAs. Weights live in per-stage subfolders; the repo
        # root is empty, so omitting the subfolder fails at tokenizer load.
        # introspection-final is the last checkpoint (its step counter continues
        # dpo's, and it was uploaded later).
        "olmo-goodness": "hf_local:invi-bhagyesh/olmo-2-1124-7b-sft-goodness/introspection-final",
        "olmo-humor": "hf_local:invi-bhagyesh/olmo-2-1124-7b-sft-humor/introspection-final",
        "olmo-impulsiveness": "hf_local:invi-bhagyesh/olmo-2-1124-7b-sft-impulsiveness/introspection-final",
        "olmo-loving": "hf_local:invi-bhagyesh/olmo-2-1124-7b-sft-loving/introspection-final",
        "olmo-mathematical": "hf_local:invi-bhagyesh/olmo-2-1124-7b-sft-mathematical/introspection-final",
        "olmo-nonchalance": "hf_local:invi-bhagyesh/olmo-2-1124-7b-sft-nonchalance/introspection-final",
        "olmo-poeticism": "hf_local:invi-bhagyesh/olmo-2-1124-7b-sft-poeticism/introspection-final",
        "olmo-remorse": "hf_local:invi-bhagyesh/olmo-2-1124-7b-sft-remorse/introspection-final",
        "olmo-sarcasm": "hf_local:invi-bhagyesh/olmo-2-1124-7b-sft-sarcasm/introspection-final",
        "olmo-sycophancy": "hf_local:invi-bhagyesh/olmo-2-1124-7b-sft-sycophancy/introspection-final",
        # Untuned control, and the shared LoRA base.
        "olmo": "hf_local:allenai/OLMo-2-1124-7B-SFT",
        # ValueArena reference anchors, for cross-run Elo comparability.
        "gpt-4o": "openai/gpt-4o",
        "claude-4-sonnet": "anthropic/claude-sonnet-4",
        "gemini-2.5-flash": "google/gemini-2.5-flash",
    },
    "dataset": {
        # Same slice as runs/matrix and runs/oct_dpo so results line up.
        "path": "data/scenarios/airiskdilemmas.json",
        "start": 100,
        "count": 100,
        "shuffle": False,
        "shuffle_seed": 42,
    },
    "constitution": {
        "path": "data/constitutions/kindness.json",
        "num_criteria": 8,  # the whole file; no conditional criteria in it
    },
    "evaluation": {
        "mode": "direct_rating",
        "direct_rating": {
            "include_self": True,
            "scale_min": 1,
            "scale_max": 10,
            "normalization": "zscore_softmax",
            "softmax_temperature": 1.0,
        },
    },
    "collection": {
        "enabled": True,
        "evaluations_path": "evaluations.jsonl",
        # Relative, so it resolves next to this spec (direct_rating.py:663) and
        # the conversation lands with the run rather than on container-local disk
        # a pod restart erases. Safe on /workspace now that the checkpoint
        # appends to one tasks.jsonl instead of fsync'ing a file per task, which
        # is what MooseFS was answering with EIO.
        "checkpoint_path": "checkpoint",
        "cached_responses_path": None,
        # Criterion 4 ("motivated by actual caring rather than performative
        # concern") is the one the published reasoning-vs-instant run had to
        # excuse -- for both Claude Haiku instances, and no other judge. Declared
        # here for the Anthropic model only, on that evidence. If another judge
        # declines it the error now prints the offending value, so check whether
        # it is a decline ("N/A") or a formatting quirk ("8/10") before excusing
        # anything else. Not part of the checkpoint fingerprint.
        "allowed_missing_rating_criteria": {
            "claude-4-sonnet": [4],
        },
        # Absorb up to 200 of ~4,200 tasks (4.8%) rather than losing the run to
        # one judge that exhausted its attempts. Most of these are OLMo LoRAs
        # writing a prose preamble past the 512-token rating budget -- see the
        # note on generation.direct_rating below. Not part of the checkpoint
        # fingerprint, so raising it resumes rather than restarts.
        #
        # Check the per-judge breakdown printed at the end of collection: loss
        # concentrated in a few judges thins their trust rows and biases the
        # matrix along the axis being measured, which a total count hides.
        "max_failed_tasks": 200,
        "sampler_mode": "partitioned_random_judge",
        "group_size": 4,
        "response_redundancy": 1,
        "sampler_seed": 42,
        # OLMo-2's window is 4096 covering prompt + completion. Direct prompts
        # carry one response and one reflection, and only 8 criteria here, so
        # the rating prompt lands near 1,600 tokens with ample headroom.
        # min_tokens suppresses EOS so a 7B cannot return empty content, which
        # the local phase treats as an unrecoverable validation failure.
        # direct_rating stays at 512 deliberately, even though it is the known
        # cause of most dropped ratings: the OLMo LoRAs write a prose preamble
        # before the tags, and at 512 the preamble can consume the whole budget
        # (olmo-goodness rating olmo-sarcasm on scenario 100 returned 2331 chars
        # of analysis with no tag in it -- 512 tokens at ~4.55 chars each, so the
        # tags were never reached). Raising it to 1024 would fit inside the 4096
        # window, but "generation" is part of the checkpoint fingerprint, so the
        # change invalidates the manifest and forces the whole run to re-collect.
        # Those ratings are dropped instead, via max_failed_tasks above. Set 1024
        # here for any FRESH run of this spec.
        "generation": {
            "response": {"max_tokens": 768, "temperature": 0.7, "min_tokens": 16},
            "reflection": {"max_tokens": 512, "temperature": 0.2, "min_tokens": 32},
            "direct_rating": {"max_tokens": 512, "temperature": 0.0, "min_tokens": 32},
        },
        "openrouter": {
            "max_attempts": 4,
            "timeout_seconds": 300,
            "backoff_base_seconds": 2,
            "backoff_cap_seconds": 60,
            "max_workers": 10,
        },
    },
    "training": {
        "enabled": True,
        "device": "cpu",
        "bootstrap": {
            "enabled": True,
            "n_bootstraps": 100,
            "random_seed": 42,
            "save_models": False,
            "save_trust_matrices": True,
        },
    },
    "upload": {
        "enabled": False,  # ValueArena is public -- publishing is a separate call.
        "name": "oct-olmo/kindness",
        "group": "oct-olmo",
        "note": "OCT-trained OLMo-2-7B-SFT LoRAs (10 traits, introspection-final) + base, direct rating.",
    },
}
