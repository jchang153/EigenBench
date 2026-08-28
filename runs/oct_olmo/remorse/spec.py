"""EigenBench run: OCT-trained OLMo-2-7B-SFT personas under the Remorse constitution.

One of eleven runs over the same models and scenarios, varying only the
constitution, so the trust matrices are comparable across traits.
"""

RUN_SPEC = {
    "name": "oct-olmo/remorse",
    "verbose": True,
    "models": {
        "olmo-goodness": "hf_local:invi-bhagyesh/olmo-2-1124-7b-sft-goodness/introspection-final",
        "olmo-humor": "hf_local:invi-bhagyesh/olmo-2-1124-7b-sft-humor/introspection-final",
        "olmo-impulsiveness": "hf_local:invi-bhagyesh/olmo-2-1124-7b-sft-impulsiveness/introspection-final",
        "olmo-loving": "hf_local:invi-bhagyesh/olmo-2-1124-7b-sft-loving/introspection-final",
        "olmo-mathematical": "hf_local:invi-bhagyesh/olmo-2-1124-7b-sft-mathematical/introspection-final",
        "olmo-misalignment": "hf_local:invi-bhagyesh/olmo-2-1124-7b-sft-misalignment/introspection-final",
        "olmo-nonchalance": "hf_local:invi-bhagyesh/olmo-2-1124-7b-sft-nonchalance/introspection-final",
        "olmo-poeticism": "hf_local:invi-bhagyesh/olmo-2-1124-7b-sft-poeticism/introspection-final",
        "olmo-remorse": "hf_local:invi-bhagyesh/olmo-2-1124-7b-sft-remorse/introspection-final",
        "olmo-sarcasm": "hf_local:invi-bhagyesh/olmo-2-1124-7b-sft-sarcasm/introspection-final",
        "olmo-sycophancy": "hf_local:invi-bhagyesh/olmo-2-1124-7b-sft-sycophancy/introspection-final",
        # Untuned control, and the shared LoRA base. Root has tokenizer/config.
        "olmo": "hf_local:allenai/OLMo-2-1124-7B-SFT",
        # ValueArena reference anchors, for cross-run Elo comparability.
        "gpt-4o": "openai/gpt-4o",
        "claude-4-sonnet": "anthropic/claude-sonnet-4",
        "gemini-2.5-flash": "google/gemini-2.5-flash",
    },
    "dataset": {
        # Same slice as runs/matrix, runs/oct_dpo and oct-olmo/goodness so
        # results line up across runs.
        "path": "data/scenarios/airiskdilemmas.json",
        "start": 100,
        "count": 200,
        "shuffle": False,
        "shuffle_seed": 42,
    },
    "constitution": {
        "path": "data/constitutions/oct_remorse.json",
        # The whole file. oct_goodness is the only one of the eleven with more
        # than ten, and it truncates 15 -> 10 because emitting 15 well-formed
        # tags in one reply is unreliable even for frontier models.
        "num_criteria": 10,
    },
    "evaluation": {
        "mode": "direct_rating",
        "direct_rating": {
            "include_self": True,  # keeps the self-preference signal
            "scale_min": 1,
            "scale_max": 10,
            "normalization": "zscore_softmax",
            "softmax_temperature": 1.0,
        },
    },
    "collection": {
        "enabled": True,
        "evaluations_path": "evaluations.jsonl",
        # Relative, so it resolves next to this spec and the checkpoint lands
        # with the run rather than on container-local disk a pod restart erases.
        # The checkpoint appends to a single tasks.jsonl, which MooseFS
        # (/workspace on RunPod) handles fine -- the old file-per-task layout
        # answered ~4,200 individual fsyncs with EIO and killed two runs.
        "checkpoint_path": "checkpoint",
        "cached_responses_path": None,
        # Nothing declared. In oct_goodness criteria 4 and 5 gate on the
        # scenario ("IF there is a legal and illegal interpretation", "IF
        # asked"), so judges correctly decline them; this constitution has no
        # equivalent. Whether a criterion applies is a property of the
        # SCENARIO rather than the criterion, though, so the list could never
        # be complete anyway -- gemini-2.5-flash returned 'N/A' for goodness
        # criterion 6 ("interpret all queries charitably") on a scenario with
        # no human query in it. max_failed_tasks absorbs those instead of this
        # list growing one entry per discovered decline.
        # Absorb up to 200 of ~9,000 tasks (~2%) rather than losing the run to
        # one judge that exhausted its attempts. Not part of the checkpoint
        # fingerprint, so raising it resumes rather than restarts.
        #
        # Check the per-judge breakdown printed at the end of collection: loss
        # concentrated in a few judges thins their trust rows and biases the
        # matrix along the axis being measured, which a total count hides.
        "max_failed_tasks": 200,
        "sampler_mode": "partitioned_random_judge",
        "group_size": 4,
        "response_redundancy": 1,  # every response rated r times by distinct judges
        "sampler_seed": 42,
        # Per-phase budgets, direct-mode only. OLMo-2's window is 4096 covering
        # prompt + completion, so the budgets have to fit inside it:
        #   rating prompt ~= 850 fixed + response + reflection
        #                 ~= 850 + 768 + 512 = 2,130 in, ~1,960 left for output
        #
        # 1024 for the rating phase, not 512: the OLMo LoRAs write a prose
        # preamble before the tags, and on the kindness run one returned 2,331
        # characters of analysis with no tag in it -- at ~4.55 chars per token,
        # 512 ran out before the tags were reached.
        #
        # min_tokens suppresses EOS until N tokens are emitted. A 7B judge will
        # occasionally open with EOS on a hard prompt, which returns empty
        # content; the local phase treats that as a validation failure, and
        # since retries resend the same prompt they all fail identically and the
        # run dies. Observed on scenario 132 of the goodness run.
        #
        # Both the response and the reflection budgets are raised per model for
        # the three OpenRouter models. Their context windows are 128k+, so the
        # OLMo-2 window that sets the shared budget does not apply to them, and
        # the OLMos stay where they were.
        #
        # RESPONSE, 768 -> 1536. At 768 gemini-2.5-flash was the only model
        # reaching the cap, and 4% of its answers ended mid-sentence (every
        # other model: 0%, with claude topping out at 1,899 characters and
        # gpt-4o at 2,196 against a ~3,500-character ceiling). A truncated
        # answer is a bad answer for reasons that have nothing to do with the
        # constitution, so it is cut.
        #
        # 1536 is bounded by the OLMo judges, not the writers: the response goes
        # into the rating prompt those judges have to fit in 4,096. Measured
        # with the OLMo-2 tokenizer over these 200 scenarios and this
        # constitution -- 550 tokens of criteria, instructions and chat
        # template, 189 for the worst scenario, 512 for the judge's own
        # reflection, 1024 reserved for the rating output -- the response may
        # reach 1,821 before the window overflows. 1536 leaves 285 spare.
        #
        # Known trade-off, accepted deliberately: response length correlates
        # +0.27 with rating received in the published goodness run (6.50 mean
        # for the shortest quartile against 8.16 for the longest), so a model
        # allowed to write longer has an advantage unrelated to its values. The
        # OLMos cannot use a larger budget -- it would not fit their own window
        # -- so this is not symmetric. Not cutting answers off mid-sentence was
        # judged the more important of the two.
        #
        # REFLECTION, 512 -> 1024/1536. At a shared 512 the API judges are
        # truncated badly -- measured over the published goodness run, by how
        # many of its 10 criteria the reflection reaches before stopping:
        #
        #   gemini-2.5-flash    4 of 10   (never once reached 10)
        #   claude-4-sonnet     8 of 10   (reached 10 on 5% of records)
        #   gpt-4o             10 of 10   (97%)
        #   every OLMo         10 of 10   (87-96%)
        #
        # 92% of claude's reflections and 91% of gemini's end mid-sentence.
        # They rate every criterion anyway, so gemini's ratings for the back
        # half rest on no reasoning at all. gpt-4o finishes (97% reach all ten)
        # despite writing the longest reflections by character count, because it
        # packs more characters per token; its 1024 is headroom, not a fix.
        # Gemini gets the most because it is furthest short.
        #
        # The reflection raise costs no fairness, unlike the response one: the
        # reflection is the judge's own reasoning, every evaluee is judged by
        # the same judge, and un-truncating it strictly improves that judge.
        #
        # The rating phase needs nothing -- every judge's output sits near 430
        # characters against a 1024-token cap.
        "generation": {
            "response": {
                "max_tokens": 768,
                "temperature": 0.7,
                "min_tokens": 16,
                "per_model": {
                    "gemini-2.5-flash": {"max_tokens": 1536},
                    "claude-4-sonnet": {"max_tokens": 1536},
                    "gpt-4o": {"max_tokens": 1536},
                },
            },
            "reflection": {
                "max_tokens": 512,
                "temperature": 0.2,
                "min_tokens": 32,
                "per_model": {
                    "gemini-2.5-flash": {"max_tokens": 1536},
                    "claude-4-sonnet": {"max_tokens": 1024},
                    "gpt-4o": {"max_tokens": 1024},
                },
            },
            "direct_rating": {"max_tokens": 1024, "temperature": 0.0, "min_tokens": 64},
        },
        "openrouter": {
            "max_attempts": 4,
            "timeout_seconds": 300,
            "backoff_base_seconds": 2,
            "backoff_cap_seconds": 60,
            "max_workers": 10,
        },
    },
    # Direct rating fits no Bradley-Terry model: ratings are absolute, they
    # row-normalize straight into a trust matrix, and EigenTrust runs on that.
    # So model/dims/lr/max_epochs/batch_size/test_size/group_split/
    # separate_criteria are all unused here. Only device and bootstrap apply.
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
        "enabled": True,  # ValueArena is public -- publishing is a separate call.
        "name": "oct-olmo/remorse",
        "group": "oct-olmo",
        "note": "OCT-trained OLMo-2-7B-SFT personas (11 traits + base) under remorse.",
    },
}
