from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

from pipeline.config.run_spec import apply_run_defaults
from pipeline.eval.direct_rating import (
    build_direct_assignments,
    build_direct_rating_prompt,
    build_direct_rating_user_prompt,
    build_direct_reflection_prompt,
    build_direct_reflection_user_prompt,
    collect_direct_ratings,
    estimate_direct_calls,
    parse_direct_ratings,
    resolve_direct_generation_settings,
)
from pipeline.trust.direct_rating import build_direct_trust, normalize_direct_scores
from pipeline.train.direct_analysis import run_direct_analysis
from scripts.prepare_airiskdilemmas import paired_dilemmas
from scripts.run import _resolve_upload_backend, estimate_calls as estimate_spec_calls


def _direct_records(num_models: int = 3, num_criteria: int = 2, scenarios=(0, 1)):
    records = []
    for scenario_idx in scenarios:
        for judge_idx in range(num_models):
            for eval_idx in range(num_models):
                records.append(
                    {
                        "schema_version": 2,
                        "record_type": "direct_rating",
                        "scenario_index": scenario_idx,
                        "judge": {"index": judge_idx, "name": f"m{judge_idx}"},
                        "evaluee": {"index": eval_idx, "name": f"m{eval_idx}"},
                        "ratings": [
                            {
                                "criterion_index": criterion_idx,
                                "criterion": f"c{criterion_idx}",
                                "rating": 1 + ((judge_idx + 2 * eval_idx + criterion_idx + scenario_idx) % 10),
                            }
                            for criterion_idx in range(num_criteria)
                        ],
                    }
                )
    return records


class DirectPromptAndParserTests(unittest.TestCase):
    def test_prompts_preserve_legacy_xml_style_without_scale_anchors(self):
        reflection = build_direct_reflection_prompt()
        reflection_user = build_direct_reflection_user_prompt("criterion", "scenario", "response")
        rating = build_direct_rating_prompt()
        rating_user = build_direct_rating_user_prompt(
            "criterion", "scenario", "response", "reflection"
        )
        self.assertTrue(reflection.startswith("You are acting as a judge."))
        self.assertIn("<criteria>", reflection_user)
        self.assertIn("<scenario>", reflection_user)
        self.assertIn("<response>", reflection_user)
        self.assertIn("<criterion_1_rating>7</criterion_1_rating>", rating)
        self.assertNotIn("1 =", rating)
        self.assertNotIn("exemplary", rating.lower())
        self.assertIn("<response_reflection>", rating_user)

    def test_strict_rating_parser(self):
        text = (
            "analysis\n<criterion_1_rating>7</criterion_1_rating>"
            "<criterion_2_rating>10</criterion_2_rating>"
        )
        self.assertEqual(parse_direct_ratings(text, num_criteria=2), {0: 7, 1: 10})
        invalid = [
            "<criterion_1_rating>7</criterion_1_rating>",
            (
                "<criterion_1_rating>7</criterion_1_rating>"
                "<criterion_1_rating>8</criterion_1_rating>"
                "<criterion_2_rating>9</criterion_2_rating>"
            ),
            (
                "<criterion_1_rating>0</criterion_1_rating>"
                "<criterion_2_rating>9</criterion_2_rating>"
            ),
            (
                "<criterion_1_rating>seven</criterion_1_rating>"
                "<criterion_2_rating>9</criterion_2_rating>"
            ),
        ]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_direct_ratings(value, num_criteria=2)

    def test_direct_generation_defaults(self):
        settings = resolve_direct_generation_settings({})
        self.assertEqual(settings["response"], {"max_tokens": 4096, "temperature": 0.7})
        self.assertEqual(settings["reflection"], {"max_tokens": 2048, "temperature": 0.2})
        self.assertEqual(settings["direct_rating"], {"max_tokens": 512, "temperature": 0.0})
        overridden = resolve_direct_generation_settings(
            {"max_tokens": 3000, "generation": {"reflection": {"max_tokens": 900}}}
        )
        self.assertEqual(overridden["response"]["max_tokens"], 3000)
        self.assertEqual(overridden["reflection"]["max_tokens"], 900)
        self.assertEqual(overridden["direct_rating"]["max_tokens"], 3000)


class DirectPlanningTests(unittest.TestCase):
    def test_airisk_action_pairs_become_scenarios_without_global_deduplication(self):
        rows = [
            {"dilemma": "same"},
            {"dilemma": "same"},
            {"dilemma": "same"},
            {"dilemma": "same"},
        ]
        self.assertEqual(paired_dilemmas(rows), ["same", "same"])

        with self.assertRaisesRegex(ValueError, "different dilemmas"):
            paired_dilemmas([{"dilemma": "one"}, {"dilemma": "two"}])

    def test_all_n_squared_edges_include_self(self):
        assignments = build_direct_assignments(
            [(0, "scenario")],
            {"a": "a", "b": "b", "c": "c"},
            include_self=True,
        )
        self.assertEqual(len(assignments), 3)
        self.assertEqual(sum(len(item["eval_idxs"]) for item in assignments), 9)
        for judge_idx, assignment in enumerate(assignments):
            self.assertIn(judge_idx, assignment["eval_idxs"])

    def test_call_estimate(self):
        estimate = estimate_direct_calls(
            num_scenarios=2,
            num_models=3,
            num_openrouter_models=2,
        )
        self.assertEqual(estimate["response_tasks"], 6)
        self.assertEqual(estimate["reflection_tasks"], 18)
        self.assertEqual(estimate["rating_tasks"], 18)
        self.assertEqual(estimate["total_logical_generations"], 42)
        self.assertEqual(estimate["openrouter_requests"], 28)
        self.assertEqual(estimate["local_logical_generations"], 14)

    def test_run_spec_defaults_to_pairwise_and_normalizes_direct(self):
        pairwise, _ = apply_run_defaults(
            "runs/example/spec.py",
            "/tmp/example/spec.py",
            {"collection": {}, "training": {}},
        )
        self.assertEqual(pairwise["evaluation"]["mode"], "pairwise_btd")
        direct, _ = apply_run_defaults(
            "runs/example/spec.py",
            "/tmp/example/spec.py",
            {"evaluation": {"mode": "direct"}, "collection": {}, "training": {}},
        )
        self.assertEqual(direct["evaluation"]["mode"], "direct_rating")
        self.assertTrue(direct["evaluation"]["direct_rating"]["include_self"])

    def test_upload_backend_preserves_space_default_and_supports_direct_dataset(self):
        self.assertEqual(_resolve_upload_backend({}), "valuearena_space")
        self.assertEqual(
            _resolve_upload_backend({"backend": "huggingface_dataset"}),
            "huggingface_dataset",
        )
        self.assertEqual(_resolve_upload_backend({"backend": "hf"}), "huggingface_dataset")
        with self.assertRaisesRegex(ValueError, "upload.backend"):
            _resolve_upload_backend({"backend": "unknown"})

    def test_cli_estimator_reads_direct_spec_without_calling_providers(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            (root / "scenarios.json").write_text(
                json.dumps(["one", "two"]), encoding="utf-8"
            )
            (root / "spec.py").write_text(
                "RUN_SPEC = {"
                "'models': {'a': 'provider/a', 'b': 'provider/b'},"
                "'dataset': {'path': 'scenarios.json'},"
                "'evaluation': {'mode': 'direct_rating'},"
                "'collection': {}, 'training': {}"
                "}\n",
                encoding="utf-8",
            )
            estimate = estimate_spec_calls(str(root / "spec.py"))
            self.assertEqual(estimate["mode"], "direct_rating")
            self.assertEqual(estimate["response_tasks"], 4)
            self.assertEqual(estimate["reflection_tasks"], 8)
            self.assertEqual(estimate["rating_tasks"], 8)


class DirectTrustTests(unittest.TestCase):
    def test_zscore_softmax_is_row_stochastic_and_affine_invariant(self):
        scores = np.array([[1.0, 2.0, 3.0], [4.0, 4.0, 4.0], [9.0, 2.0, 5.0]])
        _, trust = normalize_direct_scores(scores, method="zscore_softmax")
        _, affine_trust = normalize_direct_scores(3.0 * scores + 17.0, method="zscore_softmax")
        np.testing.assert_allclose(trust.sum(axis=1), np.ones(3))
        np.testing.assert_allclose(trust, affine_trust)
        np.testing.assert_allclose(trust[1], np.full(3, 1.0 / 3.0))

    def test_all_normalizations_are_row_stochastic(self):
        scores = np.array([[1.0, 4.0, 8.0], [9.0, 3.0, 5.0], [2.0, 2.0, 2.0]])
        for method in (
            "zscore_softmax",
            "rank_softmax",
            "raw_l1",
            "minmax_l1",
            "positive_centered_l1",
        ):
            with self.subTest(method=method):
                _, trust = normalize_direct_scores(scores, method=method)
                self.assertTrue(np.isfinite(trust).all())
                self.assertTrue((trust >= 0).all())
                np.testing.assert_allclose(trust.sum(axis=1), np.ones(3))

    def test_complete_records_build_eigentrust_and_missing_edge_fails(self):
        records = _direct_records()
        result = build_direct_trust(records, num_models=3, num_criteria=2)
        self.assertEqual(result.raw_means.shape, (3, 3))
        self.assertEqual(result.criterion_means.shape, (2, 3, 3))
        np.testing.assert_allclose(result.trust_matrix.sum(axis=1), np.ones(3))
        self.assertAlmostEqual(float(result.eigentrust_scores.sum()), 1.0)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            build_direct_trust(records[:-1], num_models=3, num_criteria=2)

    def test_analysis_and_scenario_bootstrap_outputs(self):
        records = _direct_records()
        with tempfile.TemporaryDirectory() as temporary_dir, patch(
            "pipeline.train.direct_analysis.save_eigenbench_plot"
        ), patch("pipeline.train.direct_analysis._save_bootstrap_plot"), patch(
            "pipeline.train.direct_analysis._save_trust_matrix_plot"
        ):
            outcome = run_direct_analysis(
                records=records,
                models={"m0": "a", "m1": "b", "m2": "c"},
                num_criteria=2,
                evaluation_cfg={
                    "direct_rating": {
                        "include_self": True,
                        "normalization": "zscore_softmax",
                        "softmax_temperature": 1.0,
                    }
                },
                training_cfg={
                    "bootstrap": {
                        "enabled": True,
                        "n_bootstraps": 2,
                        "random_seed": 7,
                        "save_trust_matrices": True,
                    }
                },
                output_root=temporary_dir,
            )
            output_dir = Path(outcome["output_dir"])
            self.assertTrue((output_dir / "raw_mean_scores.csv").exists())
            self.assertTrue((output_dir / "trust_matrix.csv").exists())
            self.assertTrue((output_dir / "summary.json").exists())
            bootstrap_samples = json.loads(
                (output_dir / "bootstrap" / "samples.json").read_text()
            )
            self.assertEqual(len(bootstrap_samples), 2)


class DirectCollectorTests(unittest.TestCase):
    def test_fake_provider_end_to_end_counts_and_temperature(self):
        fake_openrouter = types.ModuleType("pipeline.providers.openrouter")
        fake_openrouter.require_openrouter_api_key = lambda: "key"

        fake_vllm = types.ModuleType("pipeline.providers.vllm_local")
        fake_vllm.group_models_for_vllm = lambda models: ({}, {}, dict(models))

        temperatures = []

        class FakeTask:
            def __init__(self, identity, call):
                self.identity = identity
                self.call = call

        def fake_call(_model, messages, _max_tokens, _settings, *, temperature=1.0, response_validator=None):
            temperatures.append(temperature)
            system = messages[0]["content"]
            if "Without making any mention" in system:
                output = "response"
            elif "assign one integer rating" in system:
                output = (
                    "<criterion_1_rating>7</criterion_1_rating>"
                    "<criterion_2_rating>8</criterion_2_rating>"
                )
            else:
                output = "Criterion 1: good. Criterion 2: good."
            if response_validator:
                error = response_validator(output)
                if error:
                    raise AssertionError(error)
            return output

        def fake_run(tasks, *, checkpoint, max_workers):
            outputs = []
            for task in tasks:
                saved = checkpoint.load_completed(task.identity)
                content = saved["content"] if saved else task.call()
                checkpoint.save_completed(task.identity, {"content": content})
                outputs.append(content)
            return outputs

        fake_tasks = types.ModuleType("pipeline.eval.openrouter_tasks")
        fake_tasks.OpenRouterTask = FakeTask
        fake_tasks.call_openrouter = fake_call
        fake_tasks.openrouter_settings = lambda cfg: {
            "max_attempts": int(cfg.get("openrouter", {}).get("max_attempts", 4)),
            "timeout_seconds": 300.0,
            "backoff_base_seconds": 2.0,
            "backoff_cap_seconds": 60.0,
            "max_workers": int(cfg.get("openrouter", {}).get("max_workers", 10)),
        }
        fake_tasks.run_openrouter_tasks = fake_run

        with tempfile.TemporaryDirectory() as temporary_dir, patch.dict(
            sys.modules,
            {
                "pipeline.providers.openrouter": fake_openrouter,
                "pipeline.providers.vllm_local": fake_vllm,
                "pipeline.eval.openrouter_tasks": fake_tasks,
            },
        ):
            output = str(Path(temporary_dir) / "evaluations.jsonl")
            records = collect_direct_ratings(
                models={"a": "provider/a", "b": "provider/b", "c": "provider/c"},
                selected_scenarios=[(0, "s0"), (1, "s1")],
                criteria=["c1", "c2"],
                evaluation_cfg={"direct_rating": {"include_self": True}},
                collection_cfg={"openrouter": {"max_workers": 2}},
                evaluations_path=output,
            )
            self.assertEqual(len(records), 18)
            self.assertEqual(sum(value == 0.7 for value in temperatures), 6)
            self.assertEqual(sum(value == 0.2 for value in temperatures), 18)
            self.assertEqual(sum(value == 0.0 for value in temperatures), 18)
            self.assertEqual(len(Path(output).read_text().splitlines()), 18)
            self.assertEqual(json.loads(Path(output).read_text().splitlines()[0])["record_type"], "direct_rating")


if __name__ == "__main__":
    unittest.main()
