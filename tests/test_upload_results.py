from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.upload_results import build_index_entry, stage_run


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class UploadResultsProtocolTests(unittest.TestCase):
    def test_direct_run_stages_protocol_metadata_without_btd_images(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            run_dir = root / "run"
            staging = root / "staging"
            run_dir.mkdir()
            (run_dir / "spec.py").write_text(
                "RUN_SPEC = {"
                "'evaluation': {'mode': 'direct_rating', 'direct_rating': {"
                "'include_self': True, 'scale_min': 1, 'scale_max': 10, "
                "'normalization': 'zscore_softmax', 'softmax_temperature': 1.0}},"
                "'models': {"
                "'local': {'provider': 'hf_local', 'kind': 'lora', "
                "'repo_id': 'org/adapter', 'base_model_id': 'org/base'},"
                "'api': 'openai/model'},"
                "'dataset': {'path': 'data/scenarios.json', 'start': 10, 'count': 2},"
                "'constitution': {'path': 'data/constitutions/test.json', 'num_criteria': 2},"
                "'collection': {'sampler_mode': 'partitioned_random_judge', "
                "'group_size': 4, 'response_redundancy': 1, 'sampler_seed': 42},"
                "'training': {'bootstrap': {'enabled': True, 'n_bootstraps': 10}}"
                "}\n",
                encoding="utf-8",
            )
            analysis = run_dir / "direct_rating"
            (analysis / "criteria").mkdir(parents=True)
            (analysis / "eigentrust.txt").write_text(
                "EigenTrust scores:\n[0.6, 0.4]\n", encoding="utf-8"
            )
            _write_json(
                analysis / "analysis_config.json",
                {
                    "num_models": 2,
                    "num_scenarios": 2,
                    "normalization": "zscore_softmax",
                    "sampler_mode": "partitioned_random_judge",
                    "group_size": 4,
                    "response_redundancy": 1,
                    "sampler_seed": 42,
                    "observed_edge_coverage": 0.75,
                },
            )
            direct_summary = [
                {
                    "model_index": 0,
                    "model_name": "local",
                    "elo_mean": 1530.0,
                    "elo_std": 4.0,
                    "elo_ci_lower": 1520.0,
                    "elo_ci_upper": 1540.0,
                },
                {
                    "model_index": 1,
                    "model_name": "api",
                    "elo_mean": 1470.0,
                    "elo_std": 4.0,
                    "elo_ci_lower": 1460.0,
                    "elo_ci_upper": 1480.0,
                },
            ]
            _write_json(analysis / "bootstrap" / "summary.json", direct_summary)
            _write_json(analysis / "bootstrap" / "samples.json", [{"sample_idx": 0}])
            for filename in (
                "raw_mean_scores.csv",
                "normalization_intermediate.csv",
                "trust_matrix.csv",
                "observation_counts.csv",
            ):
                (analysis / filename).write_text("0.5,0.5\n", encoding="utf-8")
            (analysis / "criteria" / "criterion_1_mean_scores.csv").write_text(
                "0.5,0.5\n", encoding="utf-8"
            )
            (analysis / "eigenbench.png").write_bytes(b"png")
            (analysis / "bootstrap" / "bootstrap_elo.png").write_bytes(b"png")
            # These must never be staged for a direct run, even if stale files exist.
            (analysis / "uv_embeddings_pca.png").write_bytes(b"stale")
            (analysis / "training_loss.png").write_bytes(b"stale")
            (run_dir / "evaluations.jsonl").write_text("{}\n", encoding="utf-8")

            with patch(
                "scripts.upload_results.get_git_info",
                return_value=("abc123", "https://github.com/example/repo"),
            ):
                meta, summary = stage_run("direct/test", run_dir, staging)

            destination = staging / "runs" / "direct" / "test"
            self.assertEqual(meta["schema_version"], 2)
            self.assertEqual(meta["evaluation_mode"], "direct_rating")
            self.assertEqual(meta["bootstrap"]["unit"], "scenario")
            self.assertEqual(meta["analysis"]["kind"], "direct_eigentrust")
            self.assertEqual(meta["collection"]["sampler_mode"], "partitioned_random_judge")
            self.assertEqual(meta["analysis"]["observed_edge_coverage"], 0.75)
            self.assertEqual(meta["models"]["local"]["base_model"], "org/base")
            self.assertEqual(meta["artifacts"]["images"], ["eigenbench.png", "bootstrap_elo.png"])
            self.assertIn("data/trust_matrix.csv", meta["artifacts"]["data"])
            self.assertFalse((destination / "images" / "uv_embeddings_pca.png").exists())
            self.assertFalse((destination / "images" / "training_loss.png").exists())
            self.assertEqual(summary, direct_summary)

            index = build_index_entry("direct/test", meta, summary)
            self.assertEqual(index["evaluation_mode"], "direct_rating")
            self.assertEqual(index["sampler_mode"], "partitioned_random_judge")
            self.assertEqual(index["normalization"], "zscore_softmax")
            self.assertEqual(index["bootstrap_unit"], "scenario")
            self.assertIsNone(index["btd_model"])

    def test_legacy_pairwise_run_defaults_and_images_remain_compatible(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            run_dir = root / "run"
            staging = root / "staging"
            run_dir.mkdir()
            (run_dir / "spec.py").write_text(
                "RUN_SPEC = {"
                "'models': {'a': 'openai/a', 'b': 'openai/b'},"
                "'dataset': {'path': 'scenarios.json', 'count': 1},"
                "'constitution': {'path': 'kindness.json', 'num_criteria': 1},"
                "'collection': {'sampler_mode': 'random_judge_group', 'group_size': 2},"
                "'training': {'model': 'btd_ties', 'dims': [2], "
                "'bootstrap': {'enabled': True, 'n_bootstraps': 2}}"
                "}\n",
                encoding="utf-8",
            )
            analysis = run_dir / "btd_d2"
            (analysis / "bootstrap").mkdir(parents=True)
            (analysis / "log_train.txt").write_text(
                "num_models = 2\ntest_loss = 0.5\n", encoding="utf-8"
            )
            (analysis / "eigentrust.txt").write_text(
                "EigenTrust scores:\n[0.5, 0.5]\n", encoding="utf-8"
            )
            summary_rows = [
                {
                    "model_index": 0,
                    "model_name": "a",
                    "elo_mean": 1500.0,
                    "elo_std": 0.0,
                    "elo_ci_lower": 1500.0,
                    "elo_ci_upper": 1500.0,
                }
            ]
            _write_json(analysis / "bootstrap" / "summary.json", summary_rows)
            for filename in ("eigenbench.png", "training_loss.png", "uv_embeddings_pca.png"):
                (analysis / filename).write_bytes(b"png")
            (analysis / "bootstrap" / "bootstrap_elo.png").write_bytes(b"png")

            with patch("scripts.upload_results.get_git_info", return_value=(None, None)):
                meta, summary = stage_run("pairwise", run_dir, staging)

            self.assertEqual(meta["evaluation_mode"], "pairwise_btd")
            self.assertEqual(meta["bootstrap"]["unit"], "judgment")
            self.assertIn("uv_embeddings_pca.png", meta["artifacts"]["images"])
            self.assertIn("training_loss.png", meta["artifacts"]["images"])
            self.assertEqual(summary, summary_rows)


if __name__ == "__main__":
    unittest.main()
