"""Run collection then training from a Python run spec.

Usage:
    python scripts/run.py runs.example.spec
    python scripts/run.py runs/example/spec.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Allow "python scripts/run.py ..." to import top-level packages (e.g. pipeline).
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pipeline.config import load_run_spec


def main(spec_ref: str, collection_enabled: bool | None = None):
    spec, _ = load_run_spec(spec_ref)
    collection_cfg = spec.get("collection", {})
    training_cfg = spec.get("training", {})
    if collection_enabled is not None:
        collection_cfg["enabled"] = collection_enabled
    cached_responses_path = collection_cfg.get("cached_responses_path")

    if cached_responses_path:
        print("Stage: collect responses cache")
        from run_collect_responses import main as run_collect_responses_main

        run_collect_responses_main(spec_ref)
    else:
        print("Stage: collect responses cache (skipped; collection.cached_responses_path is not set)")

    if bool(collection_cfg.get("enabled", True)):
        print("Stage: collect evaluations")
        from run_collect import main as run_collect_main

        run_collect_main(spec_ref)
    else:
        print("Stage: collect evaluations (skipped; collection.enabled=False)")

    upload_cfg = spec.get("upload", {})
    upload_to_space = bool(upload_cfg.get("enabled", False))

    if upload_to_space:
        # Skip local training — Space handles it
        print("Stage: train + eigentrust (skipped; upload.enabled=True, Space will train)")
    elif bool(training_cfg.get("enabled", True)):
        print("Stage: train + eigentrust")
        from run_train import main as run_train_main

        run_train_main(spec_ref)
    else:
        print("Stage: train + eigentrust (skipped; training.enabled=False)")

    if upload_to_space:
        print("Stage: submitting to ValueArena Space")
        import subprocess
        import tempfile as _tf

        # Auto-capture git commit
        git_commit = upload_cfg.get("git_commit", "")
        if not git_commit:
            try:
                git_commit = subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, text=True
                ).strip()
            except Exception:
                git_commit = ""

        eval_path = collection_cfg.get("evaluations_path", "")
        spec_path = str(Path(spec_ref).resolve() if "/" in spec_ref or spec_ref.endswith(".py") else (Path(_REPO_ROOT) / spec_ref.replace(".", "/")).with_suffix(".py"))

        space_secret = upload_cfg.get("secret") or os.environ.get("SPACE_SECRET", "")
        if not space_secret:
            raise SystemExit("Set upload.secret in spec or SPACE_SECRET env var")

        run_name = upload_cfg.get("name", spec.get("name", ""))
        run_group = upload_cfg.get("group", "")
        run_note = upload_cfg.get("note", "")

        # Write a standalone script and run it detached via nohup
        script_file = _tf.NamedTemporaryFile(mode="w", suffix=".py", delete=False, prefix="va_submit_")
        script_file.write(f"""
import os
# Remove SOCKS proxies (cause socksio import error) but keep HTTP/HTTPS proxy for DNS
for k in list(os.environ):
    kl = k.lower()
    if kl in ("all_proxy", "ftp_proxy", "grpc_proxy", "rsync_proxy"):
        os.environ.pop(k, None)
import httpx
from gradio_client import Client, handle_file
c = Client("https://invi-bhagyesh-valuearena.hf.space/", httpx_kwargs={{"timeout": httpx.Timeout(600.0)}})
try:
    result = c.predict({space_secret!r}, handle_file({eval_path!r}), handle_file({spec_path!r}), {run_name!r}, {run_group!r}, {run_note!r}, {git_commit!r})
    print("Done!", result[0] if result else result)
except Exception as e:
    print("Error:", e)
""")
        script_file.close()

        log_file = script_file.name.replace(".py", ".log")
        subprocess.Popen(
            f"nohup {sys.executable} -u {script_file.name} > {log_file} 2>&1 &",
            shell=True,
        )
        print(f"Submitted! Job running on Space in background.")
        print(f"  Log: {log_file}")
        print(f"  Track: https://huggingface.co/spaces/invi-bhagyesh/ValueArena")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("spec", help="Path to run spec")
    parser.add_argument("--collection-enabled", type=str, default=None,
                        help="Override collection.enabled (True/False)")
    args = parser.parse_args()
    collection_override = None
    if args.collection_enabled is not None:
        collection_override = args.collection_enabled.lower() == "true"
    main(args.spec, collection_enabled=collection_override)
