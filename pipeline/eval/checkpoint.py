"""Durable task checkpoints for long-running mixed collection jobs."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
from typing import Any, Iterable

from pipeline.utils import load_records


CHECKPOINT_VERSION = 1


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _write_jsonl_temporary(path: Path, records: Iterable[dict]) -> tuple[Path, int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    digest = hashlib.sha256()
    count = 0
    try:
        with os.fdopen(fd, "wb") as handle:
            for record in records:
                encoded = (json.dumps(record, ensure_ascii=True) + "\n").encode("utf-8")
                handle.write(encoded)
                digest.update(encoded)
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        return temporary_path, count, digest.hexdigest()
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


class CollectionCheckpoint:
    """File-per-task checkpoint with atomic writes and deterministic IDs."""

    def __init__(self, path: str | Path):
        self.root = Path(path)
        self.manifest_path = self.root / "manifest.json"
        self.completed_dir = self.root / "completed"
        self.failed_dir = self.root / "failed"
        self.finalizing_path = self.root / "finalizing.json"
        self.finalized_path = self.root / "finalized.json"
        self._lock = threading.Lock()

    @staticmethod
    def default_path(evaluations_path: str | Path) -> Path:
        path = Path(evaluations_path)
        return path.with_name(f"{path.name}.checkpoint")

    def _task_path(self, directory: Path, identity: dict[str, Any]) -> Path:
        return directory / f"{_sha256_json(identity)}.json"

    def has_manifest(self) -> bool:
        return self.manifest_path.exists()

    def initialize_or_resume(
        self,
        *,
        context: dict[str, Any],
        assignments: list[dict] | None = None,
    ) -> list[dict]:
        """Create a manifest or validate and return its saved assignments."""

        context_fingerprint = _sha256_json(context)
        with self._lock:
            if self.manifest_path.exists():
                manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
                if manifest.get("version") != CHECKPOINT_VERSION:
                    raise RuntimeError(
                        f"Unsupported checkpoint version in {self.manifest_path}: "
                        f"{manifest.get('version')}"
                    )
                if manifest.get("context_fingerprint") != context_fingerprint:
                    raise RuntimeError(
                        "Collection inputs do not match the existing checkpoint. "
                        f"Use the original spec or a new checkpoint path: {self.root}"
                    )
                saved_assignments = manifest.get("assignments")
                if not isinstance(saved_assignments, list):
                    raise RuntimeError(f"Checkpoint manifest has no assignments: {self.manifest_path}")
                return saved_assignments

            if assignments is None:
                raise ValueError("assignments are required when creating a checkpoint")
            manifest = {
                "version": CHECKPOINT_VERSION,
                "context_fingerprint": context_fingerprint,
                "context": context,
                "assignments": assignments,
            }
            _atomic_write_json(self.manifest_path, manifest)
            return assignments

    def load_completed(self, identity: dict[str, Any]) -> dict[str, Any] | None:
        path = self._task_path(self.completed_dir, identity)
        if not path.exists():
            return None
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("identity") != identity:
            raise RuntimeError(f"Checkpoint task identity mismatch: {path}")
        payload = record.get("payload")
        if not isinstance(payload, dict):
            raise RuntimeError(f"Checkpoint task payload is invalid: {path}")
        return payload

    def save_completed(self, identity: dict[str, Any], payload: dict[str, Any]) -> None:
        completed_path = self._task_path(self.completed_dir, identity)
        failed_path = self._task_path(self.failed_dir, identity)
        with self._lock:
            _atomic_write_json(
                completed_path,
                {"identity": identity, "status": "completed", "payload": payload},
            )
            failed_path.unlink(missing_ok=True)

    def save_failed(self, identity: dict[str, Any], error: dict[str, Any]) -> None:
        failed_path = self._task_path(self.failed_dir, identity)
        with self._lock:
            _atomic_write_json(
                failed_path,
                {"identity": identity, "status": "failed", "error": error},
            )

    def has_finalized_output(self) -> bool:
        self._recover_finalization_if_possible()
        return self.finalized_path.exists()

    def _recover_finalization_if_possible(self) -> None:
        if self.finalized_path.exists() or not self.finalizing_path.exists():
            return
        metadata = json.loads(self.finalizing_path.read_text(encoding="utf-8"))
        output_value = metadata.get("evaluations_path")
        if not output_value:
            return
        output_path = Path(output_value)
        if not output_path.exists():
            return
        if _sha256_file(output_path) != metadata.get("sha256"):
            return
        if len(load_records(output_path)) != int(metadata.get("records", -1)):
            return
        _atomic_write_json(self.finalized_path, metadata)
        self.finalizing_path.unlink(missing_ok=True)

    def load_finalized_output(self, evaluations_path: str | Path) -> list[dict]:
        path = Path(evaluations_path)
        if not self.finalized_path.exists():
            raise RuntimeError("Checkpoint is not finalized")
        metadata = json.loads(self.finalized_path.read_text(encoding="utf-8"))
        if not path.exists():
            raise RuntimeError(f"Finalized evaluations file is missing: {path}")
        actual_hash = _sha256_file(path)
        if actual_hash != metadata.get("sha256"):
            raise RuntimeError(
                f"Finalized evaluations file hash mismatch: {path}. "
                "Refusing to continue with modified output."
            )
        records = load_records(path)
        if len(records) != int(metadata.get("records", -1)):
            raise RuntimeError(f"Finalized evaluations record count mismatch: {path}")
        return records

    def assert_output_is_safe(self, evaluations_path: str | Path) -> None:
        self._recover_finalization_if_possible()
        path = Path(evaluations_path)
        if path.exists() and path.stat().st_size > 0 and not self.finalized_path.exists():
            raise RuntimeError(
                f"Evaluations output already exists without a matching finalized checkpoint: {path}. "
                "Refusing to append or overwrite it."
            )

    def finalize(self, evaluations_path: str | Path, records: list[dict]) -> None:
        path = Path(evaluations_path)
        with self._lock:
            temporary_path, record_count, output_hash = _write_jsonl_temporary(path, records)
            metadata = {
                "version": CHECKPOINT_VERSION,
                "evaluations_path": str(path.resolve()),
                "records": record_count,
                "sha256": output_hash,
            }
            try:
                _atomic_write_json(self.finalizing_path, metadata)
                os.replace(temporary_path, path)
                _atomic_write_json(self.finalized_path, metadata)
                self.finalizing_path.unlink(missing_ok=True)
            finally:
                temporary_path.unlink(missing_ok=True)


__all__ = ["CHECKPOINT_VERSION", "CollectionCheckpoint"]
