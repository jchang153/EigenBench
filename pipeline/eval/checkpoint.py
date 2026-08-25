"""Durable task checkpoints for long-running mixed collection jobs.

Task results go into a single append-only log, ``tasks.jsonl``, rather than one
file per task. The file-per-task layout wrote two metadata operations and one
fsync for every call -- roughly 4,200 of each for a 100-scenario direct run,
from ten concurrent workers -- and MooseFS (RunPod's /workspace) answers that
with EIO, which killed two runs mid-flight. One appended handle is an ordinary
write pattern for a network filesystem, so the checkpoint can live next to the
run spec instead of on container-local disk that a pod restart erases.

Checkpoints written by the previous layout are still read on resume, so a run
already in progress under ``completed/``/``failed/`` continues without redoing
work.
"""

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

# Records are flushed to the OS on every append, so losing a process costs
# nothing. fsync is what MooseFS objected to at per-task frequency, so it runs
# every FSYNC_INTERVAL records instead: ~16 calls across a 4,200-task run, which
# bounds what a machine failure can cost to the last few completions.
FSYNC_INTERVAL = 256


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
    """Append-only task log with atomic manifest writes and deterministic IDs."""

    def __init__(self, path: str | Path):
        self.root = Path(path)
        self.manifest_path = self.root / "manifest.json"
        self.tasks_path = self.root / "tasks.jsonl"
        # Read-only, for resuming checkpoints written by the file-per-task layout.
        self.legacy_completed_dir = self.root / "completed"
        self.legacy_failed_dir = self.root / "failed"
        self.finalizing_path = self.root / "finalizing.json"
        self.finalized_path = self.root / "finalized.json"
        self._lock = threading.Lock()
        self._index: dict[str, dict[str, Any]] | None = None
        self._log = None
        self._appends_since_fsync = 0

    @staticmethod
    def default_path(evaluations_path: str | Path) -> Path:
        path = Path(evaluations_path)
        return path.with_name(f"{path.name}.checkpoint")

    # -- task index ---------------------------------------------------------

    def _load_index_locked(self) -> dict[str, dict[str, Any]]:
        if self._index is not None:
            return self._index

        index: dict[str, dict[str, Any]] = {}
        if self.tasks_path.exists():
            with self.tasks_path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        # Only the final line can be torn, and only if the
                        # machine died mid-append. Anything earlier means the
                        # file was edited, which is worth refusing.
                        remaining = handle.read().strip()
                        if remaining:
                            raise RuntimeError(
                                f"Corrupt checkpoint record at {self.tasks_path}:{line_number}"
                            )
                        break
                    key = record.get("key")
                    if not isinstance(key, str):
                        raise RuntimeError(
                            f"Checkpoint record has no key at {self.tasks_path}:{line_number}"
                        )
                    # Later lines supersede earlier ones, so a retry that
                    # succeeds overrides the failure recorded before it.
                    index[key] = record

        # Legacy per-task files fill in only where the log is silent, so a log
        # entry always wins over the older layout.
        for directory, status in (
            (self.legacy_completed_dir, "completed"),
            (self.legacy_failed_dir, "failed"),
        ):
            if not directory.is_dir():
                continue
            for path in directory.glob("*.json"):
                key = path.stem
                if key in index:
                    continue
                record = json.loads(path.read_text(encoding="utf-8"))
                record["key"] = key
                record.setdefault("status", status)
                index[key] = record

        self._index = index
        return index

    def _append_locked(self, record: dict[str, Any]) -> None:
        if self._log is None:
            self.root.mkdir(parents=True, exist_ok=True)
            self._log = self.tasks_path.open("a", encoding="utf-8")
        self._log.write(json.dumps(record, ensure_ascii=True) + "\n")
        self._log.flush()
        self._appends_since_fsync += 1
        if self._appends_since_fsync >= FSYNC_INTERVAL:
            os.fsync(self._log.fileno())
            self._appends_since_fsync = 0

    def close(self) -> None:
        with self._lock:
            if self._log is None:
                return
            self._log.flush()
            os.fsync(self._log.fileno())
            self._log.close()
            self._log = None
            self._appends_since_fsync = 0

    # -- manifest -----------------------------------------------------------

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

    # -- task results -------------------------------------------------------

    def load_completed(self, identity: dict[str, Any]) -> dict[str, Any] | None:
        key = _sha256_json(identity)
        with self._lock:
            record = self._load_index_locked().get(key)
        if record is None or record.get("status") != "completed":
            return None
        if record.get("identity") != identity:
            raise RuntimeError(f"Checkpoint task identity mismatch for key {key}")
        payload = record.get("payload")
        if not isinstance(payload, dict):
            raise RuntimeError(f"Checkpoint task payload is invalid for key {key}")
        return payload

    def save_completed(self, identity: dict[str, Any], payload: dict[str, Any]) -> None:
        key = _sha256_json(identity)
        record = {"key": key, "identity": identity, "status": "completed", "payload": payload}
        with self._lock:
            index = self._load_index_locked()
            self._append_locked(record)
            index[key] = record

    def save_failed(self, identity: dict[str, Any], error: dict[str, Any]) -> None:
        key = _sha256_json(identity)
        record = {"key": key, "identity": identity, "status": "failed", "error": error}
        with self._lock:
            index = self._load_index_locked()
            self._append_locked(record)
            index[key] = record

    # -- finalization -------------------------------------------------------

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
        self.close()
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


__all__ = ["CHECKPOINT_VERSION", "FSYNC_INTERVAL", "CollectionCheckpoint"]
