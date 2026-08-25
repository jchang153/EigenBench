"""Bounded tolerance for permanently failed collection tasks.

Kept dependency-free and separate from openrouter_tasks so the local vLLM path
can share one budget without importing the API client, which pulls in openai and
dotenv at module scope.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FailureBudget:
    """How many permanently failed tasks a run may absorb before stopping.

    A run should not lose four thousand good calls because one judge exhausted
    its attempts on task 3,900. But silently absorbing failures is worse than
    stopping: coverage that thins unevenly across judges biases the trust matrix
    along the very axis the benchmark measures, and the models most likely to
    fail a format check are the small local ones, not the API judges. So
    failures are counted against an explicit budget and reported per judge, and
    exceeding the budget still stops the run.

    ``limit`` of 0 is the strict default -- any permanent failure stops
    collection, which is the behavior every caller had before this existed.

    Recorded only from the thread orchestrating a phase, never from a worker, so
    no lock is needed. One budget is shared across every phase of a run, so the
    limit bounds total loss rather than loss per phase.
    """

    limit: int = 0
    failures: list[tuple[dict, dict]] = field(default_factory=list)

    def record(self, identity: dict, error: dict) -> bool:
        """Note a permanent failure. True if collection may continue."""

        self.failures.append((identity, error))
        return len(self.failures) <= self.limit

    @property
    def spent(self) -> int:
        return len(self.failures)

    def by_model(self) -> dict[str, int]:
        """Failure counts keyed by judge, or by evaluee for response tasks."""

        counts: dict[str, int] = {}
        for identity, _ in self.failures:
            who = identity.get("judge") or identity.get("evaluee") or "unknown"
            counts[who] = counts.get(who, 0) + 1
        return counts

    def by_stage(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for identity, _ in self.failures:
            stage = identity.get("stage", "unknown")
            counts[stage] = counts.get(stage, 0) + 1
        return counts

    def summary(self) -> str:
        if not self.failures:
            return "no failed tasks"
        stages = ", ".join(f"{k}={v}" for k, v in sorted(self.by_stage().items()))
        models = ", ".join(
            f"{k}={v}" for k, v in sorted(self.by_model().items(), key=lambda kv: -kv[1])
        )
        return (
            f"{self.spent}/{self.limit} failure budget used; "
            f"by stage: {stages}; by judge/evaluee: {models}"
        )


__all__ = ["FailureBudget"]
