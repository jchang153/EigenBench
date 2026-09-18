# Model identity in the Inspect cache

EigenBench captures the requested provider model name and arguments before lazy
initialization. This identity is included in generation cache scopes, so adapters
sharing one base cannot share cache entries merely because a provider replaces
its model name with the base name. Isolated extension clients retain the same
identity when recreated; upstream's `memoize=False` behavior is preserved.

The cache namespace is versioned. Older generation-cache entries are not reused.
This does not repair responses already saved in logs, response JSON caches, or
exports. Runs affected by identity contamination need fresh collection outputs.

Validation covers adapter/base separation after simulated provider-name mutation
and the existing Inspect collection and extension regression suite. Actual GPU
throughput and multi-adapter scheduling are separate from cache correctness.
