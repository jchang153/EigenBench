"""Focused regressions for this change."""

def test_adapter_cache_survives_lazy_provider_name_change(monkeypatch):
    import asyncio
    import importlib
    from types import SimpleNamespace
    from inspect_ai.model import GenerateConfig
    from inspect_pipeline.phases import generate_validated

    module = importlib.import_module("inspect_pipeline.eigenbench")
    responses = {}
    calls = []
    scopes = {}
    class FakeModel:
        def __init__(self, name):
            self.name = self.requested = name
        def __str__(self):
            return self.name
        async def generate(self, **kwargs):
            policy = kwargs["cache"]
            scopes.setdefault(self.requested, []).append(dict(policy.scopes))
            key = (str(self), tuple(sorted(policy.scopes.items())))
            value = responses.setdefault(key, self.requested)
            self.name = "vllm/shared-base"
            return SimpleNamespace(completion=value, stop_reason="stop")
    def get_model(name, **kwargs):
        calls.append(kwargs)
        return FakeModel(name)
    monkeypatch.setattr(module, "get_model", get_model)
    models = {"a": "inspect:vllm/base:adapter-a", "b": "inspect:vllm/base:adapter-b", "base": "inspect:vllm/base"}
    async def run():
        for _ in range(2):  # Recreated isolated clients must reuse their own identities.
            resolve = module._model_resolver(models, memoize=False)
            for attempt in range(3):
                for nick, requested in models.items():
                    output = await generate_validated(resolve(nick), [], config=GenerateConfig(max_tokens=10), max_attempts=1, cache_enabled=True, validator=None, identity=nick)
                    assert output.completion == requested.removeprefix("inspect:")
    asyncio.run(run())
    assert all(call["memoize"] is False for call in calls)
    assert len({values[0]["model_identity"] for values in scopes.values()}) == 3
    assert all(all(value == values[0] for value in values) for values in scopes.values())
    assert all(values[0]["eigenbench_cache_version"] == "2" for values in scopes.values())
