"""Unit tests for the AI Router LiteLLM callback, using LiteLLM's real auth checks."""

import json
from types import SimpleNamespace

import httpx
import pytest
from litellm.proxy._types import UserAPIKeyAuth

import airouter_litellm as ar

CATALOG = {"data": [{"id": "gpt-5-mini"}, {"id": "gpt-5.2"}, {"id": "claude-4-5-sonnet"}, {"id": "gemini-3.5-flash"}]}

MODEL_LIST = [
    {"model_name": "airouter-auto", "litellm_params": {"model": "azure/prod-mini"}, "model_info": {"base_model": "azure/gpt-5-mini"}},
    {"model_name": "mini", "litellm_params": {"model": "azure/prod-mini"}, "model_info": {"base_model": "azure/gpt-5-mini"}},
    {"model_name": "big", "litellm_params": {"model": "azure/prod-big"}, "model_info": {"base_model": "azure/gpt-5.2"}},
    {"model_name": "sonnet", "litellm_params": {"model": "vertex_ai/claude-sonnet-4-5@20250929"}, "model_info": {}},
    {"model_name": "flash", "litellm_params": {"model": "gemini/gemini-3.5-flash"}, "model_info": {}},
    {"model_name": "local-qwen", "litellm_params": {"model": "hosted_vllm/Qwen/Qwen3-30B-A3B"}, "model_info": {}},
]


def real_router(model_list):
    """A real LiteLLM router over mock deployments, so auth checks run unmodified."""
    import copy

    import litellm

    deployments = copy.deepcopy(model_list)
    for d in deployments:
        d["litellm_params"].update(api_key="fake", mock_response="ok")
    return litellm.Router(model_list=deployments)


class FakeAiRouter:
    """Stands in for api.airouter.io and records what the callback sends."""

    def __init__(self, pick="gpt-5.2", effort=None, select_status=200, report_status=202):
        self.pick, self.effort = pick, effort
        self.select_status, self.report_status = select_status, report_status
        self.select_bodies, self.report_bodies = [], []

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json=CATALOG)
        if request.url.path == "/v1/select":
            body = json.loads(request.content)
            self.select_bodies.append(body)
            if self.select_status != 200:
                return httpx.Response(self.select_status, json={"detail": "error"})
            if self.pick is None or self.pick not in body["models"]:
                return httpx.Response(200, json={"decision_id": None, "candidates": [], "predicted_savings": None})
            return httpx.Response(200, json={
                "decision_id": "d-1",
                "candidates": [{"model": self.pick, "effort": self.effort, "quality": 1, "costs": 1, "latency": 1,
                                "quality_score": 1, "costs_score": 1, "latency_score": 1, "score": 1}],
                "predicted_savings": None,
            })
        if request.url.path == "/v1/usage-reports":
            self.report_bodies.append(json.loads(request.content))
            return httpx.Response(self.report_status, json={"accepted": 1, "duplicate": 0, "rejected": []})
        return httpx.Response(404)


def make_callback(fake, monkeypatch, model_list=MODEL_LIST, **settings):
    cb = ar.AiRouterSelection(
        settings=ar.Settings(api_key="test", base_url="https://airouter.test", **settings),
        transport=httpx.MockTransport(fake.handler),
    )
    router = real_router(model_list)
    monkeypatch.setattr(ar.AiRouterSelection, "_proxy_router", staticmethod(lambda: router))
    return cb


def chat(text="hello", **extra):
    return {"model": "airouter-auto", "messages": [{"role": "user", "content": text}], **extra}


MASTER = UserAPIKeyAuth(api_key="sk-master", models=[])


# -- mapping


def test_mapping_uses_base_model_and_strips_provider_and_version():
    catalog = {e["id"].lower(): e["id"] for e in CATALOG["data"]}
    mapping = ar.build_mapping(MODEL_LIST, catalog, ar.Settings(api_key="k"))
    assert mapping.to_local == {"gpt-5-mini": "mini", "gpt-5.2": "big", "gemini-3.5-flash": "flash"}
    assert mapping.unmapped == ("sonnet", "local-qwen")


def test_mapping_uses_aliases_when_catalog_exposes_them():
    catalog = {e["id"].lower(): e["id"] for e in CATALOG["data"]}
    catalog["claude-sonnet-4-5"] = "claude-4-5-sonnet"
    mapping = ar.build_mapping(MODEL_LIST, catalog, ar.Settings(api_key="k"))
    assert mapping.to_local["claude-4-5-sonnet"] == "sonnet"


def test_explicit_map_and_exclusions_win():
    catalog = {e["id"].lower(): e["id"] for e in CATALOG["data"]}
    settings = ar.Settings(api_key="k", model_map={"claude-4-5-sonnet": "sonnet"}, exclude_models=frozenset({"flash"}))
    mapping = ar.build_mapping(MODEL_LIST, catalog, settings)
    assert mapping.to_local["claude-4-5-sonnet"] == "sonnet"
    assert "gemini-3.5-flash" not in mapping.to_local
    assert mapping.unmapped == ("local-qwen",)


def test_settings_from_env():
    s = ar.Settings.from_env({"AIROUTER_API_KEY": "k", "AIROUTER_MODEL_MAP": '{"claude-4-5-sonnet": "sonnet"}',
                              "AIROUTER_SELECT_TIMEOUT_MS": "750", "AIROUTER_EXCLUDE_MODELS": "a, b"})
    assert s.model_map == {"claude-4-5-sonnet": "sonnet"} and s.select_timeout_s == 0.75
    assert s.exclude_models == {"a", "b"}
    with pytest.raises(ValueError):
        ar.Settings.from_env({"AIROUTER_MODEL_MAP": '["x"]'})


# -- prompt extraction


def test_selection_messages_for_all_surfaces():
    anthropic = {"system": [{"type": "text", "text": "be brief"}],
                 "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}, {"type": "image", "source": {}}]}]}
    assert ar.selection_messages(anthropic) == [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}]
    responses = {"instructions": "sys", "input": [{"role": "user", "content": [{"type": "input_text", "text": "q"}]},
                                                 {"type": "function_call_output", "output": "42"}]}
    assert ar.selection_messages(responses) == [{"role": "system", "content": "sys"}, {"role": "user", "content": "q"},
                                                {"role": "tool", "content": "42"}]
    assert ar.selection_messages({"input": "plain"}) == [{"role": "user", "content": "plain"}]


# -- selection


async def test_rewrites_to_selected_deployment(monkeypatch):
    fake = FakeAiRouter(pick="gpt-5.2")
    cb = make_callback(fake, monkeypatch)
    data = await cb.async_pre_call_hook(MASTER, None, chat(max_tokens=300, tools=[{"type": "function"}]), "acompletion")
    assert data["model"] == "big"
    assert data["metadata"][ar.METADATA_DECISION_ID] == "d-1"
    body = fake.select_bodies[0]
    assert set(body["models"]) == {"gpt-5-mini", "gpt-5.2", "gemini-3.5-flash"}
    assert body["model"] == "gpt-5-mini" and body["max_tokens"] == 300 and body["tools"] is True


async def test_other_models_and_call_types_pass_through(monkeypatch):
    fake = FakeAiRouter()
    cb = make_callback(fake, monkeypatch)
    direct = {"model": "mini", "messages": [{"role": "user", "content": "x"}]}
    assert (await cb.async_pre_call_hook(MASTER, None, dict(direct), "acompletion"))["model"] == "mini"
    assert (await cb.async_pre_call_hook(MASTER, None, chat(), "aembedding"))["model"] == "airouter-auto"
    assert fake.select_bodies == []


@pytest.mark.parametrize("fake", [FakeAiRouter(select_status=500), FakeAiRouter(select_status=403), FakeAiRouter(pick=None)])
async def test_fails_open_to_default(monkeypatch, fake):
    cb = make_callback(fake, monkeypatch)
    data = await cb.async_pre_call_hook(MASTER, None, chat(), "acompletion")
    assert data["model"] == "airouter-auto"


async def test_fails_open_when_airouter_unreachable(monkeypatch):
    def boom(request):
        raise httpx.ConnectError("down")
    cb = ar.AiRouterSelection(settings=ar.Settings(api_key="t", base_url="https://x.test"), transport=httpx.MockTransport(boom))
    monkeypatch.setattr(ar.AiRouterSelection, "_proxy_router", staticmethod(lambda: real_router(MODEL_LIST)))
    assert (await cb.async_pre_call_hook(MASTER, None, chat(), "acompletion"))["model"] == "airouter-auto"


async def test_only_models_the_key_may_use_are_offered(monkeypatch):
    fake = FakeAiRouter(pick="gpt-5-mini")
    cb = make_callback(fake, monkeypatch)
    key = UserAPIKeyAuth(api_key="sk-x", models=["airouter-auto", "mini", "flash"])
    data = await cb.async_pre_call_hook(key, None, chat(), "acompletion")
    assert set(fake.select_bodies[0]["models"]) == {"gpt-5-mini", "gemini-3.5-flash"}
    assert data["model"] == "mini"


async def test_team_restriction_is_respected(monkeypatch):
    fake = FakeAiRouter(pick="gpt-5-mini")
    cb = make_callback(fake, monkeypatch)
    key = UserAPIKeyAuth(api_key="sk-y", models=[], team_id="t1", team_models=["airouter-auto", "big"])
    monkeypatch.setattr("litellm.proxy.auth.auth_checks.get_team_object", _raise)
    await cb.async_pre_call_hook(key, None, chat(), "acompletion")
    assert fake.select_bodies[0]["models"] == ["gpt-5.2"]


async def _raise(*args, **kwargs):
    raise RuntimeError("no db")


async def test_effort_per_surface_and_client_value_wins(monkeypatch):
    fake = FakeAiRouter(pick="gpt-5.2", effort="low")
    cb = make_callback(fake, monkeypatch)
    assert (await cb.async_pre_call_hook(MASTER, None, chat(), "acompletion"))["reasoning_effort"] == "low"
    own = await cb.async_pre_call_hook(MASTER, None, chat(reasoning_effort="high"), "acompletion")
    assert own["reasoning_effort"] == "high"
    resp = await cb.async_pre_call_hook(MASTER, None, {"model": "airouter-auto", "input": "q"}, "aresponses")
    assert resp["reasoning"] == {"effort": "low"} and resp["model"] == "big"
    anth = await cb.async_pre_call_hook(MASTER, None, chat(), "anthropic_messages")
    assert "reasoning_effort" not in anth and anth["model"] == "big"


# -- usage reporting


def success_kwargs(decision="d-1", group="big", model="gpt-5.2"):
    return {"litellm_params": {"metadata": {ar.METADATA_DECISION_ID: decision, ar.METADATA_MODEL: model}},
            "standard_logging_object": {"model_group": group, "prompt_tokens": 12, "completion_tokens": 34, "response_time": 0.5}}


async def test_usage_report_is_queued_and_flushed(monkeypatch):
    fake = FakeAiRouter()
    cb = make_callback(fake, monkeypatch)
    await cb.async_pre_call_hook(MASTER, None, chat(), "acompletion")  # loads the mapping
    await cb.async_log_success_event(success_kwargs(), None, None, None)
    await cb.async_log_success_event({"litellm_params": {"metadata": {}}}, None, None, None)  # not routed: ignored
    assert await cb.flush() is True
    [report] = fake.report_bodies[0]["reports"]
    assert report["decision_id"] == "d-1" and report["model"] == "gpt-5.2"
    assert (report["input_tokens"], report["output_tokens"], report["latency_ms"]) == (12, 34, 500)


async def test_report_uses_model_actually_executed_after_fallback(monkeypatch):
    fake = FakeAiRouter()
    cb = make_callback(fake, monkeypatch)
    await cb.async_pre_call_hook(MASTER, None, chat(), "acompletion")
    await cb.async_log_success_event(success_kwargs(group="mini"), None, None, None)
    await cb.flush()
    assert fake.report_bodies[0]["reports"][0]["model"] == "gpt-5-mini"


async def test_retryable_report_errors_keep_the_queue(monkeypatch):
    fake = FakeAiRouter(report_status=503)
    cb = make_callback(fake, monkeypatch)
    await cb.async_log_success_event(success_kwargs(), None, None, None)
    assert await cb.flush() is False and len(cb._reports) == 1
    fake.report_status = 202
    assert await cb.flush() is True and len(cb._reports) == 0
