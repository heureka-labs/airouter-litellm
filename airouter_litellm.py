"""AI Router model selection for a LiteLLM proxy.

Clients send requests to a virtual model. Any model_list entry with an ``airouter``
block in its ``model_info`` is one, so a proxy can offer as many as it needs, each
with its own candidate models and weighting (for example one per use case or per
project). The entry named by ``AIROUTER_VIRTUAL_MODEL`` (default ``airouter-auto``)
is a virtual model as well, with all models as candidates and the AI Router
organization's weighting. Before LiteLLM routes a request to a virtual model, this
callback asks the AI Router Decision API which candidate fits the prompt and
rewrites the request to that model group. Keys, deployments, budgets and spend
tracking stay in LiteLLM.

Any failure (AI Router unreachable, timeout, no evidence, no allowed candidate)
leaves the request on the virtual model, which is a regular model_list entry that
points at its default model. The AI Router can therefore never block a request.

After each executed call the token usage is queued and reported to
``/v1/usage-reports`` in the background, so the AI Router dashboard shows actual
usage and savings.

Install: put this file next to the proxy's config.yaml (or ``pip install``) and add
``airouter_litellm.selection`` to ``litellm_settings.callbacks``. Configuration is
read from environment variables, see README.md.
"""

# No "from __future__ import annotations": LiteLLM loads a callback file next to the
# config without registering it in sys.modules, and dataclasses then fail to resolve
# string annotations at import time.

import asyncio
import json
import logging
import math
import os
import re
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import httpx

try:
    from litellm.integrations.custom_logger import CustomLogger
except ImportError:  # pragma: no cover - allows unit tests without litellm
    class CustomLogger:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass


__version__ = "0.2.0"

logger = logging.getLogger("airouter_litellm")
if os.environ.get("AIROUTER_LOG_LEVEL"):
    logger.setLevel(os.environ["AIROUTER_LOG_LEVEL"].upper())
    if not logger.handlers:
        _handler = logging.StreamHandler()
        _handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
        logger.addHandler(_handler)

METADATA_DECISION_ID = "airouter_decision_id"
METADATA_MODEL = "airouter_model"
METADATA_VIRTUAL_MODEL = "airouter_virtual_model"

CONFIG_KEY = "airouter"  # model_info block that makes a model_list entry a virtual model
WEIGHTING_KEYS = frozenset({"quality", "costs", "latency"})

# Call types this callback acts on. Everything else passes through untouched.
CHAT_CALL_TYPES = frozenset({"completion", "acompletion"})
ANTHROPIC_CALL_TYPES = frozenset({"anthropic_messages", "aanthropic_messages"})
RESPONSES_CALL_TYPES = frozenset({"responses", "aresponses"})
SUPPORTED_CALL_TYPES = CHAT_CALL_TYPES | ANTHROPIC_CALL_TYPES | RESPONSES_CALL_TYPES


# --------------------------------------------------------------------------- config


@dataclass(frozen=True)
class Settings:
    api_key: str
    base_url: str = "https://api.airouter.io"
    virtual_model: str = "airouter-auto"
    select_timeout_s: float = 2.0
    model_map: dict[str, str] = field(default_factory=dict)
    exclude_models: frozenset[str] = frozenset()
    catalog_ttl_s: float = 600.0
    mapping_ttl_s: float = 60.0
    report_flush_interval_s: float = 5.0
    report_batch_size: int = 500
    report_queue_limit: int = 50_000

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "Settings":
        env = dict(os.environ if env is None else env)
        api_key = env.get("AIROUTER_API_KEY", "")
        model_map: dict[str, str] = {}
        raw_map = env.get("AIROUTER_MODEL_MAP", "").strip()
        if raw_map:
            loaded = json.loads(raw_map)
            if not isinstance(loaded, dict) or not all(
                isinstance(k, str) and isinstance(v, str) for k, v in loaded.items()
            ):
                raise ValueError("AIROUTER_MODEL_MAP must be a JSON object of strings")
            model_map = loaded
        exclude = frozenset(
            name.strip() for name in env.get("AIROUTER_EXCLUDE_MODELS", "").split(",") if name.strip()
        )
        return cls(
            api_key=api_key,
            base_url=env.get("AIROUTER_BASE_URL", cls.base_url).rstrip("/"),
            virtual_model=env.get("AIROUTER_VIRTUAL_MODEL", cls.virtual_model),
            select_timeout_s=float(env.get("AIROUTER_SELECT_TIMEOUT_MS", "2000")) / 1000,
            model_map=model_map,
            exclude_models=exclude,
        )


# ----------------------------------------------------------------- prompt extraction


def _text_of(content: Any) -> str:
    """Flatten any provider's message content (string or content-part list) to text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        content = [content]
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for part in content:
        if isinstance(part, str):
            parts.append(part)
        elif isinstance(part, dict):
            text = part.get("text")
            if isinstance(text, str):
                parts.append(text)
            elif part.get("type") == "tool_result":
                parts.append(_text_of(part.get("content")))
    return "\n".join(p for p in parts if p)


def selection_messages(data: dict[str, Any]) -> list[dict[str, str]]:
    """Build plain {role, content} messages from chat, Anthropic or Responses payloads."""
    out: list[dict[str, str]] = []

    def add(role: Any, content: Any) -> None:
        text = _text_of(content)
        if text:
            out.append({"role": role if isinstance(role, str) else "user", "content": text})

    add("system", data.get("system"))  # Anthropic /v1/messages
    add("system", data.get("instructions"))  # Responses API

    messages = data.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, dict):
                add(message.get("role"), message.get("content"))

    items = data.get("input")  # Responses API
    if isinstance(items, str):
        add("user", items)
    elif isinstance(items, list):
        for item in items:
            if isinstance(item, str):
                add("user", item)
            elif isinstance(item, dict):
                if item.get("type") in (None, "message"):
                    add(item.get("role"), item.get("content"))
                elif item.get("type") == "function_call_output":
                    add("tool", item.get("output"))
    return out


def _max_tokens(data: dict[str, Any]) -> int | None:
    for key in ("max_tokens", "max_completion_tokens", "max_output_tokens"):
        value = data.get(key)
        if isinstance(value, int) and value > 0:
            return value
    return None


# -------------------------------------------------------------------- model mapping

_DATE_SUFFIX = re.compile(r"(@\d{8}|-\d{8}|-\d{4}-\d{2}-\d{2}|-v\d+(:\d+)?)$")
_VENDOR_PREFIX = re.compile(r"^(?:[a-z]{2,4}\.)?(?:anthropic|meta|mistral|amazon|cohere|ai21|deepseek|qwen|openai|google)\.")


def model_identities(litellm_params: dict[str, Any], model_info: dict[str, Any]) -> list[str]:
    """Names under which a deployment's underlying model may appear in the AI Router catalog.

    ``model_info.base_model`` wins (LiteLLM uses it for cost tracking of e.g. Azure
    deployments with custom names), then ``litellm_params.model``. Provider prefixes
    (``azure/``, ``vertex_ai/``, ``bedrock/anthropic.``) and version suffixes are stripped.
    """
    raw: list[str] = []
    for value in (model_info.get("base_model"), litellm_params.get("model")):
        if isinstance(value, str) and value:
            raw.append(value)
    identities: list[str] = []
    for value in raw:
        last = value.split("/")[-1]
        candidates = [value, last, _VENDOR_PREFIX.sub("", last)]  # bedrock "anthropic.claude-..."
        for candidate in list(candidates):
            stripped = candidate
            while True:
                shorter = _DATE_SUFFIX.sub("", stripped)
                if shorter == stripped:
                    break
                stripped = shorter
            candidates.append(stripped)
        for candidate in candidates:
            if candidate and candidate not in identities:
                identities.append(candidate)
    return identities


@dataclass(frozen=True)
class Mapping:
    to_local: dict[str, str]  # AI Router catalog name -> proxy model_name
    to_catalog: dict[str, str]  # proxy model_name -> AI Router catalog name
    unmapped: tuple[str, ...]  # proxy model_names without a catalog match


def build_mapping(
    deployments: list[dict[str, Any]],
    catalog: dict[str, str],
    settings: Settings,
) -> Mapping:
    """Match the proxy's model groups to AI Router catalog names.

    ``catalog`` maps every known name (catalog id and, when the AI Router exposes them,
    aliases) in lower case to its catalog id. Explicit ``AIROUTER_MODEL_MAP`` entries
    (catalog name -> proxy model_name) take precedence over automatic matches. When two
    model groups resolve to the same catalog model, the first one in the model list wins.
    Virtual models are never candidates themselves.
    """
    virtual = find_virtual_models(deployments, settings)
    to_local: dict[str, str] = {}
    seen_groups: list[str] = []
    automatic: dict[str, str] = {}
    for deployment in deployments:
        group = deployment.get("model_name")
        if not isinstance(group, str) or group in virtual:
            continue
        if group in settings.exclude_models:
            continue
        if group not in seen_groups:
            seen_groups.append(group)
        if group in automatic.values():
            continue
        names = [group] + model_identities(
            deployment.get("litellm_params") or {}, deployment.get("model_info") or {}
        )
        for name in names:
            catalog_id = catalog.get(name.lower())
            if catalog_id and catalog_id not in automatic:
                automatic[catalog_id] = group
                break
    to_local.update(automatic)
    for catalog_name, group in settings.model_map.items():
        catalog_id = catalog.get(catalog_name.lower(), catalog_name)
        to_local = {k: v for k, v in to_local.items() if v != group}
        to_local[catalog_id] = group
    to_catalog = {group: catalog_id for catalog_id, group in to_local.items()}
    unmapped = tuple(g for g in seen_groups if g not in to_catalog)
    return Mapping(to_local=to_local, to_catalog=to_catalog, unmapped=unmapped)


# ------------------------------------------------------------------- virtual models


@dataclass(frozen=True)
class VirtualModel:
    """A model group whose requests the AI Router routes.

    ``candidates`` are proxy model_names (None: every mapped model group), ``weighting``
    the boost factors sent to the Decision API (None: the AI Router organization's
    preferences). ``error`` is set when the ``airouter`` block is invalid; such a virtual
    model is never routed and its requests stay on its default model.
    """

    name: str
    candidates: tuple[str, ...] | None = None
    weighting: dict[str, float] | None = None
    default_identities: tuple[str, ...] = ()
    error: str | None = None


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def parse_virtual_model(name: str, raw: Any, deployment: dict[str, Any]) -> VirtualModel:
    """Validate one ``model_info.airouter`` block.

    Accepted forms: absent or ``true`` (all models, organization weighting) or a mapping
    with the optional keys ``models`` (list of proxy model_names) and ``weighting``
    (``quality``, ``costs`` and ``latency``, each a non-negative number, all three required
    as in the Decision API).
    """
    identities = tuple(model_identities(deployment.get("litellm_params") or {}, deployment.get("model_info") or {}))

    def invalid(reason: str) -> VirtualModel:
        return VirtualModel(name=name, default_identities=identities, error=reason)

    if raw is None or raw is True:
        return VirtualModel(name=name, default_identities=identities)
    if not isinstance(raw, dict):
        return invalid("model_info.airouter must be true or a mapping with 'models' and/or 'weighting'")
    unknown = sorted(set(raw) - {"models", "weighting"})
    if unknown:
        return invalid(f"unknown key(s) in model_info.airouter: {', '.join(unknown)}")

    candidates: tuple[str, ...] | None = None
    if raw.get("models") is not None:
        models = raw["models"]
        if not isinstance(models, list) or not models or not all(isinstance(m, str) and m for m in models):
            return invalid("model_info.airouter.models must be a non-empty list of model names")
        candidates = tuple(dict.fromkeys(models))

    weighting: dict[str, float] | None = None
    if raw.get("weighting") is not None:
        w = raw["weighting"]
        if not isinstance(w, dict) or set(w) != WEIGHTING_KEYS or not all(_number(v) for v in w.values()):
            return invalid(
                "model_info.airouter.weighting needs quality, costs and latency, each a non-negative number"
            )
        weighting = {key: float(w[key]) for key in ("quality", "costs", "latency")}

    return VirtualModel(name=name, candidates=candidates, weighting=weighting, default_identities=identities)


def find_virtual_models(deployments: list[dict[str, Any]], settings: Settings) -> dict[str, VirtualModel]:
    """Virtual models of the proxy: entries with an ``airouter`` block plus AIROUTER_VIRTUAL_MODEL.

    The first deployment of a model group defines its configuration; give every deployment
    of a virtual model group the same block.
    """
    found: dict[str, VirtualModel] = {}
    for deployment in deployments:
        name = deployment.get("model_name")
        if not isinstance(name, str) or name in found:
            continue
        info = deployment.get("model_info") or {}
        raw = info.get(CONFIG_KEY) if hasattr(info, "get") else None
        if raw is None and name != settings.virtual_model:
            continue
        if raw is False:  # explicit opt-out, also for the AIROUTER_VIRTUAL_MODEL name
            continue
        found[name] = parse_virtual_model(name, raw, deployment)
    return found


# ------------------------------------------------------------------------ callback


class AiRouterSelection(CustomLogger):
    def __init__(self, settings: Settings | None = None, transport: httpx.AsyncBaseTransport | None = None):
        super().__init__()
        self._settings = settings
        self._transport = transport
        self._client: httpx.AsyncClient | None = None
        self._catalog: dict[str, str] = {}
        self._catalog_at = 0.0
        self._catalog_task: asyncio.Task[None] | None = None
        self._mapping: Mapping | None = None
        self._mapping_at = 0.0
        self._warned_unmapped: set[str] = set()
        self._virtual: dict[str, VirtualModel] = {}
        self._virtual_at: float | None = None
        self._warned: set[str] = set()
        self._reports: deque[dict[str, Any]] = deque()
        self._flusher: asyncio.Task[None] | None = None

    # -- infrastructure

    @property
    def settings(self) -> Settings:
        if self._settings is None:
            self._settings = Settings.from_env()
        return self._settings

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.settings.base_url,
                headers={"Authorization": f"Bearer {self.settings.api_key}"},
                transport=self._transport,
            )
        return self._client

    @staticmethod
    def _proxy_router() -> Any:
        from litellm.proxy import proxy_server

        return proxy_server.llm_router

    async def _fetch_catalog(self) -> None:
        try:
            response = await self._http().get("/v1/models", timeout=10.0)
            response.raise_for_status()
            catalog: dict[str, str] = {}
            for entry in response.json().get("data", []):
                model_id = entry.get("id")
                if not isinstance(model_id, str):
                    continue
                catalog.setdefault(model_id.lower(), model_id)
                for alias in entry.get("aliases") or []:
                    if isinstance(alias, str):
                        catalog.setdefault(alias.lower(), model_id)
            self._catalog, self._catalog_at = catalog, time.monotonic()
            self._mapping = None  # rebuild against the new catalog
        except Exception as exc:  # keep the last known catalog, retry in 30 s
            logger.warning("airouter: catalog refresh failed, keeping previous catalog: %r", exc)
            self._catalog_at = time.monotonic() - self.settings.catalog_ttl_s + 30.0

    async def _load_catalog(self) -> dict[str, str]:
        """Return the catalog without making a request wait for a refresh.

        A stale catalog is refreshed in the background while requests keep using it. Only
        the very first request after start waits for the catalog, inside the selection budget.
        """
        stale = time.monotonic() - self._catalog_at >= self.settings.catalog_ttl_s
        if (not self._catalog or stale) and (self._catalog_task is None or self._catalog_task.done()):
            self._catalog_task = asyncio.get_running_loop().create_task(self._fetch_catalog())
        if not self._catalog and self._catalog_task is not None:
            await asyncio.shield(self._catalog_task)  # a budget timeout must not cancel the fetch
        return self._catalog

    async def mapping(self, llm_router: Any) -> Mapping:
        catalog = await self._load_catalog()  # cheap; starts a background refresh when stale
        now = time.monotonic()
        if self._mapping is not None and now - self._mapping_at < self.settings.mapping_ttl_s:
            return self._mapping
        deployments = list(getattr(llm_router, "model_list", None) or [])
        mapping = build_mapping(deployments, catalog, self.settings)
        for group in mapping.unmapped:
            if group not in self._warned_unmapped:
                self._warned_unmapped.add(group)
                logger.warning(
                    "airouter: model group %r has no AI Router catalog match and is never selected; "
                    "map it with AIROUTER_MODEL_MAP or ask AI Router support to add the model",
                    group,
                )
        if catalog:
            self._mapping, self._mapping_at = mapping, now
        return mapping

    def _warn_once(self, message: str, *args: Any, level: int = logging.WARNING) -> None:
        key = message % args
        if key not in self._warned:
            self._warned.add(key)
            logger.log(level, message, *args)

    def virtual_models(self, llm_router: Any) -> dict[str, VirtualModel]:
        """Virtual models of the proxy, re-read with the mapping TTL (models added via UI or API)."""
        now = time.monotonic()
        if self._virtual_at is not None and now - self._virtual_at < self.settings.mapping_ttl_s:
            return self._virtual
        deployments = list(getattr(llm_router, "model_list", None) or [])
        self._virtual, self._virtual_at = find_virtual_models(deployments, self.settings), now
        for virtual in self._virtual.values():
            if virtual.error:
                self._warn_once(
                    "airouter: virtual model %r is not routed, its requests stay on its default model: %s",
                    virtual.name, virtual.error, level=logging.ERROR,
                )
        return self._virtual

    def _candidate_groups(self, virtual: VirtualModel, mapping: Mapping) -> list[str]:
        if virtual.candidates is None:
            return list(mapping.to_catalog)
        groups: list[str] = []
        for group in virtual.candidates:
            if group in mapping.to_catalog:
                groups.append(group)
            elif group in self._virtual:
                self._warn_once("airouter: virtual model %r lists virtual model %r as a candidate, ignored",
                                virtual.name, group)
            else:
                self._warn_once(
                    "airouter: virtual model %r lists %r, which is no model group of this proxy with an AI Router "
                    "catalog match (or is excluded), ignored", virtual.name, group,
                )
        return groups

    async def _allowed_groups(self, groups: list[str], user_api_key_dict: Any, llm_router: Any) -> list[str]:
        """Keep only model groups the calling key (and its team) may use.

        LiteLLM checks model access before pre-call hooks run, for the virtual model
        only, so the rewritten target has to be authorized here with LiteLLM's own check.
        """
        from litellm.proxy.auth.auth_checks import can_key_call_resolved_model

        allowed: list[str] = []
        for group in groups:
            try:
                await can_key_call_resolved_model(
                    model=group,
                    llm_model_list=llm_router.model_list,
                    valid_token=user_api_key_dict,
                    llm_router=llm_router,
                )
                allowed.append(group)
            except Exception:
                continue
        return allowed

    # -- request path

    async def async_pre_call_hook(self, user_api_key_dict: Any, cache: Any, data: dict, call_type: Any) -> dict:
        # One budget for the whole selection step (catalog, access checks, decision), so
        # AIROUTER_SELECT_TIMEOUT_MS is the most a request can ever wait for the AI Router.
        # _select mutates data only after its last await, so a timeout leaves it untouched.
        try:
            return await asyncio.wait_for(
                self._select(user_api_key_dict, data, str(getattr(call_type, "value", call_type))),
                timeout=self.settings.select_timeout_s,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "airouter: selection skipped: no decision within %.0f ms", self.settings.select_timeout_s * 1000
            )
            return data
        except Exception as exc:  # never break a request because of the AI Router
            logger.warning("airouter: selection skipped: %r", exc)
            return data

    async def _select(self, user_api_key_dict: Any, data: dict, call_type: str) -> dict:
        settings = self.settings
        requested = data.get("model")
        if not isinstance(requested, str) or call_type not in SUPPORTED_CALL_TYPES:
            return data
        llm_router = self._proxy_router()
        virtual = self.virtual_models(llm_router).get(requested)
        if virtual is None or virtual.error:
            return data
        if not settings.api_key:
            logger.warning("airouter: AIROUTER_API_KEY is not set, staying on %s", requested)
            return data
        messages = selection_messages(data)
        if not messages:
            return data
        mapping = await self.mapping(llm_router)
        groups = await self._allowed_groups(self._candidate_groups(virtual, mapping), user_api_key_dict, llm_router)
        if not groups:
            return data
        body: dict[str, Any] = {
            "messages": messages,
            "models": [mapping.to_catalog[g] for g in groups],
            "tools": bool(data.get("tools")),
        }
        if virtual.weighting is not None:
            body["weighting"] = dict(virtual.weighting)
        default_catalog = self._default_catalog_model(virtual)
        if default_catalog in body["models"]:
            body["model"] = default_catalog
        max_tokens = _max_tokens(data)
        if max_tokens:
            body["max_tokens"] = max_tokens

        started = time.monotonic()
        response = await self._http().post("/v1/select", json=body, timeout=None)  # bounded by wait_for
        response.raise_for_status()
        decision = response.json()
        candidates = decision.get("candidates") or []
        if not candidates:
            return data  # no evidence for this prompt: default model
        best = candidates[0]
        target = mapping.to_local.get(best.get("model"))
        if target is None or target not in groups:
            return data
        data["model"] = target
        self._apply_effort(data, call_type, best.get("effort"))
        # On Anthropic and Responses routes "metadata" is a provider parameter that would be
        # forwarded upstream; LiteLLM's own request metadata lives in "litellm_metadata" there.
        metadata_key = "metadata" if call_type in CHAT_CALL_TYPES else "litellm_metadata"
        metadata = data.setdefault(metadata_key, {})
        if isinstance(metadata, dict):
            # The rewrite replaces the requested model, so keep the virtual model visible in
            # LiteLLM's spend logs and callbacks.
            metadata[METADATA_VIRTUAL_MODEL] = virtual.name
            if decision.get("decision_id"):
                metadata[METADATA_DECISION_ID] = decision["decision_id"]
                metadata[METADATA_MODEL] = best["model"]
        logger.info(
            "airouter: %s -> %s (%s) in %.0f ms",
            virtual.name, target, best.get("model"), (time.monotonic() - started) * 1000,
        )
        return data

    def _default_catalog_model(self, virtual: VirtualModel) -> str | None:
        """Catalog name of the model a virtual model points at, if it is in the catalog."""
        for identity in virtual.default_identities:
            catalog_id = self._catalog.get(identity.lower())
            if catalog_id:
                return catalog_id
        return None

    @staticmethod
    def _apply_effort(data: dict, call_type: str, effort: Any) -> None:
        """Pass the selected reasoning effort on, unless the client set one itself."""
        if not isinstance(effort, str):
            return
        if call_type in RESPONSES_CALL_TYPES:
            reasoning = data.get("reasoning")
            if reasoning is None:
                data["reasoning"] = {"effort": effort}
            elif isinstance(reasoning, dict):
                reasoning.setdefault("effort", effort)
        elif call_type in CHAT_CALL_TYPES:
            data.setdefault("reasoning_effort", effort)
        # Anthropic /v1/messages: no effort mapping, the provider default applies.

    # -- usage reporting

    async def async_log_success_event(self, kwargs: dict, response_obj: Any, start_time: Any, end_time: Any) -> None:
        try:
            self._queue_report(kwargs)
        except Exception as exc:
            logger.warning("airouter: usage report not queued: %r", exc)

    def _queue_report(self, kwargs: dict) -> None:
        litellm_params = kwargs.get("litellm_params") or {}
        metadata: dict[str, Any] = {}
        for source in (litellm_params.get("metadata"), litellm_params.get("litellm_metadata"), kwargs.get("litellm_metadata")):
            if isinstance(source, dict) and source.get(METADATA_DECISION_ID):
                metadata = source
                break
        decision_id = metadata.get(METADATA_DECISION_ID)
        if not decision_id:
            return
        payload = kwargs.get("standard_logging_object") or {}
        executed_group = payload.get("model_group")
        model = metadata.get(METADATA_MODEL)
        if self._mapping is not None and executed_group in self._mapping.to_catalog:
            model = self._mapping.to_catalog[executed_group]  # a LiteLLM fallback may have changed it
        if not model:
            return
        report = {
            "report_id": str(uuid.uuid4()),
            "decision_id": decision_id,
            "model": model,
            "input_tokens": int(payload.get("prompt_tokens") or 0),
            "output_tokens": int(payload.get("completion_tokens") or 0),
            "latency_ms": int(float(payload.get("response_time") or 0) * 1000),
        }
        if len(self._reports) >= self.settings.report_queue_limit:
            self._reports.popleft()
            logger.warning("airouter: usage report queue full, dropped the oldest report")
        self._reports.append(report)
        self._ensure_flusher()

    def _ensure_flusher(self) -> None:
        if self._flusher is None or self._flusher.done():
            self._flusher = asyncio.get_running_loop().create_task(self._flush_loop())

    async def _flush_loop(self) -> None:
        backoff = self.settings.report_flush_interval_s
        while self._reports:
            await asyncio.sleep(backoff)
            ok = await self.flush()
            backoff = self.settings.report_flush_interval_s if ok else min(backoff * 2, 300.0)

    async def flush(self) -> bool:
        """Send queued reports. Returns False when a retryable error left reports queued."""
        while self._reports:
            batch = [self._reports[i] for i in range(min(self.settings.report_batch_size, len(self._reports)))]
            try:
                response = await self._http().post("/v1/usage-reports", json={"reports": batch}, timeout=10.0)
            except Exception as exc:
                logger.warning("airouter: usage report delivery failed, will retry: %r", exc)
                return False
            if response.status_code == 429 or response.status_code >= 500:
                logger.warning("airouter: usage report delivery got %s, will retry", response.status_code)
                return False
            for _ in batch:
                self._reports.popleft()
            if response.status_code >= 400:
                logger.warning("airouter: usage reports rejected (%s): %s", response.status_code, response.text[:500])
                continue
            rejected = (response.json() or {}).get("rejected") or []
            if rejected:
                logger.warning("airouter: %d usage reports rejected: %s", len(rejected), str(rejected)[:500])
        return True


selection = AiRouterSelection()
