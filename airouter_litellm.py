"""AI Router model selection for a LiteLLM proxy.

Clients send requests to a virtual model (default name ``airouter-auto``). Before
LiteLLM routes such a request, this callback asks the AI Router Decision API which
model fits the prompt and rewrites the request to the matching deployment of this
proxy. Keys, deployments, budgets and spend tracking stay in LiteLLM.

Any failure (AI Router unreachable, timeout, no evidence, no allowed candidate)
leaves the request on the virtual model, which is a regular model_list entry that
points at the default model. The AI Router can therefore never block a request.

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


__version__ = "0.1.0"

logger = logging.getLogger("airouter_litellm")
if os.environ.get("AIROUTER_LOG_LEVEL"):
    logger.setLevel(os.environ["AIROUTER_LOG_LEVEL"].upper())
    if not logger.handlers:
        _handler = logging.StreamHandler()
        _handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
        logger.addHandler(_handler)

METADATA_DECISION_ID = "airouter_decision_id"
METADATA_MODEL = "airouter_model"

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
    select_timeout_s: float = 1.0
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
            select_timeout_s=float(env.get("AIROUTER_SELECT_TIMEOUT_MS", "1000")) / 1000,
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
    """
    to_local: dict[str, str] = {}
    seen_groups: list[str] = []
    automatic: dict[str, str] = {}
    for deployment in deployments:
        group = deployment.get("model_name")
        if not isinstance(group, str) or group == settings.virtual_model:
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


# ------------------------------------------------------------------------ callback


class AiRouterSelection(CustomLogger):
    def __init__(self, settings: Settings | None = None, transport: httpx.AsyncBaseTransport | None = None):
        super().__init__()
        self._settings = settings
        self._transport = transport
        self._client: httpx.AsyncClient | None = None
        self._catalog: dict[str, str] = {}
        self._catalog_at = 0.0
        self._mapping: Mapping | None = None
        self._mapping_at = 0.0
        self._warned_unmapped: set[str] = set()
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

    async def _load_catalog(self) -> dict[str, str]:
        now = time.monotonic()
        if self._catalog and now - self._catalog_at < self.settings.catalog_ttl_s:
            return self._catalog
        try:
            response = await self._http().get("/v1/models", timeout=5.0)
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
            self._catalog, self._catalog_at = catalog, now
        except Exception as exc:  # keep the last known catalog
            logger.warning("airouter: catalog refresh failed, keeping previous catalog: %r", exc)
            self._catalog_at = now - self.settings.catalog_ttl_s + 30.0
        return self._catalog

    async def mapping(self, llm_router: Any) -> Mapping:
        now = time.monotonic()
        if self._mapping is not None and now - self._mapping_at < self.settings.mapping_ttl_s:
            return self._mapping
        catalog = await self._load_catalog()
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
        try:
            return await self._select(user_api_key_dict, data, str(getattr(call_type, "value", call_type)))
        except Exception as exc:  # never break a request because of the AI Router
            logger.warning("airouter: selection skipped: %r", exc)
            return data

    async def _select(self, user_api_key_dict: Any, data: dict, call_type: str) -> dict:
        settings = self.settings
        if data.get("model") != settings.virtual_model or call_type not in SUPPORTED_CALL_TYPES:
            return data
        if not settings.api_key:
            logger.warning("airouter: AIROUTER_API_KEY is not set, staying on %s", settings.virtual_model)
            return data
        messages = selection_messages(data)
        if not messages:
            return data
        llm_router = self._proxy_router()
        mapping = await self.mapping(llm_router)
        groups = await self._allowed_groups(list(mapping.to_catalog), user_api_key_dict, llm_router)
        if not groups:
            return data
        body: dict[str, Any] = {
            "messages": messages,
            "models": [mapping.to_catalog[g] for g in groups],
            "tools": bool(data.get("tools")),
        }
        default_catalog = self._default_catalog_model(llm_router, mapping)
        if default_catalog in body["models"]:
            body["model"] = default_catalog
        max_tokens = _max_tokens(data)
        if max_tokens:
            body["max_tokens"] = max_tokens

        started = time.monotonic()
        response = await self._http().post("/v1/select", json=body, timeout=settings.select_timeout_s)
        response.raise_for_status()
        decision = response.json()
        candidates = decision.get("candidates") or []
        if not candidates:
            return data  # no evidence for this prompt: default model
        best = candidates[0]
        target = mapping.to_local.get(best.get("model"))
        if target is None:
            return data
        data["model"] = target
        self._apply_effort(data, call_type, best.get("effort"))
        # On Anthropic and Responses routes "metadata" is a provider parameter that would be
        # forwarded upstream; LiteLLM's own request metadata lives in "litellm_metadata" there.
        metadata_key = "metadata" if call_type in CHAT_CALL_TYPES else "litellm_metadata"
        metadata = data.setdefault(metadata_key, {})
        if isinstance(metadata, dict) and decision.get("decision_id"):
            metadata[METADATA_DECISION_ID] = decision["decision_id"]
            metadata[METADATA_MODEL] = best["model"]
        logger.info(
            "airouter: %s -> %s (%s) in %.0f ms",
            settings.virtual_model, target, best.get("model"), (time.monotonic() - started) * 1000,
        )
        return data

    def _default_catalog_model(self, llm_router: Any, mapping: Mapping) -> str | None:
        """Catalog name of the model the virtual model points at, if it is mapped."""
        for deployment in getattr(llm_router, "model_list", None) or []:
            if deployment.get("model_name") != self.settings.virtual_model:
                continue
            params, info = deployment.get("litellm_params") or {}, deployment.get("model_info") or {}
            for identity in model_identities(params, info):
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
