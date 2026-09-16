"""LlmGateway: the only module that talks to a model provider (D21).

One call shape for every provider: `structured(stage, schema, system, user,
images)` returns a validated pydantic object. Providers:

- ollama      local, free, nothing leaves the laptop (default)
- cloudflare  Workers AI free tier (10k neurons/day), JSON mode on text models
- anthropic   Claude, paid key; prompt caching on the last system block

Guarantees enforced here:
- No tools are ever passed (D5): the model returns data, it cannot act.
- Output is validated against the schema; invalid output gets one corrective
  retry, then raises. Categorical fields must be enums in the schema, because
  small local models confuse similarly-typed string fields (measured, D21).
- Every call writes one UsageLedger row (provider:model, tokens, cost).
- Missing provider config raises LlmUnavailable; capture keeps working.
"""

from __future__ import annotations

import base64
import copy
import json
import logging
import time
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from lecture_copilot.config import Settings
from lecture_copilot.store import Store, new_id

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# USD per million tokens. Anthropic list prices; Cloudflare paid rates (free
# tier covers 10k neurons/day first). Re-verify before publishing cost figures.
PRICES: dict[str, dict[str, float]] = {
    "claude-sonnet-5": {"input": 2.00, "output": 10.00, "cache_write": 2.50, "cache_read": 0.20},
    "claude-opus-5": {"input": 5.00, "output": 25.00, "cache_write": 6.25, "cache_read": 0.50},
    "@cf/meta/llama-3.3-70b-instruct-fp8-fast": {"input": 0.293, "output": 2.253},
    "@cf/meta/llama-3.1-8b-instruct": {"input": 0.282, "output": 0.827},
    "@cf/meta/llama-3.2-11b-vision-instruct": {"input": 0.049, "output": 0.676},
    "@cf/meta/llama-4-scout-17b-16e-instruct": {"input": 0.270, "output": 0.850},
}


class LlmUnavailable(RuntimeError):
    pass


class LlmRefused(RuntimeError):
    pass


class LlmInvalidOutput(RuntimeError):
    pass


def cost_usd(model: str, input_tokens: int, output: int, cache_write: int = 0, cache_read: int = 0) -> float:
    p = PRICES.get(model)
    if not p:
        return 0.0
    total = input_tokens * p["input"] + output * p["output"]
    total += cache_write * p.get("cache_write", 0.0) + cache_read * p.get("cache_read", 0.0)
    return round(total / 1_000_000, 6)


def json_schema_for(schema: type[BaseModel]) -> dict:
    """Pydantic JSON schema with $ref/$defs inlined (grammar-based providers
    handle flat schemas most reliably)."""
    raw = schema.model_json_schema()
    defs = raw.pop("$defs", {})

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                return walk(copy.deepcopy(defs[node["$ref"].split("/")[-1]]))
            return {k: walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node

    return walk(raw)


class LlmGateway:
    def __init__(self, settings: Settings, store: Store) -> None:
        self.s = settings
        self.store = store
        self._anthropic = None
        self._http = httpx.Client(timeout=600)

    # -- availability ----------------------------------------------------
    @property
    def provider(self) -> str:
        return self.s.llm_provider

    @property
    def available(self) -> bool:
        p = self.provider
        if p == "ollama":
            return True  # reachability is checked at call time
        if p == "cloudflare":
            return bool(self.s.cloudflare_api_token and self.s.cloudflare_account_id)
        if p == "anthropic":
            return bool(self.s.anthropic_api_key)
        return False

    def describe(self) -> dict:
        p = self.provider
        models = {
            "ollama": (self.s.ollama_model_text, self.s.ollama_model_vision),
            "cloudflare": (self.s.cloudflare_model_text, self.s.cloudflare_model_vision),
            "anthropic": (self.s.anthropic_model, self.s.anthropic_model),
        }.get(p, ("?", "?"))
        return {"provider": p, "available": self.available, "text_model": models[0], "vision_model": models[1], "local": p == "ollama"}

    # -- the one call ----------------------------------------------------
    def structured(
        self,
        *,
        stage: str,
        schema: type[T],
        system: list[str],
        user: str,
        images: list[bytes] | None = None,
        effort: str = "low",
        model: str | None = None,
        lecture_id: str | None = None,
        max_tokens: int = 4000,
        cache: bool = True,
    ) -> T:
        if not self.available:
            raise LlmUnavailable(self._unavailable_reason())
        system_text = "\n\n".join(t for t in system if t)
        js = json_schema_for(schema)
        t0 = time.perf_counter()
        attempts = 0
        hint = ""
        while True:
            attempts += 1
            if self.provider == "ollama":
                raw, meta = self._ollama(system_text, user + hint, images, js, model, max_tokens)
            elif self.provider == "cloudflare":
                raw, meta = self._cloudflare(system_text, user + hint, images, js, model, max_tokens)
            else:
                raw, meta = self._anthropic(system, user + hint, images, schema, effort, model, max_tokens, cache)
            self._ledger(stage, meta, lecture_id, effort, int((time.perf_counter() - t0) * 1000))
            if meta.get("stop_reason") == "refusal":
                raise LlmRefused(f"{stage}: the model declined this request")
            try:
                return schema.model_validate_json(raw) if isinstance(raw, str) else schema.model_validate(raw)
            except ValidationError as exc:
                log.warning("%s: invalid structured output (attempt %d): %s", stage, attempts, str(exc)[:200])
                if attempts >= 2:
                    raise LlmInvalidOutput(f"{stage}: output did not match the schema after {attempts} attempts") from exc
                hint = f"\n\nYour previous answer did not match the required JSON schema: {str(exc)[:400]}. Return only valid JSON."

    def _unavailable_reason(self) -> str:
        p = self.provider
        if p == "cloudflare":
            return "Cloudflare provider needs CLOUDFLARE_API_TOKEN and CLOUDFLARE_ACCOUNT_ID in .env"
        if p == "anthropic":
            return "Anthropic provider needs ANTHROPIC_API_KEY in .env"
        return f"unknown LLM provider '{p}' (use ollama, cloudflare or anthropic)"

    # -- providers -------------------------------------------------------
    def _ollama(self, system: str, user: str, images: list[bytes] | None, js: dict, model: str | None, max_tokens: int) -> tuple[str, dict]:
        name = model or (self.s.ollama_model_vision if images else self.s.ollama_model_text)
        msg: dict[str, Any] = {"role": "user", "content": user}
        if images:
            msg["images"] = [base64.b64encode(b).decode() for b in images]
        body = {
            "model": name,
            "stream": False,
            "format": js,
            # num_predict caps output: a small model looping inside a grammar
            # otherwise generates until the context is full (measured, D21).
            "options": {"temperature": 0, "num_ctx": self.s.ollama_num_ctx, "num_predict": max_tokens},
            "messages": [{"role": "system", "content": system}, msg],
        }
        try:
            r = self._http.post(f"{self.s.ollama_url}/api/chat", json=body)
        except httpx.ConnectError as exc:
            raise LlmUnavailable(f"Ollama is not reachable at {self.s.ollama_url} (is it running?)") from exc
        if r.status_code == 404:
            raise LlmUnavailable(f"Ollama model '{name}' is not installed: run `ollama pull {name}`")
        r.raise_for_status()
        data = r.json()
        meta = {
            "model": f"ollama:{name}",
            "input_tokens": data.get("prompt_eval_count", 0) or 0,
            "output_tokens": data.get("eval_count", 0) or 0,
            "cost_usd": 0.0,
            "stop_reason": data.get("done_reason"),
        }
        return data["message"]["content"], meta

    def _cloudflare(
        self, system: str, user: str, images: list[bytes] | None, js: dict, model: str | None, max_tokens: int
    ) -> tuple[Any, dict]:
        token, acct = self.s.cloudflare_api_token, self.s.cloudflare_account_id
        headers = {"Authorization": f"Bearer {token}"}
        base = f"https://api.cloudflare.com/client/v4/accounts/{acct}/ai"
        if images:
            # Vision models have no JSON mode on Workers AI: ask for JSON and validate.
            name = model or self.s.cloudflare_model_vision
            parts: list[dict] = [
                {"type": "text", "text": f"{user}\n\nRespond with only a JSON object matching this schema:\n{json.dumps(js)}"}
            ]
            for b in images:
                parts.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(b).decode()}})
            body = {
                "model": name,
                "max_tokens": max_tokens,
                "temperature": 0,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": parts}],
            }
            r = self._http.post(f"{base}/v1/chat/completions", headers=headers, json=body)
            self._cf_check(r)
            data = r.json()
            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage") or {}
            raw: Any = _extract_json(content)
        else:
            name = model or self.s.cloudflare_model_text
            body = {
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "response_format": {"type": "json_schema", "json_schema": js},
                "max_tokens": max_tokens,
                "temperature": 0,
            }
            r = self._http.post(f"{base}/run/{name}", headers=headers, json=body)
            self._cf_check(r)
            data = r.json()
            if not data.get("success"):
                raise RuntimeError(f"Workers AI error: {data.get('errors')}")
            result = data["result"]
            raw = result.get("response")
            usage = result.get("usage") or {}
        inp, out = int(usage.get("prompt_tokens", 0) or 0), int(usage.get("completion_tokens", 0) or 0)
        meta = {
            "model": f"cloudflare:{name}",
            "input_tokens": inp,
            "output_tokens": out,
            "cost_usd": cost_usd(name, inp, out),
            "stop_reason": None,
        }
        return raw, meta

    @staticmethod
    def _cf_check(r: httpx.Response) -> None:
        if r.status_code in (401, 403):
            raise LlmUnavailable("Cloudflare rejected the token: it needs the 'Workers AI - Read' and 'Workers AI - Edit' permissions")
        if r.status_code != 200:
            raise RuntimeError(f"Workers AI HTTP {r.status_code}: {r.text[:300]}")

    def _anthropic(
        self,
        system: list[str],
        user: str,
        images: list[bytes] | None,
        schema: type[BaseModel],
        effort: str,
        model: str | None,
        max_tokens: int,
        cache: bool,
    ) -> tuple[Any, dict]:
        if self._anthropic is None:
            import anthropic

            self._anthropic = anthropic.Anthropic(api_key=self.s.anthropic_api_key, max_retries=3)
        name = model or self.s.anthropic_model
        system_blocks: list[dict] = [{"type": "text", "text": t} for t in system if t]
        if cache and system_blocks:
            system_blocks[-1]["cache_control"] = {"type": "ephemeral"}
        content: list[dict] = [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": base64.b64encode(b).decode()}}
            for b in images or []
        ]
        content.append({"type": "text", "text": user})
        response = self._anthropic.messages.parse(
            model=name,
            max_tokens=max_tokens,
            system=system_blocks,
            messages=[{"role": "user", "content": content}],
            output_format=schema,
            output_config={"effort": effort},
        )
        u = response.usage
        cw = getattr(u, "cache_creation_input_tokens", 0) or 0
        cr = getattr(u, "cache_read_input_tokens", 0) or 0
        meta = {
            "model": f"anthropic:{name}",
            "input_tokens": u.input_tokens,
            "output_tokens": u.output_tokens,
            "cache_write": cw,
            "cache_read": cr,
            "cost_usd": cost_usd(name, u.input_tokens, u.output_tokens, cw, cr),
            "stop_reason": response.stop_reason,
        }
        parsed = response.parsed_output
        return (parsed.model_dump() if parsed is not None else "{}"), meta

    # -- ledger ----------------------------------------------------------
    def _ledger(self, stage: str, meta: dict, lecture_id: str | None, effort: str, latency_ms: int) -> None:
        self.store.add_usage(
            call_id=new_id(),
            lecture_id=lecture_id,
            stage=stage,
            model=meta["model"],
            effort=effort,
            input_tokens=meta["input_tokens"],
            cache_read=meta.get("cache_read", 0),
            cache_write=meta.get("cache_write", 0),
            output_tokens=meta["output_tokens"],
            cost_usd=meta["cost_usd"],
            latency_ms=latency_ms,
            stop_reason=meta.get("stop_reason"),
        )
        log.info(
            "%s: %s in=%d out=%d $%.5f %dms",
            stage,
            meta["model"],
            meta["input_tokens"],
            meta["output_tokens"],
            meta["cost_usd"],
            latency_ms,
        )


def _extract_json(text: str) -> Any:
    """Pull the first JSON object out of a model reply that may wrap it in prose or fences."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{") :]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return text
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return text
