"""Cliente OpenRouter: geração estruturada, embeddings e rastreio de uso."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterError(RuntimeError):
    """Falha de API, sem copiar prompts ou credenciais para a mensagem."""


@dataclass(frozen=True)
class APIResult:
    payload: dict[str, Any]
    latency_seconds: float


@dataclass
class OpenRouterClient:
    api_key: str = field(repr=False)
    max_requests: int = 100
    transport: httpx.BaseTransport | None = field(default=None, repr=False)
    calls: list[dict[str, Any]] = field(default_factory=list, init=False)
    attempts: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if not self.api_key.strip():
            raise ValueError("Defina OPENROUTER_API_KEY no .env local.")
        if self.max_requests < 1:
            raise ValueError("max_requests deve ser positivo.")

    @classmethod
    def from_env(cls, *, max_requests: int = 100) -> OpenRouterClient:
        return cls(os.environ.get("OPENROUTER_API_KEY", ""), max_requests=max_requests)

    def post(self, endpoint: str, body: dict[str, Any]) -> APIResult:
        if endpoint not in {"embeddings", "chat/completions"}:
            raise ValueError("Endpoint não suportado.")
        if self.attempts >= self.max_requests:
            raise OpenRouterError("Limite de chamadas atingido; aumente --max-api-requests.")
        self.attempts += 1
        start = time.perf_counter()
        record: dict[str, Any] = {
            "endpoint": endpoint,
            "requested_model": body["model"],
            "http_status": None,
        }
        self.calls.append(record)
        try:
            with httpx.Client(transport=self.transport, timeout=90.0) as client:
                response = client.post(
                    f"{BASE_URL}/{endpoint}",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "X-OpenRouter-Title": "rag-hatespeech-ptbr",
                    },
                    json=body,
                )
        except httpx.RequestError:
            record["latency_seconds"] = time.perf_counter() - start
            record["network_error"] = True
            raise OpenRouterError(
                "Falha de rede no OpenRouter. Sem repetição automática: "
                "a chamada pode ter sido cobrada. Retome usando o mesmo cache/execução."
            ) from None
        elapsed = time.perf_counter() - start
        record.update(latency_seconds=elapsed, http_status=response.status_code)
        if response.is_error:
            raise OpenRouterError(
                f"OpenRouter HTTP {response.status_code}: consulte saldo, "
                "modelo, provedor e limites da conta."
            )
        try:
            payload = response.json()
        except ValueError:
            raise OpenRouterError("OpenRouter retornou JSON inválido.") from None
        if not isinstance(payload, dict) or "error" in payload:
            raise OpenRouterError("OpenRouter retornou uma falha no corpo da resposta.")
        record.update(
            response_id=payload.get("id"),
            returned_model=payload.get("model"),
            provider=payload.get("provider"),
            usage=payload.get("usage", {}),
        )
        return APIResult(payload, elapsed)

    def usage_summary(self) -> dict[str, Any]:
        known_costs = [
            call.get("usage", {}).get("cost")
            for call in self.calls
            if isinstance(call.get("usage", {}).get("cost"), (int, float))
        ]
        return {
            "attempted_requests": self.attempts,
            "recorded_requests": len(self.calls),
            "prompt_tokens": sum(c.get("usage", {}).get("prompt_tokens", 0) for c in self.calls),
            "completion_tokens": sum(
                c.get("usage", {}).get("completion_tokens", 0) for c in self.calls
            ),
            "reported_cost_usd": sum(known_costs) if known_costs else None,
            "cost_complete": len(known_costs) == len(self.calls) and bool(self.calls),
        }


def provider_preferences(provider: str | None = None) -> dict[str, Any]:
    preferences: dict[str, Any] = {"data_collection": "deny", "require_parameters": True}
    if provider:
        preferences.update(only=[provider], allow_fallbacks=False)
    return preferences
