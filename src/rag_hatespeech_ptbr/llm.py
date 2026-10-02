"""Mesmo prompt/classificador para baseline e RAG, com parsing estrito."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rag_hatespeech_ptbr.openrouter import OpenRouterClient, provider_preferences
from rag_hatespeech_ptbr.retrieval import RetrievedExample

PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts/offensiveness_system_v1.txt"
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "offensive_label": {"type": "integer", "enum": [0, 1]},
        "justification": {"type": "string"},
        "evidence": {"type": "array", "items": {"type": "string"}},
        "retrieved_example_ids": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["offensive_label", "justification", "evidence", "retrieved_example_ids"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class Prediction:
    offensive_label: int
    justification: str
    evidence: list[str]
    retrieved_example_ids: list[int]


def parse_prediction(content: str, comment: str, allowed_ids: set[int]) -> Prediction:
    try:
        obj = json.loads(content)
    except (TypeError, ValueError):
        raise ValueError("Resposta do modelo não é JSON válido.") from None
    if not isinstance(obj, dict) or set(obj) != set(OUTPUT_SCHEMA["required"]):
        raise ValueError("Resposta do modelo não segue o esquema solicitado.")
    if type(obj["offensive_label"]) is not int or obj["offensive_label"] not in (0, 1):
        raise ValueError("offensive_label deve ser inteiro 0 ou 1.")
    if not isinstance(obj["justification"], str) or not obj["justification"].strip():
        raise ValueError("Justificativa ausente.")
    evidence = obj["evidence"]
    if not isinstance(evidence, list) or any(
        not isinstance(e, str) or not e or e not in comment for e in evidence
    ):
        raise ValueError("Evidência não é trecho literal do comentário avaliado.")
    ids = obj["retrieved_example_ids"]
    if (
        not isinstance(ids, list)
        or any(type(i) is not int or i not in allowed_ids for i in ids)
        or len(ids) != len(set(ids))
    ):
        raise ValueError("Modelo citou IDs de exemplos não fornecidos ou repetidos.")
    return Prediction(**obj)


class LLMClassifier:
    def __init__(
        self,
        model: str,
        *,
        client: OpenRouterClient | None = None,
        provider: str | None = None,
        dry_run: bool = False,
        prompt_path: Path = PROMPT_PATH,
        max_tokens: int = 512,
    ) -> None:
        if not dry_run and (not model or client is None):
            raise ValueError("Modelo e cliente OpenRouter obrigatórios para execução real.")
        if not dry_run and model.startswith("openrouter/"):
            raise ValueError(
                "Escolha um modelo fixo; roteadores automáticos não servem "
                "para comparar o mesmo LLM com e sem RAG."
            )
        self.model = "dry-run/fixture-v1" if dry_run else model
        self.client, self.provider, self.dry_run = client, provider, dry_run
        self.system_prompt = prompt_path.read_text(encoding="utf-8")
        self.prompt_sha256 = hashlib.sha256(self.system_prompt.encode()).hexdigest()
        self.max_tokens = max_tokens

    def messages(
        self, comment: str, examples: list[RetrievedExample], *, include_rationales: bool = True
    ) -> list[dict[str, str]]:
        context = []
        for example in examples:
            item: dict[str, Any] = {
                "id": example.id,
                "comment": example.comment,
                "offensive_label": example.offensive_label,
            }
            if include_rationales:
                item["rationales"] = example.rationales
            context.append(item)
        return [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": json.dumps(
                    {"comment": comment, "training_examples": context}, ensure_ascii=False
                ),
            },
        ]

    def predict(
        self, comment: str, examples: list[RetrievedExample], *, include_rationales: bool = True
    ) -> Prediction:
        messages = self.messages(comment, examples, include_rationales=include_rationales)
        if self.dry_run:
            # Fixture deliberadamente simples: valida apenas o fluxo e os formatos.
            return Prediction(0, "Simulação técnica; não é resultado científico.", [], [])
        assert self.client is not None
        payload = self.client.post(
            "chat/completions",
            {
                "model": self.model,
                "messages": messages,
                "temperature": 0,
                "max_tokens": self.max_tokens,
                "seed": 42,
                "provider": provider_preferences(self.provider),
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "offensiveness",
                        "strict": True,
                        "schema": OUTPUT_SCHEMA,
                    },
                },
            },
        ).payload
        try:
            choice = payload["choices"][0]
            if choice.get("finish_reason") != "stop":
                raise ValueError("Resposta recusada, incompleta ou truncada pelo modelo.")
            content = choice["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise ValueError("Resposta sem classificação utilizável.") from None
        return parse_prediction(content, comment, {e.id for e in examples})

    def configuration(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "temperature": 0,
            "seed": 42,
            "max_tokens": self.max_tokens,
            "prompt_sha256": self.prompt_sha256,
            "output_schema": OUTPUT_SCHEMA,
            "provider": provider_preferences(self.provider),
            "dry_run": self.dry_run,
        }


def prediction_record(prediction: Prediction) -> dict[str, Any]:
    return asdict(prediction)
