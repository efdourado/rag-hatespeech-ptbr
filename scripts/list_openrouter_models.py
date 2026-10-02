"""Consulta catálogo público e preços, sem chave e sem inferência."""

from __future__ import annotations

import argparse
import json

import httpx

from rag_hatespeech_ptbr.openrouter import BASE_URL


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", action="store_true")
    parser.add_argument("--free-only", action="store_true")
    parser.add_argument("--structured-only", action="store_true")
    args = parser.parse_args()
    endpoint = "embeddings/models" if args.embeddings else "models"
    response = httpx.get(f"{BASE_URL}/{endpoint}", timeout=30)
    response.raise_for_status()
    result = []
    for model in response.json()["data"]:
        pricing = model.get("pricing", {})
        free = all(float(pricing.get(key, 0)) == 0 for key in ("prompt", "completion"))
        structured = "structured_outputs" in model.get("supported_parameters", [])
        if (args.free_only and not free) or (args.structured_only and not structured):
            continue
        result.append(
            {
                "id": model["id"],
                "pricing_usd_per_token": pricing,
                "context_length": model.get("context_length"),
                "structured_outputs": structured,
                "supported_parameters": model.get("supported_parameters", []),
            }
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
