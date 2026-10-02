import json

import httpx
import numpy as np
import pandas as pd
import pytest

from rag_hatespeech_ptbr.experiments import run_comparison
from rag_hatespeech_ptbr.llm import LLMClassifier, parse_prediction
from rag_hatespeech_ptbr.openrouter import OpenRouterClient, OpenRouterError
from rag_hatespeech_ptbr.retrieval import (
    DryRunEmbedder,
    LocalIndex,
    OpenRouterEmbedder,
    build_index,
)

PROVENANCE = {"dataset_source_sha256": "dataset", "manifest_sha256": "manifest"}


@pytest.fixture
def frame():
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "comment": ["bom dia", "seu idiota", "boa tarde", "um teste"],
            "offensive_label": [0, 1, 0, 1],
            "split": ["train", "train", "validation", "test"],
            "rationales_annotator1": [None, "idiota", None, None],
            "rationales_annotator2": [None, "idiota", None, None],
        }
    )


def test_retrieval_round_trip_never_indexes_validation_or_test(frame, tmp_path):
    embedder = DryRunEmbedder()
    index = build_index(frame, embedder, PROVENANCE)
    path = tmp_path / "index.npz"
    index.save(path)
    loaded = LocalIndex.load(path)
    loaded.validate(frame, embedder, PROVENANCE)
    examples = loaded.search(embedder.embed(["seu idiota"]), frame, 2)
    assert loaded.ids.tolist() == [1, 2]
    assert examples[0].id == 2
    assert examples[0].rationales == ["idiota"]
    assert examples[1].rationales == []
    with pytest.raises(FileExistsError):
        index.save(path)


def test_retrieval_rejects_leaked_ids_changed_model_and_provenance(frame):
    embedder = DryRunEmbedder()
    index = build_index(frame, embedder, PROVENANCE)
    with pytest.raises(ValueError, match="Proveniência"):
        index.validate(frame, embedder, {"manifest_sha256": "different"})
    index.ids[1] = 3
    with pytest.raises(ValueError, match="vazamento"):
        index.validate(frame, embedder, PROVENANCE)


def test_embedding_cache_orders_response_and_survives_restart(tmp_path):
    calls = []

    def handler(request):
        body = json.loads(request.content)
        calls.append(body)
        assert body["provider"]["data_collection"] == "deny"
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": [0.0, 2.0]},
                    {"index": 0, "embedding": [3.0, 0.0]},
                ],
                "usage": {"prompt_tokens": 2},
                "model": "embedding-test",
            },
        )

    client = OpenRouterClient("secret", transport=httpx.MockTransport(handler))
    cache = tmp_path / "cache.sqlite"
    embedder = OpenRouterEmbedder(client, "embedding-test", cache)
    vectors = embedder.embed(["a", "b"])
    np.testing.assert_array_equal(vectors, [[1.0, 0.0], [0.0, 1.0]])
    restarted = OpenRouterEmbedder(client, "embedding-test", cache)
    np.testing.assert_array_equal(restarted.embed(["b", "a"]), [[0.0, 1.0], [1.0, 0.0]])
    assert len(calls) == 1
    assert client.usage_summary()["reported_cost_usd"] is None
    assert "secret" not in repr(client)


def test_embedding_cache_does_not_reuse_other_models(tmp_path):
    calls = []

    def handler(request):
        calls.append(json.loads(request.content)["model"])
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0, 1.0]}]})

    client = OpenRouterClient("key", transport=httpx.MockTransport(handler))
    cache = tmp_path / "cache.sqlite"
    for model in ("one", "two"):
        OpenRouterEmbedder(client, model, cache).embed(["same text"])
    assert calls == ["one", "two"]


@pytest.mark.parametrize(
    "data",
    [
        [{"index": 0, "embedding": [0.0, 0.0]}],
        [{"index": 2, "embedding": [1.0, 1.0]}],
        [{"index": 0, "embedding": [float("inf"), 1.0]}],
    ],
)
def test_invalid_embeddings_are_rejected(tmp_path, data):
    # Mock payloads with nonfinite floats need a pre-encoded response body.
    client = OpenRouterClient(
        "key",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, content=json.dumps({"data": data}).encode())
        ),
    )
    with pytest.raises(ValueError):
        OpenRouterEmbedder(client, "model", tmp_path / "cache.sqlite").embed(["text"])


def test_client_failure_is_sanitized_and_budget_prevents_extra_request():
    client = OpenRouterClient(
        "private-key",
        max_requests=1,
        transport=httpx.MockTransport(
            lambda request: httpx.Response(402, json={"error": "private-key raw text"})
        ),
    )
    with pytest.raises(OpenRouterError, match="402") as exc:
        client.post("embeddings", {"model": "m", "input": ["private text"]})
    assert "private-key" not in str(exc.value)
    with pytest.raises(OpenRouterError, match="Limite"):
        client.post("embeddings", {"model": "m"})
    assert len(client.calls) == 1


def test_network_error_is_recorded_without_automatic_retry():
    def failure(request):
        raise httpx.ReadTimeout("sensitive text", request=request)

    client = OpenRouterClient("key", transport=httpx.MockTransport(failure))
    with pytest.raises(OpenRouterError, match="Sem repetição"):
        client.post("embeddings", {"model": "m"})
    assert client.calls[0]["network_error"] is True
    assert client.attempts == 1


def prediction_json(**changes):
    value = {
        "offensive_label": 1,
        "justification": "Insulto direto.",
        "evidence": ["idiota"],
        "retrieved_example_ids": [2],
    }
    value.update(changes)
    return json.dumps(value)


@pytest.mark.parametrize(
    "changes",
    [
        {"offensive_label": True},
        {"offensive_label": "1"},
        {"evidence": ["inventado"]},
        {"retrieved_example_ids": [999]},
        {"retrieved_example_ids": [2, 2]},
        {"justification": ""},
    ],
)
def test_model_response_rejects_invalid_labels_evidence_and_citations(changes):
    with pytest.raises(ValueError):
        parse_prediction(prediction_json(**changes), "seu idiota", {2})


def test_prompt_ablation_and_structured_output_use_same_classifier(frame):
    requests = []

    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        context = json.loads(body["messages"][1]["content"])["training_examples"]
        ids = [item["id"] for item in context]
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": prediction_json(retrieved_example_ids=ids)},
                    }
                ],
                "usage": {"cost": 0.01, "prompt_tokens": 10, "completion_tokens": 5},
                "model": "fixed-model",
                "provider": "test-provider",
            },
        )

    client = OpenRouterClient("key", transport=httpx.MockTransport(handler))
    classifier = LLMClassifier("fixed-model", client=client)
    embedder = DryRunEmbedder()
    examples = build_index(frame, embedder, PROVENANCE).search(
        embedder.embed(["seu idiota"]), frame, 1
    )
    for context, rationales in (([], True), (examples, True), (examples, False)):
        classifier.predict("seu idiota", context, include_rationales=rationales)
    assert len({request["messages"][0]["content"] for request in requests}) == 1
    assert all(request["response_format"]["json_schema"]["strict"] for request in requests)
    payloads = [json.loads(request["messages"][1]["content"]) for request in requests]
    assert payloads[0]["training_examples"] == []
    assert payloads[1]["training_examples"][0]["rationales"] == ["idiota"]
    assert "rationales" not in payloads[2]["training_examples"][0]
    assert "offensive_label" not in payloads[0]
    assert client.usage_summary()["reported_cost_usd"] == pytest.approx(0.03)


def test_dry_run_has_no_scientific_metrics_and_resume_skips_predictions(frame, tmp_path):
    embedder = DryRunEmbedder()
    index = build_index(frame, embedder, PROVENANCE)
    classifier = LLMClassifier("", dry_run=True)
    kwargs = {
        "modes": ["baseline", "rag", "rag_no_rationales"],
        "run_dir": tmp_path / "run",
        "provenance": PROVENANCE,
        "embedder": embedder,
        "index": index,
        "top_k": 1,
    }
    evaluation = frame.loc[frame.split.eq("validation")]
    first = run_comparison(frame, evaluation, classifier, **kwargs)
    assert first["status"] == "complete"
    assert first["metrics"] is None and first["scientific_result"] is False
    results = (kwargs["run_dir"] / "predictions.jsonl").read_bytes()
    run_comparison(frame, evaluation, classifier, **kwargs)
    assert (kwargs["run_dir"] / "predictions.jsonl").read_bytes() == results
    with pytest.raises(ValueError, match="Configuração mudou"):
        run_comparison(frame, evaluation, classifier, **{**kwargs, "top_k": 2})


def test_partial_experiment_can_resume_without_repeating_finished_call(frame, tmp_path):
    requests = []
    should_fail = True

    def handler(request):
        nonlocal should_fail
        requests.append(json.loads(request.content))
        if len(requests) == 2 and should_fail:
            should_fail = False
            return httpx.Response(429)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": prediction_json(
                                offensive_label=0, evidence=[], retrieved_example_ids=[]
                            )
                        },
                    }
                ]
            },
        )

    client = OpenRouterClient("key", transport=httpx.MockTransport(handler))
    classifier = LLMClassifier("fixed-model", client=client)
    evaluation = pd.concat(
        [
            frame.loc[frame.split.eq("validation")],
            frame.loc[frame.split.eq("test")].assign(split="validation"),
        ]
    )
    run_dir = tmp_path / "run"
    kwargs = {"modes": ["baseline"], "run_dir": run_dir, "provenance": PROVENANCE}
    with pytest.raises(OpenRouterError, match="429"):
        run_comparison(frame, evaluation, classifier, **kwargs)
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["completed_predictions"] == 1
    assert summary["metrics"] is None
    new_client = OpenRouterClient("key", transport=httpx.MockTransport(handler))
    result = run_comparison(
        frame, evaluation, LLMClassifier("fixed-model", client=new_client), **kwargs
    )
    assert len(requests) == 3
    assert result["scientific_result"] is True
    assert result["metrics"]["baseline"]["overall"]["n_samples"] == 2
    assert len(json.loads((run_dir / "api_calls.json").read_text())) == 3
    run_comparison(frame, evaluation, LLMClassifier("fixed-model", client=new_client), **kwargs)
    assert len(requests) == 3
    assert len(json.loads((run_dir / "api_calls.json").read_text())) == 3


def test_service_demo_retrieves_only_training_examples(frame):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from rag_hatespeech_ptbr.api import create_app

    embedder = DryRunEmbedder()
    app = create_app(
        {
            "frame": frame,
            "embedder": embedder,
            "index": build_index(frame, embedder, PROVENANCE),
            "classifier": LLMClassifier("", dry_run=True),
        }
    )
    with TestClient(app) as client:
        assert client.get("/health").json()["dry_run"] is True
        result = client.post("/classify", json={"comment": "seu idiota", "top_k": 1})
        assert result.status_code == 200
        assert result.json()["retrieved"][0]["id"] == 2
        assert client.post("/classify", json={"comment": " "}).status_code == 422
        assert client.post("/classify", json={"comment": "hi", "top_k": 3}).status_code == 422
