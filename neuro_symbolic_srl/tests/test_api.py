"""Integration tests for the FastAPI ``/api/v1/parse_frame`` endpoint.

Exercises the real lifespan-loaded service (ONNX session, IndoWordNet,
trained CRF/WSD weights) via Starlette's TestClient, including the three
verification sentences end-to-end through the actual HTTP interface rather
than the lower-level ``SRLPipeline`` (see ``test_pipeline.py``).
"""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import ONNX_MODEL_PATH, app

_HAS_ONNX_MODEL = Path(ONNX_MODEL_PATH).exists()

pytestmark = pytest.mark.skipif(
    not _HAS_ONNX_MODEL,
    reason="requires an exported ONNX model at checkpoints/srl_net_int8.onnx",
)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def _roles(response_json: dict) -> dict:
    return {arg["role"]: arg["text"] for arg in response_json["arguments"]}


class TestHealthCheck:
    def test_health_endpoint_reports_ok_once_started(self, client: TestClient) -> None:
        response = client.get("/healthz")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestParseFrameValidation:
    def test_empty_text_is_rejected(self, client: TestClient) -> None:
        response = client.post("/api/v1/parse_frame", json={"text": ""})
        assert response.status_code == 422

    def test_out_of_range_target_predicate_is_rejected(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/parse_frame", json={"text": "राम गया", "target_predicate": 5}
        )
        assert response.status_code == 422


class TestParseFrameVerificationSentences:
    """The same three sentences as test_pipeline.py, through the real HTTP API."""

    def test_chaabi_is_not_arg0(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/parse_frame", json={"text": "चाबी ने ताला खोला"}
        )
        assert response.status_code == 200
        roles = _roles(response.json())
        assert "ARG0" not in roles
        assert any(role.startswith("ARGM-INS") or role == "ARG2" for role in roles)

    def test_predicate_sense_is_disambiguated(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/parse_frame", json={"text": "चाबी ने ताला खोला"}
        )
        predicate = response.json()["predicate"]
        assert predicate["lemma"] == "खोलना"
        assert predicate["synset_id"] is not None

    def test_scrambled_osv_gives_correct_core_roles(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/parse_frame", json={"text": "सेब को बच्चे ने खाया"}
        )
        roles = _roles(response.json())
        assert roles.get("ARG1") == "सेब को"
        assert roles.get("ARG0") == "बच्चे ने"

    def test_instrument_attaches_to_verb_not_noun(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/parse_frame",
            json={"text": "पुलिस ने दूरबीन से चोर को देखा"},
        )
        roles = _roles(response.json())
        assert roles.get("ARGM-INS") == "दूरबीन से"
        assert roles.get("ARG0") == "पुलिस ने"
        assert roles.get("ARG1") == "चोर को"

    def test_karaka_field_is_populated_for_returned_arguments(
        self, client: TestClient
    ) -> None:
        response = client.post(
            "/api/v1/parse_frame",
            json={"text": "पुलिस ने दूरबीन से चोर को देखा"},
        )
        for argument in response.json()["arguments"]:
            assert argument["karaka"] is not None
