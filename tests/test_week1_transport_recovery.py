import ast
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import openai
import pytest

from src import week1_baseline_worker as workers
from src.week1_gateway import Gateway


def test_local_baseline_timeout_exceeds_upstream_without_retry_or_body_change(monkeypatch, tmp_path):
    seen = []

    def respond(request):
        seen.append(request)
        return httpx.Response(200, json={
            "id": "offline", "object": "chat.completion", "created": 0,
            "model": workers.MODEL,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "Done"}}],
        })

    network = workers.GatewayNetwork("http://127.0.0.1:8123", httpx.MockTransport(respond))
    original = openai.OpenAI
    monkeypatch.setattr(openai, "OpenAI", original)
    client = workers.patch_openai(network)(max_retries=9, timeout=1)
    monkeypatch.setattr(Gateway, "_pricing", lambda self: None)
    with Gateway("dummy-key", tmp_path, transport=httpx.MockTransport(respond)) as gateway:
        assert client.timeout == 240
        assert client.timeout == gateway._client.timeout.read + 60
        with network.client() as local_client:
            assert local_client.timeout.read == client.timeout
    assert client.max_retries == 0
    payload = {"model": workers.MODEL, "temperature": 0, "max_tokens": 4096,
               "messages": [{"role": "user", "content": "unchanged author prompt"}]}
    assert client.chat.completions.create(**payload).choices[0].message.content == "Done"
    assert len(seen) == 1
    assert json.loads(seen[0].content) == payload
    assert set(seen[0].extensions["timeout"].values()) == {240}
    client.close()
    network.close()


@pytest.mark.parametrize("changes, expected", [
    ({"raw_response": None, "budget_stop": "week1_cost_unknown"}, False),
    ({"raw_response": None}, False),
    ({"raw_response": "invalid content"}, True),
    ({"raw_response": "invalid content", "budget_stop": "week1_cost_unknown"}, False),
    ({"raw_response": "invalid content", "stage": "close"}, False),
    ({"raw_response": "invalid content", "method": "amem"}, False),
    ({"raw_response": "invalid content", "dataset": "locomo"}, False),
    ({"raw_response": "invalid content", "id": "other-question"}, False),
])
def test_initial_driver_skips_only_actual_invalid_ours_content(changes, expected):
    driver = (Path(__file__).resolve().parents[1] / "artifacts/week1/20261014T085344Z"
              / "initial-longmemeval-validation-2026-10-15.py")
    if not driver.is_file():
        pytest.skip("The frozen local pilot execution driver is not installed")
    tree = ast.parse(driver.read_text())
    predicate = next(node.value for node in ast.walk(tree)
                     if isinstance(node, ast.Assign)
                     and any(isinstance(target, ast.Name) and target.id == "previous_ours_failure"
                             for target in node.targets))
    row = {"method": "ours", "dataset": "longmemeval", "id": "95bcc1c8",
           "budget_stop": None, "raw_response": None, **changes}
    context = {"w": SimpleNamespace(records=lambda path: [row]), "directory": driver.parent,
               "example": {"id": "95bcc1c8"}}
    assert eval(compile(ast.Expression(predicate), str(driver), "eval"), context) is expected
