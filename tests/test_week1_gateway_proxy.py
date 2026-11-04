"""External gateway transport uses the user's existing proxy, entirely offline."""

import httpx
import pytest

from src import week1_gateway
from test_week1_gateway import body, completion, install


def mock_external_transport(monkeypatch):
    transport, requests = install(lambda request: httpx.Response(200, json=completion()))
    created = []

    def factory(**kwargs):
        created.append(kwargs)
        return transport

    monkeypatch.setattr(week1_gateway.httpx, "HTTPTransport", factory)
    return created, requests


def test_existing_https_system_proxy_configures_external_transport_without_retries(monkeypatch, tmp_path):
    proxy = "http://127.0.0.1:7890"
    discoveries, bypass_hosts = [], []

    def discover():
        discoveries.append(True)
        return {"http": "http://127.0.0.1:7891", "https": proxy}

    def bypass(host):
        bypass_hosts.append(host)
        return False

    monkeypatch.setattr(week1_gateway, "getproxies", discover)
    monkeypatch.setattr(week1_gateway, "proxy_bypass", bypass)
    created, requests = mock_external_transport(monkeypatch)
    with week1_gateway.Gateway("private-key", tmp_path) as gateway:
        status, result = gateway._request("/api/v1/chat/completions", body())
        assert status == 200
        assert result["choices"][0]["message"]["content"] == "Done"
        assert gateway.spent_usd == 0.0001
    assert discoveries == [True]
    assert bypass_hosts == ["openrouter.ai"]
    assert len(created) == 1
    assert created[0]["proxy"] == proxy  # HTTPS origin selects HTTPS, not HTTP proxy.
    assert created[0]["retries"] == 0
    assert str(requests[0].url) == week1_gateway.MODELS_URL
    assert str(requests[1].url) == week1_gateway.ORIGIN + "/api/v1/chat/completions"


def test_origin_proxy_bypass_respects_existing_exclusion(monkeypatch, tmp_path):
    bypass_hosts = []
    monkeypatch.setattr(week1_gateway, "getproxies", lambda: {"https": "http://127.0.0.1:7890"})

    def bypass(host):
        bypass_hosts.append(host)
        return host == "openrouter.ai"

    monkeypatch.setattr(week1_gateway, "proxy_bypass", bypass)
    created, requests = mock_external_transport(monkeypatch)
    with week1_gateway.Gateway("private-key", tmp_path) as gateway:
        gateway._request("/api/v1/chat/completions", body())
    assert bypass_hosts == ["openrouter.ai"]
    assert len(created) == 1
    assert created[0]["proxy"] is None
    assert created[0]["retries"] == 0
    assert len(requests) == 2  # Pricing and completion both remain mocked.


def test_explicit_transport_never_discovers_proxy_or_constructs_external_transport(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        pytest.fail("An explicitly supplied transport must not discover or construct another transport")

    monkeypatch.setattr(week1_gateway, "getproxies", forbidden)
    monkeypatch.setattr(week1_gateway, "proxy_bypass", forbidden)
    monkeypatch.setattr(week1_gateway.httpx, "HTTPTransport", forbidden)
    transport, requests = install(lambda request: httpx.Response(200, json=completion()))
    with week1_gateway.Gateway("private-key", tmp_path, transport=transport) as gateway:
        assert gateway._client._transport is transport
        status, _ = gateway._request("/api/v1/chat/completions", body())
        assert status == 200
    assert len(requests) == 2
