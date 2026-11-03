import pytest


@pytest.fixture(autouse=True)
def retry_waits(monkeypatch):
    """Retry backoff never really sleeps in offline tests; the requested waits are recorded."""
    from src import week1_gateway, week1_runner
    waits = []
    monkeypatch.setattr(week1_gateway.Gateway, "_sleep", lambda self, seconds: waits.append(seconds))
    monkeypatch.setattr(week1_runner, "pause", lambda seconds: waits.append(("pause", seconds)))
    return waits
