"""Offline deadlines, conservative retry accounting, and legacy task recovery."""
import threading
import time

import httpx
import pytest

from src.week1_gateway import Gateway, MAX_CALL_ATTEMPTS
from src.week1_tasks import TaskStore
from test_week1_gateway import install, completion, body, events


@pytest.mark.parametrize('kind', ['hang', 'ReadTimeout', '503'])
@pytest.mark.parametrize('endpoint', ['chat', 'embedding', 'relation'])
def test_attempt_deadline_and_transient_retry(tmp_path, kind, endpoint):
    release = threading.Event()
    attempts = []

    def respond(request):
        attempts.append(request)
        if len(attempts) == 1:
            if kind == 'hang':
                release.wait(10)
            elif kind == 'ReadTimeout':
                raise httpx.ReadTimeout('offline timeout')
            else:
                return httpx.Response(503, json={'error': {'message': 'offline'}})
        return httpx.Response(200, json=completion())

    transport, _ = install(respond)
    payload = body()
    path = '/api/v1/chat/completions'
    if endpoint == 'embedding':
        path, payload = '/api/v1/embeddings', {'model': 'openai/text-embedding-3-small', 'input': 'hello'}
    elif endpoint == 'relation':
        path, payload = '/api/alpha/decisions', {'model': 'typesafe/jev-1.13', 'state': {}, 'questions': {'q': 'hello'}}
    try:
        with Gateway('private-key', tmp_path, transport=transport, attempt_deadline_seconds=.03, retry_uncertain=True) as gateway:
            started = time.monotonic()
            assert gateway._request(path, payload)[0] == 200
            assert time.monotonic() - started < 2
            assert len(attempts) == 2
            ledger = events(gateway)
            assert len([e for e in ledger if e['event'] == 'reservation']) == 2
            assert gateway.summary()['reserved_usd'] > 0
            assert gateway.summary()['committed_upper_usd'] <= 20
            before = gateway.summary()
            release.set()
            assert gateway.summary() == before
    finally:
        release.set()


def test_three_legacy_builds_and_blocked_qa_requeue(tmp_path):
    store = TaskStore(tmp_path / 'tasks.sqlite3')
    groups = ['conv-26', 'conv-30', 'conv-41']
    store.seed([{'id': group, 'group': group, 'dataset': 'locomo', 'sessions': []} for group in groups], ['ours'], 'p')
    for group, message in zip(groups, ['Extraction provider request failed: ReadTimeout', 'Relation decision failed: ReadTimeout', 'week1_call_uncertain']):
        store.fail(store.build('ours', group)['task_id'], 'ingestion', {'type': 'ProviderError', 'message': message, 'retryable': False})
    assert len(store.requeue(4)) == 6
    assert all(store.build('ours', group)['state'] == 'pending' for group in groups)
    store.close()


def test_persistent_timeouts_are_bounded_and_cannot_exceed_budget(tmp_path):
    attempts = []

    def respond(request):
        attempts.append(request)
        raise httpx.ReadTimeout('offline')

    transport, _ = install(respond)
    with Gateway('private-key', tmp_path, transport=transport, retry_uncertain=True) as gateway:
        from src.week1_gateway import GatewayError
        with pytest.raises(GatewayError):
            gateway._request('/api/v1/chat/completions', body())
        assert len(attempts) == MAX_CALL_ATTEMPTS
        assert len([e for e in events(gateway) if e['event'] == 'reservation']) == MAX_CALL_ATTEMPTS
        assert gateway.summary()['reserved_usd'] > 0
        assert gateway.summary()['committed_upper_usd'] <= 20


def test_killed_inflight_checkpoint_resumes_with_old_reservation(tmp_path):
    transport, _ = install(lambda request: httpx.Response(200, json=completion()))
    context = {'run_id': 'r', 'task_id': 't', 'unit_id': 'u', 'stage': 'ingestion', 'protocol_sha256': 'p'}
    with Gateway('private-key', tmp_path, transport=transport, retry_uncertain=True) as gateway:
        gateway.set_context(context)
        path, payload, upper, estimated = gateway._prepare('/api/v1/chat/completions', body())
        identity, _ = gateway._checkpoint_call(path, payload)
        gateway._reserve_checkpoint(identity, {'call_id': 'killed', 'context': context,
            'cost_quoted_upper_usd': float(upper)})
    with Gateway('private-key', tmp_path, transport=transport, retry_uncertain=True) as gateway:
        gateway.set_context(context)
        assert gateway._request('/api/v1/chat/completions', body())[0] == 200
        assert gateway.summary()['reserved_usd'] == float(upper)
        assert gateway.summary()['spent_usd'] > 0
        assert gateway._checkpoints.execute('SELECT COUNT(*) FROM call_attempts').fetchone()[0] == 1


def test_local_clients_wait_for_the_gateway_retry_window(monkeypatch):
    from src.week1_baseline_worker import GatewayNetwork
    from src.week1_runner import GatewayTransport
    monkeypatch.setenv('WEEK1_LOCAL_REQUEST_TIMEOUT', '3300')
    network = GatewayNetwork('http://127.0.0.1:12345', httpx.MockTransport(lambda request: httpx.Response(200)))
    with network.client() as client:
        assert client.timeout.read == 3300
    network.close()
    routed = GatewayTransport('http://127.0.0.1:12345', 3300)
    routed.transport.close()
    seen = []
    routed.transport = httpx.MockTransport(lambda request: seen.append(request) or httpx.Response(200))
    routed.handle_request(httpx.Request('POST', 'https://api.openai.com/v1/embeddings',
                                      extensions={'timeout': {'read': 120}}))
    assert seen[0].extensions['timeout']['read'] == 3300
    routed.close()
