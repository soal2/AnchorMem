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


@pytest.mark.parametrize('stage', ['extraction', 'embedding', 'decision'])
@pytest.mark.parametrize('where', ['headers', 'body'])
def test_native_ingestion_socket_deadline_requeues_without_retaining_lock(tmp_path, monkeypatch, stage, where):
    import socket
    from pathlib import Path
    from src import week1_runner as runner
    from src.week1_tasks import ingestion_units
    from test_vikingjev_engine import FactModel

    left, right = socket.socketpair()
    entered = threading.Event()
    finished = threading.Event()

    class BlockedStream(httpx.SyncByteStream):
        def __iter__(self):
            entered.set()
            try:
                yield left.recv(1)  # Real C-level socket/poll; no socket timeout.
            finally:
                finished.set()

    def handle(request):
        if where == 'headers':
            entered.set()
            try:
                left.recv(1)
            finally:
                finished.set()
            return httpx.Response(200, json={})
        return httpx.Response(200, stream=BlockedStream())

    monkeypatch.setenv('OPENAI_API_KEY', 'offline-test')
    config, config_path = runner.load_config(Path('configs/service/vikingjev.yaml'), runner.ServiceConfig)
    example = {'id': 'q', 'dataset': 'locomo', 'group': 'conv-26',
               'sessions': [{'id': 'session_9', 'messages': [{'role': 'user', 'content': 'I like tea.'}]}]}
    unit = ingestion_units(example, 'ours')[0]
    store = TaskStore(tmp_path / 'tasks.sqlite3')
    store.seed([example], ['ours'], 'p')
    memory = None
    try:
        with runner.routed_clients('http://127.0.0.1:12345', 1, .05):
            memory = runner.NativeMemory(tmp_path, tmp_path / 'index', config, config_path)
            if stage == 'decision':
                from test_vikingjev_engine import Embedder
                memory.engine.llm = FactModel()
                memory.engine.embedder = Embedder()
                provider = memory.engine.decision_model
            elif stage == 'embedding':
                memory.engine.llm = FactModel()
                provider = memory.engine.embedder
            else:
                provider = memory.engine.llm
            transport = provider.client._client._transport
            transport.transport.close()
            transport.transport = httpx.MockTransport(handle)
            started = time.monotonic()
            with pytest.raises(runner.providers.ProviderError) as caught:
                memory.ingest_unit(example, unit)
            assert entered.is_set()
            assert time.monotonic() - started < 1
            assert not memory.engine._lock('paper:conv-26').locked()
            failure = runner.clean_error(caught.value)
            assert runner.retryable_error(failure)
            build = store.build('ours', 'conv-26')
            store.fail(build['task_id'], 'ingestion', failure)
            assert store.requeue(4)
            assert store.build('ours', 'conv-26')['completed_units'] == 0
            assert store.build('ours', 'conv-26')['state'] == 'pending'
            # Retry the same unit on the same engine: the old socket worker owns
            # neither its user lock nor its database transaction.
            memory.engine.llm = FactModel()
            from test_vikingjev_engine import Embedder, Relations, Selector
            memory.engine.embedder = Embedder()
            memory.engine.decision_model = Relations()
            memory.engine.directory_selector = Selector()
            memory.ingest_unit(example, unit)
    finally:
        right.close()
        if entered.is_set():
            assert finished.wait(1)
        left.close()
        store.close()


@pytest.mark.parametrize('method', ['mem0', 'memorybank'])
@pytest.mark.parametrize('endpoint', ['chat/completions', 'embeddings'])
def test_baseline_ingestion_blocked_socket_requeues(tmp_path, monkeypatch, method, endpoint):
    import socket
    from types import SimpleNamespace
    from src.week1_baseline_worker import GatewayNetwork, BaselineWorker
    from src.week1_memorybank import MemoryBankWorker
    from src.week1_runner import clean_error

    monkeypatch.setenv('WEEK1_EXCHANGE_DEADLINE', '.05')
    left, right = socket.socketpair()
    finished = threading.Event()

    class Stream(httpx.SyncByteStream):
        def __iter__(self):
            try:
                yield left.recv(1)
            finally:
                finished.set()

    network = GatewayNetwork('http://127.0.0.1:12345', httpx.MockTransport(
        lambda request: httpx.Response(200, stream=Stream())))
    client = network.client()

    def blocked(*args, **kwargs):
        client.post('https://api.openai.com/v1/' + endpoint, json={})

    example = {'id': 'q', 'dataset': 'locomo', 'group': 'conv-26',
               'sessions': [{'id': 'session_9', 'messages': [{'role': 'user', 'content': '[Session date: 2024-01-01]\nTea.'}]}]}
    store = TaskStore(tmp_path / 'tasks.sqlite3')
    store.seed([example], [method], 'p')
    if method == 'mem0':
        worker = BaselineWorker.__new__(BaselineWorker)
        worker.method, worker.network, worker.completed_adds = method, network, set()
        worker.system = SimpleNamespace(add=blocked)
        operation = lambda: worker.add({'messages': example['sessions'][0]['messages'], 'request_id': 'session_9:0'})
    else:
        worker = MemoryBankWorker.__new__(MemoryBankWorker)
        worker.clock, worker.network = None, network
        worker.memory_file = tmp_path / 'memory.json'
        worker.path = tmp_path
        worker.summary = SimpleNamespace(summarize_memory=blocked if endpoint == 'chat/completions' else lambda *a, **k: None)
        worker.retrieval = SimpleNamespace(init_memory_vector_store=blocked)
        operation = lambda: worker.ingest({'sessions': example['sessions'], 'group': 'conv-26'})
    try:
        started = time.monotonic()
        with pytest.raises(httpx.ReadTimeout) as caught:
            operation()
        assert time.monotonic() - started < 1
        assert 'ReadTimeout' in network.failed
        build = store.build(method, 'conv-26')
        store.fail(build['task_id'], 'ingestion', clean_error(caught.value))
        assert store.requeue(4)
        assert store.build(method, 'conv-26')['state'] == 'pending'
        assert store.build(method, 'conv-26')['completed_units'] == 0
        if method == 'mem0':
            assert not worker.completed_adds
    finally:
        right.close()
        assert finished.wait(1)
        left.close()
        client.close()
        network.close()
        store.close()


def test_local_deadline_bounds_gateway_retries_and_releases_ledger_lock(tmp_path):
    from src.week1_gateway import GatewayError
    release = threading.Event()
    transport, _ = install(lambda request: release.wait(10) or httpx.Response(200, json=completion()))
    try:
        with Gateway('private-key', tmp_path, transport=transport, attempt_deadline_seconds=10,
                     retry_uncertain=True) as gateway:
            started = time.monotonic()
            with pytest.raises(GatewayError):
                gateway._request('/api/v1/chat/completions', body(), deadline=started + .05)
            assert time.monotonic() - started < 1
            assert gateway.summary()['reserved_usd'] > 0
            assert len([e for e in events(gateway) if e['event'] == 'reservation']) == 1
            acquired = threading.Event()
            def inspect():
                gateway.summary()
                acquired.set()
            thread = threading.Thread(target=inspect, daemon=True)
            thread.start()
            assert acquired.wait(1)
            thread.join()
    finally:
        release.set()
