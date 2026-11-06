"""Offline regressions: child processes exit themselves; no signals are sent."""
import subprocess
import sys
import time

import pytest

from src.week1_gateway import AttemptDeadlineExceeded, Gateway
from src.week1_runner import clean_error
from src.week1_tasks import TaskStore, retryable_error


@pytest.mark.parametrize('block', ['main', 'abandoned'])
def test_deadlock_dumps_all_threads_and_exits(tmp_path, block):
    script = '''
import sys, threading
from pathlib import Path
from src.week1_runner import ProgressWatchdog
from src.week1_gateway import wall_clock_call, AttemptDeadlineExceeded
watchdog = ProgressWatchdog(Path(sys.argv[1]), .25, .1).start()
lock = threading.Lock()
lock.acquire()
def deadlocking_worker():
    lock.acquire()
if sys.argv[2] == 'abandoned':
    try:
        wall_clock_call(deadlocking_worker, .02)
    except AttemptDeadlineExceeded:
        pass
else:
    threading.Thread(target=deadlocking_worker, daemon=True).start()
lock.acquire()
'''
    process = subprocess.Popen([sys.executable, '-c', script, str(tmp_path), block])
    assert process.wait(timeout=15) == 124
    stacks = (tmp_path / 'thread-stacks.log').read_text()
    assert 'deadlocking_worker' in stacks
    assert 'Progress watchdog expired' in stacks
    assert 'Timeout' in stacks  # native periodic dump also fired
    assert 'watch' in stacks


def test_durable_progress_resets_timeout(tmp_path):
    script = '''
import sys, time
from pathlib import Path
from src.week1_runner import ProgressWatchdog
root = Path(sys.argv[1])
w = ProgressWatchdog(root, .3, 5).start()
for i in range(12):
    (root / 'runtime-status.json').write_text(str(i))
    time.sleep(.05)
w.close()
'''
    process = subprocess.Popen([sys.executable, '-c', script, str(tmp_path)])
    assert process.wait(timeout=15) == 0


@pytest.mark.parametrize('kind', ['build', 'qa'])
def test_wrapped_deadline_retries_and_requeues_with_bound(tmp_path, kind):
    try:
        try:
            raise AttemptDeadlineExceeded('expired')
        except AttemptDeadlineExceeded as error:
            raise RuntimeError('provider failed') from error
    except RuntimeError as error:
        record = clean_error(error)
    assert record['deadline_exceeded']
    assert retryable_error(record)
    path = tmp_path / 'tasks.sqlite3'
    store = TaskStore(path)
    item = {'id': 'q', 'dataset': 'locomo', 'group': 'conv-26', 'sessions': [{'id': 's', 'messages': [{'role': 'user', 'content': 'tea'}]}]}
    store.seed([item], ['ours'], 'protocol')
    build = store.build('ours', 'conv-26')
    if kind == 'qa':
        store.commit_build_unit(build['task_id'], 1, '/index/1')
        store.commit_build(build['task_id'], {'history_complete': True, 'history_sha256': build['history_sha256']})
    task = build if kind == 'build' else store.question('ours', 'q')
    for attempt in range(3):
        store.start(task['task_id'])
        # Legacy failures incorrectly carried retryable=False.
        store.fail(task['task_id'], 'ingestion' if kind == 'build' else 'answer', {**record, 'retryable': False})
        store.close()
        store = TaskStore(path)
        store.seed([item], ['ours'], 'protocol')
        queued = store.requeue(3)
        assert (task['task_id'] in queued) == (attempt < 2)
        if attempt < 2 and kind == 'build':
            assert store.question('ours', 'q')['state'] == 'pending'
    assert store.attempts(task['task_id']) == 3
    store.close()


def test_gateway_state_lock_remains_available_during_network_wait(tmp_path):
    import threading
    import httpx
    from test_week1_gateway import install, body, completion
    entered, release, done = threading.Event(), threading.Event(), threading.Event()

    def respond(request):
        entered.set()
        release.wait(2)
        return httpx.Response(200, json=completion())

    transport, _ = install(respond)
    with Gateway('private-key', tmp_path, transport=transport, attempt_deadline_seconds=1) as gateway:
        worker = threading.Thread(target=lambda: (gateway._request('/api/v1/chat/completions', body()), done.set()), daemon=True)
        worker.start()
        assert entered.wait(1)
        started = time.monotonic()
        assert gateway.summary()['reserved_usd'] > 0
        assert time.monotonic() - started < .2
        release.set()
        assert done.wait(2)
        worker.join(timeout=1)


def test_abandoned_upstream_pool_is_replaced_without_join(tmp_path, monkeypatch):
    import threading
    import httpx
    from test_week1_gateway import install, body, completion
    entered, release = threading.Event(), threading.Event()

    def blocked(request):
        entered.set()
        release.wait(5)
        return httpx.Response(200, json=completion())

    old_transport, _ = install(blocked)
    new_transport, _ = install(lambda request: httpx.Response(200, json=completion()))
    clients = [httpx.Client(transport=transport, headers={'Authorization': 'Bearer private-key'})
               for transport in (old_transport, new_transport)]
    replacements = iter(clients)
    monkeypatch.setattr(Gateway, '_new_client', lambda self: next(replacements))
    try:
        with Gateway('private-key', tmp_path, attempt_deadline_seconds=.1, retry_uncertain=True) as gateway:
            assert gateway._request('/api/v1/chat/completions', body())[0] == 200
            assert entered.is_set()
            assert gateway._client is clients[1]
            assert gateway.summary()['reserved_usd'] > 0
            assert gateway.summary()['spent_usd'] > 0
            assert not release.is_set()  # succeeded while old worker remained blocked
    finally:
        release.set()


def test_gateway_lock_wait_has_deadline(tmp_path):
    import threading
    gateway = Gateway('offline', tmp_path, attempt_deadline_seconds=.03)
    acquired, release, done = threading.Event(), threading.Event(), threading.Event()

    def own_lock():
        with gateway._lock:
            acquired.set()
            release.wait(2)
        done.set()

    worker = threading.Thread(target=own_lock, daemon=True)
    worker.start()
    assert acquired.wait(1)
    try:
        started = time.monotonic()
        with pytest.raises(AttemptDeadlineExceeded):
            gateway.summary()
        assert time.monotonic() - started < .5
    finally:
        release.set()
        assert done.wait(1)
        worker.join(timeout=1)


def test_resume_never_reuses_abandoned_worker_directory(tmp_path):
    from src.week1_runner import prepare_index_work, publish_index
    task = {'method': 'ours', 'group': 'conv-26'}
    abandoned = prepare_index_work(tmp_path, task, 's:0', None)
    (abandoned / 'state').write_text('unfinished')
    resumed = prepare_index_work(tmp_path, task, 's:0', None)
    assert resumed != abandoned
    (resumed / 'state').write_text('committed')
    version = publish_index(resumed)
    (abandoned / 'state').write_text('late worker write')
    assert (version / 'state').read_text() == 'committed'
