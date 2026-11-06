import json

import pytest

from src.week1_tasks import TaskStore, history_digest, ingestion_units


def example(question='q1', count=21):
    return {'id': question, 'dataset': 'locomo', 'group': 'history', 'source_tokens': 100,
            'sessions': [{'id': 's1', 'messages': [{'role': 'user', 'content': f'message {i}'} for i in range(count)]}]}


def complete_build(store, method='ours'):
    task = store.build(method, 'history')
    for number in range(1, task['total_units'] + 1):
        store.commit_build_unit(task['task_id'], number, f'/index/{number}')
    store.commit_build(task['task_id'], {'history_complete': True, 'history_sha256': task['history_sha256']})
    return store.question(method, 'q1')['task_id']


def test_fixed_add_units_retain_every_message_in_order():
    item = example(count=43)
    units = ingestion_units(item, 'ours')
    assert [len(unit['messages']) for unit in units] == [20, 20, 3]
    assert [message for unit in units for message in unit['messages']] == item['sessions'][0]['messages']
    assert ingestion_units(item, 'mem0') == units
    assert ingestion_units(item, 'memorybank') == [{'id': 'full-history', 'example': item}]


def test_no_stage_or_score_without_full_history_and_every_add_unit(tmp_path):
    store = TaskStore(tmp_path / 'tasks.sqlite3')
    store.seed([example()], ['ours'], 'protocol')
    build = store.build('ours', 'history')
    qa = store.question('ours', 'q1')['task_id']
    with pytest.raises(ValueError, match='incomplete'):
        store.commit_stage(qa, 'answer', {'answer': 'invalid partial history'})
    store.commit_build_unit(build['task_id'], 1, '/index/partial')
    with pytest.raises(ValueError, match='full frozen history'):
        store.commit_build(build['task_id'], {'history_complete': True, 'history_sha256': history_digest(example())})
    with pytest.raises(ValueError, match='next frozen unit'):
        store.commit_build_unit(build['task_id'], 1, '/index/repeated')
    store.close()


def test_build_failure_blocks_only_dependent_method_history_and_is_not_retried(tmp_path):
    store = TaskStore(tmp_path / 'tasks.sqlite3')
    items = [example(), example('q2'), {**example('other'), 'group': 'other-history'}]
    store.seed(items, ['rag', 'ours'], 'protocol')
    task = store.build('ours', 'history')
    store.fail(task['task_id'], 'ingestion', {'type': 'ProviderError', 'raw_response': 'invalid'})
    assert store.question('ours', 'q1')['state'] == store.question('ours', 'q2')['state'] == 'blocked'
    assert store.question('rag', 'q1')['state'] == store.question('ours', 'other')['state'] == 'pending'
    store.close()
    store = TaskStore(tmp_path / 'tasks.sqlite3')
    store.seed(items, ['rag', 'ours'], 'protocol')
    assert store.build('ours', 'history')['state'] == 'failed'
    assert len(store.failures()) == 1
    store.close()


def test_atomic_retrieval_commit_moves_native_state_pointer_once(tmp_path):
    store = TaskStore(tmp_path / 'tasks.sqlite3')
    store.seed([example(count=1)], ['memorybank'], 'protocol')
    qa = complete_build(store, 'memorybank')
    old = store.build('memorybank', 'history')['index_path']
    assert old == '/index/1'
    payload = {'data': [{'id': 'note', 'content': 'recalled'}]}
    store.commit_stage(qa, 'retrieval', payload, '/index/reinforced')
    assert store.stage(qa, 'retrieval') == payload
    assert store.build('memorybank', 'history')['index_path'] == '/index/reinforced'
    store.commit_stage(qa, 'retrieval', payload, '/index/should-not-overwrite')
    assert store.build('memorybank', 'history')['index_path'] == '/index/reinforced'
    store.fail(qa, 'answer', {'type': 'transport'})
    assert store.stage(qa, 'retrieval') == payload
    assert store.build('memorybank', 'history')['state'] == 'success'
    store.close()


def test_resume_preserves_stages_and_rejects_history_protocol_and_selection_changes(tmp_path):
    path = tmp_path / 'tasks.sqlite3'
    store = TaskStore(path)
    store.seed([example()], ['ours'], 'protocol')
    qa = complete_build(store)
    store.commit_stage(qa, 'answer', {'answer': 'saved', 'finish_reason': 'stop'})
    store.start(qa)
    store.close()
    store = TaskStore(path)
    store.seed([example()], ['ours'], 'protocol')
    assert store.question('ours', 'q1')['state'] == 'pending'
    assert store.stage(qa, 'answer') == {'answer': 'saved', 'finish_reason': 'stop'}
    with pytest.raises(ValueError, match='history or protocol'):
        store.seed([example()], ['ours'], 'different')
    changed = example()
    changed['sessions'][0]['messages'][-1]['content'] = 'changed final source'
    with pytest.raises(ValueError, match='history or protocol'):
        store.seed([changed], ['ours'], 'protocol')
    with pytest.raises(ValueError, match='selection changed'):
        store.seed([], ['ours'], 'protocol')
    store.close()


def test_lme_judge_and_answer_must_exist_before_score_and_scores_are_immutable(tmp_path):
    store = TaskStore(tmp_path / 'tasks.sqlite3')
    item = {**example(), 'dataset': 'longmemeval'}
    store.seed([item], ['ours'], 'protocol')
    qa = complete_build(store)
    with pytest.raises(ValueError, match='answer and official judge'):
        store.commit_stage(qa, 'score', {'score': 1})
    store.commit_stage(qa, 'answer', {'answer': 'saved', 'finish_reason': 'stop'})
    with pytest.raises(ValueError, match='answer and official judge'):
        store.commit_stage(qa, 'score', {'score': 1})
    store.commit_stage(qa, 'judge', {'judge_response': 'yes', 'finish_reason': 'stop'})
    store.commit_stage(qa, 'score', {'score': 1})
    store.commit_stage(qa, 'score', {'score': 1})
    assert store.predictions() == [{'score': 1}]
    with pytest.raises(ValueError, match='cannot be overwritten'):
        store.commit_stage(qa, 'score', {'score': 0})
    store.close()


def test_requeue_retries_failed_tasks_and_unblocks_dependents_up_to_the_attempt_bound(tmp_path):
    store = TaskStore(tmp_path / 'tasks.sqlite3')
    store.seed([example()], ['ours'], 'protocol')
    build = store.build('ours', 'history')
    store.commit_build_unit(build['task_id'], 1, '/index/1')
    store.fail(build['task_id'], 'ingestion', {'message': 'HTTP 402'})
    qa = store.question('ours', 'q1')
    assert qa['state'] == 'blocked'
    assert sorted(store.requeue(max_attempts=2)) == sorted([build['task_id'], qa['task_id']])
    resumed = store.get(build['task_id'])
    assert resumed['state'] == 'pending' and resumed['completed_units'] == 1
    assert store.get(qa['task_id'])['state'] == 'pending'
    store.fail(build['task_id'], 'ingestion', {'message': 'HTTP 402'})
    assert store.requeue(max_attempts=2) == []
    assert store.get(build['task_id'])['state'] == 'failed'
    assert store.get(qa['task_id'])['state'] == 'blocked'
    store.close()


def test_requeue_leaves_budget_stopped_tasks_alone(tmp_path):
    store = TaskStore(tmp_path / 'tasks.sqlite3')
    store.seed([example()], ['ours'], 'protocol')
    build = store.build('ours', 'history')
    store.fail(build['task_id'], 'ingestion', {'message': 'budget'}, budget_stop='week1_budget_exhausted')
    assert store.requeue(max_attempts=5) == []
    assert store.get(build['task_id'])['state'] == 'budget-stopped'
    store.close()


def test_audited_local_400_requeues_once_with_total_attempt_cap(tmp_path):
    from src.week1_tasks import retryable_error
    folder = tmp_path / '20261105T122803255750Z'
    folder.mkdir()
    store = TaskStore(folder / 'tasks.sqlite3')
    store.seed([{**example(), 'group': 'conv-26'}], ['ours'], 'protocol')
    target = store.build('ours', 'conv-26')['task_id']
    error = {'type': 'ProviderError', 'message': 'Relation decision failed', 'raw_response': None,
             'retryable': False, 'diagnostics': {'exception_type': 'BadRequestError', 'model': 'typesafe/jev-1.13',
                                                'phase': 'decision_jev', 'status_code': 400}}
    store.fail(target, 'ingestion', error)
    store.fail(target, 'ingestion', error)
    assert not retryable_error(error)
    assert target not in store.requeue(2, dry_run=True)
    assert target in store.requeue(4, dry_run=True)
    assert store.get(target)['state'] == 'failed'
    assert target in store.requeue(4)
    store.fail(target, 'ingestion', error)
    assert target not in store.requeue(4)
    store.close()
