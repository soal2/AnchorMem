"""Independent synthetic directory-routing latency experiment; no benchmark reruns."""
import hashlib
import json
import os
from pathlib import Path
import sqlite3
from statistics import median
import tempfile
from time import perf_counter
from types import SimpleNamespace

import numpy as np

from src.config import ServiceConfig, load_config
from src.contracts import SearchRequest
from src.directory_selector import JevDirectorySelector
from src.engine import MemoryEngine

OUT = Path('artifacts/vikingjev/directory-scaling')
SIZES = [10, 50, 100, 300]
PER_DIRECTORY = 100
DIMENSION = 1024


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def make_db(path, size, vectors_per_directory=PER_DIRECTORY):
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    db.executescript('''
        CREATE TABLE memory_directories(id TEXT PRIMARY KEY,path TEXT,abstract TEXT,overview TEXT);
        CREATE TABLE chunks(id TEXT PRIMARY KEY,message_id TEXT);
        CREATE TABLE message_sources(message_id TEXT PRIMARY KEY,directory_id TEXT);
        CREATE INDEX source_directory ON message_sources(directory_id);
        CREATE INDEX chunk_message ON chunks(message_id);
        CREATE TABLE embeddings(owner_type TEXT,owner_id TEXT,vector BLOB,PRIMARY KEY(owner_type,owner_id));
    ''')
    rng = np.random.default_rng(42)
    topics = ['gardening', 'cooking', 'running', 'photography', 'music', 'travel']
    for i in range(size):
        identity = f'd{i:05d}'
        topic = 'education' if i < 4 else topics[i % len(topics)]
        fact = (f'I graduated with a degree in Business Administration. Record {i} documents my university education.'
                if i < 4 else f'My {topic} hobby includes a weekly practice session. Record {i} documents supplies, preferences and plans.')
        overview = f'Session: {identity}. Topic: {topic}. {fact} This directory contains personal conversation history, dated notes and supporting details.'
        db.execute('INSERT INTO memory_directories VALUES (?,?,?,?)', (identity, f'memories/{identity}', f'Personal notes about {topic}.', overview))
        vectors = rng.normal(size=(vectors_per_directory, DIMENSION)).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        rows = [(f'{identity}-v{j:03d}', vectors[j].tobytes()) for j in range(vectors_per_directory)]
        db.executemany('INSERT INTO chunks VALUES (?,?)', [(key,key) for key,_ in rows])
        db.executemany('INSERT INTO message_sources VALUES (?,?)', [(key,identity) for key,_ in rows])
        db.executemany("INSERT INTO embeddings VALUES ('chunk',?,?)", rows)
    db.commit()
    return db


def flat_directory_selection(engine, db, request, ids):
    decisions = []
    for offset in range(0, len(ids), 10):
        batch = [dict(db.execute("SELECT id,path,abstract,overview FROM memory_directories WHERE id=?", (key,)).fetchone())
                 for key in ids[offset:offset + 10]]
        decisions.extend(engine.directory_selector.select(request.query, request.options, batch))
    decisions.sort(key=lambda d: (-d["probabilities"]["relevant"], -d["probabilities"]["uncertain"], d["id"]))
    return [d["id"] for d in decisions[:4]]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'started').open('x').close()
    (OUT/'runner.py').write_bytes(Path(__file__).read_bytes())
    for name in ['engine.py','directory_selector.py']:
        (OUT/name).write_bytes((Path('src')/name).read_bytes())
    config, _ = load_config(Path('configs/service/vikingjev.yaml'), ServiceConfig)
    context = {}
    events = []
    def telemetry(event):
        item = {**context, **event}
        events.append(item)
        with (OUT/'model-calls.jsonl').open('a') as f:
            f.write(json.dumps(item, ensure_ascii=False)+'\n')
    engine = object.__new__(MemoryEngine)
    engine.config = SimpleNamespace(embedding=SimpleNamespace(dimensions=DIMENSION))
    engine.directory_selector = JevDirectorySelector(config.directory_selector, telemetry)
    request = SearchRequest(query='What degree did I graduate with?', user_id='synthetic', top_k=20)
    query = np.random.default_rng(7).normal(size=DIMENSION).astype(np.float32)
    query /= np.linalg.norm(query)
    results = []
    # Same batching, validation, probability ranking and SQL vector code as the application.
    with tempfile.TemporaryDirectory(prefix='jev-directory-scaling-') as folder:
        dbs = {size: make_db(Path(folder)/f'{size}.sqlite3', size) for size in SIZES}
        write_json(OUT/'manifest.json', {'sizes':SIZES,'vectors_per_directory':PER_DIRECTORY,
          'dimensions':DIMENSION,'repetitions':3,'batch_size':10,'selected_directories':4,
          'seed':42,'query':request.query,'model':config.directory_selector.name,
          'retries':config.directory_selector.retries,'scope':'Warm local SQLite chunk-vector route plus live Jev; synthetic short summaries and random vectors; no embedding, BM25, fact/observation, RRF, source hydration, answer or judge.',
          'source_hashes':{name:hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in ['engine.py','directory_selector.py']},
          'thread_env':{key:os.environ.get(key) for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']}})
        for trial, order in enumerate([SIZES, list(reversed(SIZES)), [50,300,10,100]],1):
            for size in order:
                db = dbs[size]
                ids = [r[0] for r in db.execute('SELECT id FROM memory_directories ORDER BY id')]
                context.update(size=size,trial=trial)
                start_event = len(events)
                started = perf_counter()
                try:
                    chosen = flat_directory_selection(engine, db, request, ids)
                except Exception as error:
                    record={'size':size,'trial':trial,'error_type':type(error).__name__,'routing_seconds':perf_counter()-started}
                else:
                    routing = perf_counter()-started
                    timings = {'global':[],'scoped':[]}
                    for scope in [None, chosen]:
                        engine._vector_rank(db, query, 'chunk', 50, scope)
                    for iteration in range(7):
                        for mode in (['global','scoped'] if iteration%2==0 else ['scoped','global']):
                            started = perf_counter()
                            ranked = engine._vector_rank(db, query, 'chunk', 50, None if mode=='global' else chosen)
                            timings[mode].append(perf_counter()-started)
                            assert len(ranked)==50
                    record={'size':size,'trial':trial,'selected':chosen,'known_relevant_selected':len(set(chosen)&set(ids[:4])),
                      'routing_seconds':routing,'local_seconds':timings,
                      'without_jev_seconds':median(timings['global']),
                      'with_jev_seconds':routing+median(timings['scoped']),
                      'global_vectors':size*PER_DIRECTORY,'scoped_vectors':len(chosen)*PER_DIRECTORY}
                calls=events[start_event:];values=[(e.get('usage') or {}).get('cost') for e in calls]
                record.update(logical_calls=len(calls),known_credits=sum(v for v in values if v is not None),
                  missing_cost_calls=sum(v is None for v in values),call_seconds=[e['latency_seconds'] for e in calls])
                results.append(record)
                write_json(OUT/'results.json',results)
                print(json.dumps(record if 'error_type' in record else {k:record[k] for k in ['size','trial','routing_seconds','without_jev_seconds','with_jev_seconds','logical_calls','known_credits','known_relevant_selected']}),flush=True)
        for db in dbs.values(): db.close()
    engine.directory_selector.client.close()
    if any('error_type' in r for r in results):
        raise SystemExit('One or more trials failed; preserved without rerunning.')


if __name__ == '__main__':
    main()
