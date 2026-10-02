"""Two-level structural prototype: bounded parent routing, local leaf vector search."""
import argparse
import json
from pathlib import Path
from statistics import median
import tempfile
from time import perf_counter
from types import SimpleNamespace

import numpy as np

from scripts.jev_directory_scaling import make_db, write_json, DIMENSION, PER_DIRECTORY
from src.config import ServiceConfig, load_config
from src.directory_selector import CRITERIA, JevDirectorySelector
from src.engine import MemoryEngine
from src.providers import validate_choice_probabilities

OUT = Path('artifacts/vikingjev/two-level-validation')
TOPICS = {
    'education': 'Degrees, university studies, courses, graduation and educational history.',
    'work': 'Employment, employers, job roles, career changes and workplace projects.',
    'health': 'Medical history, allergies, dietary restrictions, exercise and health precautions.',
    'travel': 'Trips, destinations, itineraries, travel bookings and journey plans.',
    'finance': 'Personal budgets, savings, expenses, payments and financial plans.',
    'family': 'Relatives, household relationships, family events and responsibilities.',
    'hobbies': 'Music, gardening, photography and other leisure interests.',
    'housing': 'Residences, moving home, rent and home maintenance.',
}
CASES = [
    ('degree', 'What degree did I graduate with?', ['education']),
    ('job', 'Which company do I work for now?', ['work']),
    ('diet', 'Which foods should I avoid because of my allergies?', ['health']),
    ('career', 'How did my university degree relate to the job I took afterwards?', ['education','work']),
    ('trip', 'What health precautions do I need for my planned trip?', ['health','travel']),
    ('broad', 'Summarize changes in my education, work, health, travel, finances, family, hobbies and housing.', list(TOPICS)),
]


def route(selector, query, parents):
    decisions = selector.select(query, None, parents)
    if len(decisions) != len(parents) or {d['id'] for d in decisions} != {p['id'] for p in parents}:
        raise ValueError('invalid parent decisions')
    for d in decisions:
        validate_choice_probabilities(d['choice'], d['probabilities'], CRITERIA)
    # Broad questions may need all branches. No arbitrary top-k cap at the parent level.
    selected = [d['id'] for d in decisions if d['probabilities']['relevant'] >= .5
                or d['probabilities']['uncertain'] >= .5]
    return selected, decisions


def expand(selected, membership):
    if not set(selected).issubset(membership):
        raise ValueError('unknown parent')
    return sorted({leaf for parent in selected for leaf in membership[parent]})


def structure(db):
    membership = {topic: [] for topic in TOPICS}
    leaves = [r[0] for r in db.execute('SELECT id FROM memory_directories ORDER BY id')]
    for i, leaf in enumerate(leaves):
        topic = list(TOPICS)[i % len(TOPICS)]
        membership[topic].append(leaf)
        db.execute('UPDATE memory_directories SET path=?,abstract=?,overview=? WHERE id=?',
                   (f'{topic}/{leaf}', TOPICS[topic], f'Personal records in {topic}. {TOPICS[topic]}', leaf))
    db.commit()
    parents = [{'id':topic,'path':f'memories/{topic}','abstract':description,
                'overview':f'This branch contains personal records about: {description}'} for topic,description in TOPICS.items()]
    return parents, membership


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes', type=int, nargs='+', default=[10,100,300])
    parser.add_argument('--vectors-per-leaf', type=int, default=PER_DIRECTORY)
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    if any(size < len(TOPICS) for size in args.sizes) or args.vectors_per_leaf < 1:
        parser.error('Need at least eight directories and a positive vector count')
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    (out/'started').open('x').close()
    (out/'runner.py').write_bytes(Path(__file__).read_bytes())
    config,_ = load_config(Path('configs/service/vikingjev.yaml'), ServiceConfig)
    events=[];context={}
    def telemetry(event):
        item={**context,**event};events.append(item)
        with (out/'model-calls.jsonl').open('a') as f:f.write(json.dumps(item)+'\n')
    selector=JevDirectorySelector(config.directory_selector,telemetry)
    engine=object.__new__(MemoryEngine)
    engine.config=SimpleNamespace(embedding=SimpleNamespace(dimensions=DIMENSION))
    query_vector=np.random.default_rng(7).normal(size=DIMENSION).astype(np.float32)
    query_vector/=np.linalg.norm(query_vector)
    write_json(out/'manifest.json',{'sizes':args.sizes,'parents':TOPICS,'cases':CASES,
        'vectors_per_leaf':args.vectors_per_leaf,'dimensions':DIMENSION,'selection':'relevant >= .5 or uncertain >= .5',
        'scope':'Synthetic manual taxonomy; branch coverage and warm local chunk-vector timing; not QA accuracy or learned taxonomy validation.'})
    results=[]
    with tempfile.TemporaryDirectory(prefix='jev-two-level-') as folder:
        for size in args.sizes:
            db_path=Path(folder)/f'{size}.sqlite3'
            print(f'Building {size} directories, {size*args.vectors_per_leaf} vectors',flush=True)
            db=make_db(db_path,size,args.vectors_per_leaf)
            parents,membership=structure(db)
            write_json(out/f'structure-{size}.json',{'parents':parents,'membership':membership})
            for name,query,expected in CASES:
                context.update(size=size,case=name);before=len(events);started=perf_counter()
                try:
                    selected,decisions=route(selector,query,parents)
                    leaves=expand(selected,membership)
                except Exception as error:
                    result={'size':size,'case':name,'error_type':type(error).__name__}
                else:
                    routing=perf_counter()-started
                    timings={'global':[],'scoped':[]}
                    for scope in [None,leaves]:engine._vector_rank(db,query_vector,'chunk',50,scope)
                    for iteration in range(7):
                        for mode in (['global','scoped'] if iteration%2==0 else ['scoped','global']):
                            start=perf_counter()
                            engine._vector_rank(db,query_vector,'chunk',50,None if mode=='global' else leaves)
                            timings[mode].append(perf_counter()-start)
                    gold=set(expand(expected,membership))
                    result={'size':size,'case':name,'query':query,'expected_parents':expected,'selected_parents':selected,
                        'selected_leaves':len(leaves),'gold_leaf_coverage':len(gold&set(leaves))/len(gold),
                        'parent_precision':len(set(expected)&set(selected))/len(selected) if selected else 0,
                        'global_vectors':size*args.vectors_per_leaf,'scoped_vectors':len(leaves)*args.vectors_per_leaf,
                        'routing_seconds':routing,'without_jev_seconds':median(timings['global']),
                        'with_jev_seconds':routing+median(timings['scoped']),
                        'local_seconds':timings,'decisions':decisions}
                calls=events[before:];fees=[(c.get('usage') or {}).get('cost') for c in calls]
                result.update(logical_calls=len(calls),known_credits=sum(v for v in fees if v is not None),missing_cost_calls=sum(v is None for v in fees))
                results.append(result);write_json(out/'results.json',results)
                print(json.dumps({k:v for k,v in result.items() if k not in ['decisions','local_seconds']}),flush=True)
            db.close()
            db_path.unlink()
    selector.client.close()
    if any('error_type' in r for r in results):raise SystemExit('Failed cases preserved without rerunning')


if __name__=='__main__':main()
