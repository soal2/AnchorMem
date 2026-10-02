"""Fixed corpus, varying parent-topic count; live Jev plus local vector search."""
import json
from pathlib import Path
from statistics import median
import tempfile
from time import perf_counter
from types import SimpleNamespace

import numpy as np

from scripts.jev_directory_scaling import make_db, write_json, DIMENSION
from src.config import ServiceConfig, load_config
from src.directory_selector import JevDirectorySelector
from src.engine import MemoryEngine
from src.topics import selected_topics

OUT=Path('artifacts/vikingjev/topic-count-validation')
COUNTS=[4,8,16,32,64]
CASES=[('single','What progress did I make on project P05?', [5]),
       ('cross','Compare my progress on projects P05 and P42.',[5,42]),
       ('all','Summarize progress on all my projects P00 through P63.',list(range(64)))]


def main():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'started').open('x').close()
    (OUT/'runner.py').write_bytes(Path(__file__).read_bytes())
    config,_=load_config(Path('configs/service/vikingjev.yaml'),ServiceConfig)
    events=[];context={}
    def telemetry(event):
        item={**context,**event};events.append(item)
        with (OUT/'model-calls.jsonl').open('a') as f:f.write(json.dumps(item)+'\n')
    selector=JevDirectorySelector(config.directory_selector,telemetry)
    engine=object.__new__(MemoryEngine);engine.config=SimpleNamespace(embedding=SimpleNamespace(dimensions=DIMENSION))
    vector=np.random.default_rng(7).normal(size=DIMENSION).astype(np.float32);vector/=np.linalg.norm(vector)
    results=[]
    write_json(OUT/'manifest.json',{'directories':4096,'vectors_per_directory':20,'vectors':81920,'dimensions':DIMENSION,
        'topic_counts':COUNTS,'fine_topics':64,'cases':CASES,'scope':'Fixed synthetic project corpus and vectors. Coarse topics merge contiguous fine topics. Query routing and warm vector microbenchmark only; no summary generation or answer accuracy.'})
    with tempfile.TemporaryDirectory(prefix='jev-topic-count-') as folder:
        db=make_db(Path(folder)/'fixed.sqlite3',4096,20)
        ids=[r[0] for r in db.execute('SELECT id FROM memory_directories ORDER BY id')]
        fine={i:[] for i in range(64)}
        for i,identity in enumerate(ids):
            topic=i%64;fine[topic].append(identity)
            db.execute('UPDATE memory_directories SET abstract=?,overview=? WHERE id=?',
                       (f'Project P{topic:02d}',f'Project P{topic:02d} progress, milestones and personal working notes.',identity))
        db.commit()
        # Different orders for each query reduce confounding by request warm-up.
        for case_index,(name,query,expected) in enumerate(CASES):
            order=[COUNTS,list(reversed(COUNTS)),[16,4,64,8,32]][case_index]
            for count in order:
                width=64//count;parents=[];membership={}
                for i in range(count):
                    key=f't{i:02d}';projects=list(range(i*width,(i+1)*width))
                    description='Progress, milestones and working notes for projects '+', '.join(f'P{x:02d}' for x in projects)+'.'
                    parents.append(dict(id=key,path=f'topics/{key}',abstract=description,overview=description))
                    membership[key]=sorted({leaf for x in projects for leaf in fine[x]})
                context.update(case=name,topics=count);before=len(events);started=perf_counter()
                try:
                    decisions=[]
                    for offset in range(0,count,10):decisions.extend(selector.select(query,None,parents[offset:offset+10]))
                    selected=selected_topics(decisions,list(membership))
                    leaves=sorted({leaf for key in selected for leaf in membership[key]})
                except Exception as error:
                    result={'case':name,'topics':count,'error_type':type(error).__name__}
                else:
                    routing=perf_counter()-started;timings={'global':[],'scoped':[]}
                    for scope in [None,leaves]:engine._vector_rank(db,vector,'chunk',50,scope)
                    for repeat in range(7):
                        for mode in (['global','scoped'] if repeat%2==0 else ['scoped','global']):
                            start=perf_counter();engine._vector_rank(db,vector,'chunk',50,None if mode=='global' else leaves)
                            timings[mode].append(perf_counter()-start)
                    gold={leaf for x in expected for leaf in fine[x]}
                    result={'case':name,'topics':count,'selected_parents':selected,'scoped_vectors':len(leaves)*20,
                        'gold_leaf_coverage':len(gold&set(leaves))/len(gold),'routing_seconds':routing,
                        'global_seconds':median(timings['global']),'scoped_seconds':median(timings['scoped']),
                        'routed_seconds':routing+median(timings['scoped']),'timings':timings,'decisions':decisions}
                calls=events[before:];fees=[(x.get('usage') or {}).get('cost') for x in calls]
                result.update(calls=len(calls),known_credits=sum(v for v in fees if v is not None),missing_cost_calls=sum(v is None for v in fees))
                results.append(result);write_json(OUT/'results.json',results)
                print(json.dumps({k:v for k,v in result.items() if k not in ['timings','decisions']}),flush=True)
        db.close()
    selector.client.close()
    if any('error_type' in x for x in results):raise SystemExit('Failures preserved; no repeats')


if __name__=='__main__':main()
