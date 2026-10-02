"""One independent real Add and two Search checks, never overwrite results."""
import json
from pathlib import Path
from src.config import ServiceConfig, load_config
from src.contracts import AddRequest, Message, SearchRequest
from src.engine import MemoryEngine
from src.providers import OpenAILLM, OpenAIEmbedder, OpenAIDecision
from src.directory_selector import JevDirectorySelector

out=Path('artifacts/vikingjev/threshold-smoke').resolve()
out.mkdir(exist_ok=False)
config,path=load_config(Path('configs/service/vikingjev.yaml'),ServiceConfig)
config=config.model_copy(update={'database_dir':str(out/'index')})
events=[]
def telemetry(event):
    events.append(event)
    with (out/'model-calls.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
engine=MemoryEngine(config,path,OpenAILLM(config.llm,telemetry),OpenAIEmbedder(config.embedding,telemetry),
    OpenAIDecision(config.consolidation.decision_model,'jev',telemetry),JevDirectorySelector(config.directory_selector,telemetry))
engine.add(AddRequest(request_id='degree',user_id='smoke',session_id='school',messages=[Message(role='user',content='I graduated with a degree in Business Administration.',timestamp=1704067200000)]))
results=[]
for name,threshold in [('global',300000),('topic',0)]:
    engine.config.retrieval.directory_routing_threshold=threshold
    before=len(events)
    result=engine.search(SearchRequest(query='What degree did I graduate with?',user_id='smoke',top_k=20))
    stages=[c['stage'] for c in events[before:]]
    assert ('directory_selection_jev' in stages)==(name=='topic')
    assert 'Business Administration' in '\n'.join(e.content for e in result.data)
    results.append({'mode':name,'threshold':threshold,'stages':stages,'evidence':result.model_dump()})
with engine.store.connect('smoke') as db:
    assignments=[dict(r) for r in db.execute('SELECT topic_id,selected FROM directory_topics')]
values=[(e.get('usage') or {}).get('cost') for e in events]
report={'results':results,'assignments':assignments,'known_credits':sum(v for v in values if v is not None),'missing_cost_calls':sum(v is None for v in values)}
(out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'modes':[r['mode'] for r in results],'known_credits':report['known_credits']}))
