import pytest
from scripts.jev_two_level_validation import expand, route


class Selector:
    def select(self, query, options, parents):
        return [{'id':p['id'],'choice':p['choice'], 'probabilities':{
            key:float(key==p['choice']) for key in ['relevant','irrelevant','uncertain']}} for p in parents]


def test_multibranch_uncertainty_and_broad_queries():
    parents=[{'id':str(i),'choice':'relevant'} for i in range(8)]
    assert len(route(Selector(),'broad',parents)[0])==8
    parents[0]['choice']='uncertain';parents[1]['choice']='irrelevant'
    selected,_=route(Selector(),'cross topic',parents)
    assert '0' in selected and '1' not in selected


def test_membership_deduplicates_multitopic_leaves_and_rejects_unknown():
    assert expand(['a','b'],{'a':['x','y'],'b':['y','z']})==['x','y','z']
    assert expand([],{'a':['x']})==[]
    with pytest.raises(ValueError,match='unknown parent'):expand(['missing'],{'a':['x']})


def test_missing_parent_decision_fails():
    class Missing:
        def select(self,*args):return []
    with pytest.raises(ValueError,match='invalid parent decisions'):
        route(Missing(),'query',[{'id':'a'}])
