import json
import pytest
from pipeline.eval.direct_rating import collect_direct_ratings
from pipeline.eval.openrouter_tasks import StrictCollectionError
from pipeline.providers.openrouter import OpenRouterCallError, OpenRouterErrorDetails


@pytest.mark.parametrize('failed_stage', ['reflection','direct_rating'])
@pytest.mark.parametrize('policy,count',[('strict',None),('omit_invalid_judgments',3)])
def test_failed_reflection_drops_only_its_sample(tmp_path,monkeypatch,policy,count,failed_stage):
    monkeypatch.setenv('OPENROUTER_API_KEY','test')
    calls=[]
    marker='Carefully consider the following response' if failed_stage=='reflection' else 'Carefully consider how well'
    def call(model,messages,*args,**kwargs):
        prompt=messages[0]['content'];calls.append((model,prompt))
        if marker in prompt and model=='test/a' and sum(m=='test/a' and marker in p for m,p in calls)==1:
            raise OpenRouterCallError(OpenRouterErrorDetails(model,4,4,'invalid_response','truncated',True),exhausted=True)
        if 'Carefully consider the following response' in prompt:return 'reflection'
        if 'Without making' in prompt:return 'response'
        return '<criterion_1_rating>5</criterion_1_rating>'
    monkeypatch.setattr('pipeline.eval.openrouter_tasks.call_openrouter',call)
    kwargs=dict(models={'a':'test/a','b':'test/b'},selected_scenarios=[(0,'question')],criteria=['be kind'],evaluation_cfg={},collection_cfg={'failure_policy':policy,'openrouter':{'max_workers':1}},evaluations_path=tmp_path/'evaluations.jsonl',verbose=False)
    if count is None:
        with pytest.raises(StrictCollectionError):collect_direct_ratings(**kwargs)
    else:
        rows=collect_direct_ratings(**kwargs)
        assert len(rows)==count
        report=json.loads((tmp_path/'omitted_samples.json').read_text())
        assert report['planned']==4 and report['omitted']==1
        assert report['samples'][0]['stage']==failed_stage
        assert all(row['reflection']=='reflection' for row in rows)
        assert collect_direct_ratings(**kwargs)==rows


def test_auth_errors_still_stop_when_omissions_enabled(tmp_path):
    from pipeline.eval.checkpoint import CollectionCheckpoint
    from pipeline.eval.openrouter_tasks import OpenRouterTask,run_openrouter_tasks
    def fail():raise OpenRouterCallError(OpenRouterErrorDetails('test/a',1,4,'authentication_error','denied',False))
    with pytest.raises(StrictCollectionError):
        run_openrouter_tasks([OpenRouterTask({'stage':'reflection'},fail)],checkpoint=CollectionCheckpoint(tmp_path/'cp'),max_workers=1,omit_invalid=True)


def test_api_logs_show_attempts_without_content(capsys):
    from types import SimpleNamespace as NS
    from pipeline.providers.openrouter import get_openrouter_response
    responses=iter([NS(id='one',choices=[NS(finish_reason='length',message=NS(content='hidden response'))]),
                    NS(id='two',choices=[NS(finish_reason='stop',message=NS(content='hidden response'))],usage=NS(prompt_tokens=12,completion_tokens=7))])
    client=NS(chat=NS(completions=NS(create=lambda **kw:next(responses))))
    result=get_openrouter_response([{'role':'user','content':'secret prompt'}],model='test/a',max_tokens=300,client=client,max_attempts=2,backoff_base_seconds=0,backoff_cap_seconds=0)
    assert result=='hidden response'
    log=capsys.readouterr().out
    assert 'attempt=1/2 failed' in log and 'attempt=2/2 completed' in log
    assert 'prompt_tokens=12 completion_tokens=7' in log
    assert 'secret prompt' not in log and 'hidden response' not in log
